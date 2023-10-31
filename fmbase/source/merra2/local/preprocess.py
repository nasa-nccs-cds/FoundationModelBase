import xarray as xa
import numpy as np
from fmbase.util.config import cfg
from typing import List, Union, Tuple, Optional, Dict, Type
import glob, sys, os, time
from fmbase.source.merra2.base import MERRA2Base
from fmbase.util.ops import get_levels_config, increasing
from enum import Enum


class StatsEntry:

    def __init__(self, varname: str ):
        self._stats: Dict[str,List[xa.DataArray]] = {}
        self._varname = varname

    def add(self, statname: str, mvar: xa.DataArray, weight: int ):
        mvar.attrs['stat_weight'] = float(weight)
        elist = self._stats.setdefault(statname,[])
        elist.append( mvar )
        print( f"Add stats entry[{self._varname}.{statname}]: dims={mvar.dims}, shape={mvar.shape}, weight={weight}")

    def entries( self, statname: str ) -> Optional[List[xa.DataArray]]:
        return self._stats.get(statname)

class StatsAccumulator:

    def __init__(self):
        self._entries: Dict[str, StatsEntry] = {}

    def _entry(self, varname: str ) -> StatsEntry:
        entry: StatsEntry = self._entries.setdefault(varname,StatsEntry())
        return entry

    @property
    def varnames(self):
        return self._entries.keys()

    def add_entry(self, varname: str, mvar: xa.DataArray):
        istemporal = "time" in mvar.dims
        first_entry = varname not in self._entries
        dims = ['time', 'y', 'x'] if istemporal else ['y', 'x']
        weight =  mvar.shape[0] if istemporal else 1
        if istemporal or first_entry:
            location: xa.DataArray = mvar.mean(dim=dims, skipna=True, keep_attrs=True)
            scale: xa.DataArray = mvar.std(dim=dims, skipna=True, keep_attrs=True)
            entry: StatsEntry= self._entry( varname )
            entry.add( "location", location, weight )
            entry.add("scale",  scale, weight )

    def accumulate(self, varname: str ) -> xa.Dataset:
        varstats: StatsEntry = self._entries[varname]
        accum_stats = {}
        coords = {}
        for statname in ["location","scale"]:
            entries: Optional[List[xa.DataArray]] = varstats.entries( statname )
            squared = (statname == "scale")
            if entries is not None:
                esum, wsum = None, 0
                for entry in entries:
                    w = entry.attrs['stat_weight']
                    eterm = w*entry*entry if squared else w*entry
                    esum = eterm if (esum is None) else esum + eterm
                    wsum = wsum + w
                astat = np.sqrt( esum/wsum ) if squared else esum/wsum
                accum_stats[statname] = astat
                coords.update( astat.coords )
        return xa.Dataset( accum_stats, coords )

    def save( self, varname: str, filepath: str ):
        os.makedirs(os.path.dirname(filepath), mode=0o777, exist_ok=True)
        accum_stats: xa.Dataset = self.accumulate(varname)
        accum_stats.to_netcdf( filepath )


class MERRA2DataProcessor(MERRA2Base):

    def __init__(self):
        MERRA2Base.__init__( self )
        self.xext, self.yext = cfg().preprocess.get('xext'), cfg().preprocess.get('yext')
        self.xres, self.yres = cfg().preprocess.get('xres'), cfg().preprocess.get('yres')
        self.levels: Optional[np.ndarray] = get_levels_config( cfg().preprocess )
        self.tstep = cfg().preprocess.tstep
        self.month_range = cfg().preprocess.get('month_range',[0,12,1])
        self.year_range = cfg().preprocess.year_range
        self.vars: Dict[str, List[str]] = cfg().preprocess.vars
        self.dmap: Dict = cfg().preprocess.dims
        self.var_file_template = cfg().platform.dataset_files
        self.const_file_template = cfg().platform.constant_file
        self._subsample_coords: Dict[str,np.ndarray] = None
        self.stats = StatsAccumulator()

    def save_stats(self):
        for varname in self.stats.varnames:
            self.stats.save( varname, self.stats_filepath(varname) )

    def get_monthly_files(self, year) -> Dict[ Tuple[str,int], Tuple[List[str],List[str]] ]:
        months: List[int] = list(range(*self.month_range))
        assert "{year}" in self.var_file_template, "{year} field missing from platform.cov_files parameter"
        dset_files: Dict[ Tuple[str,int], Tuple[List[str],List[str]] ] = {}
        assert "{month}" in self.var_file_template, "{month} field missing from platform.cov_files parameter"
        for month in months:
            for collection, vlist in self.vars.items():
                if collection.startswith("const"): dset_template: str = self.const_file_template.format( collection=collection )
                else:                              dset_template: str = self.var_file_template.format(   collection=collection, year=year, month=f"{month + 1:0>2}")
                dset_paths: str = f"{self.data_dir}/{dset_template}"
                gfiles: List[str] = glob.glob(dset_paths)
                print( f" ** M{month}: Found {len(gfiles)} files for glob {dset_paths}, template={self.var_file_template}, root dir ={self.data_dir}")
                dset_files[(collection,month)] = (gfiles, vlist)
        return dset_files

    def process(self, **kwargs):
        years = list(range( *self.year_range ))
        for year in years:
            dset_files: Dict[ Tuple[str,int], Tuple[List[str],List[str]] ] = self.get_monthly_files( year )
            for (collection,month), (dfiles,dvars) in dset_files.items():
                t0 = time.time()
                for dvar in dvars:
                    self.process_subsample( collection, dvar, dfiles, year=year, month=month, **kwargs )
                print(f" -- -- Processed {len(dset_files)} files for month {month}/{year}, time = {(time.time()-t0)/60:.2f} min")

    @classmethod
    def get_varnames(cls, dset_file: str) -> List[str]:
        with xa.open_dataset(dset_file) as dset:
            return list(dset.data_vars.keys())

    def subsample_coords(self, dvar: xa.DataArray ) -> Dict[str,np.ndarray]:
        if self._subsample_coords is None:
            self._subsample_coords = {}
            if (self.levels is not None) and ('z' in dvar.dims):
                self._subsample_coords['z'] = self.levels
            if self.xres is not None:
                if self.xext is  None:
                    xc0 = dvar.coords['x'].values
                    self.xext = [ xc0[0], xc0[-1]+self.xres/2 ]
                self._subsample_coords['x'] = np.arange(self.xext[0],self.xext[1],self.xres)
            elif self.xext is not None:
                self._subsample_coords['x'] = slice(self.xext[0], self.xext[1])

            if self.yres is not None:
                if self.yext is  None:
                    yc0 = dvar.coords['y'].values
                    self.yext = [ yc0[0], yc0[-1]+self.yres/2 ]
                self._subsample_coords['y'] = np.arange(self.yext[0],self.yext[1],self.yres)
            elif self.yext is not None:
                self._subsample_coords['y'] = slice(self.yext[0], self.yext[1])
        return self._subsample_coords


    def subsample_1d(self, variable: xa.DataArray, global_attrs: Dict ) -> xa.DataArray:
        cmap: Dict[str,str] = { cn0:cn1 for (cn0,cn1) in self.dmap.items() if cn0 in list(variable.coords.keys()) }
        varray: xa.DataArray = variable.rename(**cmap)
        scoords: Dict[str,np.ndarray] = self.subsample_coords( varray )
        newvar: xa.DataArray = varray
        print(f" **** subsample {variable.name}, dims={varray.dims}, shape={varray.shape}")
        for cname, cval in scoords.items():
            if cname == 'z':
                newvar: xa.DataArray = newvar.interp(**{cname: cval}, assume_sorted=increasing(cval))
                print(f" >> zdata: {varray.coords['z'].values.tolist()}" )
                print(f" >> zconf: {cval.tolist()}")
                print(f" >> znewv: {newvar.coords['z'].values.tolist()}" )
            newvar.attrs.update( global_attrs )
            newvar.attrs.update( varray.attrs )
        return newvar.where( newvar != newvar.attrs['fmissing_value'], np.nan )

    def subsample(self, variable: xa.DataArray, global_attrs: Dict) -> xa.DataArray:
        cmap: Dict[str, str] = {cn0: cn1 for (cn0, cn1) in self.dmap.items() if cn0 in list(variable.coords.keys())}
        varray: xa.DataArray = variable.rename(**cmap)
        tattrs: Dict = variable.coords['time'].attrs
        scoords: Dict[str, np.ndarray] = self.subsample_coords(varray)
        print(f" **** subsample {variable.name}, dims={varray.dims}, shape={varray.shape}, new sizes: { {cn:cv.size for cn,cv in scoords.items()} }")

        zsorted = ('z' not in varray.coords) or increasing(varray.coords['z'].values)
        varray = varray.interp(**scoords, assume_sorted=zsorted)

        monthly = (tattrs['time_increment'] > 7000000) and (variable.shape[0] == 12)
        if   variable.shape[0] == 1:    newvar: xa.DataArray = varray
        elif monthly:                   newvar: xa.DataArray = varray.isel( time=global_attrs['month'] )
        else:                           newvar: xa.DataArray = varray.resample(time=self.tstep).mean()

        newvar.attrs.update(global_attrs)
        newvar.attrs.update(varray.attrs)
        print( f" >> NEW: shape={newvar.shape}, dims={newvar.dims}" ) # , attrs={newvar.attrs}")
        for missing in [ 'fmissing_value', 'missing_value', 'fill_value' ]:
            if missing in newvar.attrs:
                missing_value = newvar.attrs.pop('fmissing_value')
                return newvar.where( newvar != missing_value, np.nan )
        return newvar

    def process_subsample(self, collection: str, dvar: str, files: List[str], **kwargs):
        filepath: str = self.variable_cache_filepath(dvar, **kwargs)
        reprocess: bool = kwargs.pop( 'reprocess', True )
        if (not os.path.exists(filepath)) or reprocess:
            print(f" ** Processing variable {dvar} in collection {collection}, args={kwargs}: {len(files)} files")
            t0 = time.time()
            samples: List[xa.DataArray] = []
            for file in sorted(files):
                dset: xa.Dataset = xa.open_dataset(file)
                dset_attrs = dict( collection=collection, **dset.attrs, **kwargs )
                print( f"Processing var {dvar} from file {file}")
                samples.append( self.subsample( dset.data_vars[dvar], dset_attrs ) )
            if len(samples) == 0:
                print( f"Found no files for variable {dvar} in collection {collection}")
            else:
                t1 = time.time()
                if len(samples) > 1:  mvar: xa.DataArray = xa.concat( samples, dim="time" )
                else:                 mvar: xa.DataArray = samples[0]
                print(f"Saving Merged var {dvar}: shape= {mvar.shape}, dims= {mvar.dims}")
                self.stats.add_entry( dvar, mvar )
                os.makedirs(os.path.dirname(filepath), mode=0o777, exist_ok=True)
                mvar.to_netcdf( filepath, format="NETCDF4" )
                print(f" ** ** ** Saved variable {dvar} to file= {filepath} in time = {time.time()-t1} sec")
                print(f"  Completed processing in time = {(time.time()-t0)/60} min")
        else:
            print( f" ** Skipping var {dvar:12s} in collection {collection:12s} due to existence of processed file {filepath}")
