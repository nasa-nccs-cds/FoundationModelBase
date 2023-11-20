import xarray as xa
import numpy as np
from fmbase.util.config import cfg
from typing import List, Union, Tuple, Optional, Dict, Type
import glob, sys, os, time
from xarray.core.resample import DataArrayResample
from fmbase.util.ops import get_levels_config, increasing
from fmbase.source.merra2.model import variable_cache_filepath, fmbdir
np.set_printoptions(precision=3, suppress=False, linewidth=150)
from enum import Enum

def dump_dset( name: str, dset: xa.Dataset ):
    print( f"\n ---- dump_dset {name}:")
    for vname, vdata in dset.data_vars.items():
        print( f"  ** {vname}{vdata.dims}-> {vdata.shape} ")

class QType(Enum):
    Intensive = 'intensive'
    Extensive = 'extensive'

class StatsEntry:

    def __init__(self, varname: str ):
        self._stats: Dict[str,List[xa.DataArray]] = {}
        self._varname = varname

    def merge(self, entry: "StatsEntry"):
        for statname, mvars in entry._stats.items():
            for mvar in mvars:
                self.add( statname, mvar )

    def add(self, statname: str, mvar: xa.DataArray, weight: int = None ):
        if weight is not None: mvar.attrs['stat_weight'] = float(weight)
        elist = self._stats.setdefault(statname,[])
        elist.append( mvar )
#        print( f" SSS: Add stats entry[{self._varname}.{statname}]: dims={mvar.dims}, shape={mvar.shape}, size={mvar.size}, ndim={mvar.ndim}, weight={weight}")
#        if mvar.ndim > 0:  print( f"      --> sample: {mvar.values[0:8]}")
#        else:              print( f"      --> sample: {mvar.values}")

    def entries( self, statname: str ) -> Optional[List[xa.DataArray]]:
        return self._stats.get(statname)

class StatsAccumulator:
    statnames = ["mean", "std", "std_diff"]

    def __init__(self):
        self._entries: Dict[str, StatsEntry] = {}

    @property
    def entries(self) -> Dict[str, StatsEntry]:
        return self._entries

    def entry(self, varname: str) -> StatsEntry:
        return self._entries.setdefault(varname,StatsEntry(varname))

    @property
    def varnames(self):
        return self._entries.keys()

    def add_entry(self, varname: str, mvar: xa.DataArray):
        istemporal = "time" in mvar.dims
        first_entry = varname not in self._entries
        dims = ['time', 'y', 'x'] if istemporal else ['y', 'x']
        weight =  mvar.shape[0] if istemporal else 1
        if istemporal or first_entry:
            mean: xa.DataArray = mvar.mean(dim=dims, skipna=True, keep_attrs=True)
            std: xa.DataArray = mvar.std(dim=dims, skipna=True, keep_attrs=True)
            entry: StatsEntry= self.entry( varname)
            entry.add( "mean", mean, weight )
            entry.add("std",  std, weight )
            if istemporal:
                mvar_diff: xa.DataArray = mvar.diff("time")
                weight = mvar.shape[0]
                mean_diff: xa.DataArray = mvar_diff.mean( dim=dims, skipna=True, keep_attrs=True )
                std_diff: xa.DataArray  = mvar_diff.std(  dim=dims, skipna=True, keep_attrs=True )
                entry: StatsEntry = self.entry( varname)
                entry.add("mean_diff", mean_diff, weight )
                entry.add("std_diff",  std_diff,  weight )

    def accumulate(self, statname: str ) -> xa.Dataset:
        accum_stats = {}
        coords = {}
        for varname in self.varnames:
            varstats: StatsEntry = self._entries[varname]
            entries: Optional[List[xa.DataArray]] = varstats.entries( statname )
            squared = statname.startswith("std")
            if entries is not None:
                esum, wsum = None, 0
                for entry in entries:
                    w = entry.attrs['stat_weight']
                    eterm = w*entry*entry if squared else w*entry
                    esum = eterm if (esum is None) else esum + eterm
                    wsum = wsum + w
                astat = np.sqrt( esum/wsum ) if squared else esum/wsum
                accum_stats[varname] = astat
                coords.update( astat.coords )
        return xa.Dataset( accum_stats, coords )

    def save( self, statname: str, filepath: str ):
        os.makedirs(os.path.dirname(filepath), mode=0o777, exist_ok=True)
        accum_stats: xa.Dataset = self.accumulate(statname)
        accum_stats.to_netcdf( filepath )
        print(f" SSS: Save stats[{statname}] to {filepath}: {list(accum_stats.data_vars.keys())}")
        for vname, vstat in accum_stats.data_vars.items():
            print(f"   >> Entry[{statname}.{vname}]: dims={vstat.dims}, shape={vstat.shape}")
            if vstat.ndim > 0:  print(f"      --> sample: {vstat.values[0:8]}")
            else:               print(f"      --> sample: {vstat.values}")


class MERRA2DataProcessor:

    def __init__(self):
        self.xext, self.yext = cfg().preprocess.get('xext'), cfg().preprocess.get('yext')
        self.xres, self.yres = cfg().preprocess.get('xres'), cfg().preprocess.get('yres')
        self.levels: Optional[np.ndarray] = get_levels_config( cfg().preprocess )
        self.tstep = cfg().preprocess.tstep
        self.month_range = cfg().preprocess.get('month_range',[0,12,1])
        self.vars: Dict[str, List[str]] = cfg().preprocess.vars
        self.dmap: Dict = cfg().preprocess.dims
        self.var_file_template =  cfg().platform.dataset_files
        self.const_file_template =  cfg().platform.constant_file
        self.stats = StatsAccumulator()

    @classmethod
    def get_qtype( cls, vname: str) -> QType:
        extensive_vars = cfg().preprocess.get('extensive',[])
        return QType.Extensive if vname in extensive_vars else QType.Intensive

    def merge_stats( self, stats: List[StatsAccumulator] = None ):
        for stats_accum in ([] if stats is None else stats):
            for varname, new_entry in stats_accum.entries.items():
                entry: StatsEntry = self.stats.entry(varname)
                entry.merge( new_entry )

    def save_stats(self, ext_stats: List[StatsAccumulator]=None ):
        self.merge_stats( ext_stats )
        for statname in self.stats.statnames:
            filepath = stats_filepath( cfg().preprocess.version, statname )
            self.stats.save( statname, filepath )

    def get_monthly_files(self, year) -> Dict[ Tuple[str,int], Tuple[List[str],List[str]] ]:
        months: List[int] = list(range(*self.month_range))
        dsroot: str = fmbdir('dataset_root')
        assert "{year}" in self.var_file_template, "{year} field missing from platform.cov_files parameter"
        dset_files: Dict[ Tuple[str,int], Tuple[List[str],List[str]] ] = {}
        assert "{month}" in self.var_file_template, "{month} field missing from platform.cov_files parameter"
        print( f"get_monthly_files({year})-> months: {months}:")
        for month in months:
            for collection, vlist in self.vars.items():
                if collection.startswith("const"): dset_template: str = self.const_file_template.format( collection=collection )
                else:                              dset_template: str = self.var_file_template.format(   collection=collection, year=year, month=f"{month + 1:0>2}")
                dset_paths: str = f"{dsroot}/{dset_template}"
                gfiles: List[str] = glob.glob(dset_paths)
                print( f" ** M{month}: Found {len(gfiles)} files for glob {dset_paths}, template={self.var_file_template}, root dir ={dsroot}")
                dset_files[(collection,month)] = (gfiles, vlist)
        return dset_files

    def process_year(self, year: int, **kwargs ):
        dset_files: Dict[Tuple[str, int], Tuple[List[str], List[str]]] = self.get_monthly_files(year)
        for (collection, month), (dfiles, dvars) in dset_files.items():
            print(f" -- -- Procesing collection {collection} for month {month}/{year}: {len(dset_files)} files, {len(dvars)} vars")
            t0 = time.time()
            for dvar in dvars:
                self.process_subsample(collection, dvar, dfiles, year=year, month=month, **kwargs)
            print(f" -- -- Processed {len(dset_files)} files for month {month}/{year}, time = {(time.time() - t0) / 60:.2f} min")
        return self.stats

    @classmethod
    def get_varnames(cls, dset_file: str) -> List[str]:
        with xa.open_dataset(dset_file) as dset:
            return list(dset.data_vars.keys())

    def subsample_coords(self, dvar: xa.DataArray ) -> Dict[str,np.ndarray]:
        subsample_coords = {}
        if (self.levels is not None) and ('z' in dvar.dims):
            subsample_coords['z'] = self.levels
        if self.xres is not None:
            if self.xext is  None:
                xc0 = dvar.coords['x'].values
                self.xext = [ xc0[0], xc0[-1]+self.xres/2 ]
            subsample_coords['x'] = np.arange(self.xext[0],self.xext[1],self.xres)
        elif self.xext is not None:
            subsample_coords['x'] = slice(self.xext[0], self.xext[1])

        if self.yres is not None:
            if self.yext is  None:
                yc0 = dvar.coords['y'].values
                self.yext = [ yc0[0], yc0[-1]+self.yres/2 ]
            subsample_coords['y'] = np.arange(self.yext[0],self.yext[1],self.yres)
        elif self.yext is not None:
            subsample_coords['y'] = slice(self.yext[0], self.yext[1])
        return subsample_coords


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

    def subsample(self, variable: xa.DataArray, global_attrs: Dict, qtype: QType) -> xa.DataArray:
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
        else:
            resampled: DataArrayResample = varray.resample(time=self.tstep)
            newvar: xa.DataArray = resampled.mean() if qtype == QType.Intensive else resampled.sum()

        newvar.attrs.update(global_attrs)
        newvar.attrs.update(varray.attrs)
        print( f" >> NEW: shape={newvar.shape}, dims={newvar.dims}" ) # , attrs={newvar.attrs}")
        for missing in [ 'fmissing_value', 'missing_value', 'fill_value' ]:
            if missing in newvar.attrs:
                missing_value = newvar.attrs.pop('fmissing_value')
                return newvar.where( newvar != missing_value, np.nan )
        return newvar

    def process_subsample(self, collection: str, dvar: str, files: List[str], **kwargs):
        filepath: str = variable_cache_filepath( cfg().preprocess.version, dvar, **kwargs )
        reprocess: bool = kwargs.pop( 'reprocess', False )
        if (not os.path.exists(filepath)) or reprocess:
            print(f" ** Processing variable {dvar} in collection {collection}, args={kwargs}: {len(files)} files")
            t0 = time.time()
            samples: List[xa.DataArray] = []
            for file in sorted(files):
                dset: xa.Dataset = xa.open_dataset(file)
                dset_attrs = dict( collection=collection, **dset.attrs, **kwargs )
                qtype: QType = self.get_qtype(dvar)
                print( f"Processing {qtype.value} var {dvar} from file {file}")
                samples.append( self.subsample( dset.data_vars[dvar], dset_attrs, qtype) )
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

def stats_filepath( version: str, statname: str ) -> str:
    return f"{fmbdir('processed')}/{version}/stats/{statname}.nc"

def load_stats( task_config: Dict , statname: str, **kwargs ) -> xa.Dataset:
    version = task_config['dataset_version']
    filepath = stats_filepath(version,statname)
    varstats: xa.Dataset = xa.open_dataset(filepath,**kwargs)
    model_varname_map = { v: k for k, v in task_config['input_variables'].items() if v in varstats.data_vars }
    model_coord_map   = { k: v for k, v in task_config['coords'].items() if k in varstats.coords }
    print( f" load_stats:\n   model_coord_map = {model_coord_map}\n   varstats coords = {list(varstats.coords.keys())}")
    return varstats.rename( **model_varname_map, **model_coord_map )
def load_norm_data( task_config: Dict ) -> Dict[str,xa.Dataset]:     #     version = cfg().task.dataset_version
    model_statnames: Dict[str,str] = task_config.get( 'statnames' )
    stats = { model_statnames[statname]: load_stats(task_config,statname) for statname in StatsAccumulator.statnames }
    print( f" \n ---------------->>>> load_norm_data:     ")
    for sname, sdata in stats.items():
        dump_dset(sname,sdata)
    return stats
