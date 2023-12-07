import xarray as xa, pandas as pd
import numpy as np
from fmbase.util.config import cfg, Date
from typing import List, Union, Tuple, Optional, Dict, Type, Any
import glob, sys, os, time, traceback
from xarray.core.resample import DataArrayResample
from fmbase.util.ops import get_levels_config, increasing, replace_nans
from fmbase.source.merra2.model import cache_var_filepath, cache_const_filepath, fmbdir, add_derived_vars
np.set_printoptions(precision=3, suppress=False, linewidth=150)
from enum import Enum

def nnan(varray: xa.DataArray) -> int: return np.count_nonzero(np.isnan(varray.values))

def nodata_test(varray: xa.DataArray, vname: str, date: Date):
    num_nodata = nnan(varray)
    assert num_nodata == 0, f"ERROR: {num_nodata} Nodata values found in variable {vname} for date {date}"
def nmissing(varray: xa.DataArray) -> int:
    mval = varray.attrs.get('fmissing_value',-9999)
    return np.count_nonzero(varray.values == mval)
def pctnan(varray: xa.DataArray) -> str: return f"{nnan(varray) * 100.0 / varray.size:.2f}%"
def pctmissing(varray: xa.DataArray) -> str:
    return f"{nmissing(varray) * 100.0 / varray.size:.2f}%"

def dump_dset( name: str, dset: xa.Dataset ):
    print( f"\n ---- dump_dset {name}:")
    for vname, vdata in dset.data_vars.items():
        print( f"  ** {vname}{vdata.dims}-> {vdata.shape} ")

def get_day_from_filename( filename: str ) -> int:
    sdate = filename.split(".")[-2]
    return int(sdate[-2:])

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
                times: List[str] = [str(pd.Timestamp(dt64)) for dt64 in mvar.coords['time'].values.tolist()]

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

class DailyFiles:

    def __init__(self, collection: str, variables: List[str], day: int, month: int, year: int ):
        self.collection = collection
        self.vars = variables
        self.day = day
        self.month = month
        self.year = year
        self.files = []

    def add(self, file: str ):
        self.files.append( file )

class MERRA2DataProcessor:

    def __init__(self):
        self.xext, self.yext = cfg().preprocess.get('xext'), cfg().preprocess.get('yext')
        self.xres, self.yres = cfg().preprocess.get('xres'), cfg().preprocess.get('yres')
        self.levels: Optional[np.ndarray] = get_levels_config( cfg().preprocess )
        self.tstep = cfg().preprocess.tstep
        self.month_range = cfg().preprocess.get('month_range',[0,12,1])
        self.vars: Dict[str, List[str]] = cfg().preprocess.vars
        self.dmap: Dict = cfg().preprocess.dims
        self.corder = ['t','z','y','x']
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

    def get_monthly_files(self, year: int, month: int) -> Dict[ str, Tuple[List[str],List[str]] ]:
        dsroot: str = fmbdir('dataset_root')
        assert "{year}" in self.var_file_template, "{year} field missing from platform.cov_files parameter"
        dset_files: Dict[str, Tuple[List[str],List[str]] ] = {}
        assert "{month}" in self.var_file_template, "{month} field missing from platform.cov_files parameter"
        for collection, vlist in self.vars.items():
            if collection.startswith("const"): dset_template: str = self.const_file_template.format( collection=collection )
            else:                              dset_template: str = self.var_file_template.format(   collection=collection, year=year, month=f"{month + 1:0>2}")
            dset_paths: str = f"{dsroot}/{dset_template}"
            gfiles: List[str] = glob.glob(dset_paths)
#            print( f" ** M{month}: Found {len(gfiles)} files for glob {dset_paths}, template={self.var_file_template}, root dir ={dsroot}")
            dset_files[collection] = (gfiles, vlist)
        return dset_files

    def get_daily_files(self, date: Date ) -> Tuple[ Dict[str, Tuple[str, List[str]]], Dict[str, Tuple[str, List[str]]] ]:
        dsroot: str = fmbdir('dataset_root')
        dset_files:  Dict[str, Tuple[str, List[str]]] = {}
        const_files: Dict[str, Tuple[str, List[str]]] = {}
        for collection, vlist in self.vars.items():
            isconst = collection.startswith("const")
            if isconst : fpath: str = self.const_file_template.format(collection=collection)
            else:        fpath: str = self.var_file_template.format(collection=collection, **date.skw )
            file_path = f"{dsroot}/{fpath}"
            if os.path.exists( file_path ):
                dset_list = const_files if isconst else dset_files
                dset_list[collection] = (file_path, vlist)
        return dset_files, const_files

    def process_day(self, date: Date, **kwargs):
        reprocess: bool = kwargs.pop('reprocess', False)
        cache_fvpath: str = cache_var_filepath(cfg().preprocess.version, date)
        os.makedirs(os.path.dirname(cache_fvpath), mode=0o777, exist_ok=True)
        if (not os.path.exists(cache_fvpath)) or reprocess:
            cache_fcpath: str = cache_const_filepath(cfg().preprocess.version)
            dset_files, const_files = self.get_daily_files(date)
            ncollections = len(dset_files.keys())
            if ncollections == 0:
                print( f"No collections found for date {date}")
            else:
                collection_dsets: List[xa.Dataset] = []
                for collection, (file_path, dvars) in dset_files.items():
                    collection_dset: xa.Dataset = self.load_collection(  collection, file_path, dvars, date, **kwargs )
                    if collection_dset is not None: collection_dsets.append(collection_dset)
                if len(collection_dsets) > 0:
                    xa.merge(collection_dsets).to_netcdf(cache_fvpath, format="NETCDF4")
                    print(f" >> Saving collection data for {date} to file '{cache_fvpath}'")
                else:
                    print(f" >> No collection data found for date {date}")

                if not os.path.exists(cache_fcpath):
                    const_dsets: List[xa.Dataset] = []
                    for collection, (file_path, dvars) in const_files.items():
                        collection_dset: xa.Dataset = self.load_collection(  collection, file_path, dvars, date, isconst=True, **kwargs )
                        if collection_dset is not None: const_dsets.append( collection_dset )
                    if len( const_dsets ) > 0:
                        xa.merge(const_dsets).to_netcdf(cache_fcpath, format="NETCDF4")
                        print(f" >> Saving const data to file '{cache_fcpath}'")
                    else:
                        print(f" >> No constant data found")
        else:
            print( f" ** Skipping date {date} due to existence of processed file '{cache_fvpath}'")

    def load_collection(self, collection, file_path: str, dvars: List[str], date: Date, **kwargs) -> Optional[xa.Dataset]:
        dset: xa.Dataset = xa.open_dataset(file_path)
        isconst = kwargs.pop( 'isconst', False )
        dset_attrs = dict(collection=collection, **dset.attrs, **kwargs)
        mvars = {}
        for dvar in dvars:
            darray: xa.DataArray = dset.data_vars[dvar]
            qtype: QType = self.get_qtype(dvar)
            mvar: xa.DataArray = self.subsample( darray, dset_attrs, qtype, isconst )
            self.stats.add_entry(dvar, mvar)
            print(f" ** Processing variable {dvar}{mvar.dims}: {mvar.shape} for {date}")
            mvars[dvar] = mvar
        dset.close()
        if len( mvars ) > 0:
            result = xa.Dataset(mvars)
            if not isconst:
                add_derived_vars(result)
            return result

    @classmethod
    def get_varnames(cls, dset_file: str) -> List[str]:
        with xa.open_dataset(dset_file) as dset:
            return list(dset.data_vars.keys())

    def subsample_coords(self, dvar: xa.DataArray ) -> Dict[str,np.ndarray]:
        subsample_coords: Dict[str,Any] = {}
        if (self.levels is not None) and ('z' in dvar.dims):
            subsample_coords['z'] = self.levels
        if self.xres is not None:
            if self.xext is  None:
                xc0 = dvar.coords['x'].values
                self.xext = [ xc0[0], xc0[-1] ]
            subsample_coords['x'] = np.arange(self.xext[0],self.xext[1],self.xres)
        elif self.xext is not None:
            subsample_coords['x'] = slice(self.xext[0], self.xext[1])

        if self.yres is not None:
            if self.yext is  None:
                yc0 = dvar.coords['y'].values
                self.yext = [ yc0[0], yc0[-1] ]
            subsample_coords['y'] = np.arange(self.yext[0],self.yext[1]+self.yres/2,self.yres)
        elif self.yext is not None:
            subsample_coords['y'] = slice(self.yext[0], self.yext[1])
        return subsample_coords


    def subsample_1d(self, variable: xa.DataArray, global_attrs: Dict ) -> xa.DataArray:
        cmap: Dict[str,str] = { cn0:cn1 for (cn0,cn1) in self.dmap.items() if cn0 in list(variable.coords.keys()) }
        varray: xa.DataArray = variable.rename(**cmap)
        scoords: Dict[str,np.ndarray] = self.subsample_coords( varray )
        newvar: xa.DataArray = varray
 #       print(f" **** subsample {variable.name}, dims={varray.dims}, shape={varray.shape}")
        for cname, cval in scoords.items():
            if cname == 'z':
                newvar: xa.DataArray = newvar.interp(**{cname: cval}, assume_sorted=increasing(cval))
                print(f" >> zdata: {varray.coords['z'].values.tolist()}" )
                print(f" >> zconf: {cval.tolist()}")
                print(f" >> znewv: {newvar.coords['z'].values.tolist()}" )
            newvar.attrs.update( global_attrs )
            newvar.attrs.update( varray.attrs )
        return newvar.where( newvar != newvar.attrs['fmissing_value'], np.nan )

    def subsample(self, variable: xa.DataArray, global_attrs: Dict, qtype: QType, isconst: bool) -> xa.DataArray:
        cmap: Dict[str, str] = {cn0: cn1 for (cn0, cn1) in self.dmap.items() if cn0 in list(variable.coords.keys())}
        varray: xa.DataArray = variable.rename(**cmap)
        if isconst and ("time" in varray.dims):
            varray = varray.isel( time=0, drop=True )
        scoords: Dict[str, np.ndarray] = self.subsample_coords(varray)
        print(f" **** subsample {variable.name}, dims={varray.dims}, shape={varray.shape}, new sizes: { {cn:cv.size for cn,cv in scoords.items()} }")
        varray = varray.interp( x=scoords['x'], y=scoords['y'], assume_sorted=True)
        if 'z' in scoords:
            varray = varray.interp( z=scoords['z'], assume_sorted=False )
        resampled: DataArrayResample = varray.resample(time=self.tstep)
        newvar: xa.DataArray = resampled.mean() if qtype == QType.Intensive else resampled.sum()
        newvar.attrs.update(global_attrs)
        newvar.attrs.update(varray.attrs)
        for missing in [ 'fmissing_value', 'missing_value', 'fill_value' ]:
            if missing in newvar.attrs:
                missing_value = newvar.attrs.pop('fmissing_value')
                return newvar.where( newvar != missing_value, np.nan )
        return replace_nans(newvar, 'y').transpose(*self.corder, missing_dims="ignore" )

def stats_filepath( version: str, statname: str ) -> str:
    return f"{fmbdir('processed')}/{version}/stats/{statname}.nc"

def load_stats( task_config: Dict , statname: str, **kwargs ) -> xa.Dataset:
    version = task_config['dataset_version']
    filepath = stats_filepath(version,statname)
    varstats: xa.Dataset = xa.open_dataset(filepath,**kwargs)
    model_varname_map = { v: k for k, v in task_config['input_variables'].items() if v in varstats.data_vars }
    model_coord_map   = { k: v for k, v in task_config['coords'].items() if k in varstats.coords }
    result: xa.Dataset = varstats.rename( **model_varname_map, **model_coord_map )
    print( f"\nLoad stats({statname}): vars = {list(result.data_vars.keys())}")
    return result
def load_norm_data( task_config: Dict ) -> Dict[str,xa.Dataset]:     #     version = cfg().task.dataset_version
    model_statnames: Dict[str,str] = task_config.get( 'statnames' )
    stats = { model_statnames[statname]: load_stats(task_config,statname) for statname in StatsAccumulator.statnames }
    return stats
