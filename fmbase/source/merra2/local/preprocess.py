
import xarray as xa
import numpy as np
from omegaconf import DictConfig, OmegaConf
import linecache
from fmbase.util.ops import vrange
from pathlib import Path
from fmbase.util.config import cfg
from typing import List, Union, Tuple, Optional, Dict, Type
import hydra, glob, sys, os, time
from fmbase.io.nc4 import nc4_write_array

def year2date( year: Union[int,str] ) -> np.datetime64:
    return np.datetime64( int(year) - 1970, 'Y')

def is_float( string: str ) -> bool:
    try: float(string); return True
    except ValueError:  return False

def find_key( d: Dict, v: str ) -> str:
    return list(d.keys())[ list(d.values()).index(v) ]

def is_int( string: str ) -> bool:
    try: int(string);  return True
    except ValueError: return False

def str2num( string: str ) -> Union[float,int,str]:
    try: return int(string)
    except ValueError:
        try: return float(string)
        except ValueError:
            return string

def xmin( v: xa.DataArray ):
    return v.min(skipna=True).values.tolist()

def xmax( v: xa.DataArray ):
    return v.max(skipna=True).values.tolist()

def xrng( v: xa.DataArray ):
    return [ xmin(v), xmax(v) ]

def srng( v: xa.DataArray ):
    return f"[{xmin(v):.5f}, {xmax(v):.5f}]"

class MERRADataProcessor:

    def __init__(self):
        self.xext, self.yext = cfg().scenario.get('xext'), cfg().scenario.get('yext')
        self.xres, self.yres = cfg().scenario.get('xres'), cfg().scenario.get('yres')
        self.levels = np.array( list(cfg().scenario.get('levels')) ).sort()
        self.dmap: Dict = cfg().scenario.dims
        self.year_range = cfg().scenario.year_range
        self.month_range = cfg().scenario.get('month_range',[0,12,1])
        self.file_template = cfg().platform.dataset_files
        self.group = cfg().scenario.get('group','Nv')
        self.freq = cfg().scenario.get('freq', 'tave3')
        self.collections = cfg().scenario.collections
        self.cache_file_template = "{varname}_{year}-{month}.nc"
        self.cfgId = cfg().scenario.id
        self._subsample_coords: Dict[str,np.ndarray] = None

    @property
    def data_dir(self):
        return cfg().platform.dataset_root.format( root=cfg().platform.root )

    @property
    def results_dir(self):
        base_dir = cfg().platform.processed.format( root=cfg().platform.root )
        return  f"{base_dir}/data/merra2"

    def get_monthly_files(self, collection, year) -> Dict[int,List[str]]:
        months = list(range(*self.month_range))
        assert "{year}" in self.file_template, "{year} field missing from platform.cov_files parameter"
        dset_files = {}
        assert "{month}" in self.file_template, "{month} field missing from platform.cov_files parameter"
        for month in months:
            dset_template = self.file_template.format(collection=collection, year=year, month=f"{month+1:0>2}", group=self.group, freq=self.freq)
            dset_paths = f"{self.data_dir}/{dset_template}"
            gfiles = glob.glob(dset_paths)
            print( f" ** M{month}: Found {len(gfiles)} files for glob {dset_paths}, template={self.file_template}, root dir ={self.data_dir}" )
            dset_files[month] = gfiles
        return dset_files

    @classmethod
    def load_asc(cls, filepath: str, flipud=True ) -> xa.DataArray:
        raster_data: np.array = np.loadtxt( filepath, skiprows=6 )
        if flipud: raster_data = np.flipud( raster_data )
        header, varname = {}, Path(filepath).stem
        for hline in range(6):
            header_line = linecache.getline(filepath, hline).split(' ')
            if len(header_line) > 1:
                header[ header_line[0].strip() ] = str2num( header_line[1] )
        nodata: float = header.get( 'NODATA_VALUE', -9999.0 )
        raster_data[ raster_data==nodata ] = np.nan
        cs, xlc, ylc, nx, ny = header['CELLSIZE'], header['XLLCORNER'], header['YLLCORNER'], header['NCOLS'], header['NROWS']
        xc = np.array( [ xlc + cs*ix for ix in range(nx)] )
        yc = np.array([ylc + cs * iy for iy in range(ny)])
        header['_FillValue'] = np.nan
        header['long_name'] = varname
        header['varname'] = varname
        header['xres'] = cs
        header['yres'] = cs
        return xa.DataArray( raster_data, name=varname, dims=['lat','lon'], coords=dict(lat=yc,lon=xc), attrs=header )

    def process(self, **kwargs):
        years = list(range( *self.year_range ))
        for collection in self.collections:
            print(f"\n --------------- Processing collection {collection}  --------------- ")
            for year in years:
                t0 = time.time()
                dset_files: Dict[int,List[str]] = self.get_monthly_files( collection, year )
                for month, dfiles in dset_files.items():
                    dvars: List[str] = self.get_varnames( dfiles[0] )
                    if len( dvars ) == 0:
                        print(f" ** No dvars in this collection" )
                    else:
                        for dvar in dvars:
                            self.process_subsample( collection, dvar, dfiles, year=year, month=month, **kwargs )
                    print(f" -- -- Processed {len(dset_files)} files for month {month}/{year}, time = {(time.time()-t0)/60:.2f} ")


    def cache_files_exist(self, varnames: List[str], year: int, month: int ) -> bool:
        for vname in varnames:
            filepath = self.variable_cache_filepath(vname, year, month )
            if not os.path.exists( filepath ): return False
        return True

    @classmethod
    def get_varnames(cls, dset_file: str) -> List[str]:
        dset: xa.Dataset = xa.open_dataset(dset_file)
        covnames = list(dset.data_vars.keys())
        dset.close()
        return covnames

    def subsample_coords(self, dvar: xa.DataArray ):
        if self._subsample_coords is None:
            self._subsample_coords = {} if self.levels is None else dict(z=self.levels)
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


    def subsample(self, variable: xa.DataArray, global_attrs: Dict ) -> xa.DataArray:
        cmap = { cn0:cn1 for (cn0,cn1) in self.dmap.items() if cn0 in list(variable.coords.keys()) }
        varray: xa.DataArray = variable.rename(**cmap)
        scoords = self.subsample_coords( varray )
        newvar: xa.DataArray = varray
        for cname, cval in scoords.items():
            newvar: xa.DataArray = newvar.interp( **{cname:cval}, assume_sorted=(cname!='z') )
            newvar.attrs.update( global_attrs )
            newvar.attrs.update( varray.attrs )
        return newvar.where( newvar != newvar.attrs['fmissing_value'], np.nan )

    def process_subsample(self, collection: str, dvar: str, files: List[str], **kwargs):
        filepath = self.variable_cache_filepath(dvar, collection, **kwargs)
        reprocess = kwargs.pop( 'reprocess', True )
        if (not os.path.exists(filepath)) or reprocess:
            print(f" ** Processing variable {dvar} in collection {collection}, month={kwargs['month']}, year={kwargs['year']}: {len(files)} files")
            t0 = time.time()
            samples: List[xa.DataArray] = []
            for file in sorted(files):
                dset: xa.Dataset = xa.open_dataset(file)
                dset_attrs = dict( collection=os.path.basename(collection), **dset.attrs, **kwargs )
                print( f"Processing var {dvar} from file {file}")
                samples.append( self.subsample( dset.data_vars[dvar], dset_attrs ) )
            if len(samples) == 0:
                print( f"Found no files for variable {dvar} in collection {collection}")
            else:
                t1 = time.time()
                mvar: xa.DataArray = xa.concat( samples, dim="time" ) if (len(samples) > 1) else samples[0]
                print(f"Saving Merged var {dvar}: shape= {mvar.shape}, dims= {mvar.dims}")
                os.makedirs(os.path.dirname(filepath), mode=0o777, exist_ok=True)
                mvar.to_netcdf( filepath, format="NETCDF4" )
                print(f" ** ** ** Saved variable {dvar} to file= {filepath} in time = {time.time()-t1} sec")
                print(f"  Completed processing in time = {(time.time()-t0)/60} min")
        else:
            print( f" ** Skipping {collection}:{dvar} due to existence of processed file {filepath}")

    def variable_cache_filepath(self, vname: str, collection: str, **kwargs ) -> str:
        filename = self.cache_file_template.format( varname=vname, year=kwargs['year'], month=kwargs['month'] )
        return f"{self.results_dir}/merra2/{self.cfgId}/{self.freq}_{collection}_{self.group}/{filename}"
