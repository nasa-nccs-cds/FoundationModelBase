
import xarray as xa
import numpy as np
from omegaconf import DictConfig, OmegaConf
import linecache
from fmbase.util.ops import vrange
from pathlib import Path
from fmbase.util.config import cfg
from typing import List, Union, Tuple, Optional, Dict, Type
import hydra, glob, sys, os, time

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
        self.levels = cfg().scenario.get('levels')
        self.dmap: Dict = cfg().scenario.dims
        self.year_range = cfg().scenario.year_range
        self.month_range = cfg().scenario.get('month_range',[0,12,1])
        self.file_template = cfg().platform.dataset_files
        self.cache_file_template = cfg().scenario.cache_file_template
        self.cfgId = cfg().scenario.id

    @property
    def data_dir(self):
        return cfg().platform.dataset_root.format( root=cfg().platform.root )

    @property
    def cache_dir(self):
        return cfg().platform.cache.format( root=cfg().platform.root )

    def get_monthly_files(self, collection, year) -> Dict[int,List[str]]:
        months = list(range(*self.month_range))
        assert "{year}" in self.file_template, "{year} field missing from platform.cov_files parameter"
        dset_files = {}
        assert "{month}" in self.file_template, "{month} field missing from platform.cov_files parameter"
        for month in months:
            dset_template = self.file_template.format(collection=collection, year=year, month=f"{month+1:0>2}")
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

    def process(self, collection: str = None, **kwargs):
        years = list(range( *self.year_range ))
        reprocess = kwargs.get( 'reprocess', False )
        print(f"\n --------------- Processing collection {collection}  --------------- ")
        for year in years:
            t0 = time.time()
            dset_files: Dict[int,List[str]] = self.get_monthly_files( collection, year )
            for month, dfiles in dset_files.items():
                dvars: List[str] = self.get_varnames( dfiles[0] )
                if len( dvars ) == 0:
                    print(f" ** No dvars in this collection")
                    return
                if not reprocess and self.cache_files_exist( dvars, year, month ):
                    print(f" ** Skipping already processed year {year}")
                else:
                    print(f" ** Loading dataset {len(dfiles)} files for dvars {collection}:{dvars}, month={month}, year={year}")
                    agg_dataset: xa.Dataset =  self.open_collection( collection, dfiles, year=year, month=month )
                    print(f" -- -- Processing {len(dset_files)} files, load time = {time.time()-t0:.2f} ")
                    for dvar in dvars:
                        self.proccess_variable( dvar, agg_dataset, **kwargs )
                    agg_dataset.close()

    def cache_files_exist(self, varnames: List[str], year: int, month: int ) -> bool:
        for vname in varnames:
            filepath = self.variable_cache_filepath(vname, year, month )
            if not os.path.exists( filepath ): return False
        return True

    @classmethod
    def get_varnames(cls, dset_file: str) -> List[str]:
        dset: xa.Dataset = xa.open_dataset(dset_file)
        covnames = [vname for vname in dset.data_vars.keys() if vname in cfg().scenario.vars]
        dset.close()
        return covnames

    @classmethod
    def get_dvariates(cls, dset: xa.Dataset ) -> Dict[str,xa.DataArray]:
        dvariates: Dict[str,xa.DataArray] = { vid: dvar for vid, dvar in dset.data_vars.items() if vid in cfg().scenario.vars }
        return { vid: dvar.where(dvar != dvar.attrs['fmissing_value'], np.nan) for vid, dvar in dvariates.items()}

    def open_collection(self, collection, files: List[str], **kwargs) -> xa.Dataset:
        print( f" -----> open_collection[{collection}:{kwargs['year']}-{kwargs['month']}]>> {len(files)} files ", end="")
        tave =  kwargs.get( 'tave', False )
        t0 = time.time()
        dset: xa.Dataset = xa.open_mfdataset(files)
        dset_attrs = dict( collection=os.path.basename(collection), **dset.attrs, **kwargs )
        sampled_dset = xa.Dataset( self.get_dvariates( dset ), coords=dset.coords, attrs=dset_attrs )
        if tave: sampled_dset = sampled_dset.resample(time='AS').mean('time')
        print( f" Loaded {len(sampled_dset.data_vars)} in time = {time.time()-t0:.2f} sec, VARS:")
        for vid, dvar in sampled_dset.data_vars.items():
            dvar.attrs.update( dset.data_vars[vid].attrs )
            print( f" ** {vid}: {dvar.shape}")
        return sampled_dset

    # if (self.yext is None) or (self.yres is None):
    #     self.yci, self.xci = None, None
    # else:
    #     self.yci = np.arange( self.yext[0], self.yext[1]+self.yres/2, self.yres )
    #     self.xci = np.arange( self.xext[0], self.xext[1]+self.xres/2, self.xres )

    def resample_variable(self, variable: xa.DataArray) -> xa.DataArray:
        print( f"Rename, coords: {list(variable.coords.keys())}, map: {self.dmap}")
        dvar: xa.DataArray =  variable.rename( **self.dmap )
        if self.yres is not None:
            xc0, yc0 = dvar.coords['x'].values,  dvar.coords['y'].values
            if self.yext is  None:
                self.xext = [ xc0[0], xc0[-1]+self.xres/2 ]
                self.yext = [ yc0[0], yc0[-1]+self.yres/2 ]
            xc1, yc1 = np.arange(self.xext[0],self.xext[1],self.xres), np.arange(self.yext[0],self.yext[1],self.yres)
            new_coords = dict( x=xc1, y=yc1 )
            if self.levels is not None: new_coords['z'] = self.levels
            newvar: xa.DataArray = dvar.interp( **new_coords, assume_sorted=True)
            newvar.attrs.update(dvar.attrs)
            print( f"xc0[{xc0.size}] = {xc0.tolist()}")
            print( f"xres = {self.xres}")
            print( f"xc1[{len(xc1)}] = {xc1}")
        elif self.yext is not None:
            scoords = {'x': slice(self.xext[0], self.xext[1]), 'y': slice(self.yext[0], self.yext[1])}
            newvar: xa.DataArray = dvar.sel(**scoords)
            newvar.attrs.update(dvar.attrs)
        else:
            newvar: xa.DataArray = dvar
        xc, yc = newvar.coords['x'].values, newvar.coords['y'].values
        newvar.attrs['xres'], newvar.attrs['yres'] = (xc[1]-xc[0]).tolist(), (yc[1]-yc[0]).tolist()
        newvar.attrs['fmissing_value'] = np.nan
        return newvar

    def variable_cache_filepath(self, vname: str, **kwargs ) -> str:
        filename = self.cache_file_template.format( varname=vname, collection=kwargs['collection'], year=kwargs['year'], month=kwargs['month'] )
        return f"{self.cache_dir}/{self.cfgId}/{filename}"

    def proccess_variable(self, varname: str, agg_dataset: xa.Dataset, **kwargs ):
        t0 = time.time()
        reprocess = kwargs.get('reprocess',False)
        variable: xa.DataArray = agg_dataset.data_vars[varname]
        interp_var: xa.DataArray = self.resample_variable(variable)
        filepath = self.variable_cache_filepath( varname, **agg_dataset.attrs )
        if reprocess or not os.path.exists(filepath):
            print(f" ** ** ** Processing variable {variable.name}, shape= {interp_var.shape}, dims= {interp_var.dims}, file= {filepath}")
            dset: xa.Dataset = self.create_cache_dset(interp_var, agg_dataset.attrs )
            os.makedirs(os.path.dirname(filepath), mode=0o777, exist_ok=True)
            print(f" ** ** ** >> Writing cache data file: {filepath}")
            dset.to_netcdf( filepath, format="NETCDF4", engine="netcdf4" )
            print(f" >> Completed in time= {time.time()-t0} sec.")
        else:
            print(f" ** ** ** >> Skipping existing variable {variable.name}, file= {filepath} ")

    @classmethod
    def create_cache_dset( cls, vdata: xa.DataArray, dset_attrs: Dict ) -> xa.Dataset:
        print(f" create_cache_dset, shape={vdata.shape}, dims={vdata.dims}, coords = { {k:v.shape for k,v in vdata.coords.items()} } " )
        return xa.Dataset( {vdata.name: vdata}, coords=vdata.coords, attrs=dset_attrs )