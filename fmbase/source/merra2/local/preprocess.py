import xarray as xa
import numpy as np
from fmbase.util.config import cfg
from typing import List, Union, Tuple, Optional, Dict, Type
import hydra, glob, sys, os, time
from fmbase.source.merra2.base import MERRA2Base
from fmbase.util.ops import get_levels_config, increasing


class MERRA2DataProcessor(MERRA2Base):

    def __init__(self):
        MERRA2Base.__init__( self )
        self.xext, self.yext = cfg().preprocess.get('xext'), cfg().preprocess.get('yext')
        self.xres, self.yres = cfg().preprocess.get('xres'), cfg().preprocess.get('yres')
        self.levels: Optional[np.ndarray] = get_levels_config( cfg().preprocess )
        self.dmap: Dict = cfg().preprocess.dims
        self.year_range = cfg().preprocess.year_range
        self.month_range = cfg().preprocess.get('month_range',[0,12,1])
        self.file_template = cfg().platform.dataset_files
        self.collections = cfg().preprocess.collections
        self._subsample_coords: Dict[str,np.ndarray] = None

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

    @classmethod
    def get_varnames(cls, dset_file: str) -> List[str]:
        dset: xa.Dataset = xa.open_dataset(dset_file)
        covnames = list(dset.data_vars.keys())
        dset.close()
        return covnames

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
        scoords: Dict[str, np.ndarray] = self.subsample_coords(varray)
        print(f" **** subsample {variable.name}, dims={varray.dims}, shape={varray.shape}, new sizes: { {cn:cv.size for cn,cv in scoords.items()} }")
        newvar: xa.DataArray = varray.interp(**scoords, assume_sorted=True)
        newvar.attrs.update(global_attrs)
        newvar.attrs.update(varray.attrs)
        return newvar.where(newvar != newvar.attrs['fmissing_value'], np.nan)

    def process_subsample(self, collection: str, dvar: str, files: List[str], **kwargs):
        filepath: str = self.variable_cache_filepath(dvar, collection, **kwargs)
        reprocess: bool = kwargs.pop( 'reprocess', True )
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
            print( f" ** Skipping var {dvar:12s} in collection {collection:12s} due to existence of processed file {filepath}")
