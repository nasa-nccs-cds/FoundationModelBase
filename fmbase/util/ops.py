import os, glob, numpy as np
from .parse import parse
from omegaconf import DictConfig, OmegaConf
from typing import Any, Dict, List, Tuple, Type, Optional, Union
from .config import cfg
import xarray as xa

def xextent( raster: xa.DataArray ) -> Tuple[float,float,float,float]:
    xc, yc = raster.coords['lon'].values, raster.coords['lat'].values
    extent = xc[0], xc[-1]+(xc[1]-xc[0]), yc[0], yc[-1]+(yc[1]-yc[0])
    return extent

def dsextent( dset: xa.Dataset ) -> Tuple[float,float,float,float]:
    xc, yc = dset.lon.values, dset.lat.values
    extent = xc[0], xc[-1]+(xc[1]-xc[0]), yc[0], yc[-1]+(yc[1]-yc[0])
    return extent

def vrange(vdata: xa.DataArray) -> Tuple[float,float]:
    return vdata.min(skipna=True).values.tolist(), vdata.max(skipna=True).values.tolist()

def dsrange(vdata: xa.Dataset) -> Dict[str,Tuple[float,float]]:
    return { vid: vrange(v) for vid,v in vdata.data_vars.items() }

def year2date( year: Union[int,str] ) -> np.datetime64:
    return np.datetime64( int(year) - 1970, 'Y')

def extract_year( filename: str ) -> int:
    for template in cfg().platform.occ_files:
        fields = parse( template, filename )
        if (fields is not None) and ('year' in fields):
            try:     return int(fields['year'])
            except:  pass
    return 0

def extract_species( filename: str ) -> Optional[str]:
    for template in cfg().platform.occ_files:
        fields = parse( template, filename )
        if (fields is not None) and ('species' in fields):
            try:     return fields['species']
            except:  pass
    return None

def get_cfg_dates() -> List[np.datetime64]:
    return [year2date(y) for y in range(*cfg().platform.year_range) ]

def get_obs_dates() -> List[np.datetime64]:
    files = glob.glob(f"{cfg().platform.cov_data_dir}/*.jay")
    years = set([ extract_year(os.path.basename(file)) for file in files])
    return [year2date(y) for y in years if y > 0]

def get_dates( year_range: List[int] ) -> List[np.datetime64]:
    return [ year2date(y) for y in range(*year_range) ]

def get_obs_species() -> List[str]:
    files = glob.glob(f"{cfg().platform.occ_data_dir}/*.jay")
    species = set([ extract_species(os.path.basename(file)) for file in files])
    species.discard(None)
    return list(species)

def obs_dates_for_cov_date( covdate: np.datetime64 ) -> List[np.datetime64]:
    return [ covdate ]