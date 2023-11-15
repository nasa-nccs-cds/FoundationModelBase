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

def get_levels_config( config: Dict ) -> Optional[np.ndarray]:
    levs = config.get('levels')
    if levs is not None:
        levels = np.array(levs)
        levels.sort()
        return levels
    levr = config.get('level_range')
    if levr is not None:
        levels = np.arange(*levr)
        return levels


def increasing( data: np.ndarray ) -> bool:
    xl = data.tolist()
    return xl[-1] > xl[0]


def format_timedelta( td: np.timedelta64, form: str ) -> str:
	s = td.astype('timedelta64[s]').astype(np.int32)
	hours, remainder = divmod(s, 3600)
	if form == "full":
		minutes, seconds = divmod(remainder, 60)
		return '{:02}:{:02}:{:02}'.format(int(hours), int(minutes), int(seconds))
	elif form == "hr":
		return f'{hours}hr'
	else: raise Exception( f"format_timedelta: unknown form: {form}" )

def format_timedeltas( tds: xa.DataArray, form: str = "hr" ) -> str:
	if tds is None: return " NA "
	return str( [format_timedelta(td,form) for td in tds.values] ).replace('"','')

def print_dict( title: str, data: Dict ):
	print( f"\n -----> {title}:")
	for k,v in data.items():
		print( f"   ** {k}: {v}")

def parse_file_parts(file_name):
	return dict(part.split("-", 1) for part in file_name.split("_"))
def resolve_links( pdict: DictConfig, pkey: str ) -> str:
	pval = pdict[pkey]
	while '{' in pval:
		for key,val in pdict:
			if '{' not in val:
				try: pval = pval.format( key=val )
				except KeyError: pass
	return pval

def fmbdir( dtype: str ) -> str:
	return resolve_links( cfg().platform, dtype )