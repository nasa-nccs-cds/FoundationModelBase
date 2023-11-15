import xarray as xa
import numpy as np
from fmbase.util.config import cfg
from typing import List, Dict
from fmbase.util.ops import fmbdir
from fmbase.source.merra2.preprocess import StatsAccumulator
from fmbase.util.ops import get_levels_config
from dataclasses import dataclass

@dataclass(eq=True,repr=True,frozen=True,order=True)
class YearMonth:
	year: int
	month: int

def variable_cache_filepath(version: str, vname: str, **kwargs) -> str:
	if "year" in kwargs:
		if "month" in kwargs:   filename = "{varname}_{year}-{month}.nc".format(varname=vname, **kwargs)
		else:                   filename = "{varname}_{year}.nc".format(varname=vname, **kwargs)
	else:                        filename = "{varname}.nc".format(varname=vname, **kwargs)
	return f"{fmbdir('results')}/{version}/{filename}"

def stats_filepath( version: str, statname: str ) -> str:
	return f"{fmbdir('results')}/{version}/stats/{statname}.nc"

def load_cache_var( version: str, dvar: str, year: int, month: int, **kwargs  ) -> xa.DataArray:
	coord_map: Dict = kwargs.pop('coords', {})
	filepath = variable_cache_filepath( version, dvar, year=year, month=month )
	darray: xa.DataArray = xa.open_dataarray(filepath,**kwargs)
	cmap: Dict = { k:v for k,v in coord_map.items() if k in darray.coords.keys()}
	return darray.rename(cmap)

def load_stats( version: str, statname: str, **kwargs ) -> xa.Dataset:
	filepath = stats_filepath(version,statname)
	varstats: xa.Dataset = xa.open_dataset(filepath,**kwargs)
	return varstats

def merge_batch( slices: List[xa.Dataset] ) -> xa.Dataset:
	merged: xa.Dataset = xa.concat( slices, dim="time", coords = "minimal" )
	sample: xa.Dataset = slices[0]
	datetime = None
	for vname, dvar in sample.data_vars.items():
		if vname not in merged.data_vars.keys():
			merged[vname] = dvar
		elif datetime is None:
			datetime = dvar.coords['time']
	merged['datetime'] = datetime
	return merged

def load_timestep( year: int, month: int, **kwargs ) -> xa.Dataset:
	vlist: List[str] = cfg().task.input_variables + cfg().task.forcing_variables
	levels: np.ndarray = get_levels_config(cfg().task)
	version = cfg().task.dataset_version
	tsdata, coords, taxis = {}, {}, None
	print(f"load_timestep({month}/{year})")
	for vname in vlist:
		varray: xa.DataArray = load_cache_var(version, vname, year, month, **kwargs)
		coords.update( varray.coords )
		if taxis is None:
			assert 'time' in varray.coords.keys(), f"Constant DataArray can't be first in model vars configuration: {vname}"
			taxis = varray.coords['time']
		print( f"load_var({vname}): name={varray.name}, shape={varray.shape}, dims={varray.dims}, levels={levels}")
		if "time" not in varray.dims:
			varray = varray.expand_dims( dim={'time':taxis} )
		if 'z' in varray.dims:
			print(f" ---> Levels Coord= {varray.coords['z'].values.tolist()}")
			levs: List[str] = varray.coords['z'].values.tolist() if levels is None else levels
			for iL, lev in enumerate(levs):
				level_array: xa.DataArray = varray.sel( z=lev, method="nearest", drop=True )
				level_array.attrs['level'] = lev
				tsdata[f"{vname}.{iL}"] = level_array
		else:
			tsdata[vname] = varray
	features = xa.DataArray( data=list(tsdata.keys()), name="features" )
	print( f"Created coord {features.name}: shape={features.shape}, dims={features.dims}, Features:" )
	for fname, fdata in tsdata.items():
		print( f" ** {fname}{fdata.dims}: shape={fdata.shape}")
	return xa.Dataset( tsdata, coords )

def load_batch( start: YearMonth, end: YearMonth, **kwargs ) -> xa.Dataset:
	slices: List[xa.Dataset] = []
	for year in range( start.year,end.year+1):
		month_range = [0,12]
		if year == start.year: month_range[0] = start.month
		if year == end.year:   month_range[1] = end.month + 1
		for month in range( *month_range ):
			slices.append( load_timestep( year, month, **kwargs ) )
	return merge_batch( slices ).expand_dims( "batch" )

def to_feature_array( data_batch: xa.Dataset) -> xa.DataArray:
	features = xa.DataArray(data=list(data_batch.data_vars.keys()), name="features")
	result = xa.concat( list(data_batch.data_vars.values()), dim=features )
	result = result.transpose(..., "features")
	return result

def load_norm_data() -> Dict[str,xa.Dataset]:
	version = cfg().task.dataset_version
	stats = { statname: load_stats(version,statname) for statname in StatsAccumulator.statnames }
	return stats





