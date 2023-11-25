import xarray as xa
import numpy as np
from fmbase.util.hydra_config import cfg
from typing import Any, Dict, List, Tuple, Type, Optional, Union
from fmbase.util.ops import fmbdir
from fmbase.util.ops import get_levels_config
from dataclasses import dataclass

def nnan(varray: xa.DataArray) -> int: return np.count_nonzero(np.isnan(varray.values))
def pctnan(varray: xa.DataArray) -> str: return f"{nnan(varray)*100.0/varray.size:.2f}%"

@dataclass(eq=True,repr=True,frozen=True,order=True)
class YearMonth:
	year: int
	month: int

def variable_cache_filepath(version: str, vname: str, **kwargs) -> str:
	if "year" in kwargs:
		if "month" in kwargs:   filename = "{varname}_{year}-{month}.nc".format(varname=vname, **kwargs)
		else:                   filename = "{varname}_{year}.nc".format(varname=vname, **kwargs)
	else:                        filename = "{varname}.nc".format(varname=vname, **kwargs)
	return f"{fmbdir('processed')}/{version}/{filename}"

def load_cache_var( version: str, dvar: str, year: int, month: int, task: Dict, **kwargs  ) -> xa.DataArray:
	coord_map: Dict = task.get('coords',{})
	filepath = variable_cache_filepath( version, dvar, year=year, month=month )
	darray: xa.DataArray = xa.open_dataarray(filepath,**kwargs)
	cmap: Dict = { k:v for k,v in coord_map.items() if k in darray.coords.keys()}
	return darray.rename(cmap)

def merge_batch( slices: List[xa.Dataset] ) -> xa.Dataset:
	cvars = [vname for vname, vdata in slices[0].data_vars.items() if "time" not in vdata.dims]
	merged: xa.Dataset = xa.concat( slices, dim="time", coords = "minimal" )
	merged = merged.drop_vars(cvars)
	if 'datetime' not in merged.coords:
		merged.coords['datetime'] = merged.coords['time'].expand_dims("batch")
	sample: xa.Dataset = slices[0]
	for vname, dvar in sample.data_vars.items():
		if vname not in merged.data_vars.keys():
			merged[vname] = dvar
	return merged

def load_timestep( year: int, month: int, task: Dict, **kwargs ) -> xa.Dataset:
	vlist: Dict[str, str] = task['input_variables']
#	clist: Dict[str, str] = task['constant_variables']
	levels: Optional[np.ndarray] = get_levels_config(task)
	version = task['dataset_version']
	tsdata, coords = {}, {}
	print(f"load_timestep({month}/{year})")
	for vname,dsname in vlist.items():
		varray: xa.DataArray = load_cache_var( version, dsname, year, month, task, **kwargs )
		coords.update( varray.coords )
		print( f"load_var({dsname}): name={vname}, shape={varray.shape}, dims={varray.dims}, mean={varray.values.mean()}, nnan={pctnan(varray)}")
		if 'z' in varray.dims:
			levs: List[str] = varray.coords['z'].values.tolist() if levels is None else levels
			for iL, lev in enumerate(levs):
				level_array: xa.DataArray = varray.sel( z=lev, method="nearest", drop=True )
				level_array.attrs['level'] = lev
				level_array.attrs['dset_name'] = dsname
				tsdata[f"{vname}.{iL}"] = level_array
		else:
			varray.attrs['dset_name'] = dsname
			tsdata[vname] = varray
	return xa.Dataset( tsdata, coords )

def load_batch( start: YearMonth, end: YearMonth, task_config: Dict, **kwargs ) -> xa.Dataset:
	slices: List[xa.Dataset] = []
	for year in range( start.year,end.year+1):
		month_range = [0,12]
		if year == start.year: month_range[0] = start.month
		if year == end.year:   month_range[1] = end.month + 1
		for month in range( *month_range ):
			slices.append( load_timestep( year, month, task_config, **kwargs ) )
	return merge_batch( slices )

def to_feature_array( data_batch: xa.Dataset) -> xa.DataArray:
	features = xa.DataArray(data=list(data_batch.data_vars.keys()), name="features")
	result = xa.concat( list(data_batch.data_vars.values()), dim=features )
	result = result.transpose(..., "features")
	return result






