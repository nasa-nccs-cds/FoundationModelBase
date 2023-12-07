import xarray as xa, pandas as pd
import os, numpy as np
from typing import Any, Dict, List, Tuple, Type, Optional, Union, Sequence, Mapping
from fmbase.util.ops import fmbdir
from fmbase.util.ops import get_levels_config
from fmbase.util.config import cfg, Date
from fmbase.util.config import cfg

_SEC_PER_HOUR = 3600
_HOUR_PER_DAY = 24
SEC_PER_DAY = _SEC_PER_HOUR * _HOUR_PER_DAY
_AVG_DAY_PER_YEAR = 365.24219
AVG_SEC_PER_YEAR = SEC_PER_DAY * _AVG_DAY_PER_YEAR

def nnan(varray: xa.DataArray) -> int: return np.count_nonzero(np.isnan(varray.values))
def pctnan(varray: xa.DataArray) -> str: return f"{nnan(varray)*100.0/varray.size:.2f}%"

def cache_var_filepath(version: str, date: Date) -> str:
	return f"{fmbdir('processed')}/{version}/{repr(date)}.nc"

def cache_const_filepath(version: str) -> str:
	return f"{fmbdir('processed')}/{version}/const.nc"

def get_year_progress(seconds_since_epoch: np.ndarray) -> np.ndarray:
	years_since_epoch = ( seconds_since_epoch / SEC_PER_DAY / np.float64(_AVG_DAY_PER_YEAR) )
	yp = np.mod(years_since_epoch, 1.0).astype(np.float32)
	return yp

def get_day_progress( seconds_since_epoch: np.ndarray, longitude: np.ndarray ) -> np.ndarray:
	day_progress_greenwich = ( np.mod(seconds_since_epoch, SEC_PER_DAY) / SEC_PER_DAY )
	longitude_offsets = np.deg2rad(longitude) / (2 * np.pi)
	day_progress = np.mod( day_progress_greenwich[..., np.newaxis] + longitude_offsets, 1.0 )
	return day_progress.astype(np.float32)

def featurize_progress( name: str, dims: Sequence[str], progress: np.ndarray ) -> Mapping[str, xa.Variable]:
	if len(dims) != progress.ndim:
		raise ValueError( f"Number of dimensions in feature {name}{dims} must be equal to the number of dimensions in progress{progress.shape}." )
	else: print( f"featurize_progress: {name}{dims} --> progress{progress.shape} ")
	progress_phase = progress * (2 * np.pi)
	return { name: xa.Variable(dims, progress), name + "_sin": xa.Variable(dims, np.sin(progress_phase)), name + "_cos": xa.Variable(dims, np.cos(progress_phase)) }

def add_derived_vars(data: xa.Dataset) -> None:
	if 'datetime' not in data.coords:
		data.coords['datetime'] = data.coords['time'].expand_dims("batch")
	seconds_since_epoch = ( data.coords["datetime"].data.astype("datetime64[s]").astype(np.int64) )
	batch_dim = ("batch",) if "batch" in data.dims else ()
	year_progress = get_year_progress(seconds_since_epoch)
	data.update( featurize_progress( name=cfg().preprocess.year_progress, dims=batch_dim + ("time",), progress=year_progress ) )
	longitude_coord = data.coords["x"]
	day_progress = get_day_progress(seconds_since_epoch, longitude_coord.data)
	data.update( featurize_progress( name=cfg().preprocess.day_progress, dims=batch_dim + ("time",) + longitude_coord.dims, progress=day_progress ) )


def merge_batch( slices: List[xa.Dataset] ) -> xa.Dataset:
	cvars = [vname for vname, vdata in slices[0].data_vars.items() if "time" not in vdata.dims]
	merged: xa.Dataset = xa.concat( slices, dim="time", coords = "minimal" )
	merged = merged.drop_vars(cvars)
	sample: xa.Dataset = slices[0]
	for vname, dvar in sample.data_vars.items():
		if vname not in merged.data_vars.keys():
			merged[vname] = dvar
	return merged

# def load_timestep( date: Date, task: Dict, **kwargs ) -> xa.Dataset:
# 	vnames = kwargs.pop('vars',None)
# 	vlist: Dict[str, str] = task['input_variables']
# 	constants: List[str] = task['constants']
# 	levels: Optional[np.ndarray] = get_levels_config(task)
# 	version = task['dataset_version']
# 	cmap = task['coords']
# 	zc, yc, corder = cmap['z'], cmap['y'], [ cmap[cn] for cn in ['t','z','y','x'] ]
# 	tsdata = {}
# 	filepath = cache_var_filepath(version, date)
# #	if not os.path.exists( filepath ):
# 	dataset: xa.Dataset = xa.open_dataset(filepath, **kwargs)
# 	print(f"  load_timestep({date}), constants={constants}, kwargs={kwargs} ")
# 	for vname,dsname in vlist.items():
# 		if (vnames is None) or (vname in vnames):
# 			varray: xa.DataArray = dataset.data_vars[vname]
# 			if (vname in constants) and ("time" in varray.dims):
# 				varray = varray.mean( dim="time", skipna=True, keep_attrs=True )
# 			varray.attrs['dset_name'] = dsname
# 			print( f" >> Load_var({dsname}): name={vname}, shape={varray.shape}, dims={varray.dims}, zc={zc}, mean={varray.values.mean()}, nnan={nnan(varray)} ({pctnan(varray)})")
# 			tsdata[vname] = varray
# 	return xa.Dataset( tsdata )


def load_batch( dates: List[Date], task_config: Dict, **kwargs ) -> xa.Dataset:
	slices: List[xa.Dataset] = []
	version = task_config['dataset_version']
	for date in dates:
		filepath = cache_var_filepath(version, date)
		#	if not os.path.exists( filepath ):
		dataset: xa.Dataset = xa.open_dataset(filepath, **kwargs)
		var_map = {vid:newid for vid,newid in task_config.get('coords',{}).items() if vid in dataset.data_vars.keys() }
		slices.append( dataset.rename(var_map) )
	return merge_batch( slices )

def to_feature_array( data_batch: xa.Dataset) -> xa.DataArray:
	features = xa.DataArray(data=list(data_batch.data_vars.keys()), name="features")
	result = xa.concat( list(data_batch.data_vars.values()), dim=features )
	result = result.transpose(..., "features")
	return result






