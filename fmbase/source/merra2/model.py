import xarray as xa, pandas as pd
import os, numpy as np
from typing import Any, Dict, List, Tuple, Type, Optional, Union, Sequence, Mapping
from fmbase.util.ops import fmbdir
from fmbase.util.ops import get_levels_config
from dataclasses import dataclass
from fmbase.util.config import cfg

_SEC_PER_HOUR = 3600
_HOUR_PER_DAY = 24
SEC_PER_DAY = _SEC_PER_HOUR * _HOUR_PER_DAY
_AVG_DAY_PER_YEAR = 365.24219
AVG_SEC_PER_YEAR = SEC_PER_DAY * _AVG_DAY_PER_YEAR
DAY_PROGRESS = cfg().preprocess.day_progress
YEAR_PROGRESS = cfg().preprocess.year_progress
def nnan(varray: xa.DataArray) -> int: return np.count_nonzero(np.isnan(varray.values))
def pctnan(varray: xa.DataArray) -> str: return f"{nnan(varray)*100.0/varray.size:.2f}%"

@dataclass(eq=True,repr=True,frozen=True,order=True)
class YearMonth:
	year: int
	month: int

def variable_cache_filepath(version: str, vname: str, **kwargs) -> str:
	if "year" in kwargs:
		if "month" in kwargs:
			if "day" in kwargs:   filename = "{varname}_{year}-{month}-{day}.nc".format(varname=vname, **kwargs)
			else:				  filename = "{varname}_{year}-{month}.nc".format(varname=vname, **kwargs)
		else:                     filename = "{varname}_{year}.nc".format(varname=vname, **kwargs)
	else:                         filename = "{varname}.nc".format(varname=vname, **kwargs)
	return f"{fmbdir('processed')}/{version}/{filename}"

def cache_filepath(version: str, date: Tuple[int,int,int]) -> str:
	year, month, day = date
	filename = f"{year}-{month}-{day}.nc"
	return f"{fmbdir('processed')}/{version}/{filename}"

def load_cache_var( version: str, dvar: str, year: int, month: int, day: int, task: Dict, **kwargs  ) -> Optional[xa.DataArray]:
	coord_map: Dict = task.get('coords',{})
	filepath = variable_cache_filepath(version, dvar, year=year, month=month, day=day)
	if not os.path.exists( filepath ):
		filepath = variable_cache_filepath( version, dvar )
	try:
		darray: xa.DataArray = xa.open_dataarray(filepath,**kwargs)
		# if 'time' in darray.coords:
		# 	vtime: List[str] = [str(pd.Timestamp(dt64)) for dt64 in darray.coords['time'].values.tolist()]
		# 	print( f" ***>> load_cache_var[{dvar}({day}/{month}/{year})]: dims={darray.dims} shape={darray.shape} time={vtime}, filepath={filepath}" )
		cmap: Dict = { k:v for k,v in coord_map.items() if k in darray.coords.keys()}
		result = darray.rename(cmap).compute()
		darray.close()
		return result
	except FileNotFoundError:
		print( f"Not reading variable {dvar} (data file does not exist): {filepath}")

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
  for coord in ("datetime", "lon"):
    if coord not in data.coords:
      raise ValueError(f"'{coord}' must be in `data` coordinates.")
  seconds_since_epoch = ( data.coords["datetime"].data.astype("datetime64[s]").astype(np.int64) )
  batch_dim = ("batch",) if "batch" in data.dims else ()
  year_progress = get_year_progress(seconds_since_epoch)
  data.update( featurize_progress( name=YEAR_PROGRESS, dims=batch_dim + ("time",), progress=year_progress ) )
  longitude_coord = data.coords["lon"]
  day_progress = get_day_progress(seconds_since_epoch, longitude_coord.data)
  data.update( featurize_progress( name=DAY_PROGRESS, dims=batch_dim + ("time",) + longitude_coord.dims, progress=day_progress ) )

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
	add_derived_vars(merged)
	return merged

def load_timestep( year: int, month: int, day: int, task: Dict, **kwargs ) -> xa.Dataset:
	vnames = kwargs.pop('vars',None)
	vlist: Dict[str, str] = task['input_variables']
	constants: List[str] = task['constants']
	levels: Optional[np.ndarray] = get_levels_config(task)
	version = task['dataset_version']
	cmap = task['coords']
	zc, yc, corder = cmap['z'], cmap['y'], [ cmap[cn] for cn in ['t','z','y','x'] ]
	tsdata = {}
	print(f"  load_timestep({day}/{month}/{year}), constants={constants}, kwargs={kwargs} ")
	for vname,dsname in vlist.items():
		if (vnames is None) or (vname in vnames):
			varray: Optional[xa.DataArray] = load_cache_var( version, dsname, year, month, day, task, **kwargs )
			if varray is not None:
				if (vname in constants) and ("time" in varray.dims):
					varray = varray.mean( dim="time", skipna=True, keep_attrs=True )
				if zc in varray.dims:
					levs: List[str] = varray.coords[zc].values.tolist() if levels is None else levels
					level_arrays = []
					for iL, lev in enumerate(levs):
						level_array: xa.DataArray = varray.sel( **{zc:lev}, method="nearest", drop=False )
						level_array = replace_nans( level_array, yc )
						level_arrays.append( level_array )
					varray = xa.concat( level_arrays, zc ).transpose(*corder, missing_dims="ignore")
				varray.attrs['dset_name'] = dsname
				print( f" >> Load_var({dsname}): name={vname}, shape={varray.shape}, dims={varray.dims}, zc={zc}, mean={varray.values.mean()}, nnan={nnan(varray)} ({pctnan(varray)})")
				tsdata[vname] = varray
	return xa.Dataset( tsdata )

def replace_nans( level_array: xa.DataArray, dim: str ) -> xa.DataArray:
	if nnan(level_array) > 0:
		result: xa.DataArray =  level_array.interpolate_na( dim=dim, method="linear", fill_value="extrapolate" )
		assert nnan(result) == 0, "NaNs remaining after replace_nans()"
		return result
	return level_array

def load_batch( year: int, month: int, day: int, ndays: int, task_config: Dict, **kwargs ) -> xa.Dataset:
	slices: List[xa.Dataset] = []
	for day in range( day, day+ndays):
		slices.append( load_timestep( year, month, day, task_config, **kwargs ) )
	return merge_batch( slices )

def to_feature_array( data_batch: xa.Dataset) -> xa.DataArray:
	features = xa.DataArray(data=list(data_batch.data_vars.keys()), name="features")
	result = xa.concat( list(data_batch.data_vars.values()), dim=features )
	result = result.transpose(..., "features")
	return result






