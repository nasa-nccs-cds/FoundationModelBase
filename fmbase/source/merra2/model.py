import xarray as xa, pandas as pd
import os, numpy as np
from typing import Any, Dict, List, Tuple, Type, Optional, Union, Sequence, Mapping
from fmbase.util.ops import fmbdir
from fmbase.source.merra2.preprocess import StatsAccumulator
from fmbase.util.dates import drepr, dstr
from datetime import date
from fmbase.util.config import cfg

_SEC_PER_HOUR = 3600
_HOUR_PER_DAY = 24
SEC_PER_DAY = _SEC_PER_HOUR * _HOUR_PER_DAY
_AVG_DAY_PER_YEAR = 365.24219
AVG_SEC_PER_YEAR = SEC_PER_DAY * _AVG_DAY_PER_YEAR

def nnan(varray: xa.DataArray) -> int: return np.count_nonzero(np.isnan(varray.values))
def pctnan(varray: xa.DataArray) -> str: return f"{nnan(varray)*100.0/varray.size:.2f}%"
def cache_var_filepath(version: str, d: date) -> str:
	return f"{fmbdir('processed')}/{version}/{drepr(d)}.nc"
def cache_const_filepath(version: str) -> str:
	return f"{fmbdir('processed')}/{version}/const.nc"
def stats_filepath(version: str, statname: str) -> str:
	return f"{fmbdir('processed')}/{version}/stats/{statname}.nc"
def d2xa( dvals: Dict[str,float] ) -> xa.Dataset:
    return xa.Dataset( {vn: xa.DataArray( np.array(dval) ) for vn, dval in dvals.items()} )


class FMBatch:

	def __init__(self, task_config: Dict, **kwargs):
		self.batch_cache: Dict[date, xa.Dataset] = {}
		self.task_config = task_config
		self.constants: xa.Dataset = self.load_const_dataset( **kwargs )
		self.norm_data: Dict[str, xa.Dataset] = self.load_merra2_norm_data()

	def clear_catch(self):
		self.batch_cache = {}

	@classmethod
	def get_predef_norm_data(cls) -> Dict[str, xa.Dataset]:
		dstd = dict(year_progress=0.0247, year_progress_sin=0.003, year_progress_cos=0.003, day_progress=0.433, day_progress_sin=1.0, day_progress_cos=1.0)
		vmean = dict(year_progress=0.5, year_progress_sin=0.0, year_progress_cos=0.0, day_progress=0.5, day_progress_sin=0.0, day_progress_cos=0.0)
		vstd = dict(year_progress=0.29, year_progress_sin=0.707, year_progress_cos=0.707, day_progress=0.29, day_progress_sin=0.707, day_progress_cos=0.707)
		return dict(diffs_stddev_by_level=d2xa(dstd), mean_by_level=d2xa(vmean), stddev_by_level=d2xa(vstd))

	@classmethod
	def load_merra2_norm_data(cls) -> Dict[str, xa.Dataset]:
		from fmbase.source.merra2.preprocess import load_norm_data
		predef_norm_data: Dict[str, xa.Dataset] = cls.get_predef_norm_data()
		m2_norm_data: Dict[str, xa.Dataset] = cls.load_norm_data(cfg().task)
		return {nnorm: xa.merge([predef_norm_data[nnorm], m2_norm_data[nnorm]]) for nnorm in m2_norm_data.keys()}

	def load_stats(self, statname: str, **kwargs) -> xa.Dataset:
		version = self.task_config['dataset_version']
		filepath = stats_filepath(version, statname)
		varstats: xa.Dataset = xa.open_dataset(filepath, **kwargs)
		model_varname_map = {v: k for k, v in self.task_config['input_variables'].items() if v in varstats.data_vars}
		model_coord_map = {k: v for k, v in self.task_config['coords'].items() if k in varstats.coords}
		result: xa.Dataset = varstats.rename(**model_varname_map, **model_coord_map)
		print(f"\nLoad stats({statname}): vars = {list(result.data_vars.keys())}")
		return result

	def load_norm_data(self) -> Dict[str, xa.Dataset]:  # version = cfg().task.dataset_version
		model_statnames: Dict[str, str] = self.task_config.get('statnames')
		stats = {model_statnames[statname]: self.load_stats( statname) for statname in StatsAccumulator.statnames}
		return stats

	@classmethod
	def get_year_progress(cls, seconds_since_epoch: np.ndarray) -> np.ndarray:
		years_since_epoch = ( seconds_since_epoch / SEC_PER_DAY / np.float64(_AVG_DAY_PER_YEAR) )
		yp = np.mod(years_since_epoch, 1.0).astype(np.float32)
		return yp

	@classmethod
	def get_day_progress(cls, seconds_since_epoch: np.ndarray, longitude: np.ndarray ) -> np.ndarray:
		day_progress_greenwich = ( np.mod(seconds_since_epoch, SEC_PER_DAY) / SEC_PER_DAY )
		longitude_offsets = np.deg2rad(longitude) / (2 * np.pi)
		day_progress = np.mod( day_progress_greenwich[..., np.newaxis] + longitude_offsets, 1.0 )
		return day_progress.astype(np.float32)

	@classmethod
	def featurize_progress(cls, name: str, dims: Sequence[str], progress: np.ndarray ) -> Mapping[str, xa.Variable]:
		if len(dims) != progress.ndim:
			raise ValueError( f"Number of dimensions in feature {name}{dims} must be equal to the number of dimensions in progress{progress.shape}." )
		else: print( f"featurize_progress: {name}{dims} --> progress{progress.shape} ")
		progress_phase = progress * (2 * np.pi)
		return { name: xa.Variable(dims, progress), name + "_sin": xa.Variable(dims, np.sin(progress_phase)), name + "_cos": xa.Variable(dims, np.cos(progress_phase)) }

	@classmethod
	def add_derived_vars( cls, data: xa.Dataset) -> None:
		if 'datetime' not in data.coords:
			data.coords['datetime'] = data.coords['time'].expand_dims("batch")
		seconds_since_epoch = ( data.coords["datetime"].data.astype("datetime64[s]").astype(np.int64) )
		batch_dim = ("batch",) if "batch" in data.dims else ()
		year_progress = cls.get_year_progress(seconds_since_epoch)
		data.update( cls.featurize_progress( name=cfg().preprocess.year_progress, dims=batch_dim + ("time",), progress=year_progress ) )
		longitude_coord = data.coords["x"]
		day_progress = cls.get_day_progress(seconds_since_epoch, longitude_coord.data)
		data.update( cls.featurize_progress( name=cfg().preprocess.day_progress, dims=batch_dim + ("time",) + longitude_coord.dims, progress=day_progress ) )

	def merge_batch( self, slices: List[xa.Dataset], constants: xa.Dataset ) -> xa.Dataset:
		constant_vars: List[str] = self.task_config['constants']
		cvars = [vname for vname, vdata in slices[0].data_vars.items() if "time" not in vdata.dims]
		dynamics: xa.Dataset = xa.concat( slices, dim="time", coords = "minimal" )
		dynamics = dynamics.drop_vars(cvars)
		sample: xa.Dataset = slices[0].drop_dims( 'time', errors='ignore' )
		for vname, dvar in sample.data_vars.items():
			if vname not in dynamics.data_vars.keys():
				constants[vname] = dvar
			elif (vname in constant_vars) and ("time" in dvar.dims):
				dvar = dvar.mean(dim="time", skipna=True, keep_attrs=True)
				constants[vname] = dvar
		dynamics = dynamics.drop_vars(constant_vars, errors='ignore')
		return xa.merge( [dynamics, constants], compat='override' )

	def update_cache(self, dates: List[date] ):
		self.batch_cache = { d: self.batch_cache[d] for d in dates if d in self.batch_cache }

	def load_batch( self, dates: List[date], **kwargs ) -> xa.Dataset:
		self.update_cache( dates )
		time_slices: List[xa.Dataset] = [ self.load_dataset( d, **kwargs ) for d in dates ]
		return self.merge_batch( time_slices, self.constants )

	def load_dataset( self, d: date, **kwargs ):
		version = self.task_config['dataset_version']
		if d not in self.batch_cache:
			filepath =  cache_var_filepath(version, d)
			self.batch_cache[d] = self._open_dataset( filepath, **kwargs)
		return self.batch_cache.get(d)

	def _open_dataset(self, filepath: str, **kwargs) -> xa.Dataset:
		dataset: xa.Dataset = xa.open_dataset(filepath, **kwargs)
		model_varname_map = {v: k for k, v in self.task_config['input_variables'].items() if v in dataset.data_vars}
		model_coord_map = {k: v for k, v in self.task_config['coords'].items() if k in dataset.coords}
		return dataset.rename(**model_varname_map, **model_coord_map)

	def load_const_dataset( self, **kwargs ):
		version = self.task_config['dataset_version']
		filepath =  cache_const_filepath(version)
		return self._open_dataset( filepath, **kwargs )

	@classmethod
	def to_feature_array( cls, data_batch: xa.Dataset) -> xa.DataArray:
		features = xa.DataArray(data=list(data_batch.data_vars.keys()), name="features")
		result = xa.concat( list(data_batch.data_vars.values()), dim=features )
		result = result.transpose(..., "features")
		return result






	# def load_timestep( date: date, task: Dict, **kwargs ) -> xa.Dataset:
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






