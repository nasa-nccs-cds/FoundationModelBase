import xarray
import xarray as xa
import numpy as np
from omegaconf import DictConfig, OmegaConf
from fmbase.util.config import cfg
from typing import List, Union, Tuple, Optional, Dict, Type
import hydra, glob, sys, os, time
from fmbase.source.merra2.base import MERRA2Base
from fmbase.util.ops import get_levels_config
from dataclasses import dataclass

@dataclass(eq=True,repr=True,frozen=True,order=True)
class YearMonth:
	year: int
	month: int

class MERRA2DataInterface(MERRA2Base):

	def __init__(self):
		MERRA2Base.__init__(self)

	def load_batch(self, start: YearMonth, end: YearMonth) -> xa.Dataset:
		slices: List[xa.Dataset] = []
		for year in range( start.year,end.year+1):
			month_range = [0,12]
			if year == start.year: month_range[0] = start.month
			if year == end.year:   month_range[1] = end.month + 1
			for month in range( *month_range ):
				slices.append( self.load_timestep( year, month ) )
		return self.merge_batch( slices )

	@classmethod
	def merge_batch(cls, slices: List[xa.Dataset] ) -> xa.Dataset:
		merged: xa.Dataset = xa.concat( slices, dim="time", coords = "minimal" )
		sample: xa.Dataset = slices[0]
		for vname, dvar in sample.data_vars.items():
			if vname not in merged.data_vars.keys():
				merged[vname] = dvar
		return merged

	def load_timestep(self, year: int, month: int, **kwargs ) -> xa.Dataset:
		vlist: Dict[str, List] = cfg().dataset.get('vars')
		levels: np.ndarray = get_levels_config(cfg().task)
		tsdata, coords, taxis = {}, {}, None
		print(f"load_timestep({month}/{year})")
		for (collection,vlist) in vlist.items():
			for vname in vlist:
				varray: xa.DataArray = self.load_cache_var(vname, year, month, **kwargs)
				coords.update( varray.coords )
				if taxis is None:
					assert 'time' in varray.coords.keys(), f"Constant DataArray can't be first in model vars configuration: {vname}"
					taxis = varray.coords['time']
				print( f"load_var({collection}.{vname}): name={varray.name}, shape={varray.shape}, dims={varray.dims}, levels={levels}")
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

	@classmethod
	def to_feature_array(cls, data_batch: xa.Dataset) -> xa.DataArray:
		features = xa.DataArray(data=list(data_batch.data_vars.keys()), name="features")
		result = xa.concat( list(data_batch.data_vars.values()), dim=features )
		result = result.transpose(..., "features")
		return result

	def load_norm_data(self) -> Dict[str,xa.Dataset]:
		vlist: Dict[str, List] = cfg().dataset.get('vars')
		mean, std, coords, result = {}, {}, {}, {}
		for (collection,vlist) in vlist.items():
			for vname in vlist:
				stats: xa.Dataset = self.load_stats(vname)
				mean[vname] = stats['mean']
				std[vname] = stats['stds']
				coords.update( stats.coords )
		result['mean'] = xa.Dataset(mean, coords)
		result['std'] = xa.Dataset(std, coords)
		return result





