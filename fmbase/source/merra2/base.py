
import xarray as xa
import numpy as np
from omegaconf import DictConfig, OmegaConf
import linecache
from fmbase.util.ops import vrange
from pathlib import Path
from fmbase.util.config import cfg
from typing import List, Union, Tuple, Optional, Dict, Type
import hydra, glob, sys, os, time
from fmbase.io.nc4 import nc4_write_array

class MERRA2Base:

	def __init__(self):
		pass

	@property
	def data_dir(self):
		return cfg().platform.dataset_root.format(**cfg().platform)

	@property
	def results_dir(self):
		base_dir = cfg().platform.processed.format(**cfg().platform)
		return f"{base_dir}/data/merra2"

	def variable_cache_filepath(self, version: str, vname: str,  **kwargs) -> str:
		if "year" in kwargs:
			if "month" in kwargs:   filename = "{varname}_{year}-{month}.nc".format( varname=vname, **kwargs )
			else:                   filename = "{varname}_{year}.nc".format(varname=vname, **kwargs)
		else:	                    filename = "{varname}.nc".format(varname=vname, **kwargs)
		return f"{self.results_dir}/{version}/{filename}"

	def stats_filepath(self, version: str, statname: str) -> str:
		return f"{self.results_dir}/{version}/stats/{statname}.nc"

	def load_cache_var( self, version: str, dvar: str, year: int, month: int, **kwargs  ) -> xa.DataArray:
		coord_map: Dict = kwargs.pop('coords', {})
		filepath = self.variable_cache_filepath( version, dvar, year=year, month=month )
		darray: xa.DataArray = xa.open_dataarray(filepath,**kwargs)
		cmap: Dict = { k:v for k,v in coord_map.items() if k in darray.coords.keys()}
		return darray.rename(cmap)

	def load_stats( self, version: str, statname: str, **kwargs) -> xa.Dataset:
		filepath = self.stats_filepath(version,statname)
		varstats: xa.Dataset = xa.open_dataset(filepath,**kwargs)
		return varstats
