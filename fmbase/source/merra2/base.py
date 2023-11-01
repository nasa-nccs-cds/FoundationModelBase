
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
		self.cfgId = cfg().preprocess.id

	@property
	def data_dir(self):
		return cfg().platform.dataset_root.format(root=cfg().platform.root)

	@property
	def results_dir(self):
		base_dir = cfg().platform.processed.format(root=cfg().platform.root)
		return f"{base_dir}/data/merra2"

	def variable_cache_filepath(self, vname: str,  **kwargs) -> str:
		if "year" in kwargs:
			if "month" in kwargs:   filename = "{varname}_{year}-{month}.nc".format( varname=vname, **kwargs )
			else:                   filename = "{varname}_{year}.nc".format(varname=vname, **kwargs)
		else:	                    filename = "{varname}.nc".format(varname=vname, **kwargs)
		return f"{self.results_dir}/{self.cfgId}/{filename}"

	def stats_filepath(self, varname: str) -> str:
		return f"{self.results_dir}/{self.cfgId}/stats/{varname}.nc"

	def load_cache_var( self, dvar: str, year: int, month: int, **kwargs  ) -> xa.DataArray:
		filepath = self.variable_cache_filepath( dvar, year=year, month=month )
		darray: xa.DataArray = xa.open_dataarray(filepath,**kwargs)
		return darray

	def load_stats( self, dvar: str, **kwargs) -> xa.Dataset:
		filepath = self.stats_filepath(dvar)
		varstats: xa.Dataset = xa.open_dataset(filepath,**kwargs)
		return varstats
