
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

	def variable_cache_filepath(self, vname: str, collection: str, **kwargs) -> str:
		if "year" in kwargs:
			if "month" in kwargs:   filename = "{varname}_{year}-{month}.nc".format( varname=vname, **kwargs )
			else:                   filename = "{varname}_{year}.nc".format(varname=vname, **kwargs)
		else:	                    filename = "{varname}.nc".format(varname=vname, **kwargs)
		return f"{self.results_dir}/{self.cfgId}/{collection}/{filename}"