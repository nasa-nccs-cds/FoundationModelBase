import xarray as xa
import numpy as np
from omegaconf import DictConfig, OmegaConf
from fmbase.util.config import cfg
from typing import List, Union, Tuple, Optional, Dict, Type
import hydra, glob, sys, os, time
from fmbase.source.merra2.base import MERRA2Base

class MERRA2DataInterface(MERRA2Base):

	def __init__(self):
		MERRA2Base.__init__(self)
		self.levels: np.array = np.array(list(cfg().model.get('levels'))).sort()
		self.year_range: List = cfg().model.get('year_range')
		self.month_range: List = cfg().model.get('month_range')
		self.vlist: Dict[str, List] = cfg().model.get('vars')
		self.vars: Dict[str, xa.DataArray] = {}

	def load( self, dvar: str, collection: str, **kwargs  ) -> xa.DataArray:      # year: int, month: int
		filepath = self.variable_cache_filepath( dvar, collection, **kwargs )
		variable: xa.DataArray = xa.open_dataarray(filepath)
		return variable


