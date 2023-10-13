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
		levs = cfg().model.get('levels')
		self.levels: np.array = None if levs is None else np.array(list(levs)).sort()
		self.vlist: Dict[str, List] = cfg().model.get('vars')

	def load( self, dvar: str, collection: str, year, month, **kwargs  ) -> xa.DataArray:      # year: int, month: int
		filepath = self.variable_cache_filepath( dvar, collection, year=year, month=month )
		variable: xa.DataArray = xa.open_dataarray(filepath,**kwargs)
		return variable

	def load_timestep(self, year: int, month: int, **kwargs ) -> xa.DataArray:
		tsdata = []
		for (collection,vlist) in self.vlist.items():
			for var in vlist:
				varray: xa.DataArray = self.load(var, collection, year, month, **kwargs)
				print( f"load_timestep({month}/{year}): var={varray.name}, shape={varray.shape}, dims={varray.dims}")
				tsdata.append( varray )
		return None
#		return xa.concat(tsdata, dim="features")


