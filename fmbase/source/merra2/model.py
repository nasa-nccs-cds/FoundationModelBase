import xarray as xa
import numpy as np
from omegaconf import DictConfig, OmegaConf
from fmbase.util.config import cfg
from typing import List, Union, Tuple, Optional, Dict, Type
import hydra, glob, sys, os, time
from fmbase.source.merra2.base import MERRA2Base
from fmbase.util.ops import get_levels_config

class MERRA2DataInterface(MERRA2Base):

	def __init__(self):
		MERRA2Base.__init__(self)
		self.levels: np.ndarray = get_levels_config(cfg().model)
		self.vlist: Dict[str, List] = cfg().model.get('vars')

	def load( self, dvar: str, collection: str, year, month, **kwargs  ) -> xa.DataArray:      # year: int, month: int
		filepath = self.variable_cache_filepath( dvar, collection, year=year, month=month )
		variable: xa.DataArray = xa.open_dataarray(filepath,**kwargs)
		return variable

	def load_timestep(self, year: int, month: int, **kwargs ) -> xa.DataArray:
		tsdata = {}
		print(f"load_timestep({month}/{year})")
		for (collection,vlist) in self.vlist.items():
			for vname in vlist:
				varray: xa.DataArray = self.load(vname, collection, year, month, **kwargs)
				print( f"load_var({collection}.{vname}): name={varray.name}, shape={varray.shape}, dims={varray.dims}, levels={self.levels}")
				if 'z' in varray.dims:
					print(f" ---> Levels Coord= {varray.coords['z'].values.tolist()}")
					levs: List[str] = varray.coords['z'].values.tolist() if self.levels is None else self.levels
					for lev in levs:
						tsdata[f"{vname}.{lev}"] = varray.sel( z=lev, method="nearest", drop=True )
				else:
					tsdata[vname] = varray
		samples = xa.DataArray( data=list(tsdata.keys()), name="samples" )
		print( f"Created coord {samples.name}: shape={samples.shape}, dims={samples.dims} data={list(tsdata.keys())}")
		print(f" --> values={samples.values.tolist()}")
		result = xa.concat( list(tsdata.values()), dim=samples )
		return result.rename( {result.dim[0]: "samples"} )


