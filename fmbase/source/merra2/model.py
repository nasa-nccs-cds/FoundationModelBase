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
		self._stats: Dict[str,xa.DataArray] = {}
		self._snames = ['location','scale']

	def stat( self, sname ) -> Optional[xa.DataArray]:
		return self._stats.get( sname )

	def load_timestep(self, year: int, month: int, **kwargs ) -> xa.DataArray:
		vlist: Dict[str, List] = cfg().model.get('vars')
		levels: np.ndarray = get_levels_config(cfg().model)
		tsdata = {}
		print(f"load_timestep({month}/{year})")
		for (collection,vlist) in vlist.items():
			for vname in vlist:
				dset: xa.Dataset = self.load_cache_var(vname, year, month, **kwargs)
				varray: xa.DataArray = dset.data_vars[vname]
				for sname in self._snames: self._stats[sname] = dset.data_vars[sname]
				print( f"load_var({collection}.{vname}): name={varray.name}, shape={varray.shape}, dims={varray.dims}, levels={levels}")
				if 'z' in varray.dims:
					print(f" ---> Levels Coord= {varray.coords['z'].values.tolist()}")
					levs: List[str] = varray.coords['z'].values.tolist() if levels is None else levels
					for lev in levs:
						tsdata[f"{vname}.{lev}"] = varray.sel( z=lev, method="nearest", drop=True )
				else:
					tsdata[vname] = varray
		features = xa.DataArray( data=list(tsdata.keys()), name="features" )
		print( f"Created coord {features.name}: shape={features.shape}, dims={features.dims} data={list(tsdata.keys())}")
		print(f" --> values={features.values.tolist()}")
		result = xa.concat( list(tsdata.values()), dim=features )
		return result.rename( {result.dims[0]: "features"} ).transpose(..., "features")




