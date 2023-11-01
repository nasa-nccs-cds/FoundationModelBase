import xarray
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

	def load_timestep(self, year: int, month: int, **kwargs ) -> xa.DataArray:
		vlist: Dict[str, List] = cfg().model.get('vars')
		levels: np.ndarray = get_levels_config(cfg().model)
		tsdata = {}
		print(f"load_timestep({month}/{year})")
		for (collection,vlist) in vlist.items():
			for vname in vlist:
				varray: xa.DataArray = self.load_cache_var(vname, year, month, **kwargs)

				print( f"load_var({collection}.{vname}): name={varray.name}, shape={varray.shape}, dims={varray.dims}, levels={levels}")
				if 'z' in varray.dims:
					print(f" ---> Levels Coord= {varray.coords['z'].values.tolist()}")
					levs: List[str] = varray.coords['z'].values.tolist() if levels is None else levels
					for lev in levs:
						tsdata[f"{vname}.{lev}"] = varray.sel( z=lev, method="nearest", drop=True )
				else:
					tsdata[vname] = varray
		features = xa.DataArray( data=list(tsdata.keys()), name="features" )
		print( f"Created coord {features.name}: shape={features.shape}, dims={features.dims}, Features:" )
		for fname, fdata in tsdata.items():
			print( f" ** {fname}{fdata.dims}: shape={fdata.shape}")
		result = xa.concat( list(tsdata.values()), dim=features )
		return result.rename( {result.dims[0]: "features"} ).transpose(..., "features")

	def load_norm_data(self):
		vlist: Dict[str, List] = cfg().model.get('vars')
		locations, scales, coords, result = {}, {}, {}, {}
		for (collection,vlist) in vlist.items():
			for vname in vlist:
				stats: xa.Dataset = self.load_stats(vname)
				locations[vname] = stats['location']
				scales[vname] = stats['scales']
				coords.update( stats.coords )
		result['locations'] = xa.Dataset(locations, coords)
		result['scales'] = xa.Dataset(scales, coords)
		return result





