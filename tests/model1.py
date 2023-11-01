import numpy as np

from fmbase.source.merra2.model import MERRA2DataInterface
from fmbase.util.config import configure
import xarray as xa
configure( 'explore-test1' )

datasetMgr = MERRA2DataInterface()
tsdata: xa.DataArray = datasetMgr.load_timestep( 2000, 0 )
print( f"LOADED TRAIN DATA: shape={tsdata.shape}, dims={tsdata.dims}")

varname = 'T'
tstats: xa.Dataset = datasetMgr.load_stats( varname )
for statname in ['location', 'scale']:
	stat: xa.DataArray = tstats[statname]
	print( f"LOADED {varname} STATS: {statname} shape={stat.shape}, dims={stat.dims}")
