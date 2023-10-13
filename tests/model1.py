import numpy as np

from fmbase.source.merra2.model import MERRA2DataInterface
from fmbase.util.config import configure
import xarray as xa
configure( 'explore-test1' )

datasetMgr = MERRA2DataInterface()
tsdata: xa.DataArray = datasetMgr.load_timestep( 2000, 0 )
csamples: np.ndarray = tsdata.coords['samples'].values
print( f"RESULT: shape={tsdata.shape}, dims={tsdata.dims}, samples={csamples.tolist()}")
