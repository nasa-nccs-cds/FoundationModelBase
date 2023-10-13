from fmbase.source.merra2.model import MERRA2DataInterface
from fmbase.util.config import configure
import xarray as xa
configure( 'explore-test1' )

datasetMgr = MERRA2DataInterface()
tsdata: xa.DataArray = datasetMgr.load_timestep( 2000, 0 )

