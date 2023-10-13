from fmbase.source.merra2.model import MERRA2DataInterface
from fmbase.util.config import configure
configure( 'explore-test1' )

datasetMgr = MERRA2DataInterface()
datasetMgr.load_timestep()

