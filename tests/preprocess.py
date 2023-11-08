from fmbase.source.merra2.local.preprocess import MERRA2DataProcessor
from fmbase.util.config import configure
import hydra, xarray as xa
hydra.initialize( version_base=None, config_path="../config" )
configure( 'explore-test1' )

reader = MERRA2DataProcessor()
reader.process( reprocess=True )
