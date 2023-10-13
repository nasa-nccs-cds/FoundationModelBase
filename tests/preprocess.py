from fmbase.source.merra2.local.preprocess import MERRA2DataProcessor
from fmbase.util.config import configure
configure( 'explore-test1' )

reader = MERRA2DataProcessor()
reader.process( reprocess=False )