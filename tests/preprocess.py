from fmbase.source.merra2.local.preprocess import MERRADataProcessor
from fmbase.util.config import configure
configure( 'explore-test1' )
reprocess = True

reader = MERRADataProcessor()
reader.process()