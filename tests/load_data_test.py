from fmbase.source.merra2.local.reader import MERRADataProcessor
from fmbase.util.config import configure
configure( 'explore-test1' )

reader = MERRADataProcessor()
reader.process( 'inst3_3d_asm_Np', )