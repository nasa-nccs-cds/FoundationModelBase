from fmbase.source.merra2.preprocess import MERRA2DataProcessor, StatsAccumulator
from fmbase.util.config import configure, cfg
from typing import List, Tuple
from fmbase.util.config import Date
from fmbase.source.merra2.model import cache_const_filepath
from multiprocessing import Pool, cpu_count
import hydra, os

hydra.initialize( version_base=None, config_path="../config" )
configure( 'merra2-finetuning' )
reprocess=False
nproc = cpu_count()-2

def process( date: Date ) -> StatsAccumulator:
	reader = MERRA2DataProcessor()
	return reader.process_day( date, reprocess=reprocess)

if __name__ == '__main__':
	dates: List[Date] = Date.get_dates()
	print( f"Multiprocessing {len(dates)} days with {nproc} procs")
	if reprocess: os.remove( cache_const_filepath(cfg().preprocess.version) )
	with Pool(processes=nproc) as pool:
		proc_stats: List[StatsAccumulator] = pool.map( process, dates )
		MERRA2DataProcessor().save_stats(proc_stats)



