from fmbase.source.merra2.preprocess import MERRA2DataProcessor, StatsAccumulator
from fmbase.util.config import configure, cfg
from typing import List, Tuple
from multiprocessing import Pool, cpu_count
import hydra

hydra.initialize( version_base=None, config_path="../config" )
configure( 'merra2-finetuning' )
reprocess=False

nproc = cpu_count()-2
years = list( range( *cfg().preprocess.year_range ) )
months = list(range(0,12,1))
days = list(range(0,31,1))
dates = [ (day,month,year) for year in years for month in months for day in days  ]

def process( date: Tuple[int,int,int] ) -> StatsAccumulator:
	reader = MERRA2DataProcessor()
	return reader.process_day( date, reprocess=reprocess)

if __name__ == '__main__':
	print( f"Multiprocessing {len(dates)} days with {nproc} procs")
	with Pool(processes=nproc) as pool:
		proc_stats: List[StatsAccumulator] = pool.map( process, dates )
		MERRA2DataProcessor().save_stats(proc_stats)



