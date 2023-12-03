from fmbase.source.merra2.preprocess import MERRA2DataProcessor, StatsAccumulator
from fmbase.util.config import configure, cfg
from typing import List, Tuple
from multiprocessing import Pool, cpu_count
import hydra

hydra.initialize( version_base=None, config_path="../config" )
configure( 'merra2-finetuning' )
reprocess=True

nproc = cpu_count()-2
years = list( range( 1984, 1985 ) ) # list( range( *cfg().preprocess.year_range ) )
months = list(range(0,2,1))
month_years = [ (month,year) for year in years for month in months  ]

def process( month_year: Tuple[int,int] ) -> StatsAccumulator:
	reader = MERRA2DataProcessor()
	return reader.process_month( month_year[1], month_year[0], reprocess=reprocess)

if __name__ == '__main__':
	print( f"Multiprocessing {len(month_years)} months with {nproc} procs")
	with Pool(processes=nproc) as pool:
		proc_stats: List[StatsAccumulator] = pool.map( process, month_years )
		MERRA2DataProcessor().save_stats(proc_stats)



