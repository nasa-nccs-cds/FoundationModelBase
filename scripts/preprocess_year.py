from fmbase.source.merra2.preprocess import MERRA2DataProcessor, StatsAccumulator
from fmbase.util.config import configure
from typing import List, Tuple
from multiprocessing import Pool, cpu_count
import hydra

hydra.initialize( version_base=None, config_path="../config" )
configure( 'merra2-finetuning' )

nproc = cpu_count()-2
years = list( range( 1984, 2022 ) )
months = list(range(0,12,1))
month_years = [ (month,year) for year in years for month in months  ]

def process( month_year: Tuple[int,int] ) -> StatsAccumulator:
	reader = MERRA2DataProcessor()
	return reader.process_year( month_year[1], month=month_year[0], reprocess=False )

if __name__ == '__main__':
	print( f"Multiprocessing {len(month_years)} months with {nproc} procs")
	with Pool(processes=nproc) as pool:
		proc_stats: List[StatsAccumulator] = pool.map( process, month_years )
		MERRA2DataProcessor().save_stats(proc_stats)



