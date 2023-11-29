from fmbase.source.merra2.preprocess import MERRA2DataProcessor, StatsAccumulator
from fmbase.util.config import configure
from typing import List, Tuple
from multiprocessing import Pool, cpu_count
from fmbase.util.config import cfg
import hydra

hydra.initialize( version_base=None, config_path="../config" )
configure( 'merra2-test' )

nproc = round(cpu_count()*0.9)
years = list( range( 1986, 2021 ) )
months = list(range(0,12,1))
month_years = [ (month,year) for year in years for month in months  ]

def process( month_year: Tuple[int,int] ) -> StatsAccumulator:
	reader = MERRA2DataProcessor()
	return reader.process_year( month_year[1], month=month_year[0], reprocess=True )

if __name__ == '__main__':
	print( f"Multiprocessing {len(month_years)} months with {nproc} procs")
	with Pool(processes=nproc) as pool:
		proc_stats: List[StatsAccumulator] = pool.map( process, month_years )
		MERRA2DataProcessor().save_stats(proc_stats)



