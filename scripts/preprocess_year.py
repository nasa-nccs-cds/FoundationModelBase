from fmbase.source.merra2.preprocess import MERRA2DataProcessor, StatsAccumulator
from fmbase.util.config import configure
from typing import List
from multiprocessing import Pool, cpu_count
from fmbase.util.config import cfg
import hydra

hydra.initialize( version_base=None, config_path="../config" )
configure( 'merra2-test' )

nproc = round(cpu_count()*0.9)
year = 1985
months = range(1,13,1)

def process( month: int ) -> StatsAccumulator:
	reader = MERRA2DataProcessor()
	return reader.process_year( year, month=month, reprocess=True )

if __name__ == '__main__':
	print( f"Multiprocessing months {months} for year {year} with {nproc} procs")
	with Pool(processes=nproc) as pool:
		proc_stats: List[StatsAccumulator] = pool.map( process, months )
		MERRA2DataProcessor().save_stats(proc_stats)



