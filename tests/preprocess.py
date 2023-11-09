from fmbase.source.merra2.local.preprocess import MERRA2DataProcessor, StatsAccumulator
from fmbase.util.config import configure
from typing import List, Union, Tuple, Optional, Dict, Type
from multiprocessing import Pool, cpu_count
from fmbase.util.config import cfg
import hydra, xarray as xa
hydra.initialize( version_base=None, config_path="../config" )
configure( 'explore-exp1' )

year_range = cfg().preprocess.year_range
years = list(range(*year_range))
nproc = cpu_count() - 1

def process( year: int ) -> StatsAccumulator:
	reader = MERRA2DataProcessor()
	return reader.process_year( year, reprocess=False )

if __name__ == '__main__':
	print( f"Multiprocessing {len(years)} years with {nproc} procs")
	with Pool(processes=nproc) as pool:
		proc_stats: List[StatsAccumulator] = pool.map( process, years )
		MERRA2DataProcessor().save_stats(proc_stats)

