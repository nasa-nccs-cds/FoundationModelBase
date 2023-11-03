
from typing import Any, Dict, List, Tuple, Type, Optional, Union
from fmbase.source.merra2.model import MERRA2DataInterface, YearMonth
from fmbase.util.config import configure
import hydra, xarray as xa
hydra.initialize( version_base=None, config_path="../config" )
configure('explore-test1')

start = YearMonth( 2000, 0 )
end = YearMonth( 2000, 1 )

datasetMgr = MERRA2DataInterface()
example_batch: xa.Dataset = datasetMgr.load_batch(start,end)

print("Loaded Batch:")
for vname, dvar in example_batch.data_vars.items():
	print(f" {vname}{list(dvar.dims)}: shape={dvar.shape}")
