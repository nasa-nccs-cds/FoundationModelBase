import hvplot.xarray  # noqa
import xarray as xa
from fmbase.util.ops import xaformat_timedeltas
import panel.widgets as pnw


def plot( ds: xa.Dataset, vname: str, **kwargs ):
	time: xa.DataArray = xaformat_timedeltas( ds.coords['time'] )
	ds.assign_coords( time=time )
	dvar: xa.DataArray = ds.data_vars[vname]
	x = kwargs.get( 'x', 'lon' )
	y = kwargs.get( 'y', 'lat')
	z = kwargs.get( 'z', 'level')
	groupby = [ d for d in ['time',z] if d in dvar.coords ]

	time = pnw.Player(name='time', start=0, end=time.size, loop_policy='loop', interval=100)
	return dvar.interactive(loc='bottom').isel(time=time).image( x=x, y=y, groupby=groupby, cmap='jet', title=vname )