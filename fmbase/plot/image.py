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
	print( f"Plotting {vname}{dvar.dims}, shape = {dvar.shape}")
	tslider = pnw.DiscreteSlider( name='time', options =time.values.tolist() )
	return dvar.interactive(loc='bottom').sel(time=tslider).hvplot( cmap='jet', data_aspect=1 )    #.image( x=x, y=y, groupby=groupby, cmap='jet', title=vname )