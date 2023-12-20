import hvplot.xarray  # noqa
import numpy as np
import xarray as xa
from typing  import List, Tuple, Union
from fmbase.util.ops import xaformat_timedeltas
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import panel as pn
import ipywidgets as ipw

def plot( ds: xa.Dataset, vname: str, **kwargs ):
	dvar: xa.DataArray = ds.data_vars[vname]
	x = kwargs.get( 'x', 'lon' )
	y = kwargs.get( 'y', 'lat')
	z = kwargs.get( 'z', 'level')
	groupby = [ d for d in ['time',z] if d in dvar.coords ]
	return dvar.hvplot.image(x=x, y=y, groupby=groupby, cmap='jet')

def plot1( ds: xa.Dataset, vname: str, **kwargs ):
	time: xa.DataArray = xaformat_timedeltas( ds.coords['time'] )
	ds.assign_coords( time=time )
	dvar: xa.DataArray = ds.data_vars[vname].squeeze( dim="batch", drop=True )
	x = kwargs.get( 'x', 'lon' )
	y = kwargs.get( 'y', 'lat')
	z = kwargs.get( 'z', 'level')
	print( f"Plotting {vname}{dvar.dims}, shape = {dvar.shape}")
#	tslider = pnw.DiscreteSlider( name='time', options =time.values.tolist() )
#	tslider = pn.widgets.DiscreteSlider( name='time', options =list(range(time.size)) )
	tslider = ipw.IntSlider(description='time', min=0, max=time.size-1)
	return dvar.interactive(loc='bottom').isel(time=tslider).hvplot( cmap='jet', x="lon", y="lat", data_aspect=1 )
	#figure = ( dvar.interactive(loc='bottom').isel(time=tslider).hvplot( cmap='jet', x="lon", y="lat", data_aspect=1 ) )   #.image( x=x, y=y, groupby=groupby, cmap='jet', title=vname )
	#return pn.Column( f"# {vname}", figure ).servable()

def mplplot( fig: Figure, axs, target: xa.Dataset, forecast: xa.Dataset, vnames: List[str], **kwargs):
	time: xa.DataArray = xaformat_timedeltas( target.coords['time'] )
	print( str(type(axs)))
	target.assign_coords( time=time )
	forecast.assign_coords(time=time)
	ims, pvars = {}, {}
	for iv, vname in enumerate(vnames):
		tvar: xa.DataArray = target.data_vars[vname].squeeze(dim="batch", drop=True)
		fvar: xa.DataArray = forecast.data_vars[vname].squeeze(dim="batch", drop=True)
		diff: xa.DataArray = tvar - fvar
		for it, pvar in enumerate( [tvar,fvar,diff] ):
			ax = axs[ iv, it ]
			ax.set_aspect(0.5)
			ims[(iv,it)] =  pvar.isel(time=0).plot.imshow( ax=ax, x="lon", y="lat", cmap='jet', yincrease=True )
			pvars[(iv,it)] =  pvar

	def update(change):
		sindex = change['new']
		tval: str = time.values[sindex]
		for iv1, vname1 in enumerate(vnames):
			for it1, dset1 in enumerate([target, forecast]):
				ax1 = axs[iv1, it1]
				ax1.set_title(tval)
				im1, dvar1 = ims[ (iv1, it1) ], pvars[ (iv1, it1) ]
				im1.set_data( dvar1.isel(time=sindex).values )
				fig.canvas.draw_idle()

	slider = ipw.IntSlider( value=0, min=0, max=time.size-1 )
	slider.observe(update, names='value')
	return ipw.VBox([slider, fig.canvas])