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

def mplplot( target: xa.Dataset, forecast: xa.Dataset, vnames: List[str], **kwargs):
	time: xa.DataArray = xaformat_timedeltas( target.coords['time'] )
	levels: xa.DataArray = target.coords['level']
	target.assign_coords( time=time )
	forecast.assign_coords(time=time)
	ptypes = ['target', 'forecast', 'difference']
	ims, pvars, nvars = {}, {}, len(vnames)
	lslider: ipw.IntSlider = ipw.IntSlider( value=0, min=0, max=levels.size-1, description='Level Index:', )
	with plt.ioff():
		fig, axs = plt.subplots(nrows=nvars, ncols=3, sharex=True, sharey=True, figsize=[15, nvars*3], layout="tight")
	for iv, vname in enumerate(vnames):
		tvar: xa.DataArray = target.data_vars[vname].squeeze(dim="batch", drop=True)
		fvar: xa.DataArray = forecast.data_vars[vname].squeeze(dim="batch", drop=True)
		diff: xa.DataArray = tvar - fvar
		for it, pvar in enumerate( [tvar,fvar,diff] ):
			ax = axs[ iv, it ]
			ax.set_aspect(0.5)
			tslice: xa.DataArray = pvar.isel(time=0)
			if "level" in tslice.dims:
				tslice = tslice.isel(level=lslider.value)
			ims[(iv,it)] =  tslice.plot.imshow( ax=ax, x="lon", y="lat", cmap='jet', yincrease=True )
			pvars[(iv,it)] =  pvar

	def update(change):
		sindex = change['new']
		tval: str = time.values[sindex]
		for iv1, vname1 in enumerate(vnames):
			for it1 in range(3):
				ax1 = axs[iv1, it1]
				ax1.set_title( f"{vname1} {ptypes[it1]}: {tval}")
				im1, dvar1 = ims[ (iv1, it1) ], pvars[ (iv1, it1) ]
				tslice1: xa.DataArray =  dvar1.isel(time=sindex)
				if "level" in tslice1.dims:
					tslice1 = tslice1.isel(level=lslider.value)
				im1.set_data( tslice1.values )
				fig.canvas.draw_idle()

	tslider = ipw.IntSlider( value=0, min=0, max=time.size-1, description='Time Index:', )
	tslider.observe(update, names='value')
	return ipw.VBox([tslider, lslider, fig.canvas])