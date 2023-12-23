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
from fmbase.util.logging import lgm, exception_handled, log_timing

def rms( dvar: xa.DataArray, **kwargs ) -> float:
	varray: np.ndarray = dvar.isel( **kwargs, missing_dims="ignore", drop=True ).values
	return np.sqrt( np.mean( np.square( varray ) ) )

def rmse( diff: xa.DataArray, **kw ) -> xa.DataArray:
	rms_error = np.array( [ rms(diff, time=iT, **kw) for iT in range(diff.shape[0]) ] )
	return xa.DataArray( rms_error, dims=['time'], coords={'time': diff.time} )

def cscale( pvar: xa.DataArray, stretch: float = 2.0 ) -> Tuple[float,float]:
	meanv, stdv, minv = pvar.values.mean(), pvar.values.std(), pvar.values.min()
	vmin = max( minv, meanv - stretch*stdv )
	vmax = meanv + stretch*stdv
	return vmin, vmax

@exception_handled
def mplplot( target: xa.Dataset, vnames: List[str], forecast: xa.Dataset = None ):
	print( "Generating Plot")
	ims, pvars, nvars, ptypes = {}, {}, len(vnames), ['']
	time: xa.DataArray = xaformat_timedeltas( target.coords['time'] )
	levels: xa.DataArray = target.coords['level']
	target.assign_coords( time=time )
	if forecast is not None:
		forecast.assign_coords(time=time)
		ptypes = ['target', 'forecast', 'difference']
	ncols =  len( ptypes )
	lslider: ipw.IntSlider = ipw.IntSlider( value=0, min=0, max=levels.size-1, description='Level Index:', )
	tslider: ipw.IntSlider = ipw.IntSlider( value=0, min=0, max=time.size-1, description='Time Index:', )
	print( "mplplot-1")

	with plt.ioff():
		fig, axs = plt.subplots(nrows=nvars, ncols=ncols, sharex=True, sharey=True, figsize=[15, nvars*3], layout="tight")
	for iv, vname in enumerate(vnames):
		print(f" >> {vname}")
		tvar: xa.DataArray = target.data_vars[vname].squeeze(dim="batch", drop=True)
		plotvars = [ tvar ]
		if forecast is not None:
			fvar: xa.DataArray = forecast.data_vars[vname].squeeze(dim="batch", drop=True)
			diff: xa.DataArray = tvar - fvar
			rmserror: xa.DataArray = rmse(diff)
			plotvars = plotvars + [ fvar, diff ]
		vrange = None
		for it, pvar in enumerate( plotvars ):
			ax = axs[ iv, it ]
			ax.set_aspect(0.5)
			if it != 1: vrange = cscale( pvar, 2.0 )
			tslice: xa.DataArray = pvar.isel(time=tslider.value)
			if "level" in tslice.dims:
				tslice = tslice.isel(level=lslider.value)
			ims[(iv,it)] =  tslice.plot.imshow( ax=ax, x="lon", y="lat", cmap='jet', yincrease=True, vmin=vrange[0], vmax=vrange[1]  )
			pvars[(iv,it)] =  pvar
			ax.set_title(f"{vname} {ptypes[it]}")

	print( "mplplot-2")
	@exception_handled
	def time_update(change):
		sindex = change['new']
		lgm().log( f"time_update: tindex={sindex}, lindex={lslider.value}")
		for iv1, vname1 in enumerate(vnames):
			for it1 in range(ncols):
				ax1 = axs[iv1, it1]
				im1, dvar1 = ims[ (iv1, it1) ], pvars[ (iv1, it1) ]
				tslice1: xa.DataArray =  dvar1.isel( level=lslider.value, time=sindex, drop=True, missing_dims="ignore")
				im1.set_data( tslice1.values )
				ax1.set_title(f"{vname1} {ptypes[it1]}")
				lgm().log(f" >> Time-update {vname1} {ptypes[it1]}: level={lslider.value}, time={sindex}, shape={tslice1.shape}")
		fig.canvas.draw_idle()

	@exception_handled
	def level_update(change):
		lindex = change['new']
		lgm().log( f"level_update: lindex={lindex}, tindex={tslider.value}")
		for iv1, vname1 in enumerate(vnames):
			for it1 in range(ncols):
				ax1 = axs[iv1, it1]
				im1, dvar1 = ims[ (iv1, it1) ], pvars[ (iv1, it1) ]
				tslice1: xa.DataArray =  dvar1.isel( level=lindex,time=tslider.value, drop=True, missing_dims="ignore")
				im1.set_data( tslice1.values )
				ax1.set_title(f"{vname1} {ptypes[it1]}")
				lgm().log(f" >> Level-update {vname1} {ptypes[it1]}: level={lindex}, time={tslider.value}, mean={tslice1.values.mean():.4f}, std={tslice1.values.std():.4f}")
		fig.canvas.draw_idle()

	tslider.observe( time_update,  names='value' )
	lslider.observe( level_update, names='value' )
	print( "Generated Plot")
	return ipw.VBox([tslider, lslider, fig.canvas])