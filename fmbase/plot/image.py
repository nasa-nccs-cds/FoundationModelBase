import hvplot.xarray  # noqa
import xarray as xa
def plot( ds: xa.Dataset, vname: str, **kwargs ):
	dvar: xa.DataArray = ds.data_vars[vname]
	x = kwargs.get( 'x', 'lon' )
	y = kwargs.get( 'y', 'lat')
	z = kwargs.get( 'z', 'level')
	groupby = [ d for d in ['time',z] if d in dvar.coords ]
	return dvar.hvplot.image(x=x, y=y, groupby=groupby, cmap='jet')