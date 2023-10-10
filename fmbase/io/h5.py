import h5netcdf
import numpy as np

with h5netcdf.File('mydata.nc', 'w') as f:
    # set dimensions with a dictionary
    f.dimensions = {'x': 5}
    # and update them with a dict-like interface
    # f.dimensions['x'] = 5
    # f.dimensions.update({'x': 5})

    v = f.create_variable('hello', ('x',), float)
    v[:] = np.ones(5)

    # you don't need to create groups first
    # you also don't need to create dimensions first if you supply data
    # with the new variable
    v = f.create_variable('/grouped/data', ('y',), data=np.arange(10))

    # access and modify attributes with a dict-like interface
    v.attrs['foo'] = 'bar'

    # you can access variables and groups directly using a hierarchical
    # keys like h5py
    print(f['/grouped/data'])

    # add an unlimited dimension
    f.dimensions['z'] = None
    # explicitly resize a dimension and all variables using it
    f.resize_dimension('z', 3)