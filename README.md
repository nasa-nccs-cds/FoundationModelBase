
# Foundation Model Base

Framework for feeding data from various sources to FoundationModel/DigitalTwin training and inference processes.

## Objectives

- Establish common structure for various FoundationModel/DigitalTwin projects.
- Create generic data access and formating routines for use across projects

## Conda Environment

    > conda create -n fmbase -c conda-forge 
    > conda activate fmbase
    > conda install -c conda-forge pydap numpy xarray dask matplotlib scipy
    > pip install netCDF4
    > pip install hydra-core --upgrade


