
"""Module setuptools script."""

from setuptools import setup

description =  "Framework for providing reanalysis data to FoundationModel/DigitalTwin training and inference processes."

setup(
    name="FoundationModelBase",
    version="0.1",
    description=description,
    long_description=description,
    author="NASA Innovation Lab",
    license="Apache License, Version 2.0",
    keywords="Foundation Model Weather Climate",
    url="https://github.com/nasa-nccs-cds/FoundationModelBase.git",
    packages=["fmbase"],
    install_requires=[ "pydap", "numpy", "xarray", "dask", "matplotlib", "scipy", "netCDF4", "hydra-core"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
        "Topic :: Scientific/Engineering :: Physics",
    ],
)
