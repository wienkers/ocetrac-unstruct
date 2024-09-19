ocetrac-dask
==============================

Ocetrac-dask is a Python 3.6+ package based off of [ocetrack](https://github.com/ocetrac/ocetrac) used to label and track unique geospatial features from gridded datasets. This version has been modified to accept larger-tha-memory spatio-temporal datasets and process them in parallel using [dask](https://dask.org/). It avoids loop-carried dependencies in time, keeps dask arrays distributed in memory throughout, and leverages the [dask-image](https://github.com/dask/dask-image) library. These modifications has allowed preliminary scaling to 40 years of data on 1024 cores. 
Future work will support 3D tracking in time.

A citation to the original ocetrac software is [here](https://doi.org/10.5281/zenodo.5102928).


Installation
------------

**PyPI**

To install the core package run: ``pip install git+https://github.com/wienkers/ocetrac-dask.git`` 

**GitHub**

1. Clone ocetrac to your local machine: ``git clone https://github.com/wienkers/ocetrac-dask.git``
2. Change to the parent directory of ocetrac
3. Install ocetrac with ``pip install -e ./ocetrac-dask``. This will allow
   changes you make locally, to be reflected when you import the package in Python.
   
