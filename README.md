ocetrac-unstruct
==============================

`Ocetrac-unstruct` is a Python 3 package which labels and tracks unique geospatial features from gridded datasets. This version has been rewritten based on [ocetrac-dask](https://github.com/wienkers/ocetrac-dask), which has been inspired by [ocetrac](https://github.com/ocetrac/ocetrac).

These major modifications to support long daily timeseries of global _unstructured_ 3D data at increasingly high spatial resolution has been necessitated by the [EERIE project](https://eerie-project.eu). In particular, the grid structure is based on the [ICON](https://www.icon-model.org) model.

For `ocetrac-unstruct`-specific questions, please contact [Aaron Wienkers](mailto:aaron.wienkers@usys.ethz.ch)


Examples Notebooks with Dask
------------
1. ../notebooks/01_preprocess_unstruct.ipynb
2. ../notebooks/02_track_unstruct.ipynb
3. ../notebooks/03_visualise_unstruct.ipynb
   

Installation
------------

**PyPI**

To install the core package run: ``pip install git+https://github.com/wienkers/ocetrac-unstruct.git`` 

**GitHub**

1. Clone ocetrac to your local machine: ``git clone https://github.com/wienkers/ocetrac-unstruct.git``
2. Change to the parent directory of ocetrac
3. Install ocetrac with ``pip install -e ./ocetrac-unstruct``. This will allow
   changes you make locally, to be reflected when you import the package in Python


Future Work
------------
- [ ] Support for 3D MHW tracking