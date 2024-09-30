ocetrac-unstruct
==============================

`Ocetrac-unstruct` is a Python 3+ package which labels and tracks unique geospatial features from _unstructured_ datasets. This version has been rewritten based on the daskified [ocetrac-dask](https://github.com/wienkers/ocetrac-dask), which has originally been inspired by [ocetrac](https://github.com/ocetrac/ocetrac).

These major modifications to support long daily timeseries of global _unstructured_ 3D data at increasingly high spatial resolution has been necessitated by the [EERIE project](https://eerie-project.eu). In particular, the native grid structure is based on the [ICON](https://www.icon-model.org) model, although the library has been written in a sufficiently general and flexible way.

For `ocetrac-unstruct`-specific questions, please contact [Aaron Wienkers](mailto:aaron.wienkers@usys.ethz.ch)


Examples Notebooks with Dask
------------
1. ../notebooks/01_preprocess_unstruct.ipynb --- Preprocessing of data leveraging Dask. 14000 daily snapshots of 2D 5km _unstructured_ (15 million cells) model outputs processed in ~10 minutes on 512 cores.
2. ../notebooks/02_track_unstruct.ipynb --- Track MHWs using dask-powered ocetrac algorithm. 14000 daily snapshots of 2D 5km _unstructured_ (15 million cells) model outputs processed in ~6 minutes on 128 cores.
3. ../notebooks/03_visualise_unstruct.ipynb --- Some dask-backed visualisation routines
   

Installation
------------

**PyPI**

To install the core package run: ``pip install git+https://github.com/wienkers/ocetrac-unstruct.git`` 
To install optional unstructured plotting library run: ``pip install git+https://gitlab.dkrz.de/b382615/pyicon.git``

**GitHub**

1. Clone ocetrac-unstruct to your local machine: ``git clone https://github.com/wienkers/ocetrac-unstruct.git``
2. Change to the parent directory of ocetrac-unstruct
3. Install ocetrac with ``pip install -e ./ocetrac-unstruct``. This will allow
   changes you make locally, to be reflected when you import the package in Python


Future Work
------------
- [ ] Support for 3D MHW tracking