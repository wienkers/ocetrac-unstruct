from setuptools import setup


setup(
    name='ocetrac-dask',
    use_scm_version={
        "write_to": "ocetrac_dask/_version.py",
        "write_to_template": '__version__ = "{version}"',
        "tag_regex": r"^(?P<prefix>v)?(?P<version>[^\+]+)(?P<suffix>.*)?$",
    }
)
