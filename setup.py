import setuptools
from pkg_resources import DistributionNotFound, get_distribution  # type: ignore

setuptools.setup(
    use_scm_version={
        "version_scheme": "guess-next-dev",
        "local_scheme": "dirty-tag",
        "write_to": "tiledb/ml/version.py",
    }
)

try:
    __version__ = get_distribution("tiledb-ml").version
except DistributionNotFound:
    # package is not installed
    pass
