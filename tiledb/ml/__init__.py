from pkg_resources import DistributionNotFound, get_distribution  # type: ignore

# TODO change pkg_resources with importlib.metadata, as described here
#  https://pypi.org/project/setuptools-scm/#:~:text=Retrieving%20package%20version%20at%20runtime, when we stop
#  supporting Python 3.7. We the aforementioned change, we can avoid the 100ms overhead during import of the package.

try:
    __version__ = get_distribution("tiledb-ml").version
except DistributionNotFound:
    # package is not installed
    pass
