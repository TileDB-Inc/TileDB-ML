import importlib.metadata

try:
    __version__ = importlib.metadata.version("tiledb-ml")
except importlib.metadata.PackageNotFoundError:
    __version__ = ""
