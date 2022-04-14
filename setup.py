from importlib.metadata import PackageNotFoundError, version

import setuptools

setuptools.setup(
    use_scm_version={
        "version_scheme": "guess-next-dev",
        "local_scheme": "no-local-version",
        "write_to": "tiledb/ml/version.py",
    }
)

try:
    __version__ = version("package-name")
except PackageNotFoundError:
    # package is not installed
    pass
