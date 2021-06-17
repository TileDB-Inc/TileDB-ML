from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    README_MD = fh.read()

setup(
    name="tiledb-ml",
    description="Package supports all machine learning functionality for TileDB Embedded and TileDB Cloud",
    long_description=README_MD,
    long_description_content_type="text/markdown",
    author="TileDB, Inc.",
    author_email="help@tiledb.io",
    maintainers="TileDB, Inc.",
    maintainer_email="help@tiledb.io",
    url="https://github.com/TileDB-Inc/TileDB-ML",
    license="MIT",
    platforms=["any"],
    use_scm_version={
        "version_scheme": "guess-next-dev",
        "local_scheme": "dirty-tag",
        "write_to": "tiledb-ml/version.py",
    },
    project_urls={
        "Bug Tracker": "https://github.com/TileDB-Inc/TileDB-ML/issues",
    },
    test_suite="tests",
    install_requires=[
        "tiledb>=0.9.0",
    ],
    extras_require={
        "tensorflow": ["tensorflow>=2.5.0"],
        "pytorch": ["torch>=1.8.1", "torchvision>=0.9.1"],
        "sklearn": ["scikit-learn>=0.23.0"],
        "tensorflow_cloud": ["tensorflow>=2.5.0", "tiledb-cloud>=0.7.11"],
        "pytorch_cloud": ["torch>=1.8.1", "torchvision>=0.9.1", "tiledb-cloud>=0.7.11"],
        "sklearn_cloud": ["scikit-learn>=0.23.0", "tiledb-cloud>=0.7.11"],
        "full": [
            "tensorflow>=2.5.0",
            "torch>=1.8.1",
            "torchvision>=0.9.1",
            "scikit-learn>=0.23.0",
            "pytest>=6.2.4",
            "tiledb-cloud>=0.7.11",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Operating System :: Unix",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS :: MacOS X",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
    ],
    packages=find_packages(),
    python_requires=">=3.6",
)
