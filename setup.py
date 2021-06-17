import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tiledb-ml",
    version="0.1.0",
    author="George Skoumas",
    description="Package supports all machine learning functionality for TileDB Embedded and TileDB Cloud",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/TileDB-Inc/TileDB-ML",
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
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.6",
)
