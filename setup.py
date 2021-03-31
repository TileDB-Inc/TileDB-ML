import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tiledb-ml",
    version="0.1.0",
    author="George Skoumas",
    description="Package that allows saving machine learning models to and from TileDB arrays",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/TileDB-Inc/TileDB-ML",
    project_urls={
        "Bug Tracker": "https://github.com/TileDB-Inc/TileDB-ML/issues",
    },
    test_suite="tests",
    install_requires=[
        "tiledb>=0.8.5",
        "tensorflow>=2.2.0",
        "torch>=1.8.1",
        "scikit-learn>=0.23.0",
        "torchvision>=0.9.1",
    ],
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