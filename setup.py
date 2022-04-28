import setuptools

tensorflow = ["tensorflow<=2.5.0"]
pytorch = ["torch>=1.8.1, <1.10"]
sklearn = ["scikit-learn>=0.23.0"]
cloud = ["tiledb-cloud>=0.7.11"]
full = sorted({"torchvision>=0.9.1", *tensorflow, *pytorch, *sklearn, *cloud})

setuptools.setup(
    setup_requires=["setuptools_scm <= 6.0"],
    use_scm_version={
        "version_scheme": "guess-next-dev",
        "local_scheme": "dirty-tag",
        "write_to": "tiledb/ml/version.py",
    },
    install_requires=["sparse", "tiledb >= 0.14"],
    extras_require={
        "tensorflow": tensorflow,
        "pytorch": pytorch,
        "sklearn": sklearn,
        "cloud": cloud,
        "full": full,
    },
)
