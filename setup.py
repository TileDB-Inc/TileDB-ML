import setuptools

tensorflow = ["tensorflow>=2.6"]
pytorch = ["torch>=1.10"]
sklearn = ["scikit-learn>=1.0"]
cloud = ["tiledb-cloud"]
full = sorted({"torchvision", *tensorflow, *pytorch, *sklearn, *cloud})

setuptools.setup(
    setup_requires=["setuptools_scm"],
    use_scm_version={
        "version_scheme": "guess-next-dev",
        "local_scheme": "dirty-tag",
        "write_to": "tiledb/ml/version.py",
    },
    install_requires=["sparse", "tiledb>=0.14", "wrapt"],
    extras_require={
        "tensorflow": tensorflow,
        "pytorch": pytorch,
        "sklearn": sklearn,
        "cloud": cloud,
        "full": full,
    },
)
