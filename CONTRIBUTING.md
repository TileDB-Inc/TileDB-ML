# Contributing
Thank you very much for your interest in TileDB-ML!

 - Please follow the [Code of Conduct](https://github.com/TileDB-Inc/TileDB-ML/blob/master/CODE_OF_CONDUCT.md)
in all of your interactions with the project.
 - By contributing code to TileDB-ML, you are agreeing to release it under the [MIT License](https://github.com/TileDB-Inc/TileDB-ML/tree/master/LICENSE).


# Pull Requests
- Please submit [pull requests](https://help.github.com/en/desktop/contributing-to-projects/creating-a-pull-request) against the default [`master` branch of TileDB-ML](https://github.com/TileDB-Inc/TileDB-ML/tree/master)

## Formatting

We recommend formatting your PR changes with black, flake8, isort and type checked with mypy. The corresponding versions 
of the aforementioned tools can be found in `pre-commit-config.yaml`. Please, consider installing pre-commit (via `pip install pre-commit`). 
The `pre-commit-config.yaml`, located under TILEDB-ML root, will run `black`, `flake8`, `isort` and `mypy` hooks when committing 
your changes and will re-format your files if needed. After changes are applied you should again git add and git commit the reformatted files.
 
## Nice-to-Have
 
- Please provide a short but as detailed as possible description of your PR, explaining the necessity which the proposed changes cover.
- Please keep your pull requests as small as possible to accomplish a single and well-defined objective. 
- Please try to meet the test coverage standards, i.e., the same or higher test coverage statistics. 