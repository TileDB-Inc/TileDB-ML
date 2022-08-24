# Contributing

Please follow the [Code of Conduct](https://github.com/TileDB-Inc/TileDB-ML/blob/master/CODE_OF_CONDUCT.md)
in all of your interactions with the project.

# Pull Requests

## Formatting

All changes in a PR must be formatted with black, flake8, isort and type checked with mypy. The corresponding versions 
of the aforementioned tools can be found in `pre-commit-config.yaml`. Please, consider installing pre-commit (via `pip install pre-commit`). 
The `pre-commit-config.yaml`, located under TILEDB-ML root, will run `black`, `flake8`, `isort` and `mypy` hooks when committing your changes and will re-format 
your files if needed. After changes are applied you should again git add and git commit the reformatted files.

## Pull Request Annotation

If you are a prospect contributor you are advised to follow the following steps for requesting changes.

- **Title**: `[Enhancement/Feature/Fix]` followed by the PR's title.
- **Description**: A short but as detailed as possible description referring to possible issues fixed, explaining the necessity of the enhancement/feature, which the proposed changes cover.
- **Changed Files**: Files affected by the PR. This will make easier lookup and faster code path check for your reviewer.

## Branch Naming
When pushing to the repository, name your branches with a prefix that is likely unique to your _name_ or _username_ 
followed by `[feature/enhancement/fix/bug]`. For example, `kt/feature/title-of-the-branch`.

## Pull Request Best Practices
- Pull requests should be as small as possible to accomplish a single, well-defined objective. We prefer multiple small pull requests over one large pull request.
- We also expect pull requests to meet the test coverage standards. Mainly we prefer proper tested code that will keep the test coverage statistics to the same or higher levels. 
- Pull request should embody as few as possible external library requirements. It is a good principle to keep the project as self contained as we could.