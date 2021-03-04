<a href="https://tiledb.com"><img src="https://github.com/TileDB-Inc/TileDB/raw/dev/doc/source/_static/tiledb-logo_color_no_margin_@4x.png" alt="TileDB logo" width="400"></a>

# TileDB-ML

TileDB-ML is the repository that contains all machine learning oriented functionality TileDB supports. In this repo, we explain how someone can employ 
TileDB for machine learning oriented data management problems, and which are the next steps we have in mind. Here, we would firstly like to highlight our 
perspective on the relation of TileDB with general machine learning oriented data management problems and how TileDB engine could be the solution for 
efficiently storing any kind of machine learning data, i.e., from raw images, text, audio, time series and SAR to features and machine learning models. 
Before you proceed further, please take a quick look on our [medium blog post](https://medium.com/tiledb/tiledb-as-the-data-engine-for-machine-learning-b48fb0e9b147), 
which targets to explain in great detail how TileDB addresses many machine learning data format requirements, overcoming the drawbacks of the other 
candidate formats, and take this opportunity to solicit feedback and contributions from the community.

## Description

As mentioned above, this repository contains all machine learning oriented functionality TileDB supports. Specifically, code that 
can (will be able to): 

* Save machine learning models as TileDB arrays. At the moment we provide proof of concept (not optimal) implementations 
  for saving Tensorflow Keras, PyTorch and Scikit-Learn models as TileDB arrays. In our next release we will try to also support
  other machine learning frameworks like MXNET, XGBoost and CatBoost while also optimize the already existing implementations.
    
* Load features in order to train machine learning models, from TileDB arrays directly to machine learning framework's data APIs. 
  This feature is NOT supported at the moment, but we are already working on integrations with Tensorflow data API, PyTorch DataLoader API
  and Scikit-Learn Pipelines. 
  
## Structure

At them moment we provide code for saving and loading Tensorflow, PyTorch and Scikit-Learn models as and from TileDB arrays. The corresponding implementations
live in models folder. All implemented Python Classes (tensorflow_models, pytorch_models.py, sklearn_models.py ) inherit from 
Base Class (base_class) and implement save() and load() functionality. In case you would like to contribute model save/load implementations
that support other machine learning frameworks, please take a look at the current implementations and commit code accordingly. Please
also read the contributing section below.

Here we would like to highlight again that our current implementations are not optimal, and they don't support the aforementioned machine learning
frameworks 100%, e.g., serialization of model parts like numpy arrays takes place with Pickle [(which is far from optimal)](https://docs.python.org/3/library/pickle.html).
We mainly provide a proof of concept, showing the universal data management nature of TileDB, and how is elegantly applies in machine learning data of any kind.

In any case, note that the TileDB-ML repository is under development, and the API is subject to change.

## Quick links for machine learning model save/load examples.

We provide some detailed notebook examples on how to save and load machine learning models as TileDB arrays and explain why this is useful 
in order to create simple and flexible model registries with TileDB.

* [Tensorflow Keras Models](https://github.com/TileDB-Inc/TileDB-ML/blob/develop/example_notebooks/models/tensorflow_keras_tiledb_models_example.ipynb)
* [PyTorch Models](https://github.com/TileDB-Inc/TileDB-ML/blob/develop/example_notebooks/models/pytorch_tiledb_models_example.ipynb)
* [Scikit-Learn Models](https://github.com/TileDB-Inc/TileDB-ML/blob/develop/example_notebooks/models/sklearn_tiledb_models_example.ipynb)

## Roadmap

* We are already working on model save/load support for other popular machine learning frameworks.
* We are already working on readers from TileDB arrays to popular machine learning framework Data APIs, as mentioned above.

Our ultimate goal is ALL machine learning data, from raw text, images, audio, to features (Feature Store) and models (Model Registry), represented, stored and managed
in one and only format, i.e, TileDB.


## Contributing

We welcome all contributions! Please read the [contributing guidelines](https://github.com/TileDB-Inc/TileDB-ML/blob/develop/CONTRIBUTING.md) 
before submitting pull requests.

## Copyright

The TileDB-ML package is Copyright 2018-2021 TileDB, Inc

## License

MIT
