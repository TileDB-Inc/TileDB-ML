<a href="https://tiledb.com"><img src="https://github.com/TileDB-Inc/TileDB/raw/dev/doc/source/_static/tiledb-logo_color_no_margin_@4x.png" alt="TileDB logo" width="400"></a>

[![TileDB-ML CI](https://github.com/TileDB-Inc/TileDB-ML/actions/workflows/ci.yml/badge.svg)](https://github.com/TileDB-Inc/TileDB-ML/actions/workflows/ci.yml)
![Coverage Badge](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/ktsitsi/2506f6c9d3375e2d636cf594340d11bf/raw/gistfile.json)

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
can (or will be able to): 

* **Save** machine learning models as TileDB arrays (At the moment we provide implementations for saving Tensorflow Keras, PyTorch and Scikit-Learn models.)
  
* **Load** machine learning models from TileDB arrays.     

* **Read** features, in order to train machine learning models, from TileDB arrays directly to machine learning framework's data APIs. 
  We already [support](https://github.com/TileDB-Inc/TileDB-ML/blob/master/tiledb/ml/readers/) the Tensorflow and PyTorch
  data APIs with the use of Python generators for Dense and Sparse TileDB arrays, and we are similarly working on Scikit-Learn 
  Pipelines which will be out soon.
  
## Examples

[comment]: <> (## Structure)
[comment]: <> (At the moment we provide code for saving and loading models to and from TileDB arrays and for loading features from TileDB arrays )

[comment]: <> (into Tensorflow Data API. The corresponding implementations for model save/load, live in ``tiledb/ml/models`` folder. )

[comment]: <> (All implemented classes &#40;``TensorflowKerasTileDBModel``, ``PyTorchTileDBModel``, ``SklearnTileDBModel`` &#41; )

[comment]: <> (inherit from base class &#40;``TileDBModel``&#41; and implement ``save&#40;&#41;`` and ``load&#40;&#41;`` functionality. )

[comment]: <> (In case you would like to contribute model save/load implementations)

[comment]: <> (that support other machine learning frameworks, please take a look at the current implementations and commit code accordingly. Please)

[comment]: <> (also read the contributing section below.)

We provide some detailed notebook examples on how to save and load machine learning models as TileDB arrays (also on TileDB-Cloud) and explain why 
this is useful in order to create simple and flexible model registries with TileDB.

* [Example for Tensorflow Keras Models](https://github.com/TileDB-Inc/TileDB-ML/blob/master/examples/models/tensorflow_keras_tiledb_models_example.ipynb)
* [Example for PyTorch Models](https://github.com/TileDB-Inc/TileDB-ML/blob/master/examples/models/pytorch_tiledb_models_example.ipynb)
* [Example for Scikit-Learn Models](https://github.com/TileDB-Inc/TileDB-ML/blob/master/examples/models/sklearn_tiledb_models_example.ipynb)
* [Example for Tensorflow Model on TileDB-Cloud](https://github.com/TileDB-Inc/TileDB-ML/blob/master/examples/cloud/tensorflow_tiledb_cloud_ml_model_array.ipynb)
* [Example for PyTorch Model on TileDB-Cloud](https://github.com/TileDB-Inc/TileDB-ML/blob/master/examples/cloud/pytorch_tiledb_cloud_ml_model_array.ipynb)
* [Example for Scikit-Learn Model on TileDB-Cloud](https://github.com/TileDB-Inc/TileDB-ML/blob/master/examples/cloud/sklearn_tiledb_cloud_ml_model_array.ipynb)


We also provide detailed notebook examples on how to train Tensorflow and PyTorch models with the use of our Data APIs support for Dense and Sparse TileDB arrays.

* [Example on training wih Tensorflow and Dense TileDB arrays](https://github.com/TileDB-Inc/TileDB-ML/blob/master/examples/readers/tensorflow_data_api_tiledb_dense.ipynb)
* [Example on training wih Tensorflow and Sparse TileDB arrays](https://github.com/TileDB-Inc/TileDB-ML/blob/master/examples/readers/tensorflow_data_api_tiledb_sparse.ipynb)
* [Example on training wih PyTorch and Dense TileDB arrays](https://github.com/TileDB-Inc/TileDB-ML/blob/master/examples/readers/pytorch_data_api_tiledb_dense.ipynb)
* [Example on training wih PyTorch and Sparse TileDB arrays](https://github.com/TileDB-Inc/TileDB-ML/blob/master/examples/readers/pytorch_data_api_tiledb_sparse.ipynb)


## Installation

TileDB-ML can be installed:

### Quick Installation

- from source by cloning the [Git](https://github.com/TileDB-Inc/TileDB-ML) repository:

      git clone https://github.com/TileDB-Inc/TileDB-ML.git
      cd TileDB-ML
  
      # In case you want to install and check all frameworks. If you
      # use zsh replace .[full] with .\[full\]
      pip install -e .[full]

      # In case you want to install and check Tensorflow only. If you
      # use zsh replace .[tensorflow] with .\[tensorflow\]
      pip install -e .[tensorflow]

      # In case you want to install and check PyTorch only. If you
      # use zsh replace .[pytorch] with .\[pytorch\]
      pip install -e .[pytorch]

      # In case you want to install and check Scikit-Learn only. If you
      # use zsh replace .[sklearn] with .\[sklearn\]
      pip install -e .[sklearn]  

      # In case you want to try any of the aforementioned machine learning framework
      # on TileDB-Cloud try one of the follwoing.
      pip install -e .[tensorflow_cloud]
      pip install -e .[pytorch_cloud]
      pip install -e .[sklearn_cloud]

- with pip from git:

      pip install git+https://github.com/TileDB-Inc/TileDB-ML.git@master

- from PyPi:

[comment]: <> (TileDB-ML is available from either [PyPI]&#40;https://test.pypi.org/project/tiledb-ml/0.1.2.2/&#41; with ``pip``:)

  ```
  pip install tiledb-ml
  ```
  The above command will just install the basic dependency of `tiledb-ml`, hence `tiledb`.
  In order to install the integration for a specific framework you need to use:
  
  ```
  pip install tiledb-ml[pytorch] # e.g. For checking only the Pytorch integration
  ```
  
  Checking all the supported frameworks you will need to use:

  ```
  pip install tiledb-ml[full]
  ```
  
  The above commands apply to `bash` shell in case you use `zsh` you will 
  need to escape the `bracket` character like the following for example:
  
  ```
  pip install tiledb-ml\[pytorch\]
  ```
  
- You may run the test suite with:
  ```
  python setup.py test
  ```
## Roadmap

We are already working on the following:

* C++ integration of TileDB with the Tensorflow Data API through [tensorflow-io](https://github.com/tensorflow/io).
* Readers from TileDB arrays to other popular machine learning framework Data APIs, as mentioned above.
* Model save/load support for other popular machine learning frameworks like XGBoost and CatBoost.

Our ultimate goal is ALL machine learning data, from raw data (text, images, audio), to features (Feature Store) and models (Model Registry), represented, stored and managed
in one **Data Engine**, i.e, TileDB.


## Note

Here we would like to highlight that our current implementations are not optimal, and they don't support the aforementioned machine learning
frameworks 100%, e.g., serialization of model parts like numpy arrays, takes place with Pickle (which is far from optimal)
because of its ``Python Only`` nature and insecurity as described [here](https://docs.python.org/3/library/pickle.html).
We mainly provide a proof of concept, showing the universal data management ability of TileDB, and how elegantly applies in 
machine learning data of any kind. Optimizations will follow as soon as possible.

In any case, note that the TileDB-ML repository is under development, and **the API is subject to change**.


## Contributing

We welcome all contributions! Please read the [contributing guidelines](https://github.com/TileDB-Inc/TileDB-ML/blob/master/CONTRIBUTING.md) 
before submitting pull requests.

## Copyright

The TileDB-ML package is Copyright 2018-2021 TileDB, Inc

## License

MIT
