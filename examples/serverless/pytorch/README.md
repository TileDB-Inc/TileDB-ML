<a href="https://tiledb.com"><img src="https://github.com/TileDB-Inc/TileDB/raw/dev/doc/source/_static/tiledb-logo_color_no_margin_@4x.png" alt="TileDB logo" width="400"></a>

This is an end to end example of how to ingest training data, train a model and use it for predictions serverless-ly, by employing UDFs on TileDB-Cloud. 
In this example, we work on the popular mnist dataset with TileDB-ML API for dense arrays, i.e., data and labels will be stored as dense TileDB arrays. 
Firstly, we ingest all training images and labels in TileDB arrays and register them on TileDB-Cloud. We continue by serverless-ly train a model for image classification and save (and register) it as a TileDB array on TileDB-Cloud, 
and finally, we serverless-ly get some predictions using the trained model. In case you want to run the example, you will need a TileDB-Cloud account as described
[here](https://docs.tiledb.com/cloud/tutorials/start-here). After signing up, you should export your username and password 
as environmental variables (**TILEDB_USER_NAME**, **TILEDB_PASSWD**), in order to run ingestion, model training and prediction UDFs. Moreover,
please add your TileDB namespace and your **S3** bucket in each script.

# Steps

* First, run **data_ingestion.py**. This script will serverless-ly ingest mnist data in two TileDB arrays (one for images, one for labels), created in your S3 bucket, and register them on your TileDB-Cloud account. After running this script, you should be able to see a **mnist_images** array and a **mnist_labels** array in your **Arrays** asset on Tile-DB cloud.
* Continuing, run **model_training.py**, in order to serverless-ly train a model for classification on the mnist dataset. After running this script, you should be able to see a **mnist_model** array in your **ML Models** asset on TileDB-Cloud.
* Finally, run **model_load_and_predict.py**, in order to serverless-ly get some predictions using the trained model on your laptop.
