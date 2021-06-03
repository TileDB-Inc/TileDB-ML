import tiledb

from tiledb.ml.models.sklearn import SklearnTileDB

tiledb_model = SklearnTileDB(uri='tiledb-sklearn-mnist-5', ctx=tiledb.cloud.Ctx(), namespace="demo")
tiledb_model.delete_model()