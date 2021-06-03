import tiledb

from tiledb.ml.models.sklearn import SklearnTileDB

tiledb_model = SklearnTileDB(uri='tiledb-sklearn-mnist-6', ctx=tiledb.cloud.Ctx(), namespace="demo")
print(tiledb_model.load())
