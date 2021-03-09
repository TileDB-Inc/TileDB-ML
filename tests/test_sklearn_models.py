import unittest
import tempfile
import numpy as np

from sklearn import datasets
from sklearn.svm import LinearSVC, SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from tiledb.ml.models.sklearn import SklearnTileDB

iris = datasets.load_iris()


class TestSaveLoadTileDBModel(unittest.TestCase):
    def test_save_load_svc(self):
        with tempfile.TemporaryDirectory() as tiledb_array:
            model = SVC()
            model.fit(iris.data, iris.target)

            tiledb_sklearn_obj = SklearnTileDB(uri=tiledb_array)
            tiledb_sklearn_obj.save(model=model)

            loaded_model = tiledb_sklearn_obj.load()

            model_pred = model.predict(iris.data)
            loaded_model_pred = loaded_model.predict(iris.data)

            self.assertTrue(np.array_equal(model_pred, loaded_model_pred))

    def test_save_load_linear_svc(self):
        with tempfile.TemporaryDirectory() as tiledb_array:
            model = LinearSVC()
            model.fit(iris.data, iris.target)

            tiledb_sklearn_obj = SklearnTileDB(uri=tiledb_array)
            tiledb_sklearn_obj.save(model=model)

            loaded_model = tiledb_sklearn_obj.load()

            model_pred = model.predict(iris.data)
            loaded_model_pred = loaded_model.predict(iris.data)

            self.assertTrue(np.array_equal(model_pred, loaded_model_pred))

    def test_save_load_multinomial_nb(self):
        with tempfile.TemporaryDirectory() as tiledb_array:
            model = MultinomialNB()
            model.fit(iris.data, iris.target)

            tiledb_sklearn_obj = SklearnTileDB(uri=tiledb_array)
            tiledb_sklearn_obj.save(model=model)

            loaded_model = tiledb_sklearn_obj.load()

            model_pred = model.predict(iris.data)
            loaded_model_pred = loaded_model.predict(iris.data)

            self.assertTrue(np.array_equal(model_pred, loaded_model_pred))

    def test_save_load_decision_tree_regressor(self):
        with tempfile.TemporaryDirectory() as tiledb_array:
            model = DecisionTreeRegressor()
            model.fit(iris.data, iris.target)

            tiledb_sklearn_obj = SklearnTileDB(uri=tiledb_array)
            tiledb_sklearn_obj.save(model=model)

            loaded_model = tiledb_sklearn_obj.load()

            model_pred = model.predict(iris.data)
            loaded_model_pred = loaded_model.predict(iris.data)

            self.assertTrue(np.array_equal(model_pred, loaded_model_pred))

    def test_save_load_decision_tree_classifier(self):
        with tempfile.TemporaryDirectory() as tiledb_array:
            model = DecisionTreeClassifier()
            model.fit(iris.data, iris.target)

            tiledb_sklearn_obj = SklearnTileDB(uri=tiledb_array)
            tiledb_sklearn_obj.save(model=model)

            loaded_model = tiledb_sklearn_obj.load()

            model_pred = model.predict(iris.data)
            loaded_model_pred = loaded_model.predict(iris.data)

            self.assertTrue(np.array_equal(model_pred, loaded_model_pred))

    def test_save_load_with_update(self):
        with tempfile.TemporaryDirectory() as tiledb_array:
            model = LinearRegression()
            model.fit(iris.data, iris.target)

            tiledb_sklearn_obj = SklearnTileDB(uri=tiledb_array)
            tiledb_sklearn_obj.save(model=model)

            loaded_model = tiledb_sklearn_obj.load()

            model_pred = model.predict(iris.data)
            loaded_model_pred = loaded_model.predict(iris.data)

            self.assertTrue(np.array_equal(model_pred, loaded_model_pred))

            model = ElasticNet()
            model.fit(iris.data, iris.target)

            tiledb_sklearn_obj = SklearnTileDB(uri=tiledb_array)
            tiledb_sklearn_obj.save(model=model, update=True)

            loaded_model = tiledb_sklearn_obj.load()

            model_pred = model.predict(iris.data)
            loaded_model_pred = loaded_model.predict(iris.data)

            self.assertTrue(np.array_equal(model_pred, loaded_model_pred))
