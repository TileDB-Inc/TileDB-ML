import numpy as np
import tiledb

from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

from tiledb.ml.models.sklearn import SklearnTileDB


train_samples = 5000

# Load data from https://www.openml.org/d/554
print('Data fetching...')
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)

random_state = check_random_state(0)
permutation = random_state.permutation(X.shape[0])
X = X[permutation]
y = y[permutation]
X = X.reshape((X.shape[0], -1))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=train_samples, test_size=10000)

print('Data scaling...')
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

clf = LogisticRegression(
    C=50. / train_samples, penalty='l1', solver='saga', tol=0.1
)

print('Model fit...')
clf.fit(X_train, y_train)

print('Model score...')
sparsity = np.mean(clf.coef_ == 0) * 100
score = clf.score(X_test, y_test)

print("Sparsity with L1 penalty: %.2f%%" % sparsity)
print("Test score with L1 penalty: %.4f" % score)

tiledb_model_1 = SklearnTileDB(uri='tiledb-sklearn-mnist', ctx=tiledb.cloud.Ctx(), namespace="demo")

tiledb_model_1.save(model=clf,
                    meta={'Sparsity_with_L1_penalty': sparsity,
                    'score': score})