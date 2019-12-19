import unittest
from sklearn.datasets import load_digits
from sklearn.utils import check_X_y
from sklearn.feature_selection import SelectKBest
from skfeature.function.random_forest import rf_impurity, rf_permutation


class TestRandomForest(unittest.TestCase):

    def test_rf_impurity(self):
        X, y = load_digits(return_X_y=True, n_class=2)
        X_filtered = SelectKBest(rf_impurity.rf_impurity, k=10).fit_transform(X, y)
        check_X_y(X_filtered, y)

    def test_rf_permutation(self):
        X, y = load_digits(return_X_y=True, n_class=2)
        X_filtered = SelectKBest(rf_permutation.rf_permutation, k=10).fit_transform(X, y)
        check_X_y(X_filtered, y)
