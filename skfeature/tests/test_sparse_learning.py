import unittest
from sklearn.datasets import load_digits
from sklearn.utils import check_X_y, check_array
from sklearn.feature_selection import SelectKBest
from skfeature.function.sparse_learning import ll_l21, ls_l21, MCFS, NDFS, RFS, UDFS


class TestSparseLearning(unittest.TestCase):

    def test_ll_l21_proximal_gradient_descent(self):
        X, y = load_digits(return_X_y=True, n_class=2)
        X_filtered = SelectKBest(ll_l21.proximal_gradient_descent, k=10).fit_transform(X, y)
        check_X_y(X_filtered, y)

    def test_ls_l21_proximal_gradient_descent(self):
        X, y = load_digits(return_X_y=True, n_class=2)
        X_filtered = SelectKBest(ls_l21.proximal_gradient_descent, k=10).fit_transform(X, y)
        check_X_y(X_filtered, y)

    def test_RFS(self):
        X, y = load_digits(return_X_y=True, n_class=2)
        X_filtered = SelectKBest(RFS.rfs, k=10).fit_transform(X, y)
        check_X_y(X_filtered, y)

    def test_MCFS(self):
        X, y = load_digits(return_X_y=True, n_class=2)
        X_filtered = SelectKBest(MCFS.mcfs, k=10).fit_transform(X, y)
        check_X_y(X_filtered, y)

    def test_NDFS(self):
        X, y = load_digits(return_X_y=True, n_class=2)
        X_filtered = SelectKBest(NDFS.ndfs, k=10).fit_transform(X, y)
        check_X_y(X_filtered, y)

    def test_UDFS(self):
        X, y = load_digits(return_X_y=True, n_class=2)
        X_filtered = SelectKBest(UDFS.udfs, k=10).fit_transform(X, y)
        check_X_y(X_filtered, y)


if __name__ == '__main__':
    unittest.main()
