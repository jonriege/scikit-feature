import unittest
from sklearn.datasets import load_digits, load_iris
from sklearn.utils import check_X_y
from sklearn.feature_selection import SelectKBest
from skfeature.function.statistical_based import CFS, chi_square, f_score, gini_index, t_score


class TestStatisticalBased(unittest.TestCase):

    def test_CFS(self):
        X, y = load_iris(return_X_y=True)
        X_filtered = SelectKBest(CFS.cfs, k=2).fit_transform(X, y)
        check_X_y(X_filtered, y)

    def test_chi_square(self):
        X, y = load_digits(return_X_y=True, n_class=2)
        X_filtered = SelectKBest(chi_square.chi_square, k=5).fit_transform(X, y)
        check_X_y(X_filtered, y)

    def test_f_score(self):
        X, y = load_iris(return_X_y=True)
        X_filtered = SelectKBest(f_score.f_score, k=2).fit_transform(X, y)
        check_X_y(X_filtered, y)

    def test_gini_index(self):
        X, y = load_digits(return_X_y=True, n_class=2)
        X_filtered = SelectKBest(gini_index.gini_index, k=5).fit_transform(X, y)
        check_X_y(X_filtered, y)

    def test_t_score(self):
        X, y = load_digits(return_X_y=True, n_class=2)
        X_filtered = SelectKBest(t_score.t_score, k=5).fit_transform(X, y)
        check_X_y(X_filtered, y)


if __name__ == '__main__':
    unittest.main()
