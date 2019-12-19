import unittest
from sklearn.datasets import load_digits
from sklearn.utils import check_X_y
from sklearn.feature_selection import SelectKBest
from skfeature.function.similarity import fisher_score, lap_score, reliefF, trace_ratio, SPEC


class TestSimilarity(unittest.TestCase):

    def test_fisher_score(self):
        X, y = load_digits(return_X_y=True, n_class=2)
        X_filtered = SelectKBest(fisher_score.fisher_score, k=10).fit_transform(X, y)
        check_X_y(X_filtered, y)

    def test_lap_score(self):
        X, y = load_digits(return_X_y=True, n_class=2)
        X_filtered = SelectKBest(lap_score.lap_score, k=10).fit_transform(X, y)
        check_X_y(X_filtered, y)

    def test_relieff(self):
        X, y = load_digits(return_X_y=True, n_class=2)
        X_filtered = SelectKBest(reliefF.relieff, k=10).fit_transform(X, y)
        check_X_y(X_filtered, y)

    def test_trace_ratio(self):
        X, y = load_digits(return_X_y=True, n_class=3)
        X_filtered = SelectKBest(trace_ratio.trace_ratio, k=10).fit_transform(X, y)
        check_X_y(X_filtered, y)

    def test_SPEC(self):
        X, y = load_digits(return_X_y=True, n_class=3)
        X_filtered = SelectKBest(SPEC.spec, k=10).fit_transform(X, y)
        check_X_y(X_filtered, y)


if __name__ == '__main__':
    unittest.main()
