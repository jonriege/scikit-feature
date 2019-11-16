import unittest
from sklearn.datasets import load_iris
from sklearn.utils import check_X_y
from sklearn.feature_selection import SelectKBest
from skfeature.function.streaming import alpha_investing


class TestStreaming(unittest.TestCase):

    def test_alpha_investing(self):
        X, y = load_iris(return_X_y=True)
        X_filtered = SelectKBest(alpha_investing.alpha_investing, k=2).fit_transform(X, y)
        check_X_y(X_filtered, y)


if __name__ == '__main__':
    unittest.main()
