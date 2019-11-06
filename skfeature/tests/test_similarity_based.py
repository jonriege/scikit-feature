import unittest
from sklearn.datasets import load_digits
from sklearn.utils import check_X_y
from skfeature.function.similarity_based.reliefF import ReliefF


class TestSimilarityBased(unittest.TestCase):

    def test_relief_f(self):
        X, y = load_digits(return_X_y=True, n_class=2)
        relief_f = ReliefF()
        X_filtered = relief_f.fit_transform(X, y, n_features=20, threshold=0.01)
        check_X_y(X_filtered, y)


if __name__ == '__main__':
    unittest.main()
