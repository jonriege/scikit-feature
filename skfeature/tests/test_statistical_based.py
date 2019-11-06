import unittest
from sklearn.datasets import load_digits
from sklearn.utils import check_X_y
from skfeature.function.statistical_based.chi_square import ChiSquare


class TestStatisticalBased(unittest.TestCase):

    def test_chi_square(self):
        X, y = load_digits(return_X_y=True, n_class=2)
        chi_square = ChiSquare()
        X_filtered = chi_square.fit_transform(X, y, n_features=20, threshold=0.01)
        check_X_y(X_filtered, y)


if __name__ == '__main__':
    unittest.main()
