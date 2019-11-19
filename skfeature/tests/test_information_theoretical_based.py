import unittest
from sklearn.datasets import load_digits, load_iris
from sklearn.utils import check_X_y
from sklearn.feature_selection import SelectKBest
from skfeature.function.information_theoretical_based import MRMR, JMI


class MyTestCase(unittest.TestCase):

    def test_MRMR(self):
        X, y = load_digits(return_X_y=True, n_class=2)
        X_filtered = SelectKBest(MRMR.mrmr, k=10).fit_transform(X, y)
        check_X_y(X_filtered, y)

    def test_JMI(self):
        X, y = load_digits(return_X_y=True, n_class=2)
        X_filtered = SelectKBest(JMI.jmi, k=10).fit_transform(X, y)
        check_X_y(X_filtered, y)


if __name__ == '__main__':
    unittest.main()
