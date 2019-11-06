import unittest
from sklearn.datasets import load_iris
from sklearn.utils import check_X_y
from sklearn.naive_bayes import GaussianNB
from skfeature.function.wrapper.forward_selection import ForwardSelection


class TestWrapper(unittest.TestCase):

    def test_forward_selection(self):
        X, y = load_iris(return_X_y=True)
        estimator = GaussianNB()
        forward_selection = ForwardSelection(estimator)
        X_filtered = forward_selection.fit_transform(X, y)
        check_X_y(X_filtered, y)


if __name__ == '__main__':
    unittest.main()
