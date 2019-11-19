import unittest
from sklearn.datasets import load_iris
from sklearn.utils import check_X_y
from sklearn.feature_selection import SelectKBest
from skfeature.function.wrapper import forward_selection, backward_selection


class TestWrapper(unittest.TestCase):

    def test_forward_selection(self):
        X, y = load_iris(return_X_y=True)
        X_filtered = SelectKBest(forward_selection.forward_selection, k=2).fit_transform(X, y)
        check_X_y(X_filtered, y)

    def test_backward_selection(self):
        X, y = load_iris(return_X_y=True)
        X_filtered = SelectKBest(backward_selection.backward_selection, k=2).fit_transform(X, y)
        check_X_y(X_filtered, y)


if __name__ == '__main__':
    unittest.main()
