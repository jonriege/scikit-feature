import unittest
from sklearn.datasets import load_digits, load_iris
from sklearn.utils import check_X_y
from sklearn.feature_selection import SelectKBest
from skfeature.function.information_theory import MRMR, JMI, CIFE, CMIM, DISR, FCBF, ICAP, MIFS, MIM


class TestInformationTheory(unittest.TestCase):

    def test_MRMR(self):
        X, y = load_digits(return_X_y=True, n_class=2)
        X_filtered = SelectKBest(MRMR.mrmr, k=10).fit_transform(X, y)
        check_X_y(X_filtered, y)

    def test_JMI(self):
        X, y = load_digits(return_X_y=True, n_class=2)
        X_filtered = SelectKBest(JMI.jmi, k=10).fit_transform(X, y)
        check_X_y(X_filtered, y)

    def test_CIFE(self):
        X, y = load_digits(return_X_y=True, n_class=2)
        X_filtered = SelectKBest(CIFE.cife, k=10).fit_transform(X, y)
        check_X_y(X_filtered, y)

    def test_CMIM(self):
        X, y = load_digits(return_X_y=True, n_class=2)
        X_filtered = SelectKBest(CMIM.cmim, k=10).fit_transform(X, y)
        check_X_y(X_filtered, y)

    def test_DISR(self):
        X, y = load_digits(return_X_y=True, n_class=2)
        X_filtered = SelectKBest(DISR.disr, k=10).fit_transform(X, y)
        check_X_y(X_filtered, y)

    def test_FCBF(self):
        X, y = load_digits(return_X_y=True, n_class=2)
        X_filtered = SelectKBest(FCBF.fcbf, k=10).fit_transform(X, y)
        check_X_y(X_filtered, y)

    def test_ICAP(self):
        X, y = load_digits(return_X_y=True, n_class=2)
        X_filtered = SelectKBest(ICAP.icap, k=10).fit_transform(X, y)
        check_X_y(X_filtered, y)

    def test_MIFS(self):
        X, y = load_digits(return_X_y=True, n_class=2)
        X_filtered = SelectKBest(MIFS.mifs, k=10).fit_transform(X, y)
        check_X_y(X_filtered, y)

    def test_MIM(self):
        X, y = load_digits(return_X_y=True, n_class=2)
        X_filtered = SelectKBest(MIM.mim, k=10).fit_transform(X, y)
        check_X_y(X_filtered, y)


if __name__ == '__main__':
    unittest.main()
