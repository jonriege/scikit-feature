import scipy.io
from unittest import TestCase, main
from sklearn.model_selection import KFold
from sklearn import svm
from sklearn.metrics import accuracy_score
from skfeature.function.similarity import fisher_score
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning


class TestFisherScore(TestCase):

    @ignore_warnings(category=ConvergenceWarning)
    def test_fisher_score(self):
        # load data
        mat = scipy.io.loadmat('../data/COIL20.mat')
        X = mat['X']  # data
        X = X.astype(float)
        y = mat['Y']  # label
        y = y[:, 0]

        # split data into 10 folds
        kf = KFold(n_splits=10, shuffle=True)

        # perform evaluation on classification task
        clf = svm.LinearSVC()  # linear SVM

        correct = 0
        for train, test in kf.split(X):
            # obtain the score of each feature on the training set
            idx = fisher_score.fisher_score(X[train], y[train], n_selected_features=100)

            # obtain the dataset on the selected features
            selected_features = X[:, idx]

            # train a classification model with the selected features on the training dataset
            clf.fit(selected_features[train], y[train])

            # predict the class labels of test data
            y_predict = clf.predict(selected_features[test])

            # obtain the classification accuracy on the test data
            acc = accuracy_score(y[test], y_predict)
            correct = correct + acc

        # output the average classification accuracy over all 10 folds
        print('Accuracy:', float(correct) / 10)


if __name__ == '__main__':
    main()
