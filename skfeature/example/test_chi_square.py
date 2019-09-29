import scipy.io
from unittest import TestCase, main
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn import svm
from skfeature.function.statistical_based import chi_square


class TestChiSquare(TestCase):

    def test_chi_square(self):
        # load data
        mat = scipy.io.loadmat('../data/BASEHOCK.mat')
        X = mat['X']  # data
        X = X.astype(float)
        y = mat['Y']  # label
        y = y[:, 0]
        n_samples, n_features = X.shape  # number of samples and number of features

        # split data into 10 folds
        kf = KFold(n_splits=10, shuffle=True)

        # perform evaluation on classification task
        num_fea = 100  # number of selected features
        clf = svm.LinearSVC()  # linear SVM

        correct = 0
        for train, test in kf.split(X):
            # obtain the chi-square score of each feature
            idx = chi_square.chi_square(X, y, n_selected_features=num_fea)

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
