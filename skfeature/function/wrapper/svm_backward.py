import numpy as np
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning


@ignore_warnings(category=ConvergenceWarning)
def svm_backward(X, y, **kwargs):
    """
    This function implements the backward feature selection algorithm based on SVM

    Input
    -----
    X: {numpy array}, shape (n_samples, n_features)
        input data
    y: {numpy array}, shape (n_samples,)
        input class labels
    kwargs: {dictionary}
        cv: {int}
            the number of folds when calculating accuracy through cross validation (default: 5)
        n_selected_features: {int}
            the maximum number of selected features returned (default: n_features)

    Output
    ------
    F: {numpy array}, shape (n_features, )
        index of selected features
    """

    n_samples, n_features = X.shape

    cv = kwargs.get('cv', 3)
    n_selected_features = kwargs.get('n_selected_features', n_features)
    clf = LinearSVC()

    # selected feature set, initialized to contain all features
    F = list(range(n_features))
    count = n_features

    while count > n_selected_features:
        max_acc = 0
        idx = None
        for i in range(n_features):
            if i in F:
                F.remove(i)
                X_tmp = X[:, F]
                acc = cross_val_score(clf, X_tmp, y, cv=cv, n_jobs=-1).mean()
                F.append(i)
                # record the feature which results in the largest accuracy
                if acc > max_acc:
                    max_acc = acc
                    idx = i
        # delete the feature which results in the largest accuracy
        F.remove(idx)
        count -= 1
    return np.array(F)


