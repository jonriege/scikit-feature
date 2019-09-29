import numpy as np
from sklearn.feature_selection import chi2


def chi_square(X, y, **kwargs):
    """
    This function implements the chi-square feature selection (existing method for classification in scikit-learn)

    Input
    -----
    X: {numpy array}, shape (n_samples, n_features)
        input data
    y: {numpy array},shape (n_samples,)
        input class labels
    kwargs: {dictionary}
        n_selected_features: {int}
            the maximum number of selected features returned, the default is the number of input features

    Output
    ------
    F: {numpy array}, shape (n_features,)
        chi-square score for each feature
    """

    n_samples, n_features = X.shape
    n_selected_features = kwargs.get('n_selected_features', n_features)
    F, pval = chi2(X, y)
    return feature_ranking(F, n_selected_features)


def feature_ranking(F, n_selected_features):
    """
    Rank features in descending order according to chi2-score, the higher the chi2-score, the more important the feature is
    """
    idx = np.argsort(F)
    return idx[::-1][:n_selected_features]
