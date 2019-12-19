from sklearn.feature_selection import chi2


def chi_square(X, y):
    """
    This function implements the chi-square feature selection (existing method for classification in scikit-learn)
    Input
    -----
    X: {numpy array}, shape (n_samples, n_features)
        input data
    y: {numpy array},shape (n_samples,)
        input class labels
    Output
    ------
    F: {numpy array}, shape (n_features,)
        chi-square score for each feature
    """
    return chi2(X, y)