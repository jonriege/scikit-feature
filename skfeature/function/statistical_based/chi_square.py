import numpy as np
from sklearn.feature_selection import chi2
from sklearn.preprocessing import normalize


class ChiSquare:

    def __init__(self):
        self.scores = None

    def fit_transform(self, X, y, n_features=None, threshold=0):
        self.fit(X, y)
        return self.transform(X, n_features, threshold)

    def transform(self, X, n_features=None, threshold=0):
        """
        Rank features in descending order according to chi2-score, the higher the chi2-score, the more
        important the feature is
        """

        sorted_indices_rev = np.argsort(self.scores)
        sorted_indices = np.flip(sorted_indices_rev)
        filtered_indices = [i for i in sorted_indices if self.scores[i] > threshold]
        if n_features is not None:
            filtered_indices = filtered_indices[:n_features]
        return X[:, filtered_indices]

    def fit(self, X, y):
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
        """
        f, p = chi2(X, y)
        scores = np.nan_to_num(f)
        normalized_scores = normalize([scores], norm='l1')[0]
        self.scores = normalized_scores
