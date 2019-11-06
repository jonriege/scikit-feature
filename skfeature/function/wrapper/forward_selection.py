import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import normalize


class ForwardSelection:

    def __init__(self, estimator, n_features=None, threshold=0):
        self.estimator = estimator
        self.n_features = n_features
        self.threshold = threshold
        self.scores = None

    def fit_transform(self, X, y, n_features=None, threshold=0):
        self.fit(X, y)
        return self.transform(X, n_features, threshold)

    def transform(self, X, n_features=None, threshold=0):
        sorted_indices_rev = np.argsort(self.scores)
        sorted_indices = np.flip(sorted_indices_rev)
        filtered_indices = [i for i in sorted_indices if self.scores[i] > threshold]
        if n_features is not None:
            filtered_indices = filtered_indices[:n_features]
        return X[:, filtered_indices]

    def fit(self, X, y):
        x_features = X.shape[1]
        scores = np.zeros(x_features, dtype=float)
        n = 1
        n_features = self.n_features if self.n_features else x_features
        curr_score = 0.0
        selected_features = []
        while n <= n_features:
            last_score = curr_score
            curr_score = 0.0
            feature = None
            for f in range(x_features):
                if f in selected_features:
                    continue
                selected_features.append(f)
                selected_X = X[:, selected_features]
                score = cross_val_score(self.estimator, selected_X, y, cv=3, n_jobs=-1).mean()
                selected_features.pop()
                if score > curr_score:
                    curr_score = score
                    feature = f

            score_change = curr_score - last_score
            if score_change <= self.threshold:
                break

            selected_features.append(feature)
            scores[feature] = score_change
            n += 1

        normalized_scores = normalize([scores], norm='l1')[0]
        self.scores = normalized_scores
