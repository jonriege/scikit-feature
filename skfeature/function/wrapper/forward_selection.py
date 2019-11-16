import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB


def forward_selection(X, y, estimator=None, n_features=None, threshold=0.0):

    if estimator is None:
        estimator = GaussianNB()

    x_features = X.shape[1]
    scores = np.zeros(x_features, dtype=float)
    n = 1
    n_features = n_features if n_features else x_features
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
            score = cross_val_score(estimator, selected_X, y, cv=3, n_jobs=-1).mean()
            selected_features.pop()
            if score > curr_score:
                curr_score = score
                feature = f

        score_change = curr_score - last_score
        if score_change <= threshold:
            break

        selected_features.append(feature)
        scores[feature] = score_change
        n += 1

    return scores
