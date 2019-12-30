from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import numpy as np


def rf_impurity(X, y, **kwargs):

    if np.issubdtype(y.dtype, np.floating):
        clf = RandomForestRegressor(n_estimators=10)
    else:
        clf = RandomForestClassifier(n_estimators=10)

    clf.fit(X, y)
    return clf.feature_importances_
