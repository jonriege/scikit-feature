from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
import numpy as np


def rf_permutation(X, y, **kwargs):

    if np.issubdtype(y.dtype, np.floating):
        clf = RandomForestRegressor(n_estimators=10)
    else:
        clf = RandomForestClassifier(n_estimators=10)

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    clf.fit(X_train, y_train)

    result = permutation_importance(clf, X_test, y_test)

    return result['importances_mean']
