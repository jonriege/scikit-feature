import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import normalize


class ReliefF:

    def __init__(self, k=5):
        self.scores = None
        self.k = k

    def fit_transform(self, X, y, n_features=None, threshold=0):
        self.fit(X, y)
        return self.transform(X, n_features, threshold)

    def transform(self, X, n_features=None, threshold=0):
        """
        Rank features in descending order according to reliefF score, the higher the reliefF score, the more important the
        feature is
        """

        sorted_indices_rev = np.argsort(self.scores)
        sorted_indices = np.flip(sorted_indices_rev)
        filtered_indices = [i for i in sorted_indices if self.scores[i] > threshold]
        if n_features is not None:
            filtered_indices = filtered_indices[:n_features]
        return X[:, filtered_indices]

    def fit(self, X, y):
        """
        This function implements the reliefF feature selection

        Input
        -----
        X: {numpy array}, shape (n_samples, n_features)
            input data
        y: {numpy array}, shape (n_samples,)
            input class labels

        Output
        ------
        score: {numpy array}, shape (n_features,)
            reliefF score for each feature

        Reference
        ---------
        Robnik-Sikonja, Marko et al. "Theoretical and empirical analysis of relieff and rrelieff." Machine Learning 2003.
        Zhao, Zheng et al. "On Similarity Preserving Feature Selection." TKDE 2013.
        """

        n_samples, n_features = X.shape

        # calculate pairwise distances between instances
        distance = pairwise_distances(X, metric='manhattan')

        scores = np.zeros(n_features)

        # the number of sampled instances is equal to the number of total instances
        for idx in range(n_samples):
            near_hit = []
            near_miss = dict()

            self_fea = X[idx, :]
            c = np.unique(y).tolist()

            stop_dict = dict()
            for label in c:
                stop_dict[label] = 0
            del c[c.index(y[idx])]

            p_dict = dict()
            p_label_idx = float(len(y[y == y[idx]])) / float(n_samples)

            for label in c:
                p_label_c = float(len(y[y == label])) / float(n_samples)
                p_dict[label] = p_label_c / (1 - p_label_idx)
                near_miss[label] = []

            distance_sort = []
            distance[idx, idx] = np.max(distance[idx, :])

            for i in range(n_samples):
                distance_sort.append([distance[idx, i], int(i), y[i]])
            distance_sort.sort(key=lambda x: x[0])

            for i in range(n_samples):
                # find k nearest hit points
                if distance_sort[i][2] == y[idx]:
                    if len(near_hit) < self.k:
                        near_hit.append(distance_sort[i][1])
                    elif len(near_hit) == self.k:
                        stop_dict[y[idx]] = 1
                else:
                    # find k nearest miss points for each label
                    if len(near_miss[distance_sort[i][2]]) < self.k:
                        near_miss[distance_sort[i][2]].append(distance_sort[i][1])
                    else:
                        if len(near_miss[distance_sort[i][2]]) == self.k:
                            stop_dict[distance_sort[i][2]] = 1
                stop = True
                for (key, value) in stop_dict.items():
                    if value != 1:
                        stop = False
                if stop:
                    break

            # update reliefF score
            near_hit_term = np.zeros(n_features)
            for ele in near_hit:
                near_hit_term = np.array(abs(self_fea - X[ele, :])) + np.array(near_hit_term)

            near_miss_term = dict()
            for (label, miss_list) in near_miss.items():
                near_miss_term[label] = np.zeros(n_features)
                for ele in miss_list:
                    near_miss_term[label] = np.array(abs(self_fea - X[ele, :])) + np.array(near_miss_term[label])
                scores += near_miss_term[label] / (self.k * p_dict[label])
            scores -= near_hit_term / self.k

        normalized_scores = normalize([scores], norm='l1')[0]
        self.scores = normalized_scores
