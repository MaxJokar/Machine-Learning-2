"""Unsupervised Learning Algorithm K Means Clustering"""
import numpy as np
import sklearn
from sklearn.preprocessing import scale
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn import metrics
from time import time
from sklearn.decomposition import PCA


np.random.seed(42)

digits = load_digits()
data = scale(digits.data)
n_sample , n_featueres = data.shape
n_digits = len(np.unique(digits.target))

labels  = digits.target
sample_size = 300

print("n_digits, n_sample, n_featueres :\n ", n_digits, n_sample, n_featueres)

# k = len(np.unique(y))
k = len(np.unique(labels))


def bench_k_means(estimator, name, data):
    t0 = time()
    estimator.fit(data)
    print('%-9s\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, estimator.inertia_,
             metrics.homogeneity_score(labels, estimator.labels_),
             metrics.completeness_score(labels, estimator.labels_),
             metrics.v_measure_score(labels, estimator.labels_),
             metrics.adjusted_rand_score(labels, estimator.labels_),
             metrics.adjusted_mutual_info_score(labels,  estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_, metric='euclidean')))


model = KMeans(n_clusters=k, init="random", n_init=10)
bench_k_means(model, "1", data)


# Output :
# 1               69670   0.679   0.718   0.698   0.571   0.695   0.142