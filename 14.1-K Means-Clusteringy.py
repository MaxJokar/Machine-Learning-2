"""Unsupervised Learning Algorithm K Means Clustering"""
import numpy as np
import sklearn
from sklearn.preprocessing import scale
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn import metrics
from time import time
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


np.random.seed(42)

digits = load_digits()
# digits.data is all our features
# we scale all our feature between 1,-1 because our digits have a large values :RGB value 
# it saves time on the computaions specially including distance between points so smaller 
# value would be better ==> make things faster 
data = scale(digits.data)

# set amount of clusters that were gonna look for amount of centroids to make 
y = data.targets

# n_sample , n_featueres = data.shape
# n_digits = len(np.unique(digits.target))

#  dynamic: if were gonna change dataset above 
# could be this way too   , static
k = 10  
# k = len(np.unique(y))
samples , features  =  data.shape

# To get the amount of instances or amount of numbers we have that were gonna classify
#  Amount of features 
# labels  = digits.target

# sample_size = 300

# print("n_digits, n_sample, n_featueres :", n_digits, n_sample, n_featueres)

# k = len(np.unique(y))
# k = len(np.unique(labels))


# sk learn has a bunch of functions in there that automatically score like supervised learning
# unsupervised learning algorithms 
# homogeneity_score,,...are kind of range we need 
# estimator is classifier which fits data to the classifier & use bunch of different things to score :homogeneity_score 
# y is compared to the lables  estimator.labels_ that are estimated gave for each of our data 
# we dont give the Y value  when we train ,cuz its usupervised ,it automatically generates Y  value  
# for every single test data point that we give it 
# we dont have to split trainig and test cuz it doent know to start what our data is 
def bench_k_means(estimator, name, data):
    estimator.fit(data)
    print('%-9s\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, estimator.inertia_,
            metrics.homogeneity_score(y, estimator.labels_),
            metrics.completeness_score(y, estimator.labels_),
            metrics.v_measure_score(y, estimator.labels_),
            metrics.adjusted_rand_score(y, estimator.labels_),
            metrics.adjusted_mutual_info_score(y,  estimator.labels_),
            metrics.silhouette_score(data, estimator.labels_, metric='euclidean')))

#  A classifier to call our function on our classifier , it prints and trains ton of differet
# classifiers a & just score them by calling the function above 
model = KMeans(n_clusters=k, init="random", n_init=10)
bench_k_means(model, "1", data)


# Output :
# n_digits, n_sample, n_featueres : 10 1797 64
# 1               70552   0.641   0.681   0.660   0.518   0.657   0.138



