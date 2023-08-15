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


# np.random.seed(42)

digits = load_digits()
# print(digits)

# {'data': array([[ 0.,  0.,  5., ...,  0.,  0.,  0.],
#        [ 0.,  0.,  0., ..., 10.,  0.,  0.],
#        [ 0.,  0.,  0., ..., 16.,  9.,  0.],
#        ...,
#        [ 0.,  0.,  1., ...,  6.,  0.,  0.],
#        [ 0.,  0.,  2., ..., 12.,  0.,  0.],
#        [ 0.,  0., 10., ..., 12.,  1.,  0.]]), 'target': array([0, 1, 2, ..., 8, 9, 8]), 'frame': None, 'feature_names': ['pixel_0_0', 'pixel_0_1', 'pixel_0_2', 'pixel_0_3', 'pixel_0_4', 'pixel_0_5', 'pixel_0_6', 'pixel_0_7', 'pixel_1_0', 'pixel_1_1', 'pixel_1_2', 'pixel_1_3', 'pixel_1_4', 'pixel_1_5', 'pixel_1_6', 
# 'pixel_1_7', 'pixel_2_0', 'pixel_2_1', 'pixel_2_2', 'pixel_2_3', 'pixel_2_4', 'pixel_2_5', 'pixel_2_6', 'pixel_2_7', 'pixel_3_0', 'pixel_3_1', 'pixel_3_2', 'pixel_3_3', 'pixel_3_4', 'pixel_3_5', 'pixel_3_6', 'pixel_3_7', 'pixel_4_0', 'pixel_4_1', 'pixel_4_2', 'pixel_4_3', 'pixel_4_4', 'pixel_4_5', 'pixel_4_6', 'pixel_4_7', 'pixel_5_0', 'pixel_5_1', 'pixel_5_2', 'pixel_5_3', 'pixel_5_4', 'pixel_5_5', 'pixel_5_6', 'pixel_5_7', 'pixel_6_0', 'pixel_6_1', 'pixel_6_2', 'pixel_6_3', 'pixel_6_4', 'pixel_6_5', 'pixel_6_6', 'pixel_6_7', 'pixel_7_0', 'pixel_7_1', 'pixel_7_2', 'pixel_7_3', 'pixel_7_4', 'pixel_7_5', 'pixel_7_6', 'pixel_7_7'], 'target_names': array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), 'images': array([[[ 0.,  0.,  5., ...,  1.,  0.,  0.],
#         [ 0.,  0., 13., ..., 15.,  5.,  0.],
#         [ 0.,  3., 15., ..., 11.,  8.,  0.],
#         ...,
#         [ 0.,  4., 11., ..., 12.,  7.,  0.],
#         [ 0.,  2., 14., ..., 12.,  0.,  0.],
#         [ 0.,  0.,  6., ...,  0.,  0.,  0.]],

#        [[ 0.,  0.,  0., ...,  5.,  0.,  0.],
#         [ 0.,  0.,  0., ...,  9.,  0.,  0.],
#         [ 0.,  0.,  3., ...,  6.,  0.,  0.],
#         ...,
#         [ 0.,  0.,  1., ...,  6.,  0.,  0.],
#         [ 0.,  0.,  1., ...,  6.,  0.,  0.],
#         [ 0.,  0.,  0., ..., 10.,  0.,  0.]],

#        [[ 0.,  0.,  0., ..., 12.,  0.,  0.],
#         [ 0.,  0.,  3., ..., 14.,  0.,  0.],
#         [ 0.,  0.,  8., ..., 16.,  0.,  0.],
#         ...,
#         [ 0.,  9., 16., ...,  0.,  0.,  0.],
#         [ 0.,  3., 13., ..., 11.,  5.,  0.],
#         [ 0.,  0.,  0., ..., 16.,  9.,  0.]],

#        ...,

#        [[ 0.,  0.,  1., ...,  1.,  0.,  0.],
#         [ 0.,  0., 13., ...,  2.,  1.,  0.],
#         [ 0.,  0., 16., ..., 16.,  5.,  0.],
#         ...,
#         [ 0.,  0., 16., ..., 15.,  0.,  0.],
#         [ 0.,  0., 15., ..., 16.,  0.,  0.],
#         [ 0.,  0.,  2., ...,  6.,  0.,  0.]],

#        [[ 0.,  0.,  2., ...,  0.,  0.,  0.],
#         [ 0.,  0., 14., ..., 15.,  1.,  0.],
#         [ 0.,  4., 16., ..., 16.,  7.,  0.],
#         ...,
#         [ 0.,  0.,  0., ..., 16.,  2.,  0.],
#         [ 0.,  0.,  4., ..., 16.,  2.,  0.],
#         [ 0.,  0.,  5., ..., 12.,  0.,  0.]],

#        [[ 0.,  0., 10., ...,  1.,  0.,  0.],
#         [ 0.,  2., 16., ...,  1.,  0.,  0.],
#         [ 0.,  0., 15., ..., 15.,  0.,  0.],
#         ...,
#         [ 0.,  4., 16., ..., 16.,  6.,  0.],
#         [ 0.,  8., 16., ..., 16.,  8.,  0.],
#         [ 0.,  1.,  8., ..., 12.,  1.,  0.]]]), 'DESCR': ".. _digits_dataset:\n\nOptical recognition of handwritten digits dataset\n--------------------------------------------------\n\n**Data Set Characteristics:**\n\n    :Number of Instances: 1797\n    :Number of Attributes: 64\n    :Attribute Information: 8x8 image of integer pixels in the range 0..16.\n    :Missing Attribute Values: None\n    :Creator: E. Alpaydin (alpaydin '@' boun.edu.tr)\n    :Date: July; 1998\n\nThis is a copy of the test set of the UCI ML hand-written digits datasets\nhttps://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits\n\nThe data set contains images of hand-written digits: 10 classes where\neach class refers to a digit.\n\nPreprocessing programs made available by NIST were 
# used to extract\nnormalized bitmaps of handwritten digits from a preprinted form. From a\ntotal of 43 people, 30 contributed to the training set and different 13\nto the test set. 32x32 bitmaps are divided into nonoverlapping blocks of\n4x4 and the number of on pixels are counted in each block. This generates\nan input matrix of 8x8 where each element is an integer in the range\n0..16. This reduces dimensionality and gives invariance to small\ndistortions.\n\nFor info on NIST preprocessing routines, see M. D. Garris, J. L. Blue, G.\nT. Candela, D. L. Dimmick, J. Geist, P. J. Grother, S. A. Janet, and C.\nL. Wilson, NIST Form-Based Handprint Recognition System, NISTIR 5469,\n1994.\n\n.. topic:: References\n\n  - C. Kaynak (1995) Methods of Combining Multiple Classifiers and Their\n    Applications to Handwritten Digit Recognition, MSc Thesis, Institute of\n    Graduate Studies in Science and Engineering, Bogazici University.\n  - E. Alpaydin, C. Kaynak (1998) Cascading Classifiers, Kybernetika.\n  - Ken Tang and Ponnuthurai N. Suganthan and Xi Yao and A. Kai Qin.\n    Linear dimensionalityreduction using relevance weighted LDA. School of\n    Electrical and Electronic Engineering Nanyang Technological University.\n    2005.\n  - Claudio Gentile. A New Approximate Maximal Margin Classification\n    Algorithm. NIPS. 2000.\n"}




# digits.data is all our features
# we scale all our feature between 1,-1 because our digits have a large values :RGB value 
# it saves time on the computaions specially including distance between points so smaller 
# value would be better ==> make things faster 
data = scale(digits.data)
# print(data)

# [[ 0.         -0.33501649 -0.04308102 ... -1.14664746 -0.5056698 
#   -0.19600752]
#  [ 0.         -0.33501649 -1.09493684 ...  0.54856067 -0.5056698 
#   -0.19600752]
#  [ 0.         -0.33501649 -1.09493684 ...  1.56568555  1.6951369 
#   -0.19600752]
#  ...
#  [ 0.         -0.33501649 -0.88456568 ... -0.12952258 -0.5056698 
#   -0.19600752]
#  [ 0.         -0.33501649 -0.67419451 ...  0.8876023  -0.5056698 
#   -0.19600752]
#  [ 0.         -0.33501649  1.00877481 ...  0.8876023  -0.26113572
#   -0.19600752]]




# set amount of clusters that were gonna look for amount of centroids to make 
# LabelL
y = digits.target
print("amount of clusters :" , y)
# amount of clusters : [0 1 2 ... 8 9 8]

#  dynamic: if were gonna change dataset above could be this way too  : static
k = 10  
# OR 
# k = len(np.unique(y))


# Amount of features : To get the amount of instances or amount of numbers we have that were gonna classify
samples , features  =  data.shape
print("n_digits, n_sample, n_featueres :", samples, features)



# sk learn has a bunch of functions in there that automatically score like 
# supervised learning /unsupervised learning algorithms 
# homogeneity_score,,...are kind of range we need 
# estimator is classifier which fits data to the classifier & use bunch of different things to score :homogeneity_score 
# y is compared to the lables  estimator.labels_ that are estimated gave for each of our data 
# we dont give the Y value  when we train ,cuz its usupervised ,it automatically generates Y  value  
# for every single test data point that we give it 
# we dont have to split into  trainig and test data  cuz it doent know to start what our data is 
#  we just can compare the test data lables what our estimator or our classifier estimated (it predicted each label what it was  )
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

# Belown we have  A classifier(model) to call our function on our classifier , it prints and trains ton of differet
# classifiers  & just score them by calling the function above 
model = KMeans(n_clusters=k, init="random", n_init=10)
bench_k_means(model, "1", data)


# Output :
# n_digits, n_sample, n_featueres : 10 1797 64
# 1               70552   0.641   0.681   0.660   0.518   0.657   0.138



