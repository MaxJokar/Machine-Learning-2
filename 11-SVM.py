"""
Here machine learning python  I am introducing SupportVectorMachines.
This is mainly used for classification and is capable of performing 
classification for large dimensional data. 
I will also be showing you how to load datasets straight from the
sklearn module as Following:

"""

import sklearn
from sklearn import datasets   # To load Datas
from sklearn import model_selection

#1. Load data
data = datasets.load_breast_cancer()

print("feature_names :")
print(len(data.feature_names), "---->", data.feature_names, "\n")
print("target_names :")
print(len(data.target_names), "---->", data.target_names)

#############
# 2. Set up X and Ys for a data 
x = data.data
y = data.target
# To split ,to test more test  we make 0.2
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

print("x_train :")
print(x_train)
print("y_train :")
print(y_train)

# Output:
# feature_names :
# 30 ----> ['mean radius' 'mean texture' 'mean perimeter' 'mean area'       
#  'mean smoothness' 'mean compactness' 'mean concavity'
#  'mean concave points' 'mean symmetry' 'mean fractal dimension'
#  'radius error' 'texture error' 'perimeter error' 'area error'
#  'smoothness error' 'compactness error' 'concavity error'
#  'concave points error' 'symmetry error' 'fractal dimension error'        
#  'worst radius' 'worst texture' 'worst perimeter' 'worst area'
#  'worst smoothness' 'worst compactness' 'worst concavity'
#  'worst concave points' 'worst symmetry' 'worst fractal dimension']       

# target_names :
# 2 ----> ['malignant' 'benign']
# x_train :
# [[9.668e+00 1.810e+01 6.106e+01 ... 2.500e-02 3.057e-01 7.875e-02]        
#  [1.044e+01 1.546e+01 6.662e+01 ... 4.464e-02 2.615e-01 8.269e-02]        
#  [1.861e+01 2.025e+01 1.221e+02 ... 1.490e-01 2.341e-01 7.421e-02]        
#  ...
#  [1.287e+01 1.621e+01 8.238e+01 ... 5.780e-02 3.604e-01 7.062e-02]        
#  [1.940e+01 2.350e+01 1.291e+02 ... 1.564e-01 2.920e-01 7.614e-02]        
#  [9.606e+00 1.684e+01 6.164e+01 ... 8.120e-02 2.982e-01 9.825e-02]]       
# y_train :
# [1 1 0 0 1 1 0 0 0 1 1 1 0 0 1 1 0 0 0 0 1 0 0 1 1 1 0 1 1 1 1 1 1 1 1 1 0
#  0 1 0 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 0 0 0 0 1 1 1 1 0 1 0 1 1 1 1 1 0
#  1 0 1 0 0 1 0 0 0 0 1 1 1 1 1 1 1 0 0 0 1 0 1 1 1 1 1 1 0 1 1 1 0 1 1 1 1
#  1 1 1 1 0 0 1 1 1 1 1 1 1 1 0 1 0 1 1 0 0 1 1 0 0 0 0 1 1 1 1 0 1 1 1 1 1
#  1 0 1 0 0 1 1 0 0 1 1 0 0 1 0 1 1 0 1 0 1 1 0 1 1 0 1 0 1 0 1 0 0 1 1 1 1
#  1 1 1 1 1 1 1 0 1 1 1 0 0 1 1 1 1 0 0 1 1 1 1 1 0 1 1 0 1 1 1 1 0 0 0 1 1
#  1 1 0 0 1 1 0 1 1 0 0 0 1 1 1 0 1 1 0 0 1 1 0 1 1 1 1 1 0 1 0 1 1 1 1 0 1
#  0 1 1 0 0 1 0 0 1 1 1 1 1 1 1 0 0 1 1 0 1 1 0 1 0 0 0 1 1 0 1 0 0 0 1 0 1
#  0 1 1 1 1 1 0 0 0 1 1 1 1 1 1 0 0 0 0 1 0 1 1 0 1 0 0 0 0 1 1 0 1 0 1 1 1
#  1 1 0 1 1 0 0 0 1 1 0 1 0 0 1 0 1 0 0 0 1 0 0 1 0 1 1 1 1 1 1 1 0 1 0 1 1
#  0 1 0 0 0 0 0 0 1 0 1 1 1 0 0 0 0 1 1 1 0 1 1 0 0 0 0 1 0 1 0 0 1 1 1 0 1
#  1 1 1 1 1 0 1 0 1 0 1 0 0 0 0 1 1 0 1 0 1 1 1 1 1 1 1 1 1 1 0 1 0 1 1 1 0
#  0 0 1 0 0 1 0 0 1 0 1]

# To Be Continued,...