"""
Here machine learning python  I am introducing SupportVectorMachines.
This is mainly used for classification and is capable of performing 
classification for large dimensional data. 
I will also be showing you how to load datasets straight from the
sklearn module as Following:

"""

import sklearn
from sklearn import datasets
from sklearn import svm

# Load data
data = datasets.load_breast_cancer()

print(len(data.feature_names), "---->", data.feature_names, "\n")
print(len(data.target_names), "---->", data.target_names)

#############
# Set up X and Ys for a data 
x = data.data
y = data.target
# To split ,to test more test  we make 0.2
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

print(x_train)
print(y_train)

#############
# When we get 0,1 s we can print the actual answer index in this list 
classes = ["Malignant", "Benign"]




# Output:
# 30 ----> ['mean radius' 'mean texture' 'mean perimeter' 'mean area' 
#  'mean smoothness' 'mean compactness' 'mean concavity'
#  'mean concave points' 'mean symmetry' 'mean fractal dimension'     
#  'radius error' 'texture error' 'perimeter error' 'area error'      
#  'smoothness error' 'compactness error' 'concavity error'
#  'concave points error' 'symmetry error' 'fractal dimension error'  
#  'worst radius' 'worst texture' 'worst perimeter' 'worst area'      
#  'worst smoothness' 'worst compactness' 'worst concavity'
#  'worst concave points' 'worst symmetry' 'worst fractal dimension'] 

# 2 ----> ['malignant' 'benign']
# [[1.222e+01 2.004e+01 7.947e+01 ... 8.088e-02 2.709e-01 8.839e-02]  
#  [1.953e+01 3.247e+01 1.280e+02 ... 1.625e-01 2.713e-01 7.568e-02]  
#  [1.530e+01 2.527e+01 1.024e+02 ... 2.024e-01 4.027e-01 9.876e-02]
#  ...
#  [1.553e+01 3.356e+01 1.037e+02 ... 2.014e-01 3.512e-01 1.204e-01]
#  [1.495e+01 1.757e+01 9.685e+01 ... 1.667e-01 3.414e-01 7.147e-02]
#  [1.016e+01 1.959e+01 6.473e+01 ... 2.232e-02 2.262e-01 6.742e-02]]
# [1 0 0 1 1 1 1 1 0 1 1 1 0 1 0 1 0 1 1 0 1 1 0 0 1 1 0 1 0 1 1 0 0 1 1 1 1
#  0 1 1 1 1 1 1 0 1 0 1 1 1 0 0 1 1 1 0 1 0 1 1 1 1 1 0 1 1 0 1 0 1 1 1 1 1
#  0 0 1 1 1 1 0 0 0 1 0 0 0 1 1 0 1 1 1 0 1 0 0 0 1 0 1 1 0 0 0 1 0 1 1 1 0
#  1 0 1 0 0 1 1 1 0 0 1 1 1 0 0 0 0 1 1 1 0 1 1 1 0 1 1 1 1 1 0 0 1 1 1 0 0
#  1 0 0 0 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 0 1 1 1 0 1 1 1 1 1 1
#  1 0 1 1 0 0 1 0 1 1 0 0 0 0 1 0 0 1 1 1 1 1 0 1 1 0 1 1 1 1 0 1 0 0 1 1 1
#  0 1 1 0 1 1 1 0 1 0 0 1 1 1 1 0 1 0 1 0 0 0 0 0 1 0 1 0 0 1 1 1 0 1 0 0 0
#  1 1 0 0 1 1 0 1 0 0 1 1 1 1 1 1 0 0 1 1 1 0 1 1 0 1 1 0 1 0 1 0 1 0 0 0 1
#  1 1 1 0 1 1 0 1 1 1 0 1 1 0 0 0 1 0 0 1 1 1 1 1 1 1 1 0 0 1 0 1 1 1 1 1 1
#  1 0 0 0 1 1 0 1 1 1 0 0 1 1 1 0 1 1 0 0 1 0 0 1 1 0 0 1 0 0 1 1 1 0 0 1 1
#  0 0 0 1 1 1 1 0 1 0 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 0 0 1 0 0 1 0 0 1
#  0 1 0 1 1 1 1 0 1 0 0 1 1 0 1 1 1 0 1 0 1 1 1 0 1 0 1 1 0 1 1 0 0 1 1 1 0
#  1 0 1 1 0 0 1 1 0 0 1]