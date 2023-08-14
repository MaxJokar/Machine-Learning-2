"""
The Basis behind Support Vector Machine :
The larger distance  the larger  margin ,the  more we can 
separate the two classes  and do more accurate prediction 
"""


import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics

data = datasets.load_breast_cancer()

x = data.data
print("this is  x :", x)
y = data.target
print("this is  y: ", y)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)
# When we get 0,1 s we can print the actual answer index in this list
classes = ["Malignant", "Benign"]
print("classes are  ", classes)

# Support vector classification: SVC
# liner, poly, rbf , sigmod, brings our dimension up ,hyperplane give much classification
model = svm.SVC(kernel="linear")  
print("model is  :" ,model)

model.fit(x_train, y_train)
print("x_train is  :",x_train)
print("y_train is : ",y_train)

predictions = model.predict(x_test)
print("predictions is  :",predictions)
# To compare two lists 
accuracy = metrics.accuracy_score(y_test, predictions)



print("accuracy is  :",accuracy)

# Output:
# this is  x : [[1.799e+01 1.038e+01 1.228e+02 ... 2.654e-01 4.601e-01 1.189e-01]
#  [2.057e+01 1.777e+01 1.329e+02 ... 1.860e-01 2.750e-01 8.902e-02]
#  [1.969e+01 2.125e+01 1.300e+02 ... 2.430e-01 3.613e-01 8.758e-02]
#  ...
#  [1.660e+01 2.808e+01 1.083e+02 ... 1.418e-01 2.218e-01 7.820e-02]
#  [2.060e+01 2.933e+01 1.401e+02 ... 2.650e-01 4.087e-01 1.240e-01]
#  [7.760e+00 2.454e+01 4.792e+01 ... 0.000e+00 2.871e-01 7.039e-02]]
# this is  y:  [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  1 0 0 0 0 0 0 0 0 1 0 1 1 1 1 1 0 0 1 0 0 1 1 1 1 0 1 0 0 1 1 1 1 0 1 0 0
#  1 0 1 0 0 1 1 1 0 0 1 0 0 0 1 1 1 0 1 1 0 0 1 1 1 0 0 1 1 1 1 0 1 1 0 1 1
#  1 1 1 1 1 1 0 0 0 1 0 0 1 1 1 0 0 1 0 1 0 0 1 0 0 1 1 0 1 1 0 1 1 1 1 0 1
#  1 1 1 1 1 1 1 1 0 1 1 1 1 0 0 1 0 1 1 0 0 1 1 0 0 1 1 1 1 0 1 1 0 0 0 1 0
#  1 0 1 1 1 0 1 1 0 0 1 0 0 0 0 1 0 0 0 1 0 1 0 1 1 0 1 0 0 0 0 1 1 0 0 1 1
#  1 0 1 1 1 1 1 0 0 1 1 0 1 1 0 0 1 0 1 1 1 1 0 1 1 1 1 1 0 1 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 1 1 1 1 1 1 0 1 0 1 1 0 1 1 0 1 0 0 1 1 1 1 1 1 1 1 1 1 1 1
#  1 0 1 1 0 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 0 1 0 1 1 1 1 0 0 0 1 1
#  1 1 0 1 0 1 0 1 1 1 0 1 1 1 1 1 1 1 0 0 0 1 1 1 1 1 1 1 1 1 1 1 0 0 1 0 0
#  0 1 0 0 1 1 1 1 1 0 1 1 1 1 1 0 1 1 1 0 1 1 0 0 1 1 1 1 1 1 0 1 1 1 1 1 1
#  1 0 1 1 1 1 1 0 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 0 1 0 0 1 0 1 1 1 1 1 0 1 1
#  0 1 0 1 1 0 1 0 1 1 1 1 1 1 1 1 0 0 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 0 1
#  1 1 1 1 1 1 0 1 0 1 1 0 1 1 1 1 1 0 0 1 0 1 0 1 1 1 1 1 0 1 1 0 1 0 1 0 0
#  1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 0 1 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 0 0 0 0 0 0 1]
# classes are   ['Malignant', 'Benign']
# model is  : SVC(kernel='linear')
# x_train is  : [[2.522e+01 2.491e+01 1.715e+02 ... 2.867e-01 2.355e-01 1.051e-01]
#  [1.161e+01 1.602e+01 7.546e+01 ... 1.105e-01 2.787e-01 7.427e-02]
#  [1.426e+01 1.965e+01 9.783e+01 ... 1.505e-01 2.398e-01 1.082e-01]
#  ...
#  [1.305e+01 1.859e+01 8.509e+01 ... 1.258e-01 3.113e-01 8.317e-02]
#  [1.194e+01 1.824e+01 7.571e+01 ... 6.296e-02 2.785e-01 7.408e-02]
#  [1.127e+01 1.550e+01 7.338e+01 ... 8.272e-02 2.157e-01 1.043e-01]]
# y_train is :  [0 1 1 1 1 0 1 0 1 1 1 1 1 1 1 0 1 0 0 0 0 0 0 1 1 1 1 0 1 0 1 1 0 0 1 0 1
#  1 0 1 1 0 0 1 0 0 1 0 1 0 1 1 1 1 0 1 0 0 1 1 0 1 0 0 1 0 1 1 1 0 1 1 1 1
#  1 1 1 1 1 1 0 1 0 1 0 0 0 0 1 0 1 1 0 0 1 0 0 0 1 1 0 0 1 0 1 0 1 0 1 0 1
#  1 1 0 0 1 0 1 0 0 1 1 1 1 0 1 0 1 1 0 1 1 0 1 1 1 1 1 0 1 1 0 0 0 1 1 0 1
#  1 1 1 0 1 1 1 1 1 1 1 1 1 1 0 0 1 0 1 0 1 0 0 1 1 0 0 0 0 1 1 1 1 0 1 0 1
#  0 1 1 1 1 1 0 1 0 1 1 1 1 0 1 1 1 1 0 1 1 0 1 0 1 1 1 1 1 1 1 1 0 1 1 0 1
#  0 1 1 0 1 0 0 1 1 1 1 0 1 0 1 1 1 1 0 1 1 1 1 0 1 0 0 1 1 0 0 0 0 1 1 1 1
#  1 0 0 1 0 1 0 0 1 0 0 1 1 0 1 1 1 1 0 0 1 0 0 1 0 1 0 1 1 1 1 1 0 1 0 0 1
#  1 1 0 0 1 1 1 1 1 0 0 1 0 1 0 1 1 1 1 0 1 1 1 1 0 1 0 1 0 0 0 0 1 1 1 1 1
#  0 0 1 0 1 1 0 0 1 0 1 0 0 0 1 1 1 1 1 0 1 1 0 1 1 0 1 0 1 1 1 0 0 0 1 0 1
#  1 0 1 0 1 1 0 0 1 0 0 0 1 0 0 0 0 1 1 1 1 0 1 0 1 1 1 1 1 1 1 0 1 0 0 1 1
#  1 0 1 0 0 0 1 1 1 1 1 1 1 0 1 0 1 1 0 1 1 0 1 0 1 1 1 0 1 1 0 1 1 1 1 1 1
#  0 0 1 0 1 1 1 1 1 1 1]
# predictions is  : [0 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 0 1 0 1 0 0 1 1 0 1 1 0 0
#  1 1 1 0 1 0 1 1 1 1 0 0 0 0 1 0 1 0 0 1 1 1 1 0 1 1 1 0 1 0 1 1 0 1 1 1 1
#  0 1 0 1 1 1 1 1 0 1 0 1 0 1 1 0 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 1 1 1 0
#  1 0 0]
# accuracy is  : 0.9824561403508771






