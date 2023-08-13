"""
Зython machine learning I'm implementing a support vector machine to
classify data. 

"""

import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
# To compare our results using svm
from sklearn.neighbors import KNeighborsClassifier

data = datasets.load_breast_cancer()
# we define this dat point is equal to Malignant 
x = data.data
y = data.target
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

classes = ["Malignant", "Benign"]

# Support vector classification: SVC
# kernel='linear',poly brings our dimension up ,hyperplance give much classification 
model = svm.SVC(kernel="linear", C=3)
# C=1 hard margin :SVM:  0.9385964912280702 , C=3 SVM:  0.9649122807017544

model.fit(x_train, y_train)

predictions = model.predict(x_test)
# To compare two lists 
accuracy = metrics.accuracy_score(y_test, predictions)
print("SVM: ", accuracy)



model1 = KNeighborsClassifier(n_neighbors=9)
model1.fit(x_train, y_train)

prediction1 = model1.predict(x_test)
accuracy1 = metrics.accuracy_score(y_test, prediction1)



print("KNN: ", accuracy1)

# SVM:  0.9736842105263158
# KNN:  0.9473684210526315

