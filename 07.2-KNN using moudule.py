# """
# X  from a line more we  use for one sample  form X becomes x 
# For vote from counter and for distance  argsort
# We didn’t have train in KNN, we kept trained datas somewhere , saved  then after we wanted to predict we used them 

# """
import sklearn
import numpy as np
from collections import Counter
import pandas as pd
from sklearn import  preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model, preprocessing
from sklearn.model_selection import train_test_split




# To calculate between distances
# Numpy.array does for each Dimension so we dont need 
# make a loop and deminish each D then power 2, sum , sqrt
def euclidean_distance(x1,x2):
    return np.sqrt(np.sum(x1-x2)**2)


class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self,X,y):
        self.X_train = X
        self.y_train = y

#  takes a list of datas, predict something for them 
# then gives a list of labels we predicted for input predict
    def predict(self, X):
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)


    def _predict(self, x):
        # calculate Distance with all points": we can get certain numbers
        # in all self.X_train sample , separate them, name them as a x_train fin give to the above function
        #  distance include of all new points with all x_train
        distance = [euclidean_distance(x, x_train) for x_train in self.X_train] 
        # get k nearest samples, Lables 
        # k_indicies : includes index numbers of X_train ,y_train
        # k = 3
        k_indicies = np.argsort(distance)[:self.k] 
        k_nearest_labels = [self.y_train[i] for i in k_indicies]
        

        # vote : now we take that one which repeats more fore predict:1 first most repeated
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
    

# TEST:
# lst = [1,2,3,1,2,5,6,2,3,4,1,6]
# print(Counter(lst))
# print(Counter(lst).most_common(1)) --> [(2,5)]
# Counter({1: 3, 2: 3, 3: 2, 6: 2, 5: 1, 4: 1})

data = pd.read_csv("car.data")
#print(data.head())


myPreprocessor = preprocessing.LabelEncoder()
buying = myPreprocessor.fit_transform(list(data["buying"]))
maint = myPreprocessor.fit_transform(list(data["maint"]))
door = myPreprocessor.fit_transform(list(data["door"]))
persons = myPreprocessor.fit_transform(list(data["persons"]))
lug_boot = myPreprocessor.fit_transform(list(data["lug_boot"]))
safety = myPreprocessor.fit_transform(list(data["safety"]))
clas = myPreprocessor.fit_transform(list(data["class"]))

predit = "class"

X = np.array(list(zip(buying, maint, door, lug_boot, safety)))
y = np.array(list(clas))

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# print(x_train)
# make model , fit 
model = KNN(5)
model.fit(x_train,y_train )
predictions = model.predict(x_test)
 
n = 0 
for i in range(len(predictions)):
    #print(f" Real numbers : {y_test[i]} , Predicted : {predictions[i]}")

# Real numbers : 2 , Predicted : 2
# Real numbers : 2 , Predicted : 2
# Real numbers : 2 , Predicted : 2
# Real numbers : 0 , Predicted : 2
# Real numbers : 2 , Predicted : 2,...

# To score :What percent our model predicted is accurate ?

    if y_test[i] == predictions[i]:
       n +=1
print(n / len(predictions)*100) 

#  accuracy :72.83236994219652



# Compare with K nearest neighbor without our above codes as following:
sk_model = KNeighborsClassifier(n_neighbors=5)
sk_model.fit(x_train,y_train)

accuracy = sk_model.score(x_test, y_test)
print(accuracy*100)
# 0.7225433526011561