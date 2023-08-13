"""
In here we want to find out the best accuracy among several tests
This is very easy ,jut by:
Using a loop helps us to implement and find the best soultion .
"""
import numpy as np
import pandas as pd
import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model, preprocessing
import pickle


data = pd.read_csv("car.data")
# print(data)
# Output:
#      buying  maint   door persons lug_boot safety  class
# 0     vhigh  vhigh      2       2    small    low  unacc
# 1     vhigh  vhigh      2       2    small    med  unacc
# 2     vhigh  vhigh      2       2    small   high  unacc
# 3     vhigh  vhigh      2       2      med    low  unacc
# 4     vhigh  vhigh      2       2      med    med  unacc
# ...     ...    ...    ...     ...      ...    ...    ...
# 1723    low    low  5more    more      med    med   good
# 1724    low    low  5more    more      med   high  vgood
# 1725    low    low  5more    more      big    low  unacc
# 1726    low    low  5more    more      big    med   good
# 1727    low    low  5more    more      big   high  vgood


# To convert those columns into Numeric we do following:
myPreprocessor = preprocessing.LabelEncoder()
 # WE can have dimenstion as much as our features belown:
buying = myPreprocessor.fit_transform(list(data["buying"]))
maint = myPreprocessor.fit_transform(list(data["maint"]))
door = myPreprocessor.fit_transform(list(data["door"]))
persons = myPreprocessor.fit_transform(list(data["persons"]))
lug_boot = myPreprocessor.fit_transform(list(data["lug_boot"]))
safety = myPreprocessor.fit_transform(list(data["safety"]))
clas = myPreprocessor.fit_transform(list(data["class"]))

predit = "class"

x = list(zip(buying, maint, door, lug_boot, safety))
y = list(clas)

# 30 times select x, y   then train , test and choose the best accuaracy , assign it to best var
best = 0
for i in range(11,1,-1):
    print(""*5)
    print(f"To see numbers  <repeted> " ,{i})
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)


# # n_neighbors : Amount of neighbors we want 
# # 'odd' num is necessary cuz model could not decide for 'even' numbers which one is closer or vote
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(x_train,y_train)

    accuracy = model.score(x_test, y_test)
    
    print("#"*5)
    if accuracy > best:
        print(f" this is {i}   accuracy: ", accuracy)
        best = accuracy
        with open("carModel.pickle", "wb") as modelFile:
            pickle.dump(model, modelFile)
    print("="*10) 
print("Final Result : ")       
print("Best Accuracy is : ",best)
# Output:
# To see numbers  <repeted>  {11}
# #####
#  this is 11   accuracy:  0.7283236994219653
# ==========

# To see numbers  <repeted>  {10}
# #####
# ==========

# To see numbers  <repeted>  {9}
# #####
# ==========

# To see numbers  <repeted>  {8}
# #####
# ==========

# To see numbers  <repeted>  {7}
# #####
# ==========

# To see numbers  <repeted>  {6}
# #####
# ==========

# To see numbers  <repeted>  {5}
# #####
# ==========

# To see numbers  <repeted>  {4}
# #####
# ==========

# To see numbers  <repeted>  {3}
# #####
# ==========

# To see numbers  <repeted>  {2}
# #####
# ==========
# Final Result :
# Best Accuracy is :  0.7283236994219653



predicted = model.predict(x_test)
names = ["unacc", "acc", "good", "vgood"]

print("\n\n#################################")
for i in range(len(predicted)):
    print(f"Data #{i+1}")
    print("Model Prediction: ", names[predicted[i]])
    print("Input Data: ", x_test[i])
    print("Real Label: ", names[y_test[i]])
    print("#################################")
     
    
    
# Output: 
# #################################
# Data #1
# Model Prediction:  good
# Input Data:  (3, 0, 0, 1, 2)
# Real Label:  good
# #################################
# Data #2
# Model Prediction:  good
# Input Data:  (1, 0, 2, 0, 1)
# Real Label:  good
# #################################
# Data #3
# Model Prediction:  good
# Input Data:  (3, 0, 2, 0, 1)
# Real Label:  good
# #################################
# Data #4
# Model Prediction:  good
# Input Data:  (3, 0, 2, 1, 0)
# Real Label:  good
# #################################
# Data #5
# Model Prediction:  good
# Input Data:  (2, 1, 3, 2, 1)
# Real Label:  good
# #################################