"""
Preprocessing:
Using car evaluation data set :
In here we have some datas which are string , we should convert them into Numeric
to be to be used in KNN for  x axis and y axis . Also  I show how to train and test a KNN model 
and then how to look at unique data and see the neighbors for individual data points.  
in KNN  we don't have Train time but during calculating score "predict" for each point we must 
calculate with  all our  points , which is hard enough
At the end of the codes we will compare :
differ between things . for example : this pic is for cat another is dog 
this email is  spam another is a normal , classification

"""

import numpy as np
import pandas as pd
import sklearn
from sklearn.utils import shuffle
from sklearn import  preprocessing
from sklearn import model_selection


data = pd.read_csv("car.data")
# print(data.head())
# print(data)
# Output:
#   buying  maint door persons lug_boot safety  class
# 0  vhigh  vhigh    2       2    small    low  unacc
# 1  vhigh  vhigh    2       2    small    med  unacc
# 2  vhigh  vhigh    2       2    small   high  unacc
# 3  vhigh  vhigh    2       2      med    low  unacc
# 4  vhigh  vhigh    2       2      med    med  unacc
# As its shown mainly all datas are string : we must transform them into Numeric to be able to use them

# To classify data into certain category, convert 'Encode ' strings  into Numeric we do following:
myPreprocessor = preprocessing.LabelEncoder()
 # WE can have dimenstion as much as our features belown:
buying = myPreprocessor.fit_transform(list(data["buying"]))
maint = myPreprocessor.fit_transform(list(data["maint"]))
door = myPreprocessor.fit_transform(list(data["door"]))
persons = myPreprocessor.fit_transform(list(data["persons"]))
lug_boot = myPreprocessor.fit_transform(list(data["lug_boot"]))
safety = myPreprocessor.fit_transform(list(data["safety"]))
clas = myPreprocessor.fit_transform(list(data["class"]))


# print(data["door"])
# 0           2
# 1           2
# 2           2
# 3           2
# 4           2
# Bsed on data we should get class as mentioned in our documents as a result of
#  the datas  the car is  evaluated in class 
predit = "class"
print(data["door"].unique())
# ['2' '3' '4' '5more' 'more']



x = list(zip(buying, maint, door, lug_boot, safety))
print(f"this is  x   :" , x)
# this is  x   : [(3, 3, 0, 2, 1), (3, 3, 0, 2, 2)...
y = list(clas)
print(f"this is    y  :" , y)
# this is    y  : [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2...


x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

# Driver 
print(f"this is   x_train : " , x_train)
# this is   x_train :  [(3, 2, 2, 2, 2), (0, 1, 2, 0, 0),
print("=="*100)
print(f"this is   y_train : " , y_train)
# this is   y_train :  [2, 0, 0, 2, 2, 2, 2, 2, 3, 2, 0, 2, 2, 2,



# To Be Continued,...)