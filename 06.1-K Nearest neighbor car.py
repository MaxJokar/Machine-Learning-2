"""
In here we have some datas which are string , we should convert them into Numeric
to be to be used in KNN for  x axis and y axis . 
in KNN  we don't have Train time but during 
calculating score "predict" for each point we must calculate with 
all our  points , which is hard enough
in Classification we want to :
differ between things . for example : this pic is for cat another is dog 
this email is  spam another is a normal , classification
"""

import numpy as np
import pandas as pd
import sklearn
from sklearn.utils import shuffle
from sklearn import  preprocessing


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

# To convert those columns into Numeric we do following:
myPreprocessor = preprocessing.LabelEncoder()

# # WE can have dimenstion as much as our features belown:
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

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

# Driver command
print(x_train)

# Output:
# [(2, 1, 0, 2, 1), (1, 1, 2, 2, 1), (2, 2, 3, 1, 0), (0, 0, 0, 0, 1), (1, 2, 3, 0, 1), (3, 0, 1, 2, 0),
# (2, 3, 2, 1, 2), (0, 3, 2, 2, 2), (2, 0, 2, 2, 2), (0, 3, 0, 0, 1), (0, 2, 1, 0, 0), (2, 3, 0, 0, 0),...