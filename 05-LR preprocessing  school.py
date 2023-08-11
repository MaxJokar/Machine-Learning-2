""" 
Data Preprocessing helps to prepare the datas before any processing , 
here I am showing some simple methods on a  data frame 
using Pandas how to convert a  string into  Numeric it
using fit_transform

"""
import numpy as np
import pandas as pd
import sklearn
from sklearn import linear_model
import pickle
import matplotlib.pyplot as plt
from matplotlib import style
import joblib
#  import a new module for 
from sklearn import preprocessing

# Collect Data
data = pd.read_csv("student-por.csv", sep=";")
# Select Data 
data = data[[ "G2", "G3", "studytime",  "absences" , "sex", "school"]]
# Clean Data 
predict = "G3"

# Those columns we need should be made as  a model or  OBJECT preprocessing
# if we had more than 2 class in our each feachers 'Sex', F=0 M=1 we should use below method 
pre_prcossing = preprocessing.LabelEncoder()
sex = pre_prcossing.fit_transform(list(data["sex"]))
school = pre_prcossing.fit_transform(list(data["school"]))


# print(sex)
# # [0 0 0 0 0 1 1 0 1 ...
print(school)
#  0 0 0 0 0 0 0 1 1 ...

# Final Data Frame 
X = list(zip(data["absences"], data["studytime"], data["G2"], sex, school))
y = np.array(data[predict])

# #  Now we make Model and train 
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1, random_state = 123)
model = linear_model.LinearRegression()
model.fit(x_train,y_train)
score = model.score(x_test, y_test)
print(score)
# 0.9516568753859154




 