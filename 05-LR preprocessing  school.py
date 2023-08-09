""" 
helps to learn how to convert a  string into  Numeric 
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

# Data
data = pd.read_csv("student-por.csv", sep=";")
data = data[[ "G2", "G3", "studytime",  "absences" , "sex", "school"]]
predict = "G3"

# Those column we need  we should make a model or  OBJECT preprocessing
pre_prcossing = preprocessing.LabelEncoder()
sex = pre_prcossing.fit_transform(list(data["sex"]))
school = pre_prcossing.fit_transform(list(data["school"]))
X = list(zip(data["absences"], data["studytime"], data["G2"], sex, school))
y = np.array(data[predict])

#  Now we make Model and train 
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1, random_state = 123)
model = linear_model.LinearRegression()
model.fit(x_train,y_train)
score = model.score(x_test, y_test)
print(score)
# 0.9516568753859154




 