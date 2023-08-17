""" 
Unlike previous code here we could  have more than 2 classes in our feature , to convert from string into Numeric 
we must use the Below Method.Here I am showing how to convert several Columns 
having several classes and not only two  usirng  preprocessing module 
"""
import numpy as np
import pandas as pd
import sklearn
from sklearn import linear_model
import joblib
#  import a new module for 
from sklearn import preprocessing

# Collect Data
data = pd.read_csv("student-por.csv", sep=";")
# Select Data 
data = data[[ "G2", "G3", "studytime",  "absences" , "sex", "school"]]
# Clean Data 
predict = "G3"


# ****Here !!!Make a new column with variable classes, More than 2 ! 
# Those columns we need should be made as  a model or  OBJECT preprocessing
# Unlike 'sex' for example  we had only 2  classes :'Sex', F=0 M=1 ,here  we should use below method 
pre_prcossing = preprocessing.LabelEncoder()
sex = pre_prcossing.fit_transform(list(data["sex"]))
school = pre_prcossing.fit_transform(list(data["school"]))
# my_new_column = pre_prcossing.fit_transform(list(data["sex"]))
# print(f" my new column is  : {my_new_column}")

# print(sex)
# # [0 0 0 0 0 1 1 0 1 ...
# print(school)
#  0 0 0 0 0 0 0 1 1 ...

# Final Data Frame 
X = list(zip(data["absences"], data["studytime"], data["G2"], sex, school))
y = np.array(data[predict])

# #  Now we make Model and train 
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1, random_state = 123)
model = linear_model.LinearRegression()
# Train
model.fit(x_train,y_train)
# Test
score = model.score(x_test, y_test)
print(score)
# 0.9516568753859154




 