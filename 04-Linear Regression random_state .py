"""
To stabilize and have  constant or stable result , every time we dont get  
random  numbers  for rows or columns we add random_state as following"
without that  we will have very different numbers every time we run 
"""

import numpy as np
import pandas as pd
import sklearn
from sklearn import linear_model
import pickle


data = pd.read_csv("student-por.csv", sep=";")
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
predict = "G3"
x = np.array(data.drop([predict], axis=1))
y = np.array(data[predict])


# A.we can find the best predict in 20 time as shown following :

best = 0
for _ in range(20):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1 , random_state = 123)
    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)
    accuracy = linear.score(x_test,y_test)

    if accuracy > best:
        best = accuracy
        with open("studentGradeModel.pickle", "wb") as modelFile:
            pickle.dump(linear, modelFile)

print("Highest Accuracy: ", best)
print("Best Model Saved.")


# Highest Accuracy:  0.9453520920658378
# Best Model Saved.






# OR 

# B.
# EPOCH = 30
# best = 0
# for i in range(EPOCH):
#     # x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
    
#    
#     x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1 )
#     model = linear_model.LinearRegression()
#     # give TRAIN  data to Model 
#     model.fit(x_train , y_train)
#     acc = model.score(x_test , y_test)
#     if acc>best :
#         best = acc
#         # 0.9250745542256186   the best predict in 30 features 
#         print(best) 



# OR

# random_state:  helps to  have a fix row and then check the columns more accurate 
# 2.Every time we get new random numbers of different colums and rows
# x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
    
# #  To FIX  the predict , random_state  helps to  have a fix row and columns and then check the columns more accurate 
# x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1 , random_state = 123)
# model = linear_model.LinearRegression()
# Train
# model.fit(x_train , y_train)
# Test
# acc = model.score(x_test , y_test)
# print(acc)
# 0.9453520920658378







