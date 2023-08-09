"""
Data Preprocessing helps to prepare the datas before any processing , 
here I am showing some simple methods on a  data frame 
using Pandas how to convert a  string into  Numeric 
"""
import numpy as np
import pandas as pd
import sklearn
from sklearn import linear_model
import pickle
import matplotlib.pyplot as plt
from matplotlib import style
import joblib
# Data
data = pd.read_csv("student-por.csv", sep=";")
data = data[[ "G2", "G3", "studytime",  "absences" , "sex"]]
""
predict = "G3"


# print(data)
#      G2  G3  studytime  absences sex
# 0    11  11          2         4   F
# 1    11  11          2         2   F
# 2    13  12          2         6   F
# 3    14  14          3         0   F
# 4    13  13          2         0   F
# [649 rows x 5 columns]


#  To know how many unique exist in our table
# print(data['sex'].unique())
# ['F' 'M']
# print(data['sex'].unique()[1])
#  M


#  We create  a new column n_sex  with  0,1 instead of  its values F=0 , M=1
data["n_sex"] = [ 0 if  i == "F" else 1 for i in data["sex"]]
# print(data)

#      G2  G3  studytime  absences sex  n_sex
# 0    11  11          2         4   F      0
# 1    11  11          2         2   F      0
# 2    13  12          2         6   F      0
# 3    14  14          3         0   F      0
# 47  10  10           1         6    M     1
# 648  11  11          1         4   M      1


x = np.array(data.drop([predict , 'sex'], axis=1))
# print(x)

# [[11  2  4  0]
#  [11  2  2  0]
#  [13  2  6  0]


#  Label 
y = np.array(data[predict])
# print(y)

# [11 11 12 14 13 13 13 13 17 13 14 13 12 13 15 17 14 14  7 12 14 12 14 10
#  10 12 12 11 13 12 11 15 15 12 12 11 14 13 12 12 10 11 15 10 11 11 13 17
#  13 12 13 16  9 12 13 12 15 16 14 16 16 16 10 13 12 16 12 10 11 15 11 10
#  11 14 11 11 11 13 10 11 12  9 11 13 12 12 11 15 11 10 11 13 12 14 12 13
#  11 12 13 13  8 16 12 10 16 10 10 14 11 14 14 11 10 18 10 14 16 15 11 14
#  14 13 13 13 11  9 11 11 15 13 12  8 11 13 12 14 11 11 11 15 10 13 12 11
#  11 10 10 14  9 11  9 13 11 13 11  6 12 10 11 13 11  8 11  0 10 13 11 13
#   8 10 11 11  1 10  9  8 10  8  8  8 11 18 13 17 10 18 10 13 15 11 14 10
#  11 13 11 13 17 14 16 14 11 16 14 10 13 12 12 10 12 16 14 12 16 11 15 12
#  15 13 13  8 12 15 13 12 12 12 13 11 11 15 10 10 13 13 11 12 14 10 16  8
#  17 11 11 16 12 13 13 14  9 12 16 10 13 10 10  7  8  9 15 10 11 13  8  8
#  10 15 14 15 12 15 15 12 15 11 10 11 16 11 13  5 10 11  7 10  6 12 13 10
#  13 17 11 11 14 14 13 14 16 10 12 12 15 11 12 13 13  9 16 14 12 14 10 12
#  16 13 18 15 16 12 10 12 13 15 10 10 11 10 13 18 13 14 14 12 18 14 15 17
#  16 18 19 15 15 13 14 17 17 15 13  8 16 18 11 15 11 11 15 14 17 17 15 17
#  14 10 13 14 17 17 13 14 11 11  9 10 13 10 17 15 14 13 17 10 13 15 11 12
#  10 10 15 15 12 12 14 14 15 15 16 13 17 14 14 17 17 14 13 15 16 11 13 12
#  12 15 17 15 17 10 15 11 18 17 14 11 17 10 13 11 12 10 11 17  9 11 11 10
#   7 14 11 10  8 12 12 16  0  9 14  8 11  9 11  9 17 13 15 11 11  8  8  9
#  15 11 13 10 11 14 14 12 11  8 11 14 13 13 12 12 16 10 11 14  8 11  8 10
#  10 11  9 11  8 11 10 10  9 10 10  9 10 10  9 13 14 10 14 16  7 13  9 14
#  13 11 10 10  9 18 17 10  7  8  7 10 16 15  8  0  8 10  8  6  8 16 14 10
#   9 11  9 10  8 16 12 10 14 12 11 10 11 11 12  8 12  8 16 11 11 18 13 13
#  10 12 10 13 11 10 10 13 10 10 12  0 10  9  9  0  9  8  8  9  7 10 10 10
#  11 11 10  9 10  8  7  0 11  8  0  8  9 10  7 14 13 14 18 17 18  0 11 14
#  14 10 13  0 10  0 18 12 11 12  0 15 11 10 12 15 14 18 15 13 15 13  9 16
#   9 10  0 10 12  9 17 12  9 14 16  9 19  0 16  0  0 15 11 10 10 16  9 10
#  11]


# Changing string into Numeric 

# x = np.array(data.drop([predict], axis=1))
# y = np.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1, random_state = 123)


model = linear_model.LinearRegression()
model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
# print(acc)
# accuracy 0.9488380854381475
#  accuracy increased !
print(f" new accuracy {acc}")
# #  new 0.9503684647628233

# file_name = 'lr.save'
# y = joblib.dump(model, file_name)
# print(y)
# ['lr.save']




# Loading Model
# pickle_in = open("studentGradeModel.pickle", "rb")
# linear = pickle.load(pickle_in) 


# style.use("ggplot")
# plt.scatter(data["G1"],data["G3"], label="G1")
# #plt.scatter(data["G2"],data["G3"], label="G2")
# #plt.scatter(data["studytime"],data["G3"], label="studytime")
# #plt.scatter(data["failures"],data["G3"], label="Failures")
# #plt.scatter(data["absences"],data["G3"], label="Absences")
# plt.xlabel("P")
# plt.ylabel("Final Grade")
# plt.legend()
# plt.show()







