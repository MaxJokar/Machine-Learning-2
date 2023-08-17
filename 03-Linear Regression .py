import numpy as np
import pandas as pd
import sklearn
from sklearn import linear_model
import pickle

# Data Preparation
data = pd.read_csv("student-mat.csv", sep=";")
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
predict = "G3"
x = np.array(data.drop([predict],axis=1))
y = np.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)


##############################

pickle_in = open("studentGradeModel.pickle", "rb")
linear = pickle.load(pickle_in)

###############################

accuracy = linear.score(x_test,y_test)
print("Model Accuracy: ", accuracy)
print("Coefficient: ", linear.coef_) #We have a 5 dimensional space here so we have 5 Coefficients! (our line starting position depending on 5 axis )
print("Intercept: ", linear.intercept_) # y

print("\n\n#####################################")
predictions = linear.predict(x_test)
for i in range(len(predictions)):
    print(f"Data #{i+1}")
    print("Model Prediction: ", predictions[i])
    print("Input Data: ", x_test[i])
    print("Real Label: ", y_test[i])
    print("#####################################")  
# Output:
    #####################################
# Data #1
# Prediction:  9.557302298297516
# Input Data:  [10  9  3  0  2]
# Real Label:  10
# #####################################
# Data #2
# Prediction:  8.302965228849349
# Input Data:  [9 8 3 1 3]
# Real Label:  8
# #####################################
# Data #3
# Prediction:  16.908261034802674
# Input Data:  [13 17  2  0  0]
# Real Label:  17
# #####################################
# Data #4
# Prediction:  16.535368662048366
# Input Data:  [16 16  3  0  0]
# Real Label:  16
# #####################################
# Data #5
# Prediction:  11.19806703478032
# Input Data:  [10 11  1  1 16]
# Real Label:  11
# #####################################....