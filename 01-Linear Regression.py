"""
ML : I am  representing some datas taken from a 
big data frame and the  predict the Final result  using sklearn , 
im the End assure its accuracy  by testing it Manually
"""
import numpy as np
import pandas as pd
import sklearn
from sklearn import linear_model

# comma separated values
data = pd.read_csv("student-por.csv", sep=";")
# Choose features from our datas
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

predict = "G3"   # G3 is the final grade we want to predict based on above datas 

x = np.array(data.drop([predict],axis = 1)) # filterd/cleaned datas : data  without G3
y = np.array(data[predict]) # Assiagn the Predicted attribute  G3 to y

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

# Make  MODEL : and then test with our data x_train
linear = linear_model.LinearRegression()

# Give to the model the TRAIN data: by using  fit
linear.fit(x_train, y_train)

#  Test its CORRECTNESS or Accuracy , checks and predict between xtest with  ytest
accuracy = linear.score(x_test,y_test)
print("Model Accuracy: ", accuracy)
# We have a 5 dimensional space here so we have 5 Coefficients! (our line starting position depending on 5 axis )


# ====CALCULATE MANUAL PREDICT  & COMPARE WITH PREDICT ===========================

# a linear equation which exist in our  MODEL : y = mx + b
# To get  Slops for each feature for :"G1", "G2", "studytime", "failures", "absences"
print("Coefficient: ", linear.coef_)
#  [ 0.15533477  0.87065932  0.07528808 -0.17585826  0.01894131] ==> feature

 
# Our intercep/bias 
print("Intercept: ", linear.intercept_) 
# Intercept: -0.09970336081990894
#  Calculate Predict by a manual  Linear equation Manually for the equation :   y = mx + b

# ML predict :
linear.predict([x_test[0]])
print(f"this is Linear.predict ,  prediction x_test: {linear.predict([x_test[0]])}")
# [11.49214246]

print(f"this is  x_test[0] :{x_test[0]}")
# [10 10 4  0 10 ]


# CALCULATE  predict Manually:
# y = liner.coenf * x_test .............+ linear.intecept 
y = 0.15533477 * 10  +  0.87065932 * 10 + 0.07528808 * 4 + 0.17585826 * 0 + 0.01894131 *10 + -0.0986502982611448
print(f"Manual predict which is close to  liner.predict : {y}")

# ===============================================================================
print("\n\n#####################################")
predictions = linear.predict(x_test)
for i in range(len(predictions)):
    print(f"Data #{i+1}")
    print("Prediction: ", predictions[i])
    print("Input Data: ", x_test[i])
    print("Real Label: ", y_test[i])
    print("#####################################")
    
# Model Accuracy:  0.8528578562635057
# Coefficient:  [ 0.13853501  0.87960452  0.1274076  -0.26790009  0.02770761]
# Intercept:  -0.13562809981482005
# this is Linear.predict ,  prediction x_test: [10.43911741]
# this is  x_test[0] :[11 10  2  0  0]
# Manual predict which is close to  liner.predict : 10.651856021738855


# #####################################
# Data #1
# Prediction:  10.43911740635293
# Input Data:  [11 10  2  0  0]
# Real Label:  11
# #####################################
# Data #2
# Prediction:  15.446695258484224
# Input Data:  [15 15  2  0  2]
# Real Label:  15
# #####################################
# Data #3
# Prediction:  14.380179376795716
# Input Data:  [14 13  2  0 32]
# Real Label:  14
# #####################################
# Data #4
# Prediction:  7.859825724300241
# Input Data:  [6 8 1 0 0]
# Real Label:  7
# #####################################
# Data #5
# Prediction:  11.496101010944065
# Input Data:  [10 11  1  0 16]
# Real Label:  12
# #####################################