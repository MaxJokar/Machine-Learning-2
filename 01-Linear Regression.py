"""
ML : I am  representing some datas taken from a 
big data frame and the  predict the Final result/Grade for a 
situation / student  based on its activity  using sklearn , 
im the End assure its accuracy  by testing it Manually
in Regression :we estimate  
"""
import numpy as np
import pandas as pd
import sklearn
from sklearn import linear_model

# 1. comma separated values:collect data
data = pd.read_csv("student-por.csv", sep=";")
#2. Choose features from our datas:Select data 
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

predict = "G3"   # 3. Clean data :G3 is the final grade we want to predict based on above datas 

x = np.array(data.drop([predict],axis = 1)) # filterd/cleaned datas : data  without G3
y = np.array(data[predict]) # Assiagn the Predicted attribute  G3 to : y is our Label

# 4. Make  MODEL : and then test with our data x_train
#*** x_train = train , x_test(Input Data) = Prediction , score  , y_test = Label  
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
# Make model
model = linear_model.LinearRegression()

# 5. Train  Model :Give to the model the TRAIN data: by using  fit
model.fit(x_train, y_train)

# 6. Test its CORRECTNESS or Accuracy , checks and predict between xtest with  ytest
accuracy = model.score(x_test,y_test)
print("Model Accuracy: ", accuracy)  # Model Accuracy:  0.7852231040709795
# We have a 5 dimensional space here so we have 5 Coefficients! (our line starting position depending on 5 axis )


# ====CALCULATE MANUALLY The PREDICT =========TEST for one event/student =====================================================

# a model equation which exist in our  MODEL : y = mx + b
# To get  Slops for each feature for one event/student based on :"G1", "G2", "studytime", "failures", "absences"
print("Coefficient: ", model.coef_) 
# Coefficient:  [ 0.1234198   0.89513708  0.09745936 -0.17585642  0.01760637] ==> feature

# Our intercep/bias 
print("Intercept: ", model.intercept_) 
# Intercept:  -0.06175788883222211
#  Calculate Predict by a manual  model equation Manually for the equation :   y = mx + b

# ML predict :
model.predict([x_test[0]])
print(f"this is model.predict ,  prediction x_test: {model.predict([x_test[0]])}")
# this is model.predict ,  prediction x_test: [8.3520414]

print(f"this is  x_test[0] :{x_test[0]}")
# this is  x_test[0] :[8 8 2 0 4]


# CALCULATE  predict Manually:
# y = liner.coenf * x_test .............+ model.intecept 
y = 0.1234198 * 8  +   0.89513708* 8 +  0.09745936* 2 +  -0.17585642* 0 + 0.01760637*4 + -0.06175788883222211
print(f"Manual predict which is close to  liner.predict : {y}")
# Manual predict which is close to  liner.predict : 8.35204135116778

#  CONCLUSION : we can observe from above codes even we calculate manually , the numbers are the same approximately!

# =============For all  Events/Students in our data set===========================================
print("\n\n#####################################")
predictions = model.predict(x_test)
for i in range(len(predictions)):
    print(f"Data #{i+1}")
    print("Prediction: ", predictions[i])
    print("Input Data: ", x_test[i])
    print("Real Label: ", y_test[i])
    print("#####################################")
    

# Output:
# #####################################
# Data #1
# Prediction:  15.307423161224836
# Input Data:  [14 15  1  0  4]
# Real Label:  15
# #####################################
# Data #2
# Prediction:  16.266486395196836
# Input Data:  [14 16  2  0  0]
# Real Label:  16
# #####################################
# Data #3
# Prediction:  15.717796560424603
# Input Data:  [15 15  4  0  6]
# Real Label:  15
# #####################################
# Data #4
# Prediction:  9.51874715904184
# Input Data:  [11  9  2  0  0]
# Real Label:  11
# #####################################
# Data #5
# Prediction:  17.696058688912068
# Input Data:  [17 17  4  0  2]
# Real Label:  18
# #####################################

