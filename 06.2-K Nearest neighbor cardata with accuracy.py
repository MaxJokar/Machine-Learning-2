"""
Using car evaluation data set :
continues of part 6.1 with accuracy calculated given following:
k = Amount of neighbors we look for  
K should be odd number to easier to pick to classify 
"""
import numpy as np
import pandas as pd
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing



data = pd.read_csv("car.data")
#print(data.head())

myPreprocessor = preprocessing.LabelEncoder()
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

# n_neighbors : Amount of neighbors we want 
# 'odd' num is necessary cuz model could not decide for 'even' numbers which one is closer or vote
model = KNeighborsClassifier(n_neighbors=5)
# Train
model.fit(x_train,y_train)
# Test
accuracy = model.score(x_test, y_test)
print(accuracy)


# 0.7572254335260116

# DONE !
