Machine-Learning
Machine Learning Basics - Linear Regression - K Nearest Neighbor - SVM - K Means
A new journey into Machine-Learning
Linear-Regression-01 A Brief review on Machine learning : ML Supervisor , unsupervisor learning Liner gression Super vector machine Ready mathematic models give me us a learning Algorithms divided into : a computer can learn sth based on these ones 1.Supervised : yadgiri ba nezart 2.semisupervised 4.Unsupervised 4.reinforcment ###################################################### • Pandas: helps to work with datas • Numpy : for arrays and work with matrix • Matplotlib to draw graph, output graphm what model does • Sklearn: a module for AI and ML models exist pishfarz IDPL pfod github , check it just for info ##################################################### Liner Regression

In cmd: Make folder ML , inside make venv , C:\mydrive\ML>python -m venv venv C:\mydrive\ML>venv\scripts\activate !!! in ML folder install all dependencies , ML is Parent Folder (venv) C:\mydrive\ML>pip install pandas numpy sklearn notebook matplotlib pip install scikit-learn Subfolder session01 (venv) C:\mydrive\ML\ML1>python -m notebook Execute to see imports are OK ! (venv) C:\mydrive\ML>python -m notebook

import pandas as pd import numpy as np import matplotlib.pyplot as plt import sklearn

############################################################

https://archive.ics.uci.edu/dataset/320/student+performance

import pandas as pd import numpy as np import matplotlib.pyplot as plt import sklearn

df = pd.read_csv("student-por.csv",sep=";") df.head()

theory starts: We have dimension as the amount of our features , our algorithm amount increases y=m1x+m2x,….+b This style just takhmins not classified !!! remember , model exist as default in skil
