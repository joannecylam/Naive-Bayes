#!/usr/bin/env python
from sklearn.naive_bayes import GaussianNB
import numpy as np

f=open('5output.txt','w')
#load training data file
x= np.array(np.loadtxt('5train_x_convertedFIM.txt'))
y = np.array(np.loadtxt('5train_y_convertedFIM.txt'))

#Create a Gaussian Classifier
model = GaussianNB()

# Train the model using the training sets 
model.fit(x, y)

#Predict Output 
predicted= model.predict(np.loadtxt('5test_convertedFIM.txt'))
for row in predicted:
	s = str(row)	
	f.write(s+'\n')
