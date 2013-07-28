from numpy import *
from scipy.optimize import *
import numpy as np
from util import *

data =loadtxt(open("train.csv","rb"),delimiter=",",skiprows=1)
y = data[:, 0]
X = data[:,1:data.shape[1]]
dummy = np.ones(data.shape[0])
processX = column_stack((dummy, X))
theta = np.zeros(processX.shape[1])
categories = 10
alpha = 1
thetaResult = findTheta(theta, processX, y, alpha, categories)
accuracy = predict(thetaResult, processX, y)
print "Train accuracy is "
print accuracy

testData =loadtxt(open("test.csv","rb"),delimiter=",",skiprows=1)
XTest = testData[:,0:testData.shape[1]]
dummyTest = np.ones(testData.shape[0])
processXTest = column_stack((dummyTest, XTest))
result = sigmoid(dot(processXTest, thetaResult)).argmax(axis = 1)
