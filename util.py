from numpy import *
from scipy.optimize import *
import numpy as np

def sigmoid(X):
	return 1/(1 + e ** (-X))

def computeCost(theta, X, y, alpha):
	hypothesis = sigmoid( dot(X, theta) )
	tempResult = (-y * log(hypothesis)) - (1 - y) * log(1 - hypothesis)
	tempResult = sum(tempResult)
	cost = tempResult / X.shape[0]

	newTheta = theta
	newTheta[0] = 0
	sqTheta = newTheta * newTheta
	cost = cost + (sum(sqTheta) * alpha) / (2 * X.shape[0])
	return cost

def costGradient(theta, X, y, alpha):
	hypothesis = sigmoid( dot(X, theta) )
	gradTemp = dot( (hypothesis - y).T, X)
	newTheta = theta
	newTheta[0] = 0
	grad = gradTemp + (newTheta * alpha).T
	grad = grad / X.shape[0]
	return grad.T

def findTheta(theta, X, y, alpha, categories):
	allThetaResult = []
	for i in range(0,categories):
		curY = np.zeros(X.shape[0])
		curY[where(y == i)] = 1
		thetaResult = fmin_ncg(computeCost, theta, args=(X,curY,alpha), fprime=costGradient)
		if i == 0:
			allThetaResult = thetaResult
		else:
			allThetaResult = column_stack((allThetaResult, thetaResult))
	
	return allThetaResult

def predict(thetaResult, X, y):
	predictionRes = sigmoid(dot(X, thetaResult)).argmax(axis = 1)
	correctResult = sum(predictionRes == y)
	accuracy = float(correctResult)/float(X.shape[0])
	return accuracy