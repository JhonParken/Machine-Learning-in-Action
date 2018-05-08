# 打开文件并读取

import math
from numpy import *
def loadDataSet():
    dataMat = []; labelMat = []
    fr = open("textSet")  # 请使用绝对路径
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[-1]))
    return dataMat, labelMat

# sigmoid function
def sigmoid(inX):
    return 1/(1 + math.exp(-inX))

# ascent

def gradeAscent(dataMatin, classLabels):
    dataMatrix = mat(dataMatin)
    labelMat = mat(classLabels).transponse()
    m, n = shape(dataMatrix)
    weights = ones((n, 1))
    alpha = 0.001
    maxCycles = 500
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)
        error = labelMat - h
        weights = weights + alpha * error * dataMatrix.transpose()
    return weights

