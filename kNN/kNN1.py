# kNN

# 预处理
from numpy import *
import operator
def createDataSet():
    group = array([[1.0, 0.8], [0.9, 0.7], [0.5, 0.4], [0.4, 0.3]])
    labels = ("A", "A", "B", "B")
    return group, labels

# kNN 主体函数
def classify0(inX, DataSet, labels, k):
    dataSetSize = DataSet.shape[0]      # 行数
    diffMat = tile(inX, (dataSetSize, 1)) - DataSet   # 差
    sqdiffMat = diffMat ** 2
    sqdistances = sqdiffMat.sum(axis=1)
    distances = sqdistances ** 0.5
    sortedInDistances = distances.argsort()
    ClassCount = {}
    for i in range(k):
        votleLabels = labels[sortedInDistances[i]]
        ClassCount[votleLabels] = ClassCount.get(votleLabels, 0) + 1
        sortedClassCount = sorted(ClassCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

# 读取文件中的数据

def file2matrix(filename):
    fr = open(filename)
    arrayOlines = fr.readlines()
    numberOFlines = len(arrayOlines)
    returnMat = zeros((numberOFlines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOlines:
        line = line.strip()
        listFromline = line.split("\t")
        returnMat[index, :] = listFromline[0:3]
        classLabelVector.append(int(listFromline[-1]))
        index += 1
    return returnMat, classLabelVector
    fr.close()

# 数据归一化

def autoNormm(dataMat):
    minVals = dataMat.min()
    maxVals = dataMat.max()
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataMat))
    m = dataMat.shape[0]
    normDataSet = dataMat - tile(minVals, (m, 1))
    normDataSet = dataMat / tile(maxVals, (m, 1))  # tile() 在行方向重复 m 次，在列方向 重复 1 次
    return normDataSet, ranges, minVals


# 分类

def classifyPerson():
    labelsList = ["not at all", "in small does", "in large does"]
    percentTats = float(input("percentage of time spent in playing computer games"))
    ffMiles = float(input("frequent flier miles earend per year"))
    iceCream = float(input("liters of ice cream consumed per year"))
    datingDataSet, datingLabels = file2matrix("D:\Code\PycharmProjects\ML in Action\kNN\datingTestSet2.txt")
    normMat, ranges, minVals = autoNormm(datingDataSet)
    inArr = array([ffMiles, percentTats, iceCream])
    ClassifierResult = classify0(inArr, normMat, datingLabels, 3)
    print("You will probably like the person:" + labelsList[ClassifierResult - 1])




