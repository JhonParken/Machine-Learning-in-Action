# 打开文件并读取


from numpy import *
def loadDataSet():
    dataMat = []; labelMat = []
    fr = open("testSet.txt")  # 请使用绝对路径
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[-1]))
    return dataMat, labelMat

# sigmoid function
def sigmoid(inX):
    return 1.0 / (1 + exp(-inX))

# ascent

def gradeAscent(dataMatin, classLabels):
    dataMatrix = mat(dataMatin)                # convert to numpy matrix
    labelMat = mat(classLabels).transpose()    # convert to numpy matrix
    m, n = shape(dataMatrix)
    weights = ones((n, 1))
    alpha = 0.001
    maxCycles = 500
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)
        error = labelMat - h
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights


# 画出决策边界
import matplotlib
def plotBestFit(weights):
    import matplotlib.pyplot as plt
    dataMat, labelMat = loadDataSet()
    # 此处注意numpy中 mat 和 array的用法
    dataArr = array(dataMat)
    # dataArr 的行数 也就是 样本个数
    n = shape(dataArr)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    # 把样本数据分成两类，一类是y=1 另一类是y=0
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1]); ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1]), ycord2.append(dataArr[i, 2])
    # 创建一个新的 1 * 1 的子图，接下来的图样绘制在其中的第 1 块（也是唯一的一块）
    fig = plt.figure()
    ax = fig.add_subplot(111)   #
    ax.scatter(xcord1, ycord1, s=30, c="red", marker="s")
    ax.scatter(xcord2, ycord2, s=30, c="green")
    # 创建横轴的上下限以及步长
    x = arange(-3.0, 3.0, 0.1)
    # 分界线方程 (xi,x2的关系）
    y = (-weights[0] - weights[1]*x) / weights[2]
    # 画
    ax.plot(x, y)
    # 设置 x, y 轴的标志
    plt.xlabel("x1"); plt.ylabel("x2");
    plt.show()


if __name__ == "__main__":
    dataMat, labelMat = loadDataSet()
    weights = gradeAscent(dataMat, labelMat)
    plotBestFit(weights.getA())



