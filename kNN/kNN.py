# coding:utf-8


# 准备
# 使用python导入数据

# 导入科学计算包numpy 和 运算符模块

from numpy import *
import operator
def createDataSet():
    group = array([[1.0,0.9], [1.1,0.8],[0.2,0.3],[0,0]])
    labels = ('A','A','B','B')
    return group,labels

# 程序清单2-1
# k-近邻算法

# 分类代码
# 四个参数的意义：
# inX：用来分类的测试集
# dataSet : 用来训练的数据集属性
# labels: 用来分类的数据集标签
# k: k-近邻算法的k,表示选取k个有效数

def classify0 (inX, dataSet, labels, k):
    # shape[0], 返回data的行数  0- 按0维变化方向shape[i][]
    dataSetSize = dataSet.shape[0]
    # tile函数实现矩阵重复，在列方向重复1次，行方向重复dataSetSize次；
    # 对应值相减存放在 列表 diffMat中
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    # 平方
    sqDiffMat = diffMat ** 2
    # sum求和，axis = 1为按1即维变化方向操作sum[][j]，axis = 0 为按0维方向操作sum[i][]
    sqDistances = sqDiffMat.sum(axis=1)
    # 开方
    distances = sqDistances ** 0.5
    # distances.argsort(),将distances中元素即距离，从小到大排序，并返回索引值，存放在sortedDistIndicies
    sortedDistIndicies = distances.argsort()
    # 创建一个字典
    ClassCount = {}
    for i in range(k):
        # 返回距离排名第i的标签
        voteIlabel = labels[sortedDistIndicies[i]]
        # dict.get(key,default= None），字典中 get() 返回字典中指定 键 的 值
        # dict.get() 若字典中没有该标签则创建并将其值置为1，若已有标签则返回其值并将值加1
        # 如果 key不在字典中，则返回默认值default的值
        # 用 字典 记录 键（标签） 出现的次数
        ClassCount[voteIlabel] = ClassCount.get(voteIlabel, 0) + 1
        # 将 ClassCouNt 中的值进行排序
        # sorted(iterable,cmp= None,key = None,reverse = false)
        # iterable 待排序的可迭代类型的容器
        # cmp:用于比较函数；key = operator,itemgetter(1)根据字典的值进行排序
        # key=operator.itemgetter(1)根据字典的键进行排序
        # reverse = True 降序  or  reverse = False 升序|（默认）
    sortedClassCount = sorted(ClassCount.items(), key=operator.itemgetter(1), reverse=True)
    # 返回要分类的类别
    return sortedClassCount[0][0]



# 2.2.1
# 准备数据：从文本中解析数据
# 程序清单2-2
# 将文本记录转化为Numpy的解析程序

def file2matrix(filename):
    # 打开文件
    fr = open(filename)
    # readlines()方法用于读取所有行(直到结束符 EOF)并返回 列表 ;如果碰到结束符 EOF 则返回 空字符串
    arrayOLines = fr.readlines()
    # 返回文件行数
    numberOFLines = len(arrayOLines)
    # 创建返回特定形状的以0填充的 numpy 矩阵(n 行 * 每行 m 列)
    returnMat = zeros((numberOFLines, 3))
    # 创建空列表 存储标签
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        # 去掉字符串前后的空格
        line = line.strip()
        #  split() 通过指定 分隔符("\t") 对字符串进行切片，如果参数 num 有指定值，则仅分隔 num 个子字符串
        listFromLine = line.split("\t")
        # 填充 returnMat 用切片好的 listFromLines
        returnMat[index, :] = listFromLine[0:3]
        # 将 listFromLines 中的标签输入字典 classLabelVetor 中
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector


# 2.2.2
# 分析数据：使用 Matplotlib 创建散点图


# 2.2.3
# 数据归一化

def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet / tile(ranges, (m,1))
    return normDataSet, ranges, minVals










