# encoding=gbk

from numpy import *
import os
import operator

def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

# kNN classfy
# inX: input vector
# dataSet: training data
# labels: training labels
# k: top count
def classify0(inX, dataSet, labels, k):
    # calc distance
    dataSetSize = dataSet.shape[0]
    # 将inX展开成dataSet的大小，获取diff
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    # ** 获取差值平方
    sqDiffMat = diffMat ** 2
    # sum函数按行求和
    sqDistances = sqDiffMat.sum(axis=1)
    # 求和后取平方根
    distances = sqDistances ** 0.5
    # 将队列元素按小到大排序并返回index
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        # 从小到大获取对应的label
        voteIlabel = labels[sortedDistIndicies[i]]
        # 将label的个数存起来
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    # sort
    sortedClassCount = sorted(classCount.iteritems(),
            key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

# 数据加载为矩阵
def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    # 创建对应矩阵
    returnMat = zeros((len(arrayOLines), 3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        if len(listFromLine) != 4:
            continue
        # 0:3 是特征值
        returnMat[index, :] = listFromLine[0 : 3]
        # -1是label
        #classLabelVector.append(int(listFromLine[-1]))
        classLabelVector.append((listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector

# 归一化
# 公式：newValue = (oldValue - min) / (max - min)
def autoNorm(dataSet):
    # min和max是一维矩阵
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    # 创建初始矩阵
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    # 矩阵的值 = oldValue - min
    normDataSet = dataSet - tile(minVals, (m, 1))
    # 矩阵的值 = oldValue / ranges
    normDataSet = normDataSet / tile(ranges, (m, 1))
    return normDataSet, ranges, minVals

def img2vector(filename):
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect

def test_classfy0():
    intX = [0, 0]
    group, labels = createDataSet()
    print classify0(intX, group, labels, 3)

def datingClassTest():
    hoRatio = 0.1
    datingDataMat, datingLabels = file2matrix('./datingTestSet.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    # 将数据集分成了训练集和测试集
    # 测试集 0 : numTestVecs
    # 训练集 numTestVecs : m
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs : m, :], \
                datingLabels[numTestVecs : m], 5)
        print "the classifier came back with: %s, the real answer is %s"\
                % (classifierResult, datingLabels[i])
        if classifierResult != datingLabels[i]:
            errorCount += 1.0
    print "the total error rate is: %f" % (errorCount / float(numTestVecs))

def handwritingClassTest():
    hwLabels = []
    trainingFileList = os.listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        # filename: 9_9.txt
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector('trainingDigits/%s' % fileNameStr)
    testFileList = os.listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        # filename: 9_9.txt
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print "the classifier came back with: %d, the real answer is %d" \
                % (classifierResult, classNumStr)
        if classifierResult != classNumStr:
            errorCount += 1.0
    print "\nthe total number of errors is: %d" % errorCount
    print "\nthe total error rate is %f" % (errorCount / float(mTest))

if __name__ == "__main__":
    #datingClassTest()
    handwritingClassTest()
