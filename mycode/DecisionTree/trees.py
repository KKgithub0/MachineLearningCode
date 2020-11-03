# encoding=utf-8
"""
decision tree
"""

import numpy as np
from math import log
import operator

# 计算香农熵
# H = -SUM(p(x)*log(p(x)))
def calcShannonEnt(dataSet):
    # 获取data的总数目
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        # 这里选择最后一列的特征计算熵
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    # 根据公式计算熵
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt

# dataSet 最后一列代表labels
# labels表示特征
def createDataSet():
    dataSet = [[1, 1, 'yes'],
            [1, 1, 'yes'],
            [1, 0, 'no'],
            [0, 1, 'no'],
            [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels

# 将某一列的值去掉
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

# 返回值：能够使原始数据集信息熵减少最多的特征
# 矩阵的每一列均代表某一特征的值，最后一列代表label
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0; bestFeature = -1
    for i in range(numFeatures):
        # 获取第i列的所有特征值
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            # 计算某一列的某个特征在数据集的概率
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

# 多数表决
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        classCount.setdefault(vote, 0)
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet, labels):
    # 获取最后一列，分类
    classList = [example[-1] for example in dataSet]
    # 类别完全相同则停止划分
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 使用完所有的特征仍然不能把数据集划分位仅包含唯一类别的分组，则投票表决
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    # 获取最优特征
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel : {}}
    # 将label中的最优特征删除，由于是list，将对原有数据造成影响
    del(labels[bestFeat])
    # 将最优特征去除，继续挑选下一个最优特征
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        # 最终构建一颗递归树
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree

def classify(inputTree, featLabels, testVec):
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    # 适配了测试集的特征值位置
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel

def storeTree(inputTree, filename):
    import pickle
    fw = open(filename, 'w')
    pickle.dump(inputTree, fw)
    fw.close()

def grabTree(filename):
    import pickle
    fr = open(filename, 'r')
    return pickle.load(fr)

def testCreateTree():
    dataSet, labels = createDataSet()
    return createTree(dataSet, labels)

def testClassify():
    dataSet, labels = createDataSet()
    labels_bak = labels[:]
    myTree = createTree(dataSet, labels)
    print classify(myTree, labels_bak, [1, 0])

def testLenses():
    fr = open('./lenses.txt' ,'r')
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    lensesTree = creatTree(lenses, lensesLabels)

if __name__ == '__main__':
    testClassify()
    #storeTree(testCreateTree(), 'trees.txt')
