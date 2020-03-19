import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import random
import statistics as stat
import heapq

import Likelihood
import Bayes
import Header
import LinearClassifier

def normalize(data, mu):
    dataVar = np.array(data['variety'])
    dataVar = dataVar.reshape([data.shape[0], 1])
    dropData = data.drop("variety", axis=1)

    var = []
    for i in range(dropData.shape[1]):
        list = []
        list = dropData.ix[:, i]
        var.append(stat.variance(list))

    arrData = np.array(dropData)
    for i in range(arrData.shape[0]):
        for j in range(arrData.shape[1]):
            arrData[i][j] -= mu[j]
            arrData[i][j] /= var[j]
    DataF = pd.DataFrame(arrData)

    dataVarFrame = pd.DataFrame(dataVar, columns=['variety'])
    normFrame = pd.concat([DataF, dataVarFrame], axis=1)

    return normFrame

def calCov(data):
    data = pd.DataFrame(data)

    mu = data.sum()
    mu = mu.values.reshape([data.shape[1], 1])
    mu = mu / data.shape[0]

    cov = np.zeros([data.shape[1], data.shape[1]])

    for index, row in data.iterrows():
        xk = row.values.reshape((data.shape[1], 1))
        vector = xk - mu

        for i in range(data.shape[1]):
            for j in range(data.shape[1]):
                cov[i][j] += vector[i] * vector[j]

    cov = cov / data.shape[0]

    return cov

def project(A, data):
    # projection of data
    dataVar = np.array(data['variety'])
    dataVar = dataVar.reshape([data.shape[0], 1])

    newData1 = data.drop("variety", axis=1)
    newData1 = np.array(newData1)
    newData1 = np.transpose(newData1)
    projData = np.zeros([A.shape[1], newData1.shape[1]])
    np.matmul(np.transpose(A), newData1, projData)

    projData = np.transpose(projData)
    projDF = pd.DataFrame(projData)

    dataVarFrame = pd.DataFrame(dataVar, columns=['variety'])
    newFrame = pd.concat([projDF, dataVarFrame], axis=1)

    return newFrame

def PCA(dataset, dataVariety, names):
    # randomize data
    randomData = dataset.sample(frac=1)
    datasize = randomData.shape[0]

    # split the train data and test data
    trainData = randomData[:int(datasize * 0.75)]
    testData = randomData[int(datasize * 0.75):]

    # standardize train data
    dropTrainData = trainData.drop("variety", axis=1)
    trainMean = dropTrainData.sum()
    trainMean = trainMean.values.reshape([dropTrainData.shape[1], 1])
    trainMean = trainMean / dropTrainData.shape[0]
    newtrainData = normalize(trainData, trainMean)

    # standardize test data
    newtestData = normalize(testData, trainMean)

    # calculate covX
    dropTrainData2 = newtrainData.drop("variety", axis=1)
    covX = calCov(dropTrainData2)

    # calculate A
    eigValX, eigVecX = np.linalg.eig(covX)
    L = 3
    maxEigIdx = np.argsort(-eigValX)
    A = []
    for i in range(L):
        A.append(eigVecX[:, maxEigIdx[i]])
    A = np.array(A)
    A = np.transpose(A)

    # projection of train data
    projTrainFrame = project(A, newtrainData)

    # projection of test data
    projTestFrame = project(A, newtestData)


    # # classify test data by likelihood
    # g1, testIdx1, success1, confusion_mat1 = Likelihood.likelihood(projTrainFrame, projTestFrame, dataVariety)
    # Header.calAccuracy(success1, projTestFrame)
    # Header.ROC_AUC(projTestFrame, dataVariety, g1, testIdx1)
    # Header.drawConfusionMat(confusion_mat1, dataVariety)

    # # classify test data by bayes
    # names = []
    # for i in range(projTestFrame.shape[1] - 1):
    #     names.append('0')
    # names.append('variety')
    # g2, testIdx2, success2, confusion_mat2 = Bayes.bayes(projTrainFrame, projTestFrame, dataVariety, names)
    # Header.calAccuracy(success2, projTestFrame)
    # Header.ROC_AUC(projTestFrame, dataVariety, g2, testIdx2)
    # Header.drawConfusionMat(confusion_mat2, dataVariety)

    # classify test data by linear classifier
    g3, testIdx3, success3, confusion_mat3 = LinearClassifier.linearClassifier(projTrainFrame, projTestFrame, dataVariety)
    Header.calAccuracy(success3, projTestFrame)
    Header.ROC_AUC(projTestFrame, dataVariety, g3, testIdx3)
    Header.drawConfusionMat(confusion_mat3, dataVariety)
