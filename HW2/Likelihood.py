import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import statistics as stat
import random
from scipy.stats import norm
from scipy.stats import multivariate_normal
import math

def maxLikelihood(data):
    data = pd.DataFrame(data)
    newData = data.drop("variety", axis=1)

    mu = newData.sum()
    mu = mu.values.reshape([newData.shape[1], 1])
    mu = mu / newData.shape[0]

    cov = np.zeros([newData.shape[1], newData.shape[1]])

    for index, row in newData.iterrows():
        xk = row.values.reshape((newData.shape[1], 1))
        vector = xk - mu

        for i in range(newData.shape[1]):
            for j in range(newData.shape[1]):
                cov[i][j] += vector[i] * vector[j]

    cov = cov / newData.shape[0]

    return mu, cov

def gi(data, mu, cov, P, dataVarLen):
    data = pd.DataFrame(data)
    newData = data.drop("variety", axis=0)

    x = np.zeros([newData.shape[0], 1])
    for i in range(len(x)):
        x[i] = newData.ix[i]


    detCov = np.linalg.det(cov)

    expPow = -1 / 2 * np.dot(np.transpose(x - mu), np.dot(np.linalg.inv(cov), x - mu))
    div = math.pow(2 * math.pi, dataVarLen) * abs(detCov)
    pdfVal = 1 / math.sqrt(div) * math.exp(expPow)

    gi = pdfVal * P
    # gi = math.log(pdfVal * P)

    return gi


def likelihood(trainData, testData, dataVariety):
    success = 0
    confusion_mat = np.zeros([len(dataVariety), len(dataVariety)])
    g_unsort = []
    testIdX = []

    mu = np.zeros((len(dataVariety), trainData.shape[1] - 1, 1))
    cov = np.zeros((len(dataVariety), trainData.shape[1] - 1, trainData.shape[1] - 1))

    prob = []

    # train data
    for i in range(len(dataVariety)):
        classData = trainData[trainData['variety'] == dataVariety[i]]
        mu[i], cov[i] = maxLikelihood(classData)
        prob.append(classData.shape[0] / trainData.shape[0])

    # test data
    for index, row in testData.iterrows():
        g = np.zeros([len(dataVariety)])
        for i in range(len(dataVariety)):
            g[i] = gi(row, mu[i], cov[i], prob[i], len(dataVariety))
        g_unsort.append(g[0] - g[1])
        testIdX.append(index)
        predict = np.argmax(g)


        if (dataVariety[predict] == row["variety"]):
            success += 1

        real = -1
        for i in range(len(dataVariety)):
            if dataVariety[i] == row["variety"]:
                real = i
        confusion_mat[predict][real] += 1
    return g_unsort, testIdX, success, confusion_mat
