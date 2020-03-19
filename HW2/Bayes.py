import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import statistics as stat
import random
from scipy.stats import norm
import math

def pdf(x, mean, var):
    pdfVal = 1 / math.sqrt(2 * math.pi * var) * math.exp(-pow((x - mean), 2) / (2 * var))
    return pdfVal


def bayes(trainData, testData, dataVariety, names):
    success = 0
    confusion_mat = np.zeros([len(dataVariety), len(dataVariety)])
    g = []
    testIdX = []

    # train data
    mean = np.zeros((len(dataVariety), trainData.shape[1] - 1))
    var = np.zeros((len(dataVariety), trainData.shape[1] - 1))

    prob = []
    for i in range(len(dataVariety)):
        classData = trainData[trainData['variety'] == dataVariety[i]]
        classData = classData.drop("variety", axis=1)
        prob.append(classData.shape[0] / trainData.shape[0])
        for j in range(classData.shape[1]):
            classDataAttr = classData.iloc[:, j]
            mean[i][j] = stat.mean(classDataAttr)
            var[i][j] = stat.variance(classDataAttr)

    # test data
    for index, row in testData.iterrows():
        pdfVal = np.zeros((len(dataVariety), testData.shape[1] - 1))
        post = np.zeros([len(dataVariety)])
        for i in range(len(dataVariety)):
            multiPdf = 1
            newj = 0
            for j in range(testData.shape[1]):
                if names[j] == "variety":
                    continue
                pdfVal[i][newj] = pdf(row.ix[j], mean[i][newj], var[i][newj])
                multiPdf = multiPdf * pdfVal[i][newj]
                newj = newj + 1
            post[i] = prob[i] * multiPdf
        g.append(post[0] - post[1])
        testIdX.append(index)
        predict = np.argmax(post)

        if (dataVariety[predict] == row["variety"]):
            success += 1

        real = -1
        for i in range(len(dataVariety)):
            if dataVariety[i] == row["variety"]:
                real = i
        confusion_mat[predict][real] += 1

    return g, testIdX, success, confusion_mat
