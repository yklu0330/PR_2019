import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import random


# perception algorithm
def sign(z):
    if z > 0:
        return 1
    else:
        return -1

# inner product
def dot(w, yy, x):
    for i in range(len(w)):
        w[i] = w[i] + yy[0] * x[i]
    return w

def linearClassifier(trainData, testData, dataVariety):
    success = 0
    confusion_mat = np.zeros([len(dataVariety), len(dataVariety)])
    g = []
    testIdx = []

    w = np.zeros((trainData.shape[1]))

    error = 1
    iterator = 0

    while error != 0 and iterator < 10:
        error = 0
        for index, row in trainData.iterrows():
            arr = np.array(row[:trainData.shape[1] - 1])
            x = np.concatenate((np.array([1.]), arr))
            if row['variety'] == dataVariety[0]:
                y = np.array([1.])
            elif row['variety'] == dataVariety[1]:
                y = np.array([-1.])
            if sign(np.dot(w, x)) != y[0]:
                w = dot(w, y, x)
                error += 1
                iterator += 1

    for index, row in testData.iterrows():
        arr = np.array(row[:trainData.shape[1] - 1])
        x = np.concatenate((np.array([1.]), arr))
        if row['variety'] == dataVariety[0]:
            real = 0
            y = np.array([1.])
        elif row['variety'] == dataVariety[1]:
            real = 1
            y = np.array([-1.])

        g.append(np.dot(w, x))
        testIdx.append(index)

        predict = sign(np.dot(w, x))
        if predict == y[0]:
            success += 1

        if predict == 1:
            confusion_mat[0][real] += 1
        elif predict == -1:
            confusion_mat[1][real] += 1

    return g, testIdx, success, confusion_mat
