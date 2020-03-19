import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import random

def sign(z):
    if z > 0:
        return 1
    else:
        return -1

def dot(w, y, x):
    for i in range(len(w)):
        w[i] = w[i] + y[0] * x[i]
    return w

# # load ionos dataset from csv
# dataset = pd.read_csv('./ionosphere.csv', names=['c', 'd', 'e', 'f', 'g',
#                                                'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
#                                                      'u', 'v', 'w', 'x', 'y', 'z', 'ab', 'bc', 'cd', 'de', 'ef',
#                                                      'fg', 'gh', 'hi', 'variety'], skiprows=0)


# load vertebral dataset from csv
dataset = pd.read_csv('./vertebral.csv', names=['a', 'b', 'c', 'd', 'e', 'f', 'variety'], skiprows=0)

dataVariety = dataset['variety'].unique()

# randomize ionos data
randomData = dataset.sample(frac=1)
datasize = randomData.shape[0]

KFold = 10
success = 0
confusion_mat = np.zeros([len(dataVariety), len(dataVariety)])
g = []

for i in range(KFold):
    # split the train data and test data
    trainData = randomData[:int(datasize * i / KFold) + int(datasize * (i + 1) / KFold):]
    testData = randomData[int(datasize * i / KFold):int(datasize * (i + 1) / KFold)]

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
            if sign(np.dot(w, x)) != y:
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

        predict = sign(np.dot(w, x))
        if predict == y:
            success += 1

        if predict == 1:
            confusion_mat[0][real] += 1
        elif predict == -1:
            confusion_mat[1][real] += 1

accuracy = success / dataset.shape[0]
print("accuracy: %f" %(accuracy))

# draw confusion matrix
fig, axis = plt.subplots(figsize=(5, 5))
axis.matshow(confusion_mat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confusion_mat.shape[0]):
    for j in range(confusion_mat.shape[1]):
        axis.text(x=j, y=i, s=confusion_mat[i,j], va='center', ha='center')

tick_marks = np.arange(len(dataVariety))
plt.xticks(tick_marks, dataVariety, rotation=0)
plt.yticks(tick_marks, dataVariety)

plt.xlabel('predicted label')
plt.ylabel('true label')
plt.title('Confusion matrix')
plt.show()


if (len(dataVariety) == 2):
    # draw ROC curve
    g_sort = sorted(g, reverse=True)
    TPR = []
    FPR = []
    labels = np.zeros([dataset.shape[0]])
    predPosProb = np.zeros([dataset.shape[0]])

    for i in range(len(g_sort)):
        TP = 0
        FP = 0
        FN = 0
        TN = 0
        for index, row in dataset.iterrows():
            arr = np.array(row[:dataset.shape[1] - 1])
            x = np.concatenate((np.array([1.]), arr))
            g2 = np.dot(w, x)
            if g2 >= g_sort[i] and row['variety'] == dataVariety[0]:
                TP += 1
                predPosProb[index] += 1
                labels[index] = 1
            elif g2 >= g_sort[i] and row['variety'] == dataVariety[1]:
                FP += 1
                predPosProb[index] += 1
                labels[index] = 0
            elif g2 < g_sort[i] and row['variety'] == dataVariety[0]:
                FN += 1
                labels[index] = 1
            else:
                TN += 1
                labels[index] = 0
        TPR.append(TP / (TP + FN))
        FPR.append(FP / (FP + TN))

    plt.xlabel('FA')
    plt.ylabel('PD')
    plt.plot(FPR, TPR)
    plt.show()

    auc = 0
    for i in range(len(TPR) - 1):
        auc += (TPR[i] + TPR[i + 1]) * (FPR[i + 1] - FPR[i]) / 2
    print("AUC: %f" %(auc))

