import numpy as np
import random
from sklearn import metrics

import Header

def FCM(randomData, dataVariety, datasize, c, fuzzyFac, threshold, noise):
    # dataframe to numpy array
    dataLabel = np.array(randomData['variety'])
    dataLabel = dataLabel.reshape([datasize])

    dropData = randomData.drop("variety", axis=1)
    dataArr = np.array(dropData)
    dataList = dataArr.tolist()

    # randomly choose means
    means = random.sample(dataList, c)

    iteration = 0
    d = np.zeros([datasize, c])
    lastU = np.zeros([datasize, c])
    cluster = np.zeros([datasize])
    while 1:
        iteration += 1

        # recalculate uij
        u = np.zeros([datasize, c])
        for i in range(datasize):
            for j in range(c):
                d[i][j] = Header.calEucDist(dataArr[i], means[j])

        for i in range(datasize):
            for j in range(c):
                temp = 0
                if d[i][j] == 0:
                    u[i][j] = 1
                else:
                    for k in range(c):
                        flag = 0
                        if d[i][k] == 0:
                            u[i][j] = 0
                        else:
                            flag = 1
                            temp += np.power((d[i][j] / d[i][k]), (2 / (fuzzyFac - 1)))
                    if flag == 1:
                        u[i][j] = 1 / temp
            cluster[i] = np.argmax(u[i])

        # recalculate means
        for j in range(c):
            temp1 = np.zeros([1, dataArr.shape[1]])
            temp2 = 0
            for i in range(datasize):
                temp1 += np.power(u[i][j], fuzzyFac) * dataArr[i]
                temp2 += np.power(u[i][j], fuzzyFac)
            means[j] = temp1 / temp2

        # stopping criteria
        dif = u - lastU
        if abs(np.max(dif)) < threshold:
            break
        lastU = u
    print("FCM")
    print("iteration: %d" % iteration)

    predData, predLabel = Header.clustSorting(dataArr, cluster)
    if noise == 0:
        trueData, trueLabel = Header.clustSorting(dataArr, dataLabel)
    else:
        trueData, trueLabel, noiseArr = Header.realDataSorting(dataArr, dataLabel, dataVariety, noise)

    Header.plotFig(predData, "FCM_figure1")
    if noise == 0:
        Header.plotFig(trueData, "FCM_figure2")
    else:
        Header.plotNoiseFig(trueData, "FCM_figure2", noiseArr)

    ARI = metrics.adjusted_rand_score(predLabel, trueLabel)
    print("ARI: %g" % ARI)