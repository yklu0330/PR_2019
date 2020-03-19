import numpy as np
import random
from sklearn import metrics

import Header

def PCM(randomData, dataVariety, datasize, c, fuzzyFac, threshold, noise):
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
    n = np.ones([c])
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
                if d[i][j] == 0:
                    u[i][j] = 1
                else:
                    temp = np.power(d[i][j], 2) / n[j]
                    u[i][j] = 1 / (1 + np.power(temp, 1 / (fuzzyFac - 1)))
            cluster[i] = np.argmax(u[i])

        # calculate n
        for j in range(c):
            temp1 = 0
            temp2 = 0
            for i in range(datasize):
                temp1 += np.power(u[i][j], fuzzyFac) * np.power(d[i][j], 2)
                temp2 += np.power(u[i][j], fuzzyFac)
            n[j] = temp1 / temp2

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
    print("PCM")
    print("iteration: %d" % iteration)

    predData, predLabel = Header.clustSorting(dataArr, cluster)
    if noise == 0:
        trueData, trueLabel = Header.clustSorting(dataArr, dataLabel)
    else:
        trueData, trueLabel, noiseArr = Header.realDataSorting(dataArr, dataLabel, dataVariety, noise)

    Header.plotFig(predData, "PCM_figure1")
    if noise == 0:
        Header.plotFig(trueData, "PCM_figure2")
    else:
        Header.plotNoiseFig(trueData, "PCM_figure2", noiseArr)

    ARI = metrics.adjusted_rand_score(predLabel, trueLabel)
    print("ARI: %g" % ARI)