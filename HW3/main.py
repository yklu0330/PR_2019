import Header
import PCA
import HCM
import FCM
import PCM

noise = 0

randomData, dataVariety, datasize = Header.loadIris(noise)
# randomData, dataVariety, datasize = Header.loadWine(noise)
# randomData, dataVariety, datasize = Header.loadIonos(noise)
# randomData, dataVariety, datasize = Header.loadVertebral(noise)

# randomData = PCA.PCA(randomData)

HCM.HCM(randomData, dataVariety, datasize, 2, noise)
FCM.FCM(randomData, dataVariety, datasize, 2, 2, 0.001, noise)
PCM.PCM(randomData, dataVariety, datasize, 2, 2, 0.001, noise)
