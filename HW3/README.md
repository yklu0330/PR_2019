## Task
### The c-means family: HCM, FCM, and PCM

- Experiment with different parameter settings, such as fuzzification factors in FCM and HCM.
- These all give results that are dependent on initialization. Try to devise measures for the variability of the clustering results (example: pairwise ARI).
- Add different levels of random noise to the data and see how the clustering results are affected.
- Try to see if you can determine the number of clusters using some validity measure.
- Add different levels of random noise to the data and see how the clustering results are affected.

## Methods I have implemented
### I. HCM

1. 先將原始dataset的順序隨機排序，接著再隨機取c個sample座標當作群的中心點，也就是初始的prototype
2. 將每筆sample和c個群中⼼點的歐⽒距離算出來，把該筆sample分配到距離最近的群
3. 接著更新prototype，把新的群裡的所有點座標取平均，當作新的群中心點座標
4. 重複以上2, 3步驟，直到prototype不再改變

### II. FCM

1. 先將原始dataset的順序隨機排序，接著再隨機取c個sample座標當作群的中⼼點，也就是初始的prototype
2. 將每筆sample和c個群中心點的歐⽒距離算出來，再依據公式算出每個sample和每個群的membership

<div align=center>
<img src="https://i.imgur.com/bn7zbTb.png" width="20%"/>
</div>

3. 依據公式算出prototype

<div align=center>
<img src="https://i.imgur.com/NXmodW1.png" width="20%"/>
</div>

4. 重複以上2, 3步驟，直到membership改變的最大值⼩於設定的threshold

### III. PCM

1. 先將原始dataset的順序隨機排序，接著再隨機取c個sample座標當作群的中⼼點，也就是初始的prototype

2. 將每筆sample和c個群中心點的歐⽒距離算出來，再依據公式算出每個sample和每個群的membership

<div align=center>
<img src="https://i.imgur.com/yTDxPxp.png" width="20%"/>
</div>

3. 依據公式算出prototype

<div align=center>
<img src="https://i.imgur.com/5zu5g3C.png" width="20%"/>
</div>

4. 重複以上2, 3步驟，直到membership改變的最⼤值⼩於設定的threshold

## Experiments I have done, and the results

為了⽅便觀察分群的效果，我將迭代的次數還有ARI印出來觀察。另外，我將sample的其中兩維座標拿出來來畫成⼀張⼆維的圖，並將每個點分群的結果以顏⾊表⽰，相同顏⾊代表分到相同的群。為了對照sample正確的分類，我也將sample實際的類別畫成⼆維的圖，將類別以顏⾊表示。此外，FCM和PCM經過測試後，threshold = 0.1, 0.01, 0.001的結果都差不多，因此後面的測試皆將threshold設為0.001。

### I. FCM和PCM在不同fuzzy factor下的結果

<img src="https://i.imgur.com/HZbTj1x.png" width="40%">
<img src="https://i.imgur.com/tVQ5VJs.png" width="100%">

![](https://i.imgur.com/KpPvgzQ.png)
![](https://i.imgur.com/x1IJsHa.png)

<img src="https://i.imgur.com/kmbjmOA.png" width="40%">

![](https://i.imgur.com/cBCHIiO.png)
![](https://i.imgur.com/fmbB4pV.png)
![](https://i.imgur.com/OUAzpIa.png)

以fuzzy factor = 1.5, 2, 3分別做測試後發現，FCM和PCM在fuzzy factor = 2的時候表現都不錯，在兩個dataset中，ARI都是最高的，分類出的結果與sample真實類別最接近，因此之後的測試都以fuzzy factor = 2做其他項⽬的測試。

### II. 加入noise的結果

將原有的dataset再加入⼀些雜訊，雜訊的取法是在每維座標的最大值和最⼩值之間隨機取一個數當作新的座標點，⽽雜訊的真實類別將隨機分配。為了觀察時得以區別雜訊，雜訊在真實類別的圖中以黑⾊表示，但在分群時雜訊將與原本的sample點一起被分群。

<img src="https://i.imgur.com/ERQDVuM.png" width="80%">
<img src="https://i.imgur.com/rb6Yrhw.png" width="90%">

從結果可以發現，不管是HCM, FCM, PCM，加入雜訊後ARI都降低了，表⽰分群的表現都變差，因此可以推論加入雜訊確實會影響分群表現。然⽽，雖然經過多次的測試，我無法證實PCM受到雜訊的影響最小，因為每次雜訊的影響⼤小都不一，難以做出整體的判斷。不過，我倒是有發現PCM的分類結果常常變化很⼤，有時ARI很⾼，有時ARI很低，也許是因為PCM對於初始取的群中心好壞很敏感，才會有這樣子的表現。


### III. 先⽤PCA降維再分群

為了讓圖能夠真實呈現sample所在的位置，⽽不是只⽤其中兩維的座標來畫圖，我先用PCA將dataset降成2維再去分類，也順便觀察降維後的分類表現。

<img src="https://i.imgur.com/KpggVlA.png" width="80%">
<img src="https://i.imgur.com/QYxx5cZ.png" width="80%">
<img src="https://i.imgur.com/0XDnQuc.png" width="80%">
<img src="https://i.imgur.com/BGVuhAA.png" width="80%">

透過真實類別的圖可以發現，用PCA降維後，許多不同類別的sample會重疊在一起，導致分群上的困難，容易分錯，因此降維後的ARI也比較低。

## Analysis - Are the results what I expect? Why?

經過三種測試後，測試的結果與我預期的其實差不多。上網查相關資料，許多網路上的⾼手都說fuzzy factor取2最適合，結果測試後發現真的是如此，我推測原因可能是fuzzy factor如果太低會分得太細，如果太⾼會分得越模糊。加入noise後，分群會受到noise的影響，容易將其他sample分錯，原因我想也是⼗分直觀，因為noise會⼲擾分群的過程。⽽降維後分群表現比較不好，仔細想想還蠻合理，降維本來就會失去⼀些資訊，導致分群上的困難。因此，這次作業的結果⼤致上與我想的相差不遠。

