# Pythonで始める機械学習

常に書くっぽいやつはめもっておく

```
import sys
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import IPython
import sklearn
import mglearn
%matplotlib inline
```


## 2章：教師あり学習


### pythonぽい書き方のやつ
```
print("Sample counts per class : \n {}".format(
    {n: v for n , v in zip(cancer.target_names , np.bincount(cancer.target))})
)
>> Sample counts per class : 
>> {'malignant': 212, 'benign': 357}
```

#### np.bincount
```
array = [0,1,1,5]
count = np.bincount(array)  -> [1,2,0,0,0,1] 
ans = np.argmax(count)  -> 1
```
bincountの引数の配列の要素内の値が何個あるかを調べてくれる。
cancer例題の場合はtarget_nameの順に0か1（陽性か陰性か）が入っている。


### k-最近傍法の特徴
(以下の部分、ほんとか怪しいと思っている)

!["knn"](img/k-neighbors.png)

左から近傍点1,3,9

この境界線を決定境界という


* 近傍点の数を1とかにすると、学習データに対して最適になる
    * = 複雑度の高いモデルに対応する
* 近傍点の数を増やすと滑らかになる
    * 滑らかな境界はより単純なモデルに最適化する
* 極端なケースとして、近傍点を全てのデータポイントと同じにすると、そのデータセットに対して最適なモデルになる
    * これが過学習？

#### いい感じの決定境界探し

```
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
X_train , X_test , y_train , y_test = train_test_split(
    cancer.data , cancer.target , stratify = cancer.target , random_state = 66)

training_accuracy = []
test_accuracy = []

neighbors_settings = range(1,11)

for n_neighbors in neighbors_settings:
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X_train,y_train)
    training_accuracy.append(clf.score(X_train,y_train))
    test_accuracy.append(clf.score(X_test,y_test))
    
plt.plot(neighbors_settings , training_accuracy , label="training accuracy")
plt.plot(neighbors_settings , test_accuracy , label = "test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()
```

!["knn2"](img/k-neighbors2.png)

近傍点が6のあたりが、テストのデータを試した感じいい感じ。


### k-近傍回帰

k-最近傍法の亜種

KNeighborsRegressorで使える

```
fig , axes = plt.subplots(1,3,figsize=(15,4))
line = np.linspace(-3,3,1000).reshape(-1,1)
for n_neighbors , ax in zip([1,3,9], axes):
    reg = KNeighborsRegressor(n_neighbors=n_neighbors)
    reg.fit(X_train,y_train)
    ax.plot(line , reg.predict(line))
    ax.plot(X_train,y_train, '^' , c=mglearn.cm2(0) , markersize=8)
    ax.plot(X_test,y_test, 'v' , c=mglearn.cm2(1) , markersize=8)
    
    ax.set_title(
        "{} neighbors train score : {:.2f} test score : {:.2f}".format(
            n_neighbors , reg.score(X_train , y_train),
            reg.score(X_test,y_test)))
    
    ax.set_xlabel("Feature")
    ax.set_ylabel("Target")

axes[0].legend(["Model predictions" , "Training data/target","Test data/target"] , loc="best")
```

!["knn3"](img/k-neighbors3.png)

3あたりが、テストデータにフィットしていそう

### 最近傍法、近傍回帰の注意

* データポイントや特徴量の多いデータには向かない（遅い）ので、実際にはあまり使われない
* けど、単純にデフォでやっただけでも精度は結構出るので、まずはここを起点に考えると良い


### 線形回帰

#### 最小二乗法


```
from sklearn.linear_model import LinearRegression
X,y = mglearn.datasets.make_wave(n_samples=60)
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=42)

lr = LinearRegression().fit(X_train,y_train)

display(lr.coef_)
display(lr.intercept_)

>> array([0.39390555])
>> -0.031804343026759746
```

線形回帰に調整パラメーターはない。

* 傾きを表すパラメーター(w)は、重み、もしくは係数(coefficient)といわれ coef_ に格納される
* オフセットもしくは切片(intercept , b)はintercept_ に格納される
* coef_ とか intercept_ の最後のアンダーバーは訓練データから得られたパラメーターを表す
* scikit-learnではユーザーが設定したパラメーターと明確に区分するために、こういう仕様になっている。


#### 線形回帰の注意

* データセットが多くの特徴量を持つ場合（ボストンの例だと104の特徴量で506のサンプル)は過剰適合してしまいがち
* 訓練セットのscoreは高いがテストセットへの適合率が低い時とかはその兆候

#### そんな時に、リッジ回帰

```
from sklearn.linear_model import Ridge

ridge = Ridge().fit(X_train,y_train)
display(ridge.score(X_train,y_train))
display(ridge.score(X_test,y_test))

>> 0.8857966585170941
>> 0.7527683481744755
```

* リッジ回帰は alpha 引数で正則化(せいそくか)できる
* デフォルトは alpha = 1 
* ↑で見れるように、リッジ回帰は訓練データへの適合は低いけど、テストデータへの適合は高い（より汎化している）
* リッジ回帰は過剰適合の危険は少ない
* 最良の alpha はデータセットに依存する


```
ridge10 = Ridge(alpha=10).fit(X_train,y_train)
display(ridge10.score(X_train,y_train))
display(ridge10.score(X_test,y_test))

>> 0.7882787115369614
>> 0.6359411489177311
```

* alphaが増えるほと過剰適合していく

```
ridge01 = Ridge(alpha=0.1).fit(X_train,y_train)
display(ridge01.score(X_train,y_train))
display(ridge01.score(X_test,y_test))

>> 0.9282273685001987
>> 0.7722067936479814
``` 

* alphaが減ると適合不足になる。

それをグラフでみるには以下

```
plt.plot(ridge.coef_ , 's' , label="Ridge alpha=1")
plt.plot(ridge10.coef_ , 's' , label="Ridge alpha=10")
plt.plot(ridge01.coef_ , 's' , label="Ridge alpha=0.1")

plt.plot(lr.coef_ , 'o' ,label="LinearRegression")
plt.xlabel("Coefficient index")
plt.ylabel("Coefficient magnitude")
plt.hlines(0,0,len(lr.coef_))
plt.ylim(-25,25)
plt.legend()
```

!["ridge"](img/ridge.png)

* x軸は coef_ の要素を表す
* x=0 は最初の特徴量に対する係数、x=1は2番目の特徴量に対する係数となって、100まで続いている
* alpha=10 の時は殆どの特徴量が-3から3の範囲に収まっている
* alpha=1 はちょっと上下の範囲がばらついている
* alpha=0.1 の時はかなりばらついている



