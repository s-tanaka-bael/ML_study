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


#### Lasso

Ridgeとは別の線形回帰のやつ
* いくつかのデータを完全に無視するのが特徴

```
from sklearn.linear_model import Lasso

lasso = Lasso().fit(X_train , y_train)
display(lasso.score(X_train,y_train))
display(lasso.score(X_test,y_test))
np.sum(lasso.coef_ != 0)

>> 0.29323768991114596
>> 0.20937503255272272
>> 4
```

* デフォだとめちゃくちゃ弱い
* 4つしか特徴量使ってない

```
lasso001 = Lasso(alpha=0.01 , max_iter=100000).fit(X_train,y_train)

display(lasso001.score(X_train,y_train))
display(lasso001.score(X_test,y_test))
display(np.sum(lasso001.coef_ != 0))

>> 0.8962226511086497
>> 0.7656571174549983
>> 33
```

* alphaが小さくなるとより複雑なデータに適合するようになる

```
lasso00001 = Lasso(alpha=0.0001 , max_iter=100000).fit(X_train,y_train)

display(lasso00001.score(X_train,y_train))
display(lasso00001.score(X_test,y_test))
display(np.sum(lasso00001.coef_ != 0))

>> 0.9507158754515462
>> 0.6437467421273558
>> 96
```
* alpha小さくしすぎると正則化の効果が薄れて過剰適合する


#### クラス分類の為の線形モデル

* ロジスティック回帰と線形サポートベクターマシン

```
# ロジスティック回帰
linear_model.LogisticRegression 

# 線形サポートベクタマシン
svm.LinearSVC
```

forgeデータでおためし

```
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

X,y = mglearn.datasets.make_forge()

fig, axes = plt.subplots(1,2, figsize=(10,3))

for model, ax in zip([LinearSVC(),LogisticRegression()], axes):
    clf = model.fit(X,y)
    mglearn.plots.plot_2d_separator(clf , X , fill=False , eps=0.5 , ax=ax , alpha=.7)
    mglearn.discrete_scatter(X[:,0] , X[:,1] , y , ax=ax)
    ax.set_title("{}".format(clf.__class__.__name__))
    ax.set_xlabel("Feature 0")
    ax.set_ylabel("Feature 1")

axes[0].legend
```

!["LRvsSVC"](img/LRvsSVC.png)

* LogisticRegressionとLinearSVCの正則化強度を決定するパラメータはCと言われる
* Cが大きくなると正則化が弱くなる（訓練データに適合度をあげる）
* Cが小さくなるとデータポイントの「大多数」に対して適合しようとする（この辺が面白い特徴らしい）


!["LRvsSVC2"](img/LRvsSVC2.png)

* Cが小さい〜大きいまでの決定境界
* 小さすぎると境界が水平に近くなって、大きすぎると過剰適合


別のデータでやってみる

```
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
X_train,X_test,y_train,y_test = train_test_split(cancer.data , cancer.target , stratify=cancer.target , random_state = 42)
logreg = LogisticRegression().fit(X_train,y_train)

display(logreg.score(X_train,y_train))
display(logreg.score(X_test,y_test))

>> 0.9530516431924883
>> 0.958041958041958
```

#### 線形モデルによる多クラス分類

* 基本線形モデルは多クラス分類できないので、1対nでそれっぽい感じにする

```
from sklearn.datasets import make_blobs

X,y = make_blobs(random_state=42)
mglearn.discrete_scatter(X[:,0] , X[:,1] , y)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.legend(["Class 0", "Class 1" , "Class 2"])
```

!["LRvsSVC3"](img/LRvsSVC3.png)

```
linear_svm = LinearSVC().fit(X,y)
display(linear_svm.coef_.shape)
display(linear_svm.intercept_.shape)

>> (3, 2)
>> (3,)
```


```
mglearn.discrete_scatter(X[:,0] , X[:,1] , y)
line = np.linspace(-15,15)
for coef , intercept , color in zip(linear_svm.coef_ , linear_svm.intercept_ , ['b','r','g']):
    plt.plot(line, -(line * coef[0] + intercept) / coef[1], c=color)

plt.ylim(-10 , 15)
plt.xlim(-10 , 8)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.legend(["Class 0","Class 1","Class 2", "Line class 0","Line class 1","Line class 2"], loc=(1.01,0.3))

```

* それぞれのクラスの決定境界はこう

!["LRvsSVC4"](img/LRvsSVC4.png)


```
mglearn.plots.plot_2d_classification(linear_svm , X , fill=True , alpha=.7)
mglearn.discrete_scatter(X[:,0] , X[:,1] , y)
ine = np.linspace(-15,15)
for coef , intercept , color in zip(linear_svm.coef_ , linear_svm.intercept_ , ['b','r','g']):
    plt.plot(line, -(line * coef[0] + intercept) / coef[1], c=color)

plt.ylim(-10 , 15)
plt.xlim(-10 , 8)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.legend(["Class 0","Class 1","Class 2", "Line class 0","Line class 1","Line class 2"], loc=(1.01,0.3))
```

* こんな感じ

!["LRvsSVC5"](img/LRvsSVC5.png)


#### 利点・欠点・パラメータ

* 回帰モデルではalpha
* LinbearSVCとLogisticRegressionではC
* alphaが大きいとき、Cが小さいときは単純なモデルに適合する
* 線形モデルは高速、大きいデータセットにも適用できる
* デメリット何書いてるかよくわからんかった



#### ナイーブベイズクラス分類器

* 線形モデルによく似た分類器
* 高速だけど、LogisticRegressionとかLinearSVCとかより汎化性能が劣る場合が多い
* scikit-learn には GaussianNB , BernoulliNB , MultinomialNB が実装されている
* GaussianNBは任意の連続値データに適用できる
* BernoulliNBは2値データを仮定されている
* MultinomialNBはカウントデータを仮定している
    * カウントデータは例えば、文中に出てくる単語の出現数など


#### 決定木

* Yes/No で答えられる質問で構成された、階層的な木構造を学習する
* 決定木の学習は正解に最も早く辿り着けるようなYes/No型の質問を学習すること
* 個々のテストは1つの特徴量しかもたない(Yes/Noなので)常に軸に平行な境界を持つ





