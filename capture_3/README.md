# 教師なし学習と前処理

教師なし学習ではアルゴリズムには入力データだけ与えられ、そこから知識を抽出する事が要求される。


## 教師なし学習の種類

* データセットの変換

    * データセットの教師なし変換(Unsupervised transformations)

        元のデータ表現を変換して人間や他の機械学習アルゴにとって、わかりやすい新しいデータ表現を作るアルゴ

        #### 最も一般的な利用法は次元削減である

        他には、データを「構成する」部品、成分を見つける事。

        #### 例えば、文章データからトピックを抽出する

* クラスタリング

    * クラスタリングアルゴリズム(Clustering algorithms)

        データを似たような要素から構成されるグループに分けるアルゴリズム

        #### 例えば、SNSサイトにアップされた写真の中から同じ人を抽出する

の2つがある。

## 教師なし学習の難しさ

アルゴリズムが学習したことの有用性の評価。結果を人間が確かめるしかない場合とか。

なので、教師あり学習の前処理ステップとして利用する事があったりする。


## 前処理とスケール変換

教師あり学習では、ニューラルネットワークやVMなどのアルゴはデータのスケール変換が重要って話があった。

よく使われるのは、特徴量ごとにスケールを変換してずらす方法


```
mglearn.plots.plot_scaling()
```

!["sample"](img/sample.png)


scikit-learnに

* StandardScaler
* RobustScaler
* MinMaxScaler
* Normalizer

とかがある。


## データ変換やってみる


```
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
cancer = load_breast_cancer()

X_train , X_test , y_train , y_test = train_test_split(cancer.data,cancer.target , random_state=1)

print(X_train.shape)
print(X_test.shape)

>> (426, 30)
>> (143, 30)
``` 

教師なし学習なんだけど、前処理した後に構築する教師ありモデルを評価するために訓練セットとテストセットに分けてる

```
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)

>> MinMaxScaler(copy=True, feature_range=(0, 1))
```

fitメソッドを訓練データに適合させる。MinMaxScalerのfitメソッドはデータの各特徴量の最小値と最大値を計算する。教師ありのクラス分類とは違って、X_trainのみで教師データは用いない

学習した変換を実際に適応する(スケール変換する)にはスケール変換器の transform メソッドを用いる。


```
X_train_scaled = scaler.transform(X_train)

print("transformed shape : {}".format(X_train_scaled.shape))
print("pre-freature minimum before scaling:\n {}".format(X_train.min(axis=0)))
print("pre-freature maximum before scaling:\n {}".format(X_train.max(axis=0)))
print("pre-freature minimum after scaling:\n {}".format(X_train_scaled.min(axis=0)))
print("pre-freature maximum before scaling:\n {}".format(X_train_scaled.max(axis=0)))

>> transformed shape : (426, 30)
>> pre-freature minimum before scaling:
>>  [6.981e+00 9.710e+00 4.379e+01 1.435e+02 5.263e-02 1.938e-02 0.000e+00
>>  0.000e+00 1.060e-01 5.024e-02 1.153e-01 3.602e-01 7.570e-01 6.802e+00
>>  1.713e-03 2.252e-03 0.000e+00 0.000e+00 9.539e-03 8.948e-04 7.930e+00
>>  1.202e+01 5.041e+01 1.852e+02 7.117e-02 2.729e-02 0.000e+00 0.000e+00
>>  1.566e-01 5.521e-02]
>> pre-freature maximum before scaling:
>>  [2.811e+01 3.928e+01 1.885e+02 2.501e+03 1.634e-01 2.867e-01 4.268e-01
>>  2.012e-01 3.040e-01 9.575e-02 2.873e+00 4.885e+00 2.198e+01 5.422e+02
>>  3.113e-02 1.354e-01 3.960e-01 5.279e-02 6.146e-02 2.984e-02 3.604e+01
>>  4.954e+01 2.512e+02 4.254e+03 2.226e-01 9.379e-01 1.170e+00 2.910e-01
>>  5.774e-01 1.486e-01]
>> pre-freature minimum after scaling:
>>  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
>>  0. 0. 0. 0. 0. 0.]
>> pre-freature maximum before scaling:
>>  [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
>>  1. 1. 1. 1. 1. 1.]

```

変換されたデータの配列は元の形と同じだけど、特徴量がシフトされて0と1になっている。

同じく、X_testにも適応してやる。

```
X_test_scaled = scaler.transform(X_test)

print("pre-feature minimum after scaling:\n {}".format(X_test_scaled.min(axis=0)))
print("pre-feature maximum after scaling:\n {}".format(X_test_scaled.max(axis=0)))

>> pre-feature minimum after scaling:
>>  [ 0.0336031   0.0226581   0.03144219  0.01141039  0.14128374  0.04406704
>>   0.          0.          0.1540404  -0.00615249 -0.00137796  0.00594501
>>   0.00430665  0.00079567  0.03919502  0.0112206   0.          0.
>>  -0.03191387  0.00664013  0.02660975  0.05810235  0.02031974  0.00943767
>>   0.1094235   0.02637792  0.          0.         -0.00023764 -0.00182032]
>> pre-feature maximum after scaling:
>>  [0.9578778  0.81501522 0.95577362 0.89353128 0.81132075 1.21958701
>>  0.87956888 0.9333996  0.93232323 1.0371347  0.42669616 0.49765736
>>  0.44117231 0.28371044 0.48703131 0.73863671 0.76717172 0.62928585
>>  1.33685792 0.39057253 0.89612238 0.79317697 0.84859804 0.74488793
>>  0.9154725  1.13188961 1.07008547 0.92371134 1.20532319 1.63068851]

```

テストデータの方は、スケール変換後の最小と最大の値が0と1になっていない！特徴量によっては1をはみ出ている。

これは、MinMaxScalerが訓練データとテストデータに全く同じ変換を施すから。transformメソッドは訓練データの最小値と最大値を引き継ぎ、訓練データのレンジで割るからだ。


## 訓練データとテストデータを同じように変換する

テストセットを訓練セットと全く同じスケールで変換するのはめちゃくちゃ重要。

試しに、テストセットの最小値とレンジを使うと何が起こるかを示す。

```
from sklearn.datasets import make_blobs
# 合成データを作成
X, _ = make_blobs(n_samples=50,centers=5,random_state=4,cluster_std=2)

# 訓練セットとテストセットをプロット
X_train , X_test = train_test_split(X,random_state=5,test_size=.1)

fig,axes = plt.subplots(1,3,figsize=(13,4))
axes[0].scatter(X_train[:,0],X_train[:,1],c=mglearn.cm2(0),label="Training set",s=60)
axes[0].scatter(X_test[:,0],X_test[:,1],marker="^",c=mglearn.cm2(1),label="Test set",s=60)
axes[0].legend(loc="upper left")
axes[0].set_title("Original Data")

# MinMaxScalerでデータ変換
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# スケール変換されたデータの特性を可視化
axes[1].scatter(X_train_scaled[:,0],X_train_scaled[:,1],c=mglearn.cm2(0),label="Training set", s=60)
axes[1].scatter(X_test_scaled[:,0],X_test_scaled[:,1],marker="^",c=mglearn.cm2(1),label="Test set", s=60)
axes[1].set_title("Scaled Data")

# テストセットを訓練セットとは別にスケール変換
# 最小値と最大値が0,1になる。ここではわざとやっているが、「実際にはやってはいけない！」
test_scaler = MinMaxScaler()
test_scaler.fit(X_test)
X_test_scaled_badly = test_scaler.transform(X_test)

# ダメなスケール変換を可視化
axes[2].scatter(X_train_scaled[:,0],X_train_scaled[:,1],c=mglearn.cm2(0),label="Training set", s=60)
axes[2].scatter(X_test_scaled[:,0],X_test_scaled[:,1],marker="^",c=mglearn.cm2(1),label="Test set", s=60)
axes[2].set_title("Improperly Scaled Data")

for ax in axes:
    ax.set_xlabel("Feature 0")
    ax.set_ylabel("Feature 1")

```


!["sample2"](img/sample2.png)

一番左は何もいじってないやつ。スケール変換されていないので値の範囲がバラついている。

真ん中が訓練データベースでスケールされたもの。

一番右がダメなやつで、訓練データとテストデータが別々のスケールで変換されている。あかんやつ！



## 教師あり学習における前処理の結果

```
from sklearn.svm import SVC

X_train,X_test,y_train,y_test = train_test_split(cancer.data,cancer.target,random_state=0)

svm = SVC(C=100)
svm.fit(X_train,y_train)
print("Test set accuracy : {:.2f}".format(svm.score(X_test,y_test)))

>> Test set accuracy : 0.63
```
63%くらいの精度

```
# 0-1スケール変換で前処理
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 変換された訓練データで学習さす
svm.fit(X_train_scaled,y_train)

print("Scaled set accuracy : {:.2f}".format(svm.score(X_test_scaled,y_test)))

>> Scaled set accuracy : 0.97
```

めっちゃアガる

```
# 平均を0に分散を1に前処理
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

svm.fit(X_train_scaled,y_train)

print("SVM test accuracy: {:.2f}".format(svm.score(X_test_scaled,y_test)))

>> SVM test accuracy: 0.96
```

Standard Scalerでもめっちゃアガる


## 次元削減、特徴量抽出、多様体学習

教師なし学習のデータ変換の一般的な動機は、可視化、データの圧縮、以降の処理に適した表現の発見(いみわからん)

それらに最もよく使われるアルゴリズムが主成分分析(pricinpal component analysys : PCA)

主成分分析の他に、非負値行列因子分解(non-negative matrix factorization : NMF)と二次元散布図を用いたデータ可視化によく用いられる t-SNE がある。


### 主成分分析(PCA)

主成分分析とは、データセットの特徴量を相互に統計的に関連しないように回転する手法。

多くの場合、回転したあとの特徴量からデータを説明するのに重要な一部の特徴量だけを抜き出す。


!["pca"](img/pca.png)

137pあたりに説明があるが、全く意味がわからん。

#### cancerデータセットのPCAによる可視化

cancerデータは特徴量が30もあるので、たいへん。ちゃんとやると、30×29/2=435の散布図ができてしまうし、理解なんてできない。

なので、特徴量ごとに良性か悪性の2つのクラスのヒストグラムを書く。

```
fig, axes = plt.subplots(15,2,figsize=(10,20))
malignant = cancer.data[cancer.target == 0]
benign = cancer.data[cancer.target == 1]

ax = axes.ravel()

for i in range(30):
    _, bins = np.histogram(cancer.data[:,i],bins=50)
    ax[i].hist(malignant[:,i],bins=bins,color=mglearn.cm3(0),alpha=.5)
    ax[i].hist(benign[:,i],bins=bins,color=mglearn.cm3(2),alpha=.5)
    ax[i].set_title(cancer.feature_names[i])
    ax[i].set_yticks(())

ax[0].set_xlabel("Feature magnitude")
ax[0].set_ylabel("Frequency")
ax[0].legend(["malignant","benign"],loc="best")
fig.tight_layout()

```


!["pca"](img/pca2.png)

個々のデータポイントの特徴量が特定のレンジ(ビンと呼ぶ)に何回入ったか数えることで特徴量ごとにヒストグラムを作っている。

malignant：悪性(青)

benign：良性(緑)

のデータがそれぞれどういう分布になっているかがよくわかる。(これによって、どの特徴量が良性と悪性を見分けるのに使えそうかわかる)

例えば「smoothness error」とかは値が被っていて使えなさそうなのがわかる。逆に「worst concave points」はほとんど重なっていないので良さげ。

しかしこれを見てもここの特徴量の相関やそれがクラス分類に与える影響についてはわからない（わからんの？）、そこでPCAの出番。

PCAると、主な相関を捉える事ができるのでもう少し全体像が見やすくなる。


```
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()

scaler = StandardScaler()
scaler.fit(cancer.data)
X_test_scaled = scaler.transform(cancer.data)
```

PCAを用いる前に、StandardScalerでスケール変換して、個々の特徴量の分散が1になるようにする。

PCA変換の学習と適用は前処理と同じように簡単にできる。PCAオブジェクト作って、fitしてtransformするだけ。これで回転と次元削減を行ってくれる。

デフォルトだとデータの回転とシフトしか行わず、全ての主成分を維持する。データの次元削減を行うにはPCAオブジェクトを作る際に、維持する主成分の数を指定する必要がある

```
from sklearn.decomposition import PCA

# データの最初の2つの主成分だけ維持する
pca = PCA(n_components=2)
# cancerデータにPCAモデルを適合
pca.fit(X_scaled)

# 最初の2つの主成分に対してデータポイントを変換
x_pca = pca.transform(X_scaled)
print("Origin shape: {}".format(str(X_scaled.shape)))
print("Reduced shape: {}".format(str(x_pca.shape)))

>> Origin shape: (569, 30)
>> Reduced shape: (569, 2)
```

```
# 第一主成分と第二主成分のプロット、クラスごとに色分け
plt.figure(figsize=(8,8))
mglearn.discrete_scatter(x_pca[:,0],x_pca[:,1],cancer.target)
plt.legend(cancer.target_names,loc="best")
plt.gca().set_aspect("equal")
plt.xlabel("First principal component")
plt.ylabel("Second principal component")
```

!["pca"](img/pca3.png)

これはつまり、第一主成分に対して第二主成分をクラス情報を使って相関をプロットしたの図。

教師なしなんだけど、結構いい感じに別れている。悪性のデータポイントは良性に比べて広範囲に分布しているのもわかる。


#### 固有顔による特徴量抽出

PCAのもう一つの利用方法は特徴量抽出、画像関連など。

```
from sklearn.datasets import fetch_lfw_people
people = fetch_lfw_people(min_faces_per_person=20,resize=0.7)
image_shape = people.images[0].shape

fix,axes = plt.subplots(2,5,figsize=(15,8),subplot_kw={'xticks':(),'yticks':()})
for target , image , ax in zip(people.target,people.images,axes.ravel()):
    ax.imshow(image)
    ax.set_title(people.target_names[target])

```

!["pca"](img/pca4.png)


```
print("people.images.shape: {}".format(people.images.shape))
print("Number of classes: {}".format(len(people.target_names)))

>> people.images.shape: (3023, 87, 65)
>> Number of classes: 62

```

```
# 各ターゲットの出現回数をカウント
counts = np.bincount(people.target)
# ターゲット名と出現回数を並べて表示
for i , (count,name) in enumerate(zip(counts,people.target_names)):
    print("{0:25} {1:3}".format(name,count),end='   ')
    if (i + 1) % 3 == 0:
        print()


>> Alejandro Toledo           39   Alvaro Uribe               35   Amelie Mauresmo            21   
>> Andre Agassi               36   Angelina Jolie             20   Ariel Sharon               77   
>> Arnold Schwarzenegger      42   Atal Bihari Vajpayee       24   Bill Clinton               29   
>> Carlos Menem               21   Colin Powell              236   David Beckham              31   
>> Donald Rumsfeld           121   George Robertson           22   George W Bush             530   
>> Gerhard Schroeder         109   Gloria Macapagal Arroyo    44   Gray Davis                 26   
>> Guillermo Coria            30   Hamid Karzai               22   Hans Blix                  39   
>> Hugo Chavez                71   Igor Ivanov                20   Jack Straw                 28   
>> Jacques Chirac             52   Jean Chretien              55   Jennifer Aniston           21   
>> Jennifer Capriati          42   Jennifer Lopez             21   Jeremy Greenstock          24   
>> Jiang Zemin                20   John Ashcroft              53   John Negroponte            31   
>> Jose Maria Aznar           23   Juan Carlos Ferrero        28   Junichiro Koizumi          60   
>> Kofi Annan                 32   Laura Bush                 41   Lindsay Davenport          22   
>> Lleyton Hewitt             41   Luiz Inacio Lula da Silva  48   Mahmoud Abbas              29   
>> Megawati Sukarnoputri      33   Michael Bloomberg          20   Naomi Watts                22   
>> Nestor Kirchner            37   Paul Bremer                20   Pete Sampras               22   
>> Recep Tayyip Erdogan       30   Ricardo Lagos              27   Roh Moo-hyun               32   
>> Rudolph Giuliani           26   Saddam Hussein             23   Serena Williams            52   
>> Silvio Berlusconi          33   Tiger Woods                23   Tom Daschle                25   
>> Tom Ridge                  33   Tony Blair                144   Vicente Fox                32   
>> Vladimir Putin             49   Winona Ryder               24

```

↑この辺のループとか配列の処理がパッと頭に入ってこない。

データが少し偏ってるので、次で最大数50に抑える


```
mask = np.zeros(people.target.shape,dtype=np.bool)
for target in np.unique(people.target):
    mask[np.where(people.target==target)[0][:50]] = 1

X_people = people.data[mask]
y_people = people.target[mask]

# 0から255で表現されている、グレースケールの値0と1の間に変換する
# こうした方が数値的に安定する
X_people = X_people/255.
```

顔認識の一般的なタスクとして、見たことのない顔がデータベースの中と一致するかを判別するタスクがある。

しかし多くの場合、顔データベースにはたくさんの人物が登録されており、同じ人物の画像は少ない（つまり訓練データが少ない）

こういう場合、ほとんどクラス分類器は訓練が難しくなる。さらに新しい人を追加するたびに大きなモデルを再訓練するのは大変。

簡単な方法として、1-最近傍法クラス分類器を使う方法がある。クラス分類しようとしている顔に一番近いものを探す。理論的にはクラスごとに訓練サンプルが1つだけあれば機能するはず。KNeighborsClassifierがどのくらいうまく機能するか見てみる。

```
from sklearn.neighbors import KNeighborsClassifier
# 訓練データとテストデータに分割

X_train , X_test , y_train , y_test = train_test_split(X_people,y_people,stratify=y_people,random_state=0)

# KNeighborsClassifierを1-最近傍法で構築
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)
print("Test set score of 1-nn: {:.2f}".format(knn.score(X_test,y_test)))

>> Test set score of 1-nn: 0.23

```

精度が低い！

ただこれは62クラス分類なので、そこまで悪くはない（ランダムに選択したら 1/62=1.5%）

4回に1回しか当たらない程度の精度ではあるが。

そこでPCAの出番

元のピクセルの空間で距離を系さんするのは顔の近似値を測るのは全く適していない。ピクセル表現で2つの画像を比較するということは、相互の画像の対応するピクセルの値を比較する事になる。例えば同じ画像でも1ピクセルずらすだけで全く違うデータになってしまうから。

主成分に沿った距離を使うことで精度が上げれないか試してみよう。ここではPCAのwhitenオプションを使う。これを使うと主成分が同じスケールになるようにスケール変換する。

PCA変換後にStandardScalerをかけるのと同じ。whitenオプションをつけると、データを回転するだけでなく楕円ではなく円を描くようにスケール変換することになる。

!["pca"](img/pca5.png)

イメージはこんな↑感じ

PCAオブジェクトを訓練して最初の100主成分を抜き出す。そのあと訓練データとテストデータを変換する

```
pca = PCA(n_components=100,whiten=True,random_state=0).fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

print("X_train_pca.shape : {}".format(X_train_pca.shape))

>> X_train_pca.shape : (1547, 100)

```

新しいデータは100の特徴量を持つ。主成分の最初の100要素。これを使って1-最近傍法クラス分類器にかけてみる。


```
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train_pca,y_train)
print("Test set accuracy: {:.2f}".format(knn.score(X_test_pca,y_test)))

>> Test set accuracy: 0.31
```

10%くらい上がった。

画像データは見つけた主成分を簡単に可視化できる。

```
print("pca.components_.shape : {}".format(pca.components_.shape))

>> pca.components_.shape : (100, 5655)

fix, axes = plt.subplots(3,5,figsize=(15,12),subplot_kw={'xticks': (),'yticks' : ()})
for i , (component , ax) in enumerate(zip(pca.components_,axes.ravel())):
    ax.imshow(component.reshape(image_shape),cmap='viridis')
    ax.set_title("{}. component".format((i+1)))
```

!["pca"](img/pca6.png)





.