# titanicにチャンレジする

titanicチャレンジ

https://www.kaggle.com/c/titanic/


## チャレンジ1：ランダムフォレスト

### 前処理

前処理が全てなのでやっていく

```
# 乗客の名前の長さ
train['Name_length'] = train['Name'].apply(len)
test['Name_length'] = test['Name'].apply(len)

# 客室番号データがあるなら１を、欠損値なら0を
train['Has_Cabin'] = train["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
test['Has_Cabin'] = test["Cabin"].apply(lambda x: 0 if type(x) == float else 1)

# 家族の大きさを"タイタニックに同乗している兄弟/配偶者の数"と
# "タイタニックに同乗している親/子供の数"から定義
for dataset in full_data:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

```
* 文字列は数値化する
* データの欠損値は補完する
* 似たようなデータを集約する

```
# 家族がいるかどうか
# いるなら"IsAlone"が１
for dataset in full_data:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

# 出港地の欠損値を一番多い"S"としておく
for dataset in full_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')

# 料金の欠損値を中央値としておく
# 料金の大きく４つのグループに分ける
for dataset in full_data:
    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())
train['CategoricalFare'] = pd.qcut(train['Fare'], 4)
```








## ハマったとこメモ
### いきなりcsvインポートでハマる

```
import pandas as pd

train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
combine = [train_df, test_df]

>> FileNotFoundError: [Errno 2] File b'../input/train.csv' does not exist: b'../input/train.csv'

```

どうもディレクトリを掘るとそこを見つけられないっぽい。
import os とかしてos.pathgetcwd()すると、全然違うディレクトリで実行されているっぽい。

```
import os
os.getcwd()

>> '/Users/shinyatanaka/Desktop/unity_study/chap6/New Unity Project (1)'

```

なんやそれ。実行中のファイルをとる `__file__` はnotebookだとつかえないくさい。ダル。

```
# data analysis and wrangling
import pandas as pd
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
combine = [train_df, test_df]

>> FileNotFoundError: [Errno 2] File b'train.csv' does not exist: b'train.csv'

```
ファイル移動して実行ファイルと同じにしてもあかんかった。なんやそれ

と思ったら原因わかった。VSCodeでワークスペースに複数登録していると、一番先頭の箇所にカレントディレクトリが登録されるっぽい。

```
import pandas as pd
import os

current_dir = os.getcwd() + '/capture_2.5(titanic challenge)/'
train_df = pd.read_csv(current_dir + 'input/train.csv')
test_df = pd.read_csv(current_dir + 'input/test.csv')
combine = [train_df, test_df]

```

こんな感じで解決した。
os.getctw()でも実行ファイルまでのパスにならないのすごいうざい。



### 配列は参照渡しやで
配列は参照渡しのもよう

```
foo = {}
bar = {}
foo["A"] = 1
bar["A"] = 2
hoge = [foo,bar]
foo["A"] = 10
hoge

>> [{'A': 10}, {'A': 2}]
```

### グリッドサーチしてみた

```
# グリッドサーチしてみる
params = {
    "n_estimators":[i for i in range(10,100,10)],
    "criterion":["gini","entropy"],
    "max_depth":[i for i in range(1,6,1)],
    'min_samples_split': [2, 4, 10,12,16],
    "random_state":[3],
}

clf = GridSearchCV(RandomForestClassifier(), params,cv=5,n_jobs=-1)
clf_fit=clf.fit(X_train, y_train)
```

これで78までいった

