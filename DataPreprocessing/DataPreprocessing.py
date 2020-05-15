#Importing the libraries
import numpy as np #数値計算ライブラリ
import matplotlib.pyplot as plt #numpy用のグラフ描画ライブラリ
import pandas as pd #データ解析ライブラリ

#Importing the dataset
dataset = pd.read_csv('Data.csv')#csvファイルの読み込み
X = dataset.iloc[:,:-1].values#indexで指定
y = dataset.iloc[:,-1].values
print(X)
print(y)
print("######################")

#Taking care of missing data
from sklearn.impute import SimpleImputer #機械学習用ライブラリ, impute...欠損値を置き換えるため
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')#SimpleImputer(missing_values="欠損値の値は？", strategy="どう置き換えるか")
imputer.fit(X[:,1:3])#変換式を計算、この場合このカラムの平均値
X[:,1:3] = imputer.transform(X[:,1:3])#fitに基づいてデータを変換

#Encoding categorical data (カテゴリー変数を処理)
#1.Encoding the Independent Variable(説明変数を処理)....0,1問題ではないからダミー変数に変換
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder#指定した配列を(0,1)の2値で構成される行列に変換
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')#transformers=[('名前','推定器','カラム')], remainder=指定したカラム以外はどうする？
X = np.array(ct.fit_transform(X))#Xにnumpyの配列を打ち込む

#2.Encoding the Dependent Variable(目的変数を処理) 0,1問題はこっち
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)#fit and transform

#Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split#学習用データとテスト用データに分ける
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1) #(データ、データ、test_size,random_state) random_state...1だとデータそのまんま
print(X_train)
print(X_test)
print(y_train)
print(y_test)
print("###################")

#Feature Scaling
from sklearn.preprocessing import StandardScaler#標準化を行う。
sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc.transform(X_test[:, 3:])
print(X_train)
print(X_test)