#ライブラリのインポート
import  numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#データの読み込み
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,:-1] #説明変数を切り取る
y = dataset.iloc[:,-1] #目的変数を切り取る

#学習用データとテストデータに分ける
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state= 0)

#ここから単回帰分析
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#学習したもののをもとにX_testをインプットしたアウトプットを表示
y_pred = regressor.predict(X_test)

#ここからはグラフの図示(学習用)
plt.scatter(X_train, y_train, color='red') #学習用データの点をプロット
plt.plot(X_train, regressor.predict(X_train), color='blue')#直線を描画
plt.title('Salary vs Experience(training_set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#テスト用
plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color="blue")
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

