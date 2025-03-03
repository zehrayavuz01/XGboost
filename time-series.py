import kagglehub
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import os


path = kagglehub.dataset_download("sumanthvrao/daily-climate-time-series-data")

print("Path to dataset files:", path)



dataset_path = "/Users/zehra/.cache/kagglehub/datasets/sumanthvrao/daily-climate-time-series-data/versions/3"

# Specify the file you want to load
train_file = os.path.join(dataset_path, "DailyDelhiClimateTrain.csv")
test_file = os.path.join(dataset_path, "DailyDelhiClimateTest.csv")

# Load the datasets
train = pd.read_csv(train_file, parse_dates=["date"], index_col="date")
test = pd.read_csv(test_file, parse_dates=["date"], index_col="date")
print(test.columns)
print(train.columns)

def date_features(df):
    df["year"]= df.index.year
    df["month"]=df.index.month
    df["day"]=df.index.day

date_features(train)
date_features(test)
X_train=train[['humidity', 'wind_speed', 'meanpressure','year','month','day']]
y_train=train['meantemp']
X_test=test[['humidity', 'wind_speed', 'meanpressure',"year","month","day"]]
y_test=test['meantemp']


n_estimators=[50,100,150,200,250,300]
learning_rates=[0.01,0.03,0.05,0.07,0.09,0.1,0.12,0.14,0.16,0.18,0.2]

results=[]
for estimator in n_estimators:
    print("Estimator:", estimator)
    for rate in learning_rates:

        print("Rate:", rate)
        model = xgb.XGBClassifier(n_estimators=estimator, learning_rate=rate)
        print("model deined")
        model.fit(X_train, y_train)
        print("model trained")
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)


        results.append((estimator, rate, mse))
        print("The mean squared error (MSE) on test set: {:.4f}".format(mse))




df_results = pd.DataFrame(results, columns=["n_estimator","learning_rate","mse"])
df_results=df_results.pivot(index="n_estimator", columns="learning_rate", values="mse")
plt.figure(figsize=(10, 6))
sns.heatmap(df_results, annot=True, fmt=".4f", cmap="coolwarm", linewidths=0.5)
plt.title("MSE Heatmap for Different n_estimators and Learning Rates")
plt.xlabel("Learning Rate")
plt.ylabel("n_estimators")
plt.show()







