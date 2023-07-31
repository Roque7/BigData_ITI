#!/usr/bin/env python
# coding: utf-8

# # MLFlow Hyper Param Experiment
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import ElasticNet
from sklearn.datasets import load_boston

import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Hyper_param_example")


data = load_boston()
df = pd.DataFrame(data.data, columns=data.feature_names)

xtest,xtrain,ytest,ytrain = train_test_split(df.drop(['CRIM'],axis=1), df['CRIM'])

en = ElasticNet()


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

def train(alpha,l1_ratio):
    with mlflow.start_run():
            lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
            lr.fit(xtrain, ytrain)

            predicted_qualities = lr.predict(xtest)

            (rmse, mae, r2) = eval_metrics(ytest, predicted_qualities)

            print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
            print("  RMSE: %s" % rmse)
            print("  MAE: %s" % mae)
            print("  R2: %s" % r2)

            mlflow.log_param("alpha", alpha)
            mlflow.log_param("l1_ratio", l1_ratio)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2", r2)
            mlflow.log_metric("mae", mae)

            mlflow.sklearn.log_model(lr, "model")


alpha = np.arange(0,5,0.1)
l1_ratio = 0.5

for a in alpha:
    train(a,l1_ratio)