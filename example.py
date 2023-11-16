import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from urllib.parse import urlparse
import mlflow.sklearn
import mlflow
from mlflow.models.signature import infer_signature
import logging
import warnings

logging.basicConfig(level=logging.WARN)
logging = logging.getLogger(__name__)

def eval_matrix(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(33)

    data = pd.read_csv("wine.csv")
    train, test = train_test_split(data)
    train_x = train.drop(['TARGET'], axis=1)
    test_x = test.drop(['TARGET'], axis=1)

    train_y = train['TARGET']
    test_y = test['TARGET']

    print(f"shape of X_train: {train_x.shape}")
    print(f"shape of X_test: {test_x.shape}")
    print(f"shape of y_train: {train_y.shape}")
    print(f"shape of y_test: {test_y.shape}")
    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2] if len(sys.argv) >2 else 0.5)

    with mlflow.start_run():
        lr = ElasticNet(alpha=alpha, l1_ratio= l1_ratio, random_state= 41)
        lr.fit(train_x,train_y)

        pred = lr.predict(test_x)
        (rmse, mae, r2) = eval_matrix(test_y, pred)

        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric('rmse', rmse)
        mlflow.log_metric('r2', r2)
        mlflow.log_metric('mar', mae)

        preditions = lr.predict(train_x)
        signature = infer_signature(train_x, preditions)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        # Model registry does not work with file store
        if tracking_url_type_store !="file":
            mlflow.sklearn.log_model(
                lr, "model",registered_model_name = "ElasticnetWineModel",signature=signature
            )
        else:
            mlflow.sklearn.log_model(lr, "model",signature=signature)


