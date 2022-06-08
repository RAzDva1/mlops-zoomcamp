from datetime import datetime
import pickle

import pandas as pd

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from prefect import flow, task, get_run_logger
from prefect.deployments import DeploymentSpec
from prefect.orion.schemas.schedules import CronSchedule
from prefect.flow_runners import SubprocessFlowRunner

@task
def read_data(path):
    df = pd.read_parquet(path)
    return df

@task
def prepare_features(df, categorical, train=True):
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    mean_duration = df.duration.mean()
    if train:
        print(f"The mean duration of training is {mean_duration}")
    else:
        print(f"The mean duration of validation is {mean_duration}")

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df

@task
def train_model(df, categorical):

    logger = get_run_logger()
    train_dicts = df[categorical].to_dict(orient='records')
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts) 
    y_train = df.duration.values

    logger.info("The shape of X_train is {X_train.shape}")
    logger.info(f"The DictVectorizer has {len(dv.feature_names_)} features")

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_train)
    mse = mean_squared_error(y_train, y_pred, squared=False)
    logger.info("The MSE of training is: {mse}")
    return lr, dv

@task
def run_model(df, categorical, dv, lr):
    logger = get_run_logger()
    val_dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(val_dicts)
    y_pred = lr.predict(X_val)
    y_val = df.duration.values

    mse = mean_squared_error(y_val, y_pred, squared=False)
    logger.info(f"The MSE of validation is: {mse}")
    return

@task
def get_paths(date=None):
    if not date:
        # datetime.now().strftime('%Y-%m')
        month, year = datetime.now().month, datetime.now().year - 1
    else:
        date_list = date.split("-")
        month, year = int(date_list[1]), int(date_list[0])

    train_path = f"../data/fhv_tripdata_{year}-0{month-2}.parquet"
    val_path = f"../data/fhv_tripdata_{year}-0{month-1}.parquet"
    return train_path, val_path

@flow(name="NYT")
def main(date="2021-08-15"):
    train_path, val_path = get_paths(date).result()

    categorical = ['PUlocationID', 'DOlocationID']

    df_train = read_data(train_path)
    df_train_processed = prepare_features(df_train, categorical)

    df_val = read_data(val_path)
    df_val_processed = prepare_features(df_val, categorical, False)

    # train the model
    lr, dv = train_model(df_train_processed, categorical).result()
    run_model(df_val_processed, categorical, dv, lr)
    with open(f'model-{date}.bin', 'wb') as file:
        pickle.dump(lr, file)
    with open(f'dv-{date}.bin', 'wb') as file:
        pickle.dump(dv, file)


DeploymentSpec(
    flow=main,
    name="NYT_training",
    schedule=CronSchedule(cron="*/2 * * * *", timezone="Europe/Moscow"),
    flow_runner=SubprocessFlowRunner(),
    tags=["ml"]
)
# q1 - train_model
# q2 -  11.63703425376567
# q3 13,000 bytes
# q4 - 0 9 15 * *
# q5 - 3
# q6 - work-queue ls
