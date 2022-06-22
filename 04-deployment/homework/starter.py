#!/usr/bin/env python
# coding: utf-8


import pickle
import pandas as pd
import sys

def load_model():
    with open('model.bin', 'rb') as f_in:
        dv, lr = pickle.load(f_in)
    return (dv, lr)


categorical = ['PUlocationID', 'DOlocationID']

def read_data(filename):
    df = pd.read_parquet(filename)

    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    return df





def predict(df, dv, lr):
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)
    print("predict", y_pred.mean())
    return y_pred

def save_output(df, y_pred, month, year):

    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    df["predict"] = y_pred

    df_result = df[["predict", "ride_id"]]



    output_file = "fhv_tripdata_2021-02_with_ride_id_predict.parquet"

    df_result.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
    )

if __name__ == "__main__":
    year = int(sys.argv[1]) # 2021
    month = int(sys.argv[2]) # 3
    uri = f'https://nyc-tlc.s3.amazonaws.com/trip+data/fhv_tripdata_{year:04d}-{month:02d}.parquet'
    df = read_data(uri)
    dv, lr = load_model()
    predict_df = predict(df, dv, lr)
    # save_output(df, predict_df, month, year)
