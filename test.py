from API.Requests import predict_api
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from collections import defaultdict
import pandas as pd


def smape(A, F):
    A = np.array(A)
    F = np.array(F)
    tmp = 2 * np.abs(F - A) / (np.abs(A) + np.abs(F))
    len_ = np.count_nonzero(~np.isnan(tmp))
    return 100 / len_ * np.nansum(tmp)


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.nanmean(np.abs((y_true - y_pred) / y_true)) * 100


starting_date = '2016-01-01'
ending_date = '2021-06-15'
months = ['2020-07-15', '2020-08-15', '2020-09-15', '2020-10-15', '2020-11-15', '2020-12-15',
          '2021-01-15', '2021-02-15', '2021-03-15', '2021-04-15', '2021-05-15', '2021-06-15']
# months = ['2021-06-15']
n_days = 30
models = ['DEF', 'BIN1', 'NUL3', 'NUL1', 'DMDNUL1']
# models = ['DMDNUL1']
# results_df = pd.DataFrame()
data = defaultdict(list)
for end_date in months:
    for model in models:
        preds_df, truth, = predict_api(startDate='2016-01-01', endDate=end_date, days=n_days, model=model, plot=False)
        preds = preds_df["predictions"].tolist()
        mae = mean_absolute_error(truth, preds),
        mape = mean_absolute_percentage_error(truth, preds),
        rmse = mean_squared_error(truth, preds, squared=False),
        smaperes = smape(truth, preds)
        data["maes"].append(mae[0])
        data["mapes"].append(mape[0])
        data["rmses"].append(rmse[0])
        data["smaperes"].append(smaperes)
        data["days"].append(n_days)
        data["model"].append(model)
        data["end_date"].append(end_date)
results_df = pd.DataFrame(data)
results_df.to_csv('dmdnulnew.csv', index=False)
