from ETAI.API.Requests import predict_api
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from collections import defaultdict
import pandas as pd
import time


def smape(A, F):
    A = np.array(A)
    F = np.array(F)
    tmp = 2 * np.abs(F - A) / (np.abs(A) + np.abs(F))
    len_ = np.count_nonzero(~np.isnan(tmp))
    return 100 / len_ * np.nansum(tmp)


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.nanmean(np.abs((y_true - y_pred) / y_true)) * 100


print("RUNNING TESTS")
starting_date = '2016-01-01'
ending_date = '2021-08-23'
months = ['2020-09-23', '2020-10-23', '2020-11-23', '2020-12-23', '2021-01-23', '2021-02-23',
          '2021-03-23', '2021-04-23', '2021-05-23', '2021-06-23', '2021-07-23', '2021-08-23']
days = [31, 30, 31, 30, 31, 30, 31, 28, 31, 30, 31, 30, 31]
# months = ['2021-06-15']
models = ['DEF', 'BIN1', 'NUL3', 'NUL1', 'DMDNUL1']
# models = ['DMDNUL1']
# results_df = pd.DataFrame()
data = defaultdict(list)
full = time.time()
for target in ["price", "consumption", "production"]:
    start = time.time()
    for end_date in months:
        i = 0
        print("TRAINING FROM: ", starting_date)
        print("TRAINING UNTIL: ", end_date)
        for model in models:
            preds_df, truth, = predict_api(startDate='2016-01-01', endDate=end_date, days=days[i], model=model,
                                           plot=False,
                                           target=target)
            preds = preds_df["predictions"].tolist()
            mae = mean_absolute_error(truth, preds),
            mape = mean_absolute_percentage_error(truth, preds),
            rmse = mean_squared_error(truth, preds, squared=False),
            smaperes = smape(truth, preds)
            data["maes"].append(mae[0])
            data["mapes"].append(mape[0])
            data["rmses"].append(rmse[0])
            data["smaperes"].append(smaperes)
            data["days"].append(days[i])
            data["model"].append(model)
            data["end_date"].append(end_date)
        i += 1
    end = time.time()
    print("elapsed: ", end - start)
    results_df = pd.DataFrame(data)
    results_df.to_csv(target + '_new_results1.csv', index=False)
fend = time.time()
print("all 3 elapsed: ", fend - full)
