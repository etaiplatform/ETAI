#!/usr/bin/env python
# coding: utf-8

# In[1]:

from sklearn.metrics import mean_absolute_error

# In[3]:


# import Preprocessing
import pandas as pd
import numpy as np
import gc
import os
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

from source.Preprocessing import preprocess_final
from source.utils_cv import PurgedGroupTimeSeriesSplit


def smape(A, F):
    tmp = 2 * np.abs(F - A) / (np.abs(A) + np.abs(F))
    len_ = np.count_nonzero(~np.isnan(tmp))
    return 100 / len_ * np.nansum(tmp)


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.nanmean(np.abs((y_true - y_pred) / y_true)) * 100


# ### Gradually Decrease prediction size

# In[37]:


# In[34]:


params = {
    'objective': 'quantile',
    'random_state': 0,
    'learning_rate': 0.1,
    'alpha': 0.55,
    'verbose': 0,
    'force_col_wise': True
}


# In[ ]:


def predict_lastn_reg(model, days_pred, data, version):
    X = data.drop('dayAheadPrices', axis=1)
    y = data["dayAheadPrices"]
    del data
    gc.collect()
    groups = X["year"].astype('str') + "_" + X["month"].astype('str') + "_" + X["day"].astype('str')
    oof_preds = np.zeros((days_pred * 24))
    oof_test = np.zeros((days_pred * 24))
    kf = PurgedGroupTimeSeriesSplit(days_pred, group_gap=0, max_test_group_size=1, max_train_group_size=1).split(X, y,
                                                                                                                 groups=
                                                                                                                 groups.factorize()[
                                                                                                                     0])
    estimator = None
    for i, (train_index, test_index) in enumerate(kf):
        x_train_kf, x_test_kf = X.iloc[:test_index[0], :].copy(), X.iloc[test_index, :].copy()
        y_train_kf, y_test_kf = y[:test_index[0]], y[test_index]
        model.fit(x_train_kf.drop('dayAheadPrices', axis=1, errors='ignore'), y_train_kf)
        preds = model.predict(x_test_kf.drop('dayAheadPrices', axis=1, errors='ignore'))
        print("training period {} - {}".format(x_train_kf.iloc[0].name, x_train_kf.iloc[-1].name))
        print("predicting period {} - {}".format(x_test_kf.iloc[0].name, x_test_kf.iloc[-1].name))
        oof_preds[24 * i: 24 * (i + 1)] = (preds)
        oof_test[24 * i: 24 * (i + 1)] = (y_test_kf)
    return oof_preds


def predict_lastn_reg_dimdik(model, days_pred, data):
    X = data.drop('dayAheadPrices', axis=1)
    y = data["dayAheadPrices"]
    del data
    gc.collect()
    x_train_kf, x_test_kf = X[:-24 * days_pred].copy(), X[-24 * days_pred:].copy()
    print("training period {} - {}".format(x_train_kf.iloc[0].name, x_train_kf.iloc[-1].name))
    print("predicting period {} - {}".format(x_test_kf.iloc[0].name, x_test_kf.iloc[-1].name))
    y_train_kf, y_test_kf = y[:-24 * days_pred], y[-24 * days_pred:]
    model.fit(x_train_kf.drop('dayAheadPrices', axis=1, errors='ignore'), y_train_kf)
    preds = model.predict(x_test_kf.drop('dayAheadPrices', axis=1, errors='ignore'))
    # oof_preds = preds
    # oof_test = y_test_kf
    return preds


# In[1]:


def start(startDate='2016-01-01', endDate='2020-12-31', n_days=2, plot=False):
    model = lgb.LGBMRegressor(**params)
    if os.path.exists(startDate + "_" + endDate + ".csv"):
        data = pd.read_csv(startDate + "_" + endDate + ".csv")
        data = data.set_index('date')
    else:
        data = preprocess_final(startDate, endDate)
    data = data.drop(['target', 'isSpike'], axis=1, errors='ignore')
    oof_preds = predict_lastn_reg(model, int(n_days) + 1, data, 'none')
    test_res = data["dayAheadPrices"].iloc[-24 * int(n_days) + 1:].to_list()
    pred_res = oof_preds[-24 * int(n_days) + 1:]
    print("MAE of the last predicted batch for None: ", mean_absolute_error(test_res, pred_res))
    print("SMAPE of the last predicted batch for None: ", smape(test_res, pred_res))
    print("MAPE of the last predicted batch for None: ", mean_absolute_percentage_error(test_res, pred_res))
    print("RMSE of the last predicted batch for None: ", mean_squared_error(test_res, pred_res, squared=False))
    if plot:
        plt.plot(data["dayAheadPrices"].iloc[-24 * int(n_days):].to_list(), label="truth")
        plt.plot(list(oof_preds[-24 * int(n_days):]), label="predictions")
        plt.legend(loc="upper left")
        path = '../flaskApp/static/plot' + startDate + '_' + endDate + '_' + str(n_days) + '.png'
        plt.savefig(path)
        plt.clf()
        return pred_res, path, test_res
    else:
        return pred_res, test_res


def start_after(data, preds, startDate='2016-01-01', endDate='2020-12-31', n_days=2, plot=False):
    model = lgb.LGBMRegressor(**params)
    if data is None:
        data = preprocess_final(startDate, endDate)
    data = data.drop('target', axis=1, errors='ignore')
    data.iloc[-24 * int(n_days):, data.columns.get_loc("isSpike")] = preds.ravel()
    oof_preds = predict_lastn_reg(model, int(n_days) + 1, data, 'binary')
    test_res = data["dayAheadPrices"].iloc[-24 * int(n_days) + 1:].to_list()
    pred_res = oof_preds[-24 * int(n_days) + 1:]
    print("MAE of the last predicted batch for binary: ", mean_absolute_error(test_res, pred_res))
    print("SMAPE of the last predicted batch for binary: ", smape(test_res, pred_res))
    print("MAPE of the last predicted batch for binary: ", mean_absolute_percentage_error(test_res, pred_res))
    print("RMSE of the last predicted batch for binary: ", mean_squared_error(test_res, pred_res, squared=False))
    if plot:
        plt.plot(data["dayAheadPrices"].iloc[-24 * int(n_days):].to_list(), label="truth")
        plt.legend(loc="upper left")
        plt.plot(list(oof_preds[-24 * int(n_days) + 1:]), label="prediction")
        path = '../flaskApp/static/plot' + startDate + '_' + endDate + '_' + str(n_days) + '.png'
        plt.savefig(path)
        plt.clf()
        return pred_res, path, test_res
    else:
        return pred_res, test_res


def start_after_multi(data, preds, startDate='2016-01-01', endDate='2020-12-31', n_days=2, plot=False):
    model = lgb.LGBMRegressor(**params)
    if data is None:
        data = preprocess_final(startDate, endDate)
    data = data.drop('isSpike', axis=1, errors='ignore')
    data.iloc[-24 * int(n_days):, data.columns.get_loc("target")] = preds.ravel()
    oof_preds = predict_lastn_reg(model, int(n_days) + 1, data, 'multi')
    test_res = data["dayAheadPrices"].iloc[-24 * int(n_days):].to_list()
    pred_res = oof_preds[-24 * int(n_days):]
    print("MAE of the last predicted batch for multi: ", mean_absolute_error(test_res, pred_res))
    print("SMAPE of the last predicted batch for multi: ", smape(test_res, pred_res))
    print("MAPE of the last predicted batch for multi: ", mean_absolute_percentage_error(test_res, pred_res))
    print("RMSE of the last predicted batch for multi: ", mean_squared_error(test_res, pred_res, squared=False))
    if plot:
        plt.plot(data["dayAheadPrices"].iloc[-24 * int(n_days):].to_list(), label="truth")
        plt.plot(list(oof_preds[-24 * int(n_days):]), label="prediction")
        plt.legend(loc="upper left")
        path = '../flaskApp/static/plot' + startDate + '_' + endDate + '_' + str(n_days) + '.png'
        plt.savefig(path)
        plt.clf()
        return pred_res, path, test_res
    else:
        return pred_res, None, test_res


def start_after_dimdik_multi(data, preds, startDate='2016-01-01', endDate='2020-12-31', n_days=2, plot=False):
    model = lgb.LGBMRegressor(**params)
    if data is None:
        data = preprocess_final(startDate, endDate)
    data = data.drop('isSpike', axis=1, errors='ignore')
    data.iloc[-24 * int(n_days):, data.columns.get_loc("target")] = preds.ravel()
    oof_preds = predict_lastn_reg_dimdik(model, int(n_days), data)
    test_res = data["dayAheadPrices"].iloc[-24 * int(n_days):].to_list()
    pred_res = oof_preds[-24 * int(n_days):]
    print("MAE of the last predicted batch for multi: ", mean_absolute_error(test_res, pred_res))
    print("SMAPE of the last predicted batch for multi: ", smape(test_res, pred_res))
    print("MAPE of the last predicted batch for multi: ", mean_absolute_percentage_error(test_res, pred_res))
    print("RMSE of the last predicted batch for multi: ", mean_squared_error(test_res, pred_res, squared=False))
    if plot:
        plt.plot(data["dayAheadPrices"].iloc[-24 * int(n_days):].to_list(), label="truth")
        plt.plot(list(oof_preds[-24 * int(n_days):]), label="prediction")
        plt.legend(loc="upper left")
        path = '../flaskApp/static/plot' + startDate + '_' + endDate + '_' + str(n_days) + '.png'
        plt.savefig(path)
        plt.clf()
        return pred_res, path, test_res
    else:
        return pred_res, test_res
