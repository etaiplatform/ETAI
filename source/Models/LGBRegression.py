#!/usr/bin/env python
# coding: utf-8

# In[1]:

from sklearn.metrics import mean_absolute_error

import pandas as pd
import numpy as np
import gc
import os
import datetime
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

from ETAI.source.Preprocessing import preprocess_final
from ETAI.source.utils_cv import PurgedGroupTimeSeriesSplit


def smape(A, F):
    tmp = 2 * np.abs(F - A) / (np.abs(A) + np.abs(F))
    len_ = np.count_nonzero(~np.isnan(tmp))
    return 100 / len_ * np.nansum(tmp)


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.nanmean(np.abs((y_true - y_pred) / y_true)) * 100


params = {
    'objective': 'quantile',
    'random_state': 0,
    'learning_rate': 0.1,
    'alpha': 0.55,
    'verbose': 0,
    'force_col_wise': True
}

def predict_lastn_reg(model, days_pred, data, target):
    X = data.drop(target, axis=1)
    y = data[target]
    del data
    gc.collect()
    groups = X["year"].astype('str') + "_" + X["month"].astype('str') + "_" + X["day"].astype('str')
    oof_preds = np.zeros((days_pred * 24))
    oof_test = np.zeros((days_pred * 24))
    kf = PurgedGroupTimeSeriesSplit(days_pred, group_gap=0, max_test_group_size=1, max_train_group_size=1) \
        .split(X, y, groups=groups.factorize()[0])
    estimator = None
    x_train_kf, x_test_kf = None, None
    y_train_kf, y_test_kf = None, None
    for i, (train_index, test_index) in enumerate(kf):
        if not y[:test_index[0]].isnull().values.any():
            y_train_kf = y[:test_index[0]]
            x_train_kf = X.iloc[:test_index[0], :].copy()
            print("not fitting on {}".format(x_train_kf.iloc[-1].name))
        y_test_kf = y[test_index]
        x_test_kf = X.iloc[test_index, :].copy()
        model.fit(x_train_kf.drop(target, axis=1, errors='ignore'), y_train_kf)
        preds = model.predict(x_test_kf.drop(target, axis=1, errors='ignore'))
        oof_preds[24 * i: 24 * (i + 1)] = (preds)
        oof_test[24 * i: 24 * (i + 1)] = (y_test_kf)
        if sum(y_train_kf.isna()) >= 24:
            y_train_kf.values[y_train_kf.reset_index(drop=True).isnull().index[-24:].to_list()] \
                = oof_preds[24 * i: 24 * (i + 1)]
        print("training period {} - {}".format(x_train_kf.iloc[0].name, x_train_kf.iloc[-1].name))
        print("predicting period {} - {}".format(x_test_kf.iloc[0].name, x_test_kf.iloc[-1].name))

    return oof_preds


def predict_lastn_reg_dimdik(model, days_pred, data, target):
    X = data.drop(target, axis=1)
    y = data[target]
    del data
    gc.collect()
    diff = 1
    if (pd.to_datetime(X.copy().iloc[-1].name) - datetime.timedelta(
            days=days_pred)).date() > datetime.datetime.today().date():
        if datetime.datetime.today().hour > 13:
            diff = (pd.to_datetime(X.copy().iloc[-1].name) - (
                    datetime.datetime.today() + datetime.timedelta(days=1))).days
        else:
            diff = (pd.to_datetime(X.copy().iloc[-1].name - (datetime.datetime.today()))).days
    elif (pd.to_datetime(X.copy().iloc[-1].name) - datetime.timedelta(
            days=days_pred)).date() == datetime.datetime.today().date():
        diff = (pd.to_datetime(X.copy().iloc[-1].name) - (datetime.datetime.today())).days
    elif (pd.to_datetime(X.copy().iloc[-1].name) - datetime.timedelta(
            days=days_pred)).date() < datetime.datetime.today().date():
        diff = days_pred
    x_train_kf = X[:-24 * diff].copy()
    x_test_kf = X[-24 * days_pred:].copy()
    y_train_kf = y[:-24 * diff]
    y_test_kf = y[-24 * days_pred:]
    print("training period {} - {}".format(x_train_kf.iloc[0].name, x_train_kf.iloc[-1].name))
    print("predicting period {} - {}".format(x_test_kf.iloc[0].name, x_test_kf.iloc[-1].name))
    model.fit(x_train_kf.drop(target, axis=1, errors='ignore'), y_train_kf)
    preds = model.predict(x_test_kf.drop(target, axis=1, errors='ignore'))
    return preds


# In[1]:


def start(startDate='2016-01-01', endDate='2020-12-31', n_days=2, plot=False, target='dayAheadPrices'):
    model = lgb.LGBMRegressor(**params)
    if os.path.exists(startDate + "_" + endDate + ".csv"):
        data = pd.read_csv(startDate + "_" + endDate + ".csv")
        data = data.set_index('date')
    else:
        data = preprocess_final(startDate, endDate, target)
    data = data.drop(['target', 'isSpike'], axis=1, errors='ignore')
    oof_preds = predict_lastn_reg(model, int(n_days) + 1, data, target)
    test_res = data[target].iloc[-24 * int(n_days) + 1:].to_list()
    pred_res = oof_preds[-24 * int(n_days) + 1:]
    if np.isnan(test_res).all():
        test_res = [-1 for _ in range(len(test_res))]
    print("MAE of the last predicted batch for None: ", mean_absolute_error(test_res, pred_res))
    print("SMAPE of the last predicted batch for None: ", smape(test_res, pred_res))
    print("MAPE of the last predicted batch for None: ", mean_absolute_percentage_error(test_res, pred_res))
    print("RMSE of the last predicted batch for None: ", mean_squared_error(test_res, pred_res, squared=False))
    if plot:
        plt.plot(data[target].iloc[-24 * int(n_days):].to_list(), label="truth")
        plt.plot(list(oof_preds[-24 * int(n_days):]), label="predictions")
        plt.legend(loc="upper left")
        path = '../flaskApp/static/plot' + startDate + '_' + endDate + '_' + str(n_days) + '.png'
        plt.savefig(path)
        plt.clf()
        return pred_res, path, test_res
    else:
        return pred_res, None, test_res


def start_after(data, preds, startDate='2016-01-01', endDate='2020-12-31', n_days=2, plot=False,
                target='dayAheadPrices'):
    model = lgb.LGBMRegressor(**params)
    if data is None:
        data = preprocess_final(startDate, endDate, target)
    data = data.drop('target', axis=1, errors='ignore')
    data.iloc[-24 * int(n_days):, data.columns.get_loc("isSpike")] = preds.ravel()
    oof_preds = predict_lastn_reg(model, int(n_days) + 1, data, target)
    test_res = data[target].iloc[-24 * int(n_days) + 1:].to_list()
    pred_res = oof_preds[-24 * int(n_days) + 1:]
    if np.isnan(test_res).all():
        test_res = [-1 for _ in range(len(test_res))]
    print("MAE of the last predicted batch for binary: ", mean_absolute_error(test_res, pred_res))
    print("SMAPE of the last predicted batch for binary: ", smape(test_res, pred_res))
    print("MAPE of the last predicted batch for binary: ", mean_absolute_percentage_error(test_res, pred_res))
    print("RMSE of the last predicted batch for binary: ", mean_squared_error(test_res, pred_res, squared=False))
    if plot:
        plt.plot(data[target].iloc[-24 * int(n_days):].to_list(), label="truth")
        plt.legend(loc="upper left")
        plt.plot(list(oof_preds[-24 * int(n_days) + 1:]), label="prediction")
        path = '../flaskApp/static/plot' + startDate + '_' + endDate + '_' + str(n_days) + '.png'
        plt.savefig(path)
        plt.clf()
        return pred_res, path, test_res
    else:
        return pred_res, None, test_res


def start_after_multi(data, preds, startDate='2016-01-01', endDate='2020-12-31', n_days=2, plot=False,
                      target='dayAheadPrices'):
    model = lgb.LGBMRegressor(**params)
    if data is None:
        data = preprocess_final(startDate, endDate, target=target)
    data = data.drop('isSpike', axis=1, errors='ignore')
    data.iloc[-24 * int(n_days):, data.columns.get_loc("target")] = preds.ravel()
    oof_preds = predict_lastn_reg(model, int(n_days) + 1, data, target)
    test_res = data[target].iloc[-24 * int(n_days):].to_list()
    pred_res = oof_preds[-24 * int(n_days):]
    test_res = [-1 if np.isnan(i) else i for i in test_res]
    print("MAE of the last predicted batch for multi: ", mean_absolute_error(test_res, pred_res))
    print("SMAPE of the last predicted batch for multi: ", smape(test_res, pred_res))
    print("MAPE of the last predicted batch for multi: ", mean_absolute_percentage_error(test_res, pred_res))
    print("RMSE of the last predicted batch for multi: ", mean_squared_error(test_res, pred_res, squared=False))
    if plot:
        plt.plot(data[target].iloc[-24 * int(n_days):].to_list(), label="truth")
        plt.plot(list(oof_preds[-24 * int(n_days):]), label="prediction")
        plt.legend(loc="upper left")
        path = '../flaskApp/static/plot' + startDate + '_' + endDate + '_' + str(n_days) + '.png'
        plt.savefig(path)
        plt.clf()
        return pred_res, path, test_res
    else:
        return pred_res, None, test_res


def start_after_dimdik_multi(data, preds, startDate='2016-01-01', endDate='2020-12-31', n_days=2, plot=False,
                             target='dayAheadPrices'):
    model = lgb.LGBMRegressor(**params)
    if data is None:
        data = preprocess_final(startDate, endDate, target)
    data = data.drop('isSpike', axis=1, errors='ignore')
    data.iloc[-24 * int(n_days):, data.columns.get_loc("target")] = preds.ravel()
    oof_preds = predict_lastn_reg_dimdik(model, int(n_days), data, target)
    test_res = data[target].iloc[-24 * int(n_days):].to_list()
    pred_res = oof_preds[-24 * int(n_days):]
    test_res = [-1 if np.isnan(i) else i for i in test_res]
    print("MAE of the last predicted batch for multi: ", mean_absolute_error(test_res, pred_res))
    print("SMAPE of the last predicted batch for multi: ", smape(test_res, pred_res))
    print("MAPE of the last predicted batch for multi: ", mean_absolute_percentage_error(test_res, pred_res))
    print("RMSE of the last predicted batch for multi: ", mean_squared_error(test_res, pred_res, squared=False))
    if plot:
        plt.plot(data[target].iloc[-24 * int(n_days):].to_list(), label="truth")
        plt.plot(list(oof_preds[-24 * int(n_days):]), label="prediction")
        plt.legend(loc="upper left")
        path = '../flaskApp/static/plot' + startDate + '_' + endDate + '_' + str(n_days) + '.png'
        plt.savefig(path)
        plt.clf()
        return pred_res, path, test_res
    else:
        return pred_res, None, test_res
