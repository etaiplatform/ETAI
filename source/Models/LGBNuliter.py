#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import gc
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score

from ETAI.source.Preprocessing import preprocess_final
from ETAI.source.utils_cv import PurgedGroupTimeSeriesSplit

params = {
    'boosting_type': 'gbdt',
    #     "objective"                  : "binary",
    "learning_rate": 0.08,
    "num_leaves": 4,
    "subsample": 0.8,
    'subsample_for_bin': 1000000,
    'max_depth': 3,
    'reg_lambda': 2,
    'verbose': 0,
    "force_row_wise": True,
    #     "is_unbalance"               : True
}


def predict_lastn_clf(model, days_pred, data):
    X = data.drop(['target'], axis=1)
    y = data['target']
    del data
    gc.collect()
    groups = X["year"].astype('str') + "_" + X["month"].astype('str') + "_" + X["day"].astype('str')
    oof_preds = np.zeros((days_pred * 24))
    oof_test = np.zeros((days_pred * 24))
    kf = PurgedGroupTimeSeriesSplit(days_pred, group_gap=0, max_test_group_size=1, max_train_group_size=1) \
        .split(X, y, groups=groups.factorize()[0])
    for i, (train_index, test_index) in enumerate(kf):
        x_train_kf, x_test_kf = X.iloc[:test_index[0], :].copy(), X.iloc[test_index, :].copy()
        y_train_kf, y_test_kf = y[:test_index[0]], y[test_index]
        model.fit(x_train_kf.drop(['target'], axis=1, errors='ignore'), y_train_kf)
        preds = model.predict(x_test_kf.drop(['target'], axis=1, errors='ignore'))
        oof_preds[24 * i: 24 * (i + 1)] = (preds)
        oof_test[24 * i: 24 * (i + 1)] = (y_test_kf)
    print("F1 score of the last predicted batch: ", f1_score(oof_test, oof_preds, average='macro'))
    print("Accuracy of the last predicted batch: ", accuracy_score(oof_test, oof_preds))
    return oof_preds, oof_test


def start_multi_clf(startDate='2016-01-01', endDate='2020-12-31', n_days=2):
    model = lgb.LGBMClassifier(**params)
    data = preprocess_final(startDate, endDate)
    data = data.drop(['isSpike'], axis=1, errors='ignore')
    oof_preds, oof_test = predict_lastn_clf(model, int(n_days) + 1, data)
    return oof_preds[-24 * int(n_days):]


def predict_lastn_reg(model, days_pred, data, startDate='2016-01-01', endDate='2020-12-31',
                      target="dayAheadPrices"):
    X = data.drop(target, axis=1)
    y = data[target]
    del data
    gc.collect()
    groups = X["year"].astype('str') + "_" + X["month"].astype('str') + "_" + X["day"].astype('str')
    oof_preds = []
    indexes = []
    kf = PurgedGroupTimeSeriesSplit(days_pred, group_gap=0, max_test_group_size=1, max_train_group_size=1) \
        .split(X, y, groups=groups.factorize()[0])
    x_train_kf, x_test_kf = None, None
    y_train_kf, y_test_kf = None, None
    for i, (train_index, test_index) in enumerate(kf):
        if not y[:test_index[0]].isnull().values.any():
            y_train_kf = y[:test_index[0]]
            x_train_kf = X.iloc[:test_index[0], :].copy()
            # print("fitting until {}".format(x_train_kf.iloc[-1].name))
        else:
            print("not fitting {}".format(x_train_kf.iloc[-1].name))
        y_test_kf = y[test_index]
        x_test_kf = X.iloc[test_index, :].copy()
        tocheck = list(set(pd.to_datetime(x_test_kf.reset_index()["date"]).dt.date.astype('str')))
        dates = [str(pd.to_datetime(endDate).date() - pd.Timedelta(days=i)) for i in range(days_pred)]
        if len([i for i in tocheck if i in dates]) == 0:
            continue
        model.fit(x_train_kf.drop(target, axis=1, errors='ignore'), y_train_kf)
        preds = model.predict(x_test_kf.drop(target, axis=1, errors='ignore'))
        oof_preds.extend(preds)
        indexes.extend(x_test_kf.index)
    return oof_preds, indexes


# In[17]:


def start_after_normal(data, preds, startDate='2016-01-01', endDate='2020-12-31', n_days=2, target='dayAheadPrices'):
    model = lgb.LGBMRegressor(**params)
    # if os.path.exists(startDate + "_" + endDate + ".csv"):
    #     data = pd.read_csv(startDate + "_" + endDate + ".csv")
    #     data = data.set_index('date')
    # else:
    # data = preprocess_final(startDate, endDate, target)
    data = data.drop(['isSpike'], axis=1, errors='ignore')
    data.iloc[-len(preds.ravel()):, data.columns.get_loc("target")] = preds.ravel()
    data = data[data["target"] == 0]
    oof_preds, norm_index = predict_lastn_reg(model, n_days, data, startDate, endDate, target)
    return oof_preds[-len(preds.ravel()):], norm_index[-len(preds.ravel()):],


# In[18]:


def start_after_lower(data, preds, startDate='2016-01-01', endDate='2020-12-31', n_days=2, target='dayAheadPrices'):
    model = lgb.LGBMRegressor(**params)
    # if os.path.exists(startDate + "_" + endDate + ".csv"):
    #     data = pd.read_csv(startDate + "_" + endDate + ".csv")
    #     data = data.set_index('date')
    # else:
    #     data = preprocess_final(startDate, endDate)
    data = data.drop(['isSpike'], axis=1, errors='ignore')
    data.iloc[-len(preds.ravel()):, data.columns.get_loc("target")] = preds.ravel()
    data = data[data["target"] == -1]
    oof_preds, lower_index = predict_lastn_reg(model, n_days, data, startDate, endDate, target)
    return oof_preds[-len(preds.ravel()):], lower_index[-len(preds.ravel()):]


# In[19]:


def start_after_upper(data, preds, startDate='2016-01-01', endDate='2020-12-31', n_days=2, target='dayAheadPrices'):
    model = lgb.LGBMRegressor(**params)
    # if os.path.exists(startDate + "_" + endDate + ".csv"):
    #     data = pd.read_csv(startDate + "_" + endDate + ".csv")
    #     data = data.set_index('date')
    # else:
    #     data = preprocess_final(startDate, endDate)
    data = data.drop('isSpike', axis=1, errors='ignore')
    data.iloc[-len(preds.ravel()):, data.columns.get_loc("target")] = preds.ravel()
    data = data[data["target"] == 1]
    oof_preds, upper_index = predict_lastn_reg(model, n_days, data, startDate, endDate, target)
    return oof_preds[-len(preds.ravel()):], upper_index[-len(preds.ravel()):]


# In[28]:


def start_after_multi_models(data, clf_preds, startDate='2016-01-01', endDate='2021-03-11', n_days=2, plot=False,
                             target='dayAheadPrices'):
    if data is None:
        data = preprocess_final(startDate, endDate, target)
    norm_preds, norm_indexes = start_after_normal(data, clf_preds, startDate=startDate, endDate=endDate,
                                                  n_days=int(n_days) + 1, target=target)
    upper_preds, upper_indexes = start_after_upper(data, clf_preds, startDate=startDate, endDate=endDate,
                                                   n_days=int(n_days) + 1, target=target)
    lower_preds, lower_indexes = start_after_lower(data, clf_preds, startDate=startDate, endDate=endDate,
                                                   n_days=int(n_days) + 1, target=target)
    preds = pd.DataFrame()
    norm_indexes.extend(upper_indexes)
    norm_indexes.extend(lower_indexes)
    norm_preds.extend(upper_preds)
    norm_preds.extend(lower_preds)
    preds["date"] = norm_indexes
    preds["preds"] = norm_preds
    preds = preds.sort_values(by='date')
    # data = pd.read_csv(startDate + "_" + endDate + ".csv")
    if plot:
        plt.plot(data[target].iloc[-24 * int(n_days):].to_list(), label='truth')
        plt.plot(preds["preds"].to_list()[-24 * int(n_days):], label='predictions')
        plt.legend(loc="upper left")
        path = 'ETAI/flaskApp/static/plot' + startDate + '_' + endDate + '_' + str(n_days) + '.png'
        plt.savefig(path)
        plt.clf()
        return preds[-24 * int(n_days):], path, data[target].iloc[-24 * int(n_days):].to_list()
    else:
        return preds[-24 * int(n_days):], None, data[target].iloc[-24 * int(n_days):].to_list()
