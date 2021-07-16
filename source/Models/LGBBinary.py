#!/usr/bin/env python
# coding: utf-8

# In[26]:


import numpy as np
import gc
import lightgbm as lgb
from sklearn.metrics import f1_score
from source.Preprocessing import preprocess_final
from source.utils_cv import PurgedGroupTimeSeriesSplit

# In[28]:


params = {
    'boosting_type': 'gbdt',
    "objective": "binary",
    "learning_rate": 0.08,
    "num_leaves": 4,
    "subsample": 0.8,
    'subsample_for_bin': 1000000,
    'max_depth': 3,
    'reg_lambda': 2,
    'verbose': 0,
    "force_row_wise": True,
    "is_unbalance": True
}


# In[29]:


def predict_lastn_clf(model, days_pred, data):
    X = data.drop('isSpike', axis=1)
    y = data["isSpike"]
    del data
    gc.collect()
    groups = X["year"].astype('str') + "_" + X["month"].astype('str') + "_" + X["day"].astype('str')
    oof_preds = np.zeros((days_pred * 24))
    oof_test = np.zeros((days_pred * 24))
    kf = PurgedGroupTimeSeriesSplit(days_pred, group_gap=0, max_test_group_size=1, max_train_group_size=1) \
        .split(X, y, groups=groups.factorize()[0])
    estimator = None
    for i, (train_index, test_index) in enumerate(kf):
        x_train_kf, x_test_kf = X.iloc[:test_index[0], :].copy(), X.iloc[test_index, :].copy()
        y_train_kf, y_test_kf = y[:test_index[0]], y[test_index]
        model.fit(x_train_kf.drop('isSpike', axis=1, errors='ignore'), y_train_kf)
        #         model = lgb.train(params, lgb.Dataset(x_train_kf.drop('isSpike',axis=1, errors='ignore'), y_train_kf))
        preds = model.predict(x_test_kf.drop('isSpike', axis=1, errors='ignore'))
        #         preds = model.predict(x_test_kf.drop('isSpike',axis=1, errors='ignore'))
        oof_preds[24 * i: 24 * (i + 1)] = (preds)
        oof_test[24 * i: 24 * (i + 1)] = (y_test_kf)
    print("F1 score of the last predicted batch: ", f1_score(oof_test, oof_preds))
    return oof_preds, oof_test


# In[31]:


def start_clf(startDate='2016-01-01', endDate='2020-12-31', n_days=2, target='dayAheadPrices'):
    model = lgb.LGBMClassifier(**params)
    data = preprocess_final(startDate, endDate, target)
    data = data.drop('target', axis=1, errors='ignore')
    oof_preds, oof_test = predict_lastn_clf(model, int(n_days) + 1, data)
    #     print(oof_preds[-24*int(n_days):])
    #     print(oof_test[-24*int(n_days):])
    #     for xc in list(range(0,24)):
    #         plt.axvline(x=xc, label='line at x = {}'.format(xc))
    #     plt.step(list(range(0,len(data["isSpike"].iloc[-24*int(n_days):].to_list()))), list(oof_preds[-24*int(n_days):]))
    #     plt.step(list(range(0,len(data["isSpike"].iloc[-24*int(n_days):].to_list()))), data["isSpike"].iloc[-24*int(n_days):].to_list())
    #     path = 'static/plot'+startDate+'_'+endDate+'_'+n_days+'.png'
    #     plt.savefig(path)
    #     plt.clf()
    return oof_preds[-24 * int(n_days):], data

# In[ ]:
