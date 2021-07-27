#!/usr/bin/env python
# coding: utf-8

import os
import datetime
import pandas as pd
import numpy as np
from transparency_epias.production import productionClient  # üretim
from transparency_epias.consumption import consumptionClient  # tüketim
from transparency_epias.markets import dayaheadClient  # gün öncesi fiyatlar


def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object and str(col) != "date":
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df


def create_features(data, target):
    data["t-1_lag_price"] = data[target].shift(24 * 1)
    data["t-2_lag_price"] = data[target].shift(24 * 2)
    data["t-3_lag_price"] = data[target].shift(24 * 3)
    data["t-4_lag_price"] = data[target].shift(24 * 4)
    data["t-5_lag_price"] = data[target].shift(24 * 5)
    data["t-6_lag_price"] = data[target].shift(24 * 6)
    data["t-7_lag_price"] = data[target].shift(24 * 7)
    #     data["t-14_lag_price"] = data[target"].shift(24*14)
    #     data["t-21_lag_price"] = data[target"].shift(24*21)
    #     data["t-28_lag_price"] = data[target"].shift(24*28)
    data["t-30_lag_price"] = data[target].shift(24 * 30)
    data["t-365_lag_price"] = data[target].shift(24 * 365)

    # prods = ['production', 'consumption']  # 'naturalGas', 'wind',
    # for prod in prods:
    #     data["t-1_lag_" + prod] = data[prod].shift(24 * 1)
    #     # #         data["t-2_lag_"+prod] = data[prod].shift(24*2)
    #     # #         data["t-3_lag_"+prod] = data[prod].shift(24*3)
    #     # #         data["t-4_lag_"+prod] = data[prod].shift(24*4)
    #     # #         data["t-5_lag_"+prod] = data[prod].shift(24*5)
    #     # #         data["t-6_lag_"+prod] = data[prod].shift(24*6)
    #     data["t-7_lag_" + prod] = data[prod].shift(24 * 7)
    #     # #         data["t-14_lag_"+prod] = data[prod].shift(24*14)
    #     # #         data["t-21_lag_"+prod] = data[prod].shift(24*21)
    #     # #         data["t-28_lag_"+prod] = data[prod].shift(24*28)
    #     # #         data["t-365_lag_"+prod] = data[prod].shift(24*365)
    #     #         data['rolling_mean_weekly_'+prod] = data[prod].rolling(7*24).mean().shift(24)
    #     # #         data['rolling_mean_2weekly_'+prod] = data[prod].rolling(14*24).mean().shift(24)
    #     # #         data['rolling_mean_monthly_'+prod] = data[prod].rolling(30*24).mean().shift(24)
    #     # # #         data['rolling_mean_year_'+prod] = data[prod].rolling(365*24).mean().shift(24)
    #     data.drop(prod, axis=1, inplace=True)

    data['rolling_mean_weekly_price'] = data[target].rolling(7 * 24).mean().shift(24)
    data['rolling_std_weekly_price'] = data[target].rolling(7 * 24).std().shift(24)
    #     data['rolling_mean_2weekly_price'] = data['dayAheadPrices'].rolling(14*24).mean().shift(24)
    #     data['rolling_std_2weekly_price'] = data['dayAheadPrices'].rolling(14*24).std().shift(24)
    #     data['rolling_mean_monthly_price'] = data['dayAheadPrices'].rolling(30*24).mean().shift(24)
    #     data['rolling_std_monthly_price'] = data['dayAheadPrices'].rolling(30*24).std().shift(24)
    #     data['rolling_mean_yearly_price'] = data['dayAheadPrices'].rolling(365*24).mean().shift(24)
    #     data['rolling_std_yearly_price'] = data['dayAheadPrices'].rolling(365*24).std().shift(24)
    return data


# (df["dayAheadPrices"] < mean-(n-.2)*std) | 
def seperate_spikes(df, n, target):
    spike_df = pd.DataFrame()
    spikelen = 1
    while spikelen != 0:
        mean = df[target].mean()
        std = df[target].std()
        spikes = df[(df[target] < mean - (n - .2) * std) | (df[target] > mean + (n - .2) * std)]
        spike_df = pd.concat([spikes, spike_df])
        spikelen = len(spikes)
        df = df.drop(spikes.index)
    return spike_df


def seperate_upper_spikes(df, n, target):
    upper_spike_df = pd.DataFrame()
    spikelen = 1
    while spikelen != 0:
        mean = df[target].mean()
        std = df[target].std()
        upper_spikes = df[(df[target] > mean + (n - .2) * std)]
        upper_spike_df = pd.concat([upper_spikes, upper_spike_df])
        spikelen = len(upper_spikes)
        df = df.drop(upper_spikes.index)
    return upper_spike_df


def seperate_lower_spikes(df, n, target):
    lower_spike_df = pd.DataFrame()
    spikelen = 1
    while spikelen != 0:
        mean = df[target].mean()
        std = df[target].std()
        lower_spikes = df[(df[target] < mean - (n - .2) * std)]
        lower_spike_df = pd.concat([lower_spikes, lower_spike_df])
        spikelen = len(lower_spikes)
        df = df.drop(lower_spikes.index)
    return lower_spike_df


### iterate over data, classify as a spike if data is over the range: (mean-n*std, mean+n*std)
def process_spikes(data, n, period, target):
    spikelen = 21
    spikes = pd.DataFrame()
    periods = list(range(0, len(data), period * 24))
    for i in range(len(periods)):
        if i != len(periods) - 1:
            df = data.iloc[periods[i]: periods[i + 1]]
            spike_df = seperate_spikes(df, n, target)
            spikes = pd.concat([spikes, spike_df])
    return spikes


def process_lower_spikes(data, n, period, target):
    spikelen = 21
    lower_spikes = pd.DataFrame()
    periods = list(range(0, len(data), period * 24))
    for i in range(len(periods)):
        if i != len(periods) - 1:
            df = data.iloc[periods[i]: periods[i + 1]]
            lower_spike_df = seperate_lower_spikes(df, n, target)
            lower_spikes = pd.concat([lower_spikes, lower_spike_df])
    return lower_spikes


def process_upper_spikes(data, n, period, target):
    spikelen = 21
    upper_spikes = pd.DataFrame()
    periods = list(range(0, len(data), period * 24))
    for i in range(len(periods)):
        if i != len(periods) - 1:
            df = data.iloc[periods[i]: periods[i + 1]]
            upper_spike_df = seperate_upper_spikes(df, n, target)
            upper_spikes = pd.concat([upper_spikes, upper_spike_df])
    return upper_spikes


STARTDATE = "2016-01-01"
ENDDATE = "2020-12-31"


def read_real_time_consumption(startDate, endDate):
    data = consumptionClient.consumption.consumption_realtime(startDate=startDate, endDate=endDate)
    data = pd.DataFrame(data).transpose()
    data.columns = ["date", "consumption"]
    return data


def read_consumption_plan(startDate, endDate):
    data = consumptionClient.consumption.consumption_forecast(startDate=startDate, endDate=endDate)
    data = pd.DataFrame(data)
    data.columns = ["date", "lep"]
    return data


def divide_dates(start, end, intv):
    from datetime import datetime
    start = datetime.strptime(start, "%Y-%m-%d")
    end = datetime.strptime(end, "%Y-%m-%d")
    diff = (end - start) / intv
    for i in range(intv):
        yield (start + diff * i).strftime("%Y-%m-%d")
    yield end.strftime("%Y-%m-%d")


def read_real_time_gen(startDate, endDate):
    dates = list(divide_dates(startDate, endDate, int(pd.to_datetime(endDate).year - pd.to_datetime(startDate).year)))
    dfs = []
    for i in range(len(dates) - 1):
        print(dates[i], dates[i + 1])
        real_time_gen = productionClient.production.real_time_gen(startDate=dates[i], endDate=dates[i + 1])
        real_time_gen = pd.DataFrame(real_time_gen)
        print(real_time_gen)
        real_time_gen = real_time_gen[["date", "total"]]
        real_time_gen.columns = ["date", "production"]
        dfs.append(real_time_gen)
    return pd.concat(dfs)


def read_planned_gen(startDate, endDate):
    planned_gen = productionClient.production.daily_production_plan(startDate=startDate, endDate=endDate)
    planned_gen = pd.DataFrame(planned_gen)
    return planned_gen


def read_total_planned_gen(startDate, endDate):
    planned_total_gen = productionClient.production.daily_production_plan_total(startDate=startDate, endDate=endDate)
    planned_total_gen = pd.DataFrame(planned_total_gen)
    return planned_total_gen


def read_dayAhead(startDate, endDate):
    gün_öncesi_fiyatlar = dayaheadClient.dayahead.mcp(startDate=startDate, endDate=endDate)
    return gün_öncesi_fiyatlar


def create_nextDay(data):
    next_day = {}
    for i in data["date"].iloc[-24:]:
        next_day[i + pd.Timedelta(days=1)] = [0 for x in data.columns]
        next_day[i + pd.Timedelta(days=1)][0] = i + pd.Timedelta(days=1)
    data = pd.concat([data, pd.DataFrame(next_day.values(), columns=data.columns)])
    # data['date'] = data['date'].apply(lambda x: str(x)[:-9])
    data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%dT%H:%M:%S', )
    return data


def process_date(data):
    #     data['date'] = data['date'].apply(lambda x: str(x)[:-6])
    #     data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%dT%H:%M:%S')
    data['year'] = data['date'].dt.year
    data['month'] = data['date'].dt.month
    data['day'] = data['date'].dt.day
    data['week'] = data['date'].dt.isocalendar().week
    data['dayofweek_num'] = data['date'].dt.dayofweek
    data['dayofweek_name'] = data['date'].dt.day_name()
    data['Hour'] = data['date'].dt.hour
    data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d')
    return data


def preprocess_no_feature(startDate, endDate, target):
    if target == "dayAheadPrices":
        day_ahead = read_dayAhead(startDate=startDate, endDate=endDate)
        data = pd.DataFrame(day_ahead[0], columns=['date'])
        data["dayAheadPrices"] = day_ahead[1]
    elif target == "consumption":
        data = read_real_time_consumption(startDate=startDate, endDate=endDate)
        data["consumption"] = data["consumption"].astype('float')
    elif target == "production":
        data = read_real_time_gen(startDate=startDate, endDate=endDate)
        data["production"] = data["production"].astype('float')
    if data.shape[0] < 24:
        data = pd.DataFrame(pd.date_range(start=pd.to_datetime(startDate),
                                          end=(pd.to_datetime(endDate) + pd.DateOffset(days=int(1))) - pd.DateOffset(
                                              hours=1),
                                          freq='H').tolist(), columns=['date'])
    data['date'] = data['date'].apply(lambda x: str(x)[:-6])
    data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%dT%H:%M:%S')
    while str(endDate) not in data["date"].apply(lambda x: str(x)[:10]).tolist():
        data = create_nextDay(data)
    data = process_date(data)
    return data


def preprocess_features(data, target, spike_interval=30):
    spikes = process_spikes(data, 2, spike_interval, target)
    upper_spikes = process_upper_spikes(data, 2, spike_interval, target)
    lower_spikes = process_lower_spikes(data, 2, spike_interval, target)
    data["upper"] = 0
    data["lower"] = 0
    data["isSpike"] = 0
    data.loc[spikes.index, "isSpike"] = 1
    data.loc[upper_spikes.index, "upper"] = 1
    data.loc[lower_spikes.index, "lower"] = 1
    data = create_features(data, target)
    cls = data.filter(regex='^t').columns
    for c in cls:
        data[c] = data[c].fillna(data.groupby(['dayofweek_num', 'month', 'year', 'Hour'])[c].transform('mean'))
    # data = data.set_index('date')
    return data


# def prep_from_scratch(startDate, endDate):
#     data = read_consumption_plan(startDate=startDate, endDate=endDate)
#     data = data[data['lep'].notna()]
#     total_planned_gen = read_total_planned_gen(startDate=startDate, endDate=endDate)
#     day_ahead = read_dayAhead(startDate=startDate, endDate=endDate)
#     #     data = pd.merge(data, read_total_planned_gen, left_on =['date'], right_on=["date"], how='outer')
#     data["dayAheadPrices"] = day_ahead[1]
#     data["production"] = pd.to_numeric(total_planned_gen["dpp"])
#     data["consumption"] = pd.to_numeric(data["lep"])
#     data.drop('lep', axis=1, inplace=True)
#     data['date'] = data['date'].apply(lambda x: str(x)[:-6])
#     data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%dT%H:%M:%S')
#     if pd.to_datetime(endDate, format='%Y-%m-%dT%H:%M:%S') not in data["date"].to_list():
#         data = create_nextDay(data)
#     data = process_date(data)
#     spikes = process_spikes(data, 2, 30)
#     upper_spikes = process_upper_spikes(data, 2, 30)
#     lower_spikes = process_lower_spikes(data, 2, 30)
#     data["upper"] = 0
#     data["lower"] = 0
#     data["isSpike"] = 0
#     data.loc[spikes.index, "isSpike"] = 1
#     data.loc[upper_spikes.index, "upper"] = 1
#     data.loc[lower_spikes.index, "lower"] = 1
#     data = create_features(data)
#     data = data.set_index('date')
#     return data


def preprocess_final(startDate='2016-01-01', endDate='2020-12-31', target="dayAheadPrices"):
    if os.path.exists("../API/" + target + "_data.csv"):
        data = pd.read_csv("../API/" + target + "_data.csv")
        data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d')
        if pd.to_datetime(data.iloc[-1]["date"]) < pd.to_datetime(endDate):
            print("catching up to date")
            data = catch_up_2_date(data, endDate, target)
    else:
        print("Fetching and processing the data...")
        if target != "production":
            stdate = "2011-12-01"
        else:
            stdate = "2013-05-01"
        data = preprocess_no_feature(stdate,
                                     str(datetime.date.today()) if int(datetime.datetime.now().hour) < 13 else str(
                                         datetime.date.today() + datetime.timedelta(days=1)), target=target)
        print("prep_no_feat else")
        print(data)
        data = reduce_mem_usage(data)
        data.to_csv("../API/" + target + "_data.csv", index=False)
        if pd.to_datetime(data.iloc[-1]["date"]) < pd.to_datetime(endDate):
            print("catching up to date")
            data = catch_up_2_date(data, endDate, target)
    if 'date' in data.columns:
        data = data.set_index('date')
    data = data[startDate: endDate]
    data = preprocess_features(data, target=target)
    data.drop('index', axis=1, inplace=True, errors='ignore')
    print("Reducing memory usage...")
    data = reduce_mem_usage(data)
    data.drop(['fueloil', 'gasOil', 'blackCoal', 'lignite', 'geothermal', 'river',
               'dammedHydro', 'lng', 'biomass', 'naphta', 'importCoal',
               'asphaltiteCoal', 'nucklear', 'sun', 'importExport', 'dayofweek_name'], axis=1, errors='ignore',
              inplace=True)
    data["target"] = np.where(data["upper"] == 1, 1,
                              np.where(data["lower"] == 1, -1, 0))
    data.drop(["upper", "lower"], axis=1, inplace=True)
    print("Data Processed.")
    return data


def catch_up_2_date(data, catchDate, target):
    startDate = str(pd.to_datetime(data.iloc[-1]["date"]) + pd.Timedelta(days=1))[:10]
    endDate = catchDate
    # if pd.to_datetime(endDate) <= datetime.date.today() or (pd.to_datetime(endDate) == (
    #         pd.to_datetime(datetime.date.today()) + pd.Timedelta(days=1)) and datetime.datetime.now().hour > 13):
    topping = preprocess_no_feature(startDate, endDate, target).reset_index()
    data = pd.concat([data, topping])
    data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d')
    data = data.set_index('date')
    return data
