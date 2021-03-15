from datetime import date, timedelta
import gc
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from xgboost import XGBRegressor
import random
from timeit import default_timer as timer



### load train
data = pd.read_csv(
    '/favorita-grocery-sales-forecasting/train.csv', usecols=["date","store_nbr","item_nbr","unit_sales","onpromotion"],         
    dtype={'onpromotion': bool}, parse_dates=["date"], skiprows=range(1, 66458909))                                                              

###### This section is to randomly sample from data for future analysis. It is not directly repated to our current task.
# n = 23808260 #number of records in file
# s = 1000000 #desired sample size
# skip = sorted(random.sample(range(n),n-s))
# data = pd.read_csv('/favorita-grocery-sales-forecasting/train.csv',skiprows=skip)

### load test 
testing = pd.read_csv(
    "/favorita-grocery-sales-forecasting/test.csv", usecols=["id","date","store_nbr","item_nbr","onpromotion"],          ## we need id only for submission
    dtype={'onpromotion': bool}, parse_dates=["date"])

### load items and stores
items = pd.read_csv("/favorita-grocery-sales-forecasting/items.csv",)
stores = pd.read_csv("/favorita-grocery-sales-forecasting/stores.csv",) 

data["unit_sales"]=data["unit_sales"].apply(lambda u: np.log1p(float(u)) if float(u) > 0 else 0)        ##### natiral logarithm plus 1, that is log(x+1)
testing.set_index(['store_nbr', 'item_nbr', 'date'], inplace=True)
items.set_index("item_nbr", inplace=True)
stores.set_index("store_nbr", inplace=True)

le = LabelEncoder()
items['family'] = le.fit_transform(items['family'].values)          ### from item table

stores['city'] = le.fit_transform(stores['city'].values)            ### from stores table
stores['state'] = le.fit_transform(stores['state'].values)
stores['type'] = le.fit_transform(stores['type'].values)

data = data.loc[data.date>=pd.datetime(2017,1,1)]

promo_train = data.set_index(["store_nbr", "item_nbr", "date"])[["onpromotion"]].unstack(level=-1).fillna(False)
promo_train.columns = promo_train.columns.get_level_values(1)
promo_test = testing[["onpromotion"]].unstack(level=-1).fillna(False)
promo_test.columns = promo_test.columns.get_level_values(1)
promo_test = promo_test.reindex(promo_train.index).fillna(False)
promo = pd.concat([promo_train, promo_test], axis=1)
del promo_test, promo_train

data = data.set_index(["store_nbr", "item_nbr", "date"])[["unit_sales"]].unstack(level=-1).fillna(0)
data.columns = data.columns.get_level_values(1)

items = items.reindex(data.index.get_level_values(1))
stores = stores.reindex(data.index.get_level_values(0))

data_item = data.groupby('item_nbr')[data.columns].sum()
promo_item = promo.groupby('item_nbr')[promo.columns].sum()

data_store_class = data.reset_index()
data_store_class['class'] = items['class'].values
data_store_class_index = data_store_class[['class', 'store_nbr']]
data_store_class = data_store_class.groupby(['class', 'store_nbr'])[data.columns].sum()

data_store_class_index = [tuple(x) for x in data_store_class_index.to_numpy()]               

data_promo_store_class = promo.reset_index()
data_promo_store_class['class'] = items['class'].values
data_promo_store_class_index = data_promo_store_class[['class', 'store_nbr']]
data_promo_store_class = data_promo_store_class.groupby(['class', 'store_nbr'])[promo.columns].sum()

win = 16

def get_timespan(df, dt, minus, periods, freq='D'):
    return df[pd.date_range(dt - timedelta(days=minus), periods=periods, freq=freq)]

def prepare_dataset(df, promo_df, t, is_train=True, name_prefix=None):
    X = {
        "promo_14": get_timespan(promo_df, t, 14, 14).sum(axis=1).values,
        "promo_60": get_timespan(promo_df, t, 60, 60).sum(axis=1).values,
        "promo_140": get_timespan(promo_df, t, 140, 140).sum(axis=1).values,
        "promo_3_aft": get_timespan(promo_df, t + timedelta(days=win), 15, 3).sum(axis=1).values,
        "promo_7_aft": get_timespan(promo_df, t + timedelta(days=win), 15, 7).sum(axis=1).values,
        "promo_14_aft": get_timespan(promo_df, t + timedelta(days=win), 15, 14).sum(axis=1).values,
    }

    for i in [3, 7, 14, 30, 60, 140]:
        tmp1 = get_timespan(df, t, i, i)
        tmp2 = (get_timespan(promo_df, t, i, i) > 0) * 1

        X['has_promo_mean_%s' % i] = (tmp1 * tmp2.replace(0, np.nan)).mean(axis=1).values
        X['has_promo_mean_%s_decay' % i] = (tmp1 * tmp2.replace(0, np.nan) * np.power(0.9, np.arange(i)[::-1])).sum(axis=1).values

        X['no_promo_mean_%s' % i] = (tmp1 * (1 - tmp2).replace(0, np.nan)).mean(axis=1).values
        X['no_promo_mean_%s_decay' % i] = (tmp1 * (1 - tmp2).replace(0, np.nan) * np.power(0.9, np.arange(i)[::-1])).sum(axis=1).values

    for i in [3, 7, 14, 30, 60, 140]:
        tmp = get_timespan(df, t, i, i)
        X['diff_%s_mean' % i] = tmp.diff(axis=1).mean(axis=1).values
        X['mean_%s_decay' % i] = (tmp * np.power(0.9, np.arange(i)[::-1])).sum(axis=1).values
        X['mean_%s' % i] = tmp.mean(axis=1).values
        X['median_%s' % i] = tmp.median(axis=1).values
        X['min_%s' % i] = tmp.min(axis=1).values
        X['max_%s' % i] = tmp.max(axis=1).values
        X['std_%s' % i] = tmp.std(axis=1).values

    for i in [3, 7, 14, 30, 60, 140]:
        tmp = get_timespan(df, t + timedelta(days=-7), i, i)
        X['diff_%s_mean_2' % i] = tmp.diff(axis=1).mean(axis=1).values
        X['mean_%s_decay_2' % i] = (tmp * np.power(0.9, np.arange(i)[::-1])).sum(axis=1).values
        X['mean_%s_2' % i] = tmp.mean(axis=1).values
        X['median_%s_2' % i] = tmp.median(axis=1).values
        X['min_%s_2' % i] = tmp.min(axis=1).values
        X['max_%s_2' % i] = tmp.max(axis=1).values
        X['std_%s_2' % i] = tmp.std(axis=1).values

    for i in [7, 14, 30, 60, 140]:
        tmp = get_timespan(df, t, i, i)
        X['has_sales_days_in_last_%s' % i] = (tmp > 0).sum(axis=1).values
        X['last_has_sales_day_in_last_%s' % i] = i - ((tmp > 0) * np.arange(i)).max(axis=1).values
        X['first_has_sales_day_in_last_%s' % i] = ((tmp > 0) * np.arange(i, 0, -1)).max(axis=1).values

        tmp = get_timespan(promo_df, t, i, i)
        X['has_promo_days_in_last_%s' % i] = (tmp > 0).sum(axis=1).values
        X['last_has_promo_day_in_last_%s' % i] = i - ((tmp > 0) * np.arange(i)).max(axis=1).values
        X['first_has_promo_day_in_last_%s' % i] = ((tmp > 0) * np.arange(i, 0, -1)).max(axis=1).values

    tmp = get_timespan(promo_df, t + timedelta(days=win), 15, 15)
    X['has_promo_days_in_after_15_days'] = (tmp > 0).sum(axis=1).values
    X['last_has_promo_day_in_after_15_days'] = i - ((tmp > 0) * np.arange(15)).max(axis=1).values
    X['first_has_promo_day_in_after_15_days'] = ((tmp > 0) * np.arange(15, 0, -1)).max(axis=1).values

    for i in range(1, win):
        X['day_%s' % i] = get_timespan(df, t, i, 1).values.ravel()

    for i in range(7):
        X['mean_4_dow{}'.format(i)] = get_timespan(df, t, 28-i, 4, freq='7D').mean(axis=1).values
        X['mean_20_dow{}'.format(i)] = get_timespan(df, t, 140-i, 20, freq='7D').mean(axis=1).values

    for i in range(-win, win):
        X["promo_{}".format(i)] = promo_df[t + timedelta(days=i)].values.astype(np.uint8)

    X = pd.DataFrame(X)

    if is_train:
        y = df[
            pd.date_range(t, periods=win)                     
        ].values
        return X, y
    if name_prefix is not None:
        X.columns = ['%s_%s' % (name_prefix, c) for c in X.columns]
    return X


refe_time = date(2017, 6, 14)
num_days = 6
X_l, y_l = [], []
for i in range(num_days):
    delta = timedelta(days=7 * i)
    X_tmp, y_tmp = prepare_dataset(data, promo, refe_time + delta)

    X_tmp2 = prepare_dataset(data_item, promo_item, refe_time + delta, is_train=False, name_prefix='item')
    X_tmp2.index = data_item.index
    X_tmp2 = X_tmp2.reindex(data.index.get_level_values(1)).reset_index(drop=True)

    X_tmp3 = prepare_dataset(data_store_class, data_promo_store_class, refe_time + delta, is_train=False, name_prefix='store_class')
    X_tmp3.index = data_store_class.index

    X_tmp3 = X_tmp3.reindex(data_store_class_index).reset_index(drop=True)                  

    X_tmp = pd.concat([X_tmp, X_tmp2, X_tmp3, items.reset_index(), stores.reset_index()], axis=1)
    X_l.append(X_tmp)
    y_l.append(y_tmp)

    del X_tmp2
    gc.collect()

X_train = pd.concat(X_l, axis=0)
y_train = np.concatenate(y_l, axis=0)


del X_l, y_l
X_val, y_val = prepare_dataset(data, promo, date(2017, 7, 26))

X_val2 = prepare_dataset(data_item, promo_item, date(2017, 7, 26), is_train=False, name_prefix='item')
X_val2.index = data_item.index
X_val2 = X_val2.reindex(data.index.get_level_values(1)).reset_index(drop=True)

X_val3 = prepare_dataset(data_store_class, data_promo_store_class, date(2017, 7, 26), is_train=False, name_prefix='store_class')
X_val3.index = data_store_class.index
X_val3 = X_val3.reindex(data_store_class_index).reset_index(drop=True)

X_val = pd.concat([X_val, X_val2, X_val3, items.reset_index(), stores.reset_index()], axis=1)

X_test = prepare_dataset(data, promo, date(2017, 8, win), is_train=False)

X_test2 = prepare_dataset(data_item, promo_item, date(2017, 8, win), is_train=False, name_prefix='item')
X_test2.index = data_item.index
X_test2 = X_test2.reindex(data.index.get_level_values(1)).reset_index(drop=True)

X_test3 = prepare_dataset(data_store_class, data_promo_store_class, date(2017, 8, win), is_train=False, name_prefix='store_class')
X_test3.index = data_store_class.index
X_test3 = X_test3.reindex(data_store_class_index).reset_index(drop=True)

X_test = pd.concat([X_test, X_test2, X_test3, items.reset_index(), stores.reset_index()], axis=1)

del X_test2, X_val2, data_item, promo_item, data_store_class, data_promo_store_class, data_store_class_index
gc.collect()



### model's hyper-parameters and structure
num_boost_round = 500
param = {'eval_metric': 'rmsle', 'max_depth': 3, 'eta': 1, 'silent': 1, 'min_child_weight': 1,
            'colsample_bytree': 0.8,
            'objective': "reg:squarederror", 'n_estimators': 100}

print("model's hyper-parameters and structure: ", param)

val_pred = []
test_pred = []
cate_vars = []
for i in range(win):
    print("=" * 50)
    print("Step %d" % (i+1))
    print("=" * 50)
    dtrain = xgb.DMatrix(X_train,label=y_train[:, i], weight=pd.concat([items["perishable"]] * num_days) * 0.25 + 1)
    val = xgb.DMatrix(X_val, label=y_val[:, i], weight=pd.concat([items["perishable"]] * num_days) * 0.25 + 1)
    dtest = xgb.DMatrix(X_test, weight=pd.concat([items["perishable"]] * num_days) * 0.25 + 1)
    # training
    start_train = timer()
    model = xgb.train(param, dtrain, num_boost_round=num_boost_round, evals=[(val,"valid")])
    end_train = timer()
    print('Training with %d samples completed in %.2f seconds' % (X_train.shape[0], end_train - start_train))

    ###### this section is to extract feature importance. You can uncomment for this purpose. 
    # m_feature = XGBRegressor(max_depth=3, n_estimators=100)
    # m_feature.fit(X_train, y_train[:, i])
    # x_weight = m_feature.get_booster().get_score(importance_type='gain')
    # print("\n".join(("%s: %s" % x) for x in sorted(
    #     zip(x.columns, x_weight),
    #     key=lambda x: x[1], reverse=True
    # )))
 
    val_pred.append(model.predict(val))
    test_pred.append(model.predict(dtest))


print("Validation mse:", mean_squared_error(
    y_val, np.array(val_pred).transpose()))

weight = items["perishable"] * 0.25 + 1
err = (y_val - np.array(val_pred).transpose())**2
err = err.sum(axis=1) * weight
err = np.sqrt(err.sum() / weight.sum() / win)
print('nwrmsle = {}'.format(err))

y_val = np.array(val_pred).transpose()
df_preds = pd.DataFrame(
    y_val, index=data.index,
    columns=pd.date_range("2017-07-26", periods=win)
).stack().to_frame("unit_sales")
df_preds.index.set_names(["store_nbr", "item_nbr", "date"], inplace=True)
df_preds["unit_sales"] = np.clip(np.expm1(df_preds["unit_sales"]), 0, 1000)
df_preds.reset_index().to_csv('lgb_cv.csv', index=False)

print("Testing outcome...")
y_test = np.array(test_pred).transpose()
df_preds = pd.DataFrame(
    y_test, index=data.index,
    columns=pd.date_range("2017-08-16", periods=win)
).stack().to_frame("unit_sales")
df_preds.index.set_names(["store_nbr", "item_nbr", "date"], inplace=True)

submission = testing[["id"]].join(df_preds, how="left").fillna(0)
submission["unit_sales"] = np.clip(np.expm1(submission["unit_sales"]), 0, 1000)
submission.to_csv('lgb_sub.csv', float_format='%.4f', index=None)