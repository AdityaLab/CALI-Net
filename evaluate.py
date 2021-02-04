%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime as dt
import pandas as pd

def get_RMSE(path,k_wk_ahead=4):
    df_new=pd.read_csv(path)
    df_new.head()

    df_new['date'] = pd.to_datetime(df_new['date'])
    df_new = df_new[(df_new['date']>=dt.strptime('2020-02-29',"%Y-%m-%d")) & (df_new['date']<=dt.strptime('2020-04-11',"%Y-%m-%d"))]
    df_new.reset_index(inplace=True)
    df_new.drop('index',axis=1,inplace=True)
    df_new.head()

    squared_error=dict()
    for key,val in df_new.groupby(['region','date','iter_number']):
        for i in range(1,5):
            sq_err=np.square(val['val{}'.format(i)] - val['pred{}'.format(i)])
            df_new.loc[val.index.values,'squared_error_{}'.format(i)] = sq_err

    if k_wk_ahead==4:
        cols=['squared_error_1','squared_error_2','squared_error_3','squared_error_4']
    elif k_wk_ahead==2:
        cols=['squared_error_1','squared_error_2']
    sq_errs=df_new[cols]
    mean_squared_errors_across_kweek_forecasts = np.mean(sq_errs,axis=1)
    df_new['mean_sq_err'] = mean_squared_errors_across_kweek_forecasts

    rmses_per_region_per_iter_number={'region':[],'rmse':[],'iter_number':[]}

    #Mean across all Squared Errors for all dates, followed by square root of the mean.
    for key,val in df_new.groupby(['region','iter_number']):
    #     print(val.shape)
        rmse = np.sqrt(val['mean_sq_err']).mean()
        rmses_per_region_per_iter_number['region'].append(key[0])
        rmses_per_region_per_iter_number['iter_number'].append(key[1])
        rmses_per_region_per_iter_number['rmse'].append(rmse)

    rmses_per_region_per_iter_number=pd.DataFrame(rmses_per_region_per_iter_number)
    for key,val in rmses_per_region_per_iter_number.groupby('region'):
        print("Region = {}, RMSE = {}".format(key,val['rmse'].mean()))
    #%%
    print('by region')
    for key,val in rmses_per_region_per_iter_number.groupby('region'):
        print(val['rmse'].mean())

    #%%
    rmses_per_date_per_iter_number={'date':[],'rmse':[],'iter_number':[]}

    #Mean across all Squared Errors for all regions, followed by square root of the mean.
    for key,val in df_new.groupby(['date','iter_number']):
        #print(val.shape)
        rmse = np.sqrt(val['mean_sq_err']).mean()
        rmses_per_date_per_iter_number['date'].append(key[0])
        rmses_per_date_per_iter_number['iter_number'].append(key[1])
        rmses_per_date_per_iter_number['rmse'].append(rmse)

    rmses_per_date_per_iter_number=pd.DataFrame(rmses_per_date_per_iter_number)
    for key,val in rmses_per_date_per_iter_number.groupby('date'):
        print("Date = {}, RMSE = {}".format(key,val['rmse'].mean()))
    #%%
    print('by date')
    for key,val in rmses_per_date_per_iter_number.groupby('date'):
        print(val['rmse'].mean())
    print('total:',rmses_per_date_per_iter_number['rmse'].mean())


path='./rmse_results/YOUR_RESULTS_FILE.csv'
get_RMSE(path,4)