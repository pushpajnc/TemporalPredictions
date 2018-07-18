###########################################
# Suppress matplotlib user warnings
# Necessary for newer version of matplotlib
import warnings
warnings.filterwarnings("ignore", category = UserWarning, module = "matplotlib")
###########################################

import numpy as np
import pandas as pd
from time import time
from sklearn.cross_validation import ShuffleSplit
from IPython.display import display
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import calendar
import math
import io
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace import sarimax as sa
from statsmodels.stats import diagnostic as diag

def stats_day(df):
    days = \
    ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']

    df['day_of_week'] = df['visit_date'].dt.weekday_name
    df['day_of_week'] = df['day_of_week'].astype("category", categories=days, ordered=True)

    medianvisitors_day = \
    df.groupby('day_of_week', as_index=False)['visitors'].median()
    meanvisitors_day = \
    df.groupby('day_of_week', as_index=False)['visitors'].mean()
    sumvisitors_day = \
    df.groupby('day_of_week', as_index=False)['visitors'].sum()

    weekdays = [i for i in range(1,8)]

    colors = ['g', 'r', 'b', 'c']

    fig = plt.figure(figsize=(15, 8))
    fig.add_subplot(2, 2, 1)
    plt.bar(weekdays, medianvisitors_day.visitors, color = colors)
    plt.xticks(weekdays, medianvisitors_day.day_of_week)
    plt.title('Median visitors per day', fontsize = 16, color = 'red')

    fig.add_subplot(2, 2, 2)
    plt.bar(weekdays, meanvisitors_day.visitors, color = colors)
    plt.xticks(weekdays, meanvisitors_day.day_of_week)
    plt.title('Mean visitors per day', fontsize = 16, color = 'red')
    plt.show()

    fig.add_subplot(2, 2, 3)
    plt.bar(weekdays, sumvisitors_day.visitors, color = colors)
    plt.xticks(weekdays, meanvisitors_day.day_of_week)
    plt.title('Sum visitors per day', fontsize = 16, color = 'red')
    plt.show()

def stats_month(df):
    df['month'] = df['visit_date'].dt.month
    medianvisitors_month = \
    df.groupby('month', as_index=False)['visitors'].median()

    meanvisitors_month = \
    df.groupby('month', as_index=False)['visitors'].mean()

    month_name = ['Jan', 'Feb', 'March', 'April', 'May', 'June', 'July', 'August', 'Sep', 'Oct', 'Nov', 'Dec']
    colors = ['g', 'r', 'b', 'c']

    plt.figure(figsize=(12,4))
    plt.bar(medianvisitors_month.month, medianvisitors_month.visitors, color = colors)
    plt.xticks(medianvisitors_month.month, month_name )
    plt.title('Median Visitors per month', fontsize = 16, color = 'red')

    plt.figure(figsize=(12,4))
    plt.bar(meanvisitors_month.month, meanvisitors_month.visitors, color = colors)
    plt.xticks(meanvisitors_month.month, month_name )
    plt.title('Mean Visitors per month', fontsize = 16, color = 'red')

    plt.show()

def capture_image():
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return buf

def capture_image_and_not_show(fig):
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    return buf
 
def totvstors_date(df, dates, ylimit):
    fig, ax = plt.subplots(figsize=(15, 4))
    ax.plot(df.visit_date, df.visitors, \
            label = 'air_visit')
    legend = ax.legend(loc = 'upper left', shadow = True)
    plt.xlabel('visit_date')
    plt.ylabel('Total Visitors')
    plt.title('Total visitors per day', fontsize = 14, color='red')
    plt.xlim(dates)
    plt.ylim(ylimit)
    # plt.gca().xaxis.set_major_locator(months)
    plt.gcf().autofmt_xdate(rotation=30)
    #plt.close(fig)
    return capture_image()

def savefig(buf, filename):
    f = open(filename, 'wb')
    f.write(buf.read())
    f.close()

def ACF(X):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (12, 3))
    plot_acf(X, ax=ax1, lags=40)
    plot_pacf(X, ax=ax2, lags=40)
    
def totvstors_date_hol(df, holidays, title):
    fig, ax = plt.subplots(figsize=(15, 4))
    ax.plot(df.visit_date, df.visitors)
    
    for holiday in holidays:
        ax.axvline(x=holiday, color='silver')
    plt.title(title, fontsize = 14, color='red')
    plt.xlabel('visit_date')
    plt.xlim('2016-01-01', '2017-05-05')
    plt.ylabel('Total Visitors')
    # plt.gca().xaxis.set_major_locator(months)
    plt.gcf().autofmt_xdate(rotation=30)
    plt.show()

def ADF(X):
    split = len(X) / 2
    X1, X2 = X[0:split], X[split:]
    mean1, mean2 = X1.mean(), X2.mean()
    var1, var2 = X1.var(), X2.var()

    print('mean1=%f, mean2=%f' % (mean1, mean2))
    print('variance1=%f, variance2=%f' % (var1, var2))

    adf_result = adfuller(X)

    print('ADF Statistic: %f' % adf_result[0])
    print('p-value: %f' % adf_result[1])
    print('Critical Values:')
    for key, value in adf_result[4].items():
        print('\t%s: %.3f' % (key, value))

def Box(X):
    box_result = diag.acorr_ljungbox(X)
    print box_result
    
def mavg(N, mylist):
    cumsum, moving_aves = [0], []

    for i, x in enumerate(mylist, 1):
        cumsum.append(cumsum[i-1] + x)
        if i>=N:
            moving_ave = (cumsum[i] - cumsum[i-N])/N
            #can do stuff with moving_ave here
            moving_aves.append(moving_ave)
    return moving_aves

def arima(series, pdq, PDQ):
    l = []
    for order in pdq:
        for sorder in PDQ:            
            try:
                mod = sa.SARIMAX(series.values, order=order, seasonal_order=sorder)
                results = mod.fit()

                l.append({'Order':order, 'SOrder':sorder, 'AIC':results.aic, \
                          'BIC':results.bic})
            except:
                continue   
    arimaDF = pd.DataFrame(l)
    order_minAIC = arimaDF.loc[arimaDF['AIC'].idxmin()][2]
    sorder_minAIC = arimaDF.loc[arimaDF['AIC'].idxmin()][3]
    mod = sa.SARIMAX(series, order=order_minAIC, \
                     seasonal_order=sorder_minAIC)
    results = mod.fit()
    return (results, arimaDF)

def evaluation_metric(observed_series, forecasted_series):
    sumabs = 0.0
    sumsqr = 0.0
    pererr = 0.0
    for truth, forecast in zip(observed_series, forecasted_series):
        sumabs = sumabs + abs(truth - forecast)
        sumsqr = sumsqr + (truth - forecast)**2
        pererr = pererr + 100*abs(truth - forecast)/truth

    RMSE = math.sqrt(sumsqr/len(observed_series))
    MAE  = sumabs/len(observed_series)
    MAPE = pererr/len(observed_series)
    values = dict()
    values['RMSE error'] = round(RMSE, 2)
    values['MAE error'] = round(MAE, 2)
    values['MAPE error'] = round(MAPE, 2)
    return values

def vstors_by_area_plot(df):
    g = sns.FacetGrid(df, col="areaid", col_wrap=3, sharey=False,\
                      size=4, aspect=1.2, hue="areaid")
    g = g.map(plt.plot, "visit_date", "visitors")
    g.set_titles(row_template = '{row_name}', col_template = '{col_name}', \
                 size=20)
    g.set_axis_labels('Date', 'Total Visitors')
    g.set_xticklabels(rotation=30, size =20)
    g.set_yticklabels(size =15)
    
def vstors_by_genre_plot(df):
    g = sns.FacetGrid(df, col="air_genre_name", col_wrap=3, sharey=False,\
                      size=4, aspect=1.2, hue="air_genre_name")
    g = g.map(plt.plot, "visit_date", "visitors")
    g.set_titles(row_template = '{row_name}', col_template = '{col_name}', \
                 size=20)
    g.set_axis_labels('Date', 'Total Visitors')
    g.set_xticklabels(rotation=30, size =20)
    g.set_yticklabels(size =15)
    
def visitors_by_genre_area(visitor_df, store_df):
    visitors1 = visitor_df.merge(store_df[['air_store_id', 'air_genre_name', \
                      'air_area_name']], on = 'air_store_id', how= 'left')
    
    sumvstors_by_date  = visitors1.groupby(['air_area_name', 'air_genre_name', 'visit_date'], \
                         as_index=False)['visitors'].sum()
    
    sumvstors_by_genre = visitors1.groupby(['air_genre_name', 'visit_date'], \
                         as_index=False)['visitors'].sum()
    
    sumvstors_by_area  = visitors1.groupby(['air_area_name', 'visit_date'], \
                         as_index=False)['visitors'].sum()
    
    y = sumvstors_by_area.iloc[:,0]
    uniques_y, Y = np.unique(y, return_inverse=True)
    uniques_y, Y

    sumvstors_by_area['areaid'] = Y
    return sumvstors_by_genre, sumvstors_by_area, sumvstors_by_date
    