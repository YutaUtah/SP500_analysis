
from visualize_correlation import visualize_data

import bs4 as bs
import datetime as dt
import os
import pandas as pd
import numpy as np
import pandas_datareader.data as web
import pickle
import requests
from collections import Counter






def save_sp500_tickers():
    '''
    save_sp500_tickers() is to extract SP500 ticker names
    :return: tickers
    '''
    response = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    soup = bs.BeautifulSoup(response.text, "lxml")

    #specify table
    table = soup.find("table", {"class":"wikitable sortable"})
    tickers =[]

    for row in table.findAll("tr")[1:]:
        ticker = row.findAll("td")[0].text
        ticker = ticker[:-1]
        tickers.append(ticker)

    #write in pickle fidle f
    with open("sp500tickers.pickle", "wb") as f:
        pickle.dump(tickers, f)

    return tickers




def get_data_from_yahoo(reload_sp500 = False):
    '''
    get_data_from_yahoo() is to get the data from yahoo finance and save it to pickle file
    and make the folder stock_dfs to save csv file to each ticker
    :param reload_sp500:
    :return:
    '''

    if reload_sp500:
        tickers = save_sp500_tickers()
    else:
        with open("sp500tickers.pickle", "rb") as f:
                tickers = pickle.load(f)

    if not os.path.exists("stock_dfs"):
        os.mkdir("stock_dfs")



    start = dt.datetime(2000,1,1)
    end = dt.datetime(2016, 12, 31)

    for ticker in tickers[:55]:
        print(ticker)
        if not os.path.exists("stock_dfs/{}.csv".format(ticker)):
            df = web.DataReader(ticker, "yahoo", start, end)
            df.to_csv("stock_dfs/{}.csv".format(ticker))


        else:
            print("we already have {}".format(ticker))





def complie_data():
    with open("sp500tickers.pickle", "rb") as f:
        tickers = pickle.load(f)
        print(tickers)

    main_df = pd.DataFrame()

    for count, ticker in enumerate(tickers[:55]):
        # print(ticker)
        df=pd.read_csv("stock_dfs/{}.csv".format(ticker))
        df.set_index("Date", inplace= True)


        # this is to rename "Adj Close" to each ticker names
        df.rename(columns = {"Adj Close": ticker}, inplace = True)
        # 1 means axis
        df.drop(["Open","High","Low","Close","Volume"],1,inplace=True)

        #
        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.join(df, how="outer")

        if count % 10 == 0:
            print(count)

    print(main_df.head())
    main_df.to_csv("sp500_joined_closes.csv")




##Applying Machine Learning to Correlation Tables
def process_data_for_labels(ticker):  # Prepares Labels
    hm_days = 7  # how many days in the future do we have to make or lose 'x%'
    df = pd.read_csv('sp500_joined_closes.csv', index_col=0)
    tickers = df.columns.values.tolist()
    df.fillna(0, inplace=True)

    for i in range(1, hm_days + 1):
        df['{}_{}d'.format(ticker, i)] = (df[ticker].shift(-i) - df[ticker]) / df[ticker]  # generates % changes

    df.fillna(0, inplace=True)
    return tickers, df


# process_data_for_labels('TSLA')


def buy_sell_hold(*args):  # Helper Function
    cols = [c for c in args]  # mapping to pandas
    requirement = 0.02  # if stock price changes by 2% in 7 days
    for col in cols:
        if col > requirement:
            return 1
        if col < -requirement:
            return -1
    return 0


def extract_featuresets(ticker):  # Map Helper Function to DataFrame New Column
    tickers, df = process_data_for_labels(ticker)
    df['{}_target'.format(ticker)] = list(map(buy_sell_hold,
                                              df['{}_1d'.format(ticker)],
                                              df['{}_2d'.format(ticker)],
                                              df['{}_3d'.format(ticker)],
                                              df['{}_4d'.format(ticker)],
                                              df['{}_5d'.format(ticker)],
                                              df['{}_6d'.format(ticker)],
                                              df['{}_7d'.format(ticker)]
                                              ))
    vals = df['{}_target'.format(ticker)].values.tolist()
    str_vals = [str(i) for i in vals]
    print('Data spread:', Counter(str_vals))

    df.fillna(0, inplace=True)
    df = df.replace([np.inf, -np.inf], np.nan)  # replaces any infinite price changes with NAN's
    df.dropna(inplace=True)

    df_vals = df[[ticker for ticker in
                  tickers]].pct_change()  # normalizes price changes (today's value as opposed to yesterday) == % change data for all S&P500 companies including company in question
    df_vals = df.replace([np.inf, -np.inf], 0)  # replace infinite price changes with 0's
    df_vals.fillna(0, inplace=True)  # replace NAN's with 0's

    X = df_vals.values  # defines feature set  == % change data for all S&P500 companies including company in question
    y = df['{}_target'.format(ticker)].values  # defines target (0, 1 or -1)

    return X, y, df





