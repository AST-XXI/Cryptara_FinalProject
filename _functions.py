# Extract Algo & NLP results from Funcationality folders

from datetime import datetime, timedelta
import os
time = datetime.now()
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
lemmatizer = WordNetLemmatizer()
from main import Main
from tradingview_ta import TA_Handler, Interval
from finta import TA
from pyod.models.copod import COPOD
import pandas as pd 
import numpy as np
from pycoingecko import CoinGeckoAPI
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
from datetime import datetime, timedelta
from pandas import json_normalize
import re
import tweepy
import matplotlib.pyplot as plt

def clean_text(text):
    regex = re.compile("[^a-zA-Z0-9]")
    re_clean = regex.sub(' ', text)
    words = word_tokenize(re_clean)
    return words


def Average(lst):
    return sum(lst) / len(lst)

def Sum(lst):
    return sum(lst) / len(lst)

def format_convert(x):
    try:
        return "{:.0%}".format(x)
    except:
        return "{:.0%}".format(float(x))

def dollar_sign(x):
    return "${:,.2f}".format(x)

def percent_sign(x):
    return "{:,.2f}%".format(x)


def algo_strategy(time_series):
    buy_price=0
    trend_cnt=0
    dtrend_cnt=0
    maker_fee = .01
    sell_price=0
    profit =0
    accumulated_profit = 0
    previous_price = 0
    previous_returns = 0
    previous_candle = 'downtrend'
    classification = 'SELL'
    previous_classification = 'SELL'
    candle = 0
    ema = 0
    stochastic = 0
    time_series['trading'] = np.nan
    for index, row in time_series.iterrows():
        if row["candle"] < 0:
            candle = 'downtrend'
        else:
            candle = 'uptrend'
        if row["returns"] < 0:
            returns = 'downtrend'
        else: 
            returns = 'uptrend'
        if row['stoch_k'] > row['stoch_d']:
            stochastic = 'uptrend'
        else:
            stochastic = 'downtrend'
        if row['short_vama'] > row['long_vama']:
            vama = 'uptrend'
        else:
            vama = 'downtrend'
        if row['short_ema'] > row['long_ema']:
            ema = 'uptrend'
        else:
            ema = 'downtrend'
        if index == time_series.index[0]:
            time_series.loc[index, "trading"] = previous_classification
        elif (candle=='uptrend') and (returns=='uptrend') and (vama=='uptrend') and (ema=='uptrend') and (previous_classification=='SELL'):
            classification ='BUY'
            buy_price = row['close']
            time_series.loc[index, "trading"] = classification
        elif (candle=='uptrend') and (returns=='uptrend') and (vama=='uptrend') and (ema=='uptrend') and (previous_classification=='BUY'):
            classification ='HOLD'
            time_series.loc[index, "trading"] = classification
        elif (candle=='uptrend') and (returns=='uptrend') and (vama=='uptrend') and (ema=='uptrend') and (previous_classification=='HOLD'):
            classification ='HOLD'
            time_series.loc[index, "trading"] = classification
        elif (candle=='downtrend') and (previous_candle=='downtrend') and (returns=='downtrend') and (previous_returns=='downtrend') and (previous_classification=='HOLD') and (row['close'] > buy_price):
            classification ='SELL'
            sell_price = row['close']
            profit = ((sell_price - buy_price)*(1-maker_fee))
            accumulated_profit = (accumulated_profit+profit)
            time_series.loc[index, "profit"] = (profit + accumulated_profit)
            time_series.loc[index, "trading"] = classification
        elif (candle=='downtrend') and (previous_candle=='downtrend') and (returns=='downtrend') and (previous_returns=='downtrend') and (previous_classification=='BUY') and (row['close'] > buy_price):
            classification ='SELL'
            sell_price = row['close']
            profit = ((sell_price - buy_price)*(1-maker_fee))
            accumulated_profit = (accumulated_profit+profit)
            time_series.loc[index, "trading"] = classification
        elif previous_classification=='BUY' and classification==previous_classification:
            classification ='HOLD'
            time_series.loc[index, "trading"] = classification
        elif previous_classification=='SELL' and classification==previous_classification:
            classification=='SELL'
            time_series.loc[index, "trading"] = classification
        else:
            classification=previous_classification
            time_series.loc[index, "trading"] = classification
        if candle == 'uptrend' and previous_candle=='uptrend':
            trend_cnt = trend_cnt+1
            time_series.loc[index, "trend_cnt"] = trend_cnt
        else: 
            trend_cnt=0
            time_series.loc[index, "trend_cnt"] = trend_cnt
        if candle == 'downtrend' and previous_candle=='downtrend':
            dtrend_cnt = dtrend_cnt+1
            time_series.loc[index, "dtrend_cnt"] = dtrend_cnt
        else: 
            dtrend_cnt=0
            time_series.loc[index, "dtrend_cnt"] = dtrend_cnt

        if trend_cnt==4 and classification=='HOLD' and ((row['close'] > buy_price)>0.75):
            classification ='SELL'
            sell_price = row['close']
            profit = ((sell_price - buy_price)*(1-maker_fee))
            accumulated_profit = (accumulated_profit+profit)
            time_series.loc[index, "trading"] = classification

        time_series.loc[index, "profit"] = accumulated_profit
        previous_classification=classification
        previous_candle = candle
        previous_returns = returns
