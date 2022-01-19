# Live data requires an active TradingView and Twitter Developer account. See README for more information.
from datetime import datetime, timedelta
import os
time = datetime.now()
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
lemmatizer = WordNetLemmatizer()
from _functions import *
from main import Main
#Import dependencies
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
plt.style.use('fivethirtyeight')
load_dotenv()
cg = CoinGeckoAPI()
coins = cg.get_coins()
screener = 'CRYPTO'

#Activate environment variables
consumer_key = os.getenv("tapi_key")
consumer_secret = os.getenv("tapi_secret")
access_token = os.getenv("taccess_token")
access_secret = os.getenv("taccess_secret")
bearer_token = os.getenv("tbearer_token")

#Twitter credentials
authenticate = tweepy.OAuthHandler(consumer_key, consumer_secret)
authenticate.set_access_token(access_token, access_secret)
api = tweepy.API(authenticate, wait_on_rate_limit = True)

#Extract top 50 market cap coins
tickers = []
for i in coins:
  tickers.append(i['symbol'])
print('')
print('Extracted top fifty market cap coins from CoinGeckoAPI')
print('')

#Format prices for TradingView inputs
for i in range(len(tickers)):
  tickers[i] = tickers[i].upper()

denomination = 'USD'
tickers_denominated = [x + denomination for x in tickers]
print('Top Fifty Market Cap Cryptocurrencies')
print('')
print(tickers_denominated)

influencers = ['CryptoPriceCall','HourlyBTCUpdate','whale_alert', 'nftwhalealert']

#Extract 15 tweets from each twitter user timeline
recent_twitter_df = pd.DataFrame()
for influencer in influencers:
    recent_posts = api.user_timeline(screen_name = influencer, count=10, tweet_mode='extended')
    data = pd.DataFrame( [tweet.full_text for tweet in recent_posts] , columns=['Tweets'])
    recent_twitter_df = recent_twitter_df.append(data)

recent_twitter_df['Tweets'] = recent_twitter_df['Tweets'].apply(clean_text)

big_movers=recent_twitter_df["Tweets"][1:9]
bitcoin=recent_twitter_df["Tweets"][10:19]
blockchain=recent_twitter_df["Tweets"][20:29]
nft=recent_twitter_df["Tweets"][30:39]

big_movers.to_csv('Data/Functionality/Twitter/big_movers.csv')
bitcoin.to_csv('Data/Functionality/Twitter/bitcoin.csv')
blockchain.to_csv('Data/Functionality/Twitter/blockchain.csv')
nft.to_csv('Data/Functionality/Twitter/nft.csv')

print(f'Extracting analyst recommendations (buys and strong buys)')
print('')
from tradingview_ta import TA_Handler, Interval, Exchange
staging_df = pd.DataFrame()
binance_daily = pd.DataFrame()
coinbase_daily = pd.DataFrame()
for ticker in tickers_denominated:
    try:
        data = (TA_Handler(symbol=ticker,screener=screener,
                        exchange='BINANCE',interval=Interval.INTERVAL_1_DAY ).get_analysis().summary)
        symbol = ticker
        staging_df = list(data.values())
        final_df = (pd.DataFrame((data), index={ticker}))
        binance_daily = binance_daily.append(final_df)
    except:
        print(f'{ticker} not listed on Binance, checking Coinbase')
        pass

for ticker in tickers_denominated:
    try:
        data = (TA_Handler(symbol=ticker,screener=screener,
                        exchange='COINBASE',interval=Interval.INTERVAL_1_DAY ).get_analysis().summary)
        symbol = ticker
        staging_df = list(data.values())
        final_df = (pd.DataFrame((data), index={ticker}))
        coinbase_daily = coinbase_daily.append(final_df)
    except:
        pass
print(f'finished extracting analyst recommendations..')

# Filter our sells/strong sells from Analyst recommendations
indicators = pd.concat([coinbase_daily,binance_daily], axis='columns', join='outer')
indicators = indicators['RECOMMENDATION']
indicators.columns = ['binance', 'coinbase']
analyst_recommendations = indicators['binance'].combine_first(indicators['coinbase'])
analyst_recommendations = analyst_recommendations[~analyst_recommendations.str.contains("SELL", na=False)]
analyst_recommendations = analyst_recommendations[~analyst_recommendations.str.contains("STRONG_SELL", na=False)]
analyst_recommendations.to_csv('Data/tickers.csv')
cryptos = analyst_recommendations.index.values.tolist()
print(cryptos)

from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from tradingview_ta import TA_Handler, Interval
from tvDatafeed import TvDatafeed,Interval
import logging
import os
tradev_id = os.getenv("username")
tradev_secret_key = os.getenv("password_tv")
from tvDatafeed import TvDatafeed,Interval


#Get historical pricing information for total population, us FinTa to calculate oscillator and momentum indicators
#Run Algo strategy & Linear Regression analaysis on each crypto 
tradev_id = os.getenv("username")
tradev_secret_key = os.getenv("password_tv")
tv = TvDatafeed(tradev_id, tradev_secret_key, chromedriver_path=None)

crypto_data = pd.DataFrame()
predictions = {}
short_window = 10
long_window = 20
try:
    for tickers in cryptos:
        try:
            data = tv.get_hist(
                symbol=tickers,
                exchange='BINANCE',
                interval=Interval.in_2_hour,n_bars=10000)
            data['candle'] = (data['close'] - data['open'])
            bbands_df = TA.BBANDS(data)
            data['bb_upper'] = bbands_df['BB_UPPER']
            data['bb_middle'] = bbands_df['BB_MIDDLE']
            data['bb_lower'] = bbands_df['BB_LOWER']
            data['returns'] = data['close'].pct_change()
            data['stoch_k'] = TA.STOCH(data)
            data['stoch_d'] = TA.STOCHD(data)
            data['short_ema'] = TA.EMA(data, short_window)
            data['long_ema'] = TA.EMA(data, long_window)
            data['short_vama'] = TA.VAMA(data, short_window)
            data['long_vama'] = TA.VAMA(data, long_window)
            data.dropna(inplace=True)
            data[['exchange', 'ticker']] = data['symbol'].str.split(':',expand=True)
            data=data.drop(columns=["symbol"])
            data=data.drop(columns=["exchange"])
            clf = COPOD()
            X = data.iloc[:,:15]
            clf.fit(X)
            y_train_pred = clf.labels_
            data['anomoly'] = y_train_pred
            algo_strategy(data)
            data.to_csv(f'Data/Functionality/Algo_Bot/{tickers}.csv')
            crypto_data = crypto_data.append(data)
        except:
            data = tv.get_hist(
                symbol=tickers,
                exchange='COINBASE',
                interval=Interval.in_2_hour,n_bars=10000)
            data['candle'] = (data['close'] - data['open'])
            bbands_df = TA.BBANDS(data)
            data['bb_upper'] = bbands_df['BB_UPPER']
            data['bb_middle'] = bbands_df['BB_MIDDLE']
            data['bb_lower'] = bbands_df['BB_LOWER']
            data['returns'] = data['close'].pct_change()
            data['stoch_k'] = TA.STOCH(data)
            data['stoch_d'] = TA.STOCHD(data)
            data['short_ema'] = TA.EMA(data, short_window)
            data['long_ema'] = TA.EMA(data, long_window)
            data['short_vama'] = TA.VAMA(data, short_window)
            data['long_vama'] = TA.VAMA(data, long_window)
            data.dropna(inplace=True)
            data[['exchange', 'ticker']] = data['symbol'].str.split(':',expand=True)
            data=data.drop(columns=["symbol"])
            data=data.drop(columns=["exchange"])
            algo_strategy(data)
            data.to_csv(f'Data/Functionality/TradingView/{tickers}.csv')
            crypto_data = crypto_data.append(data)
except:
    print('You must have TradingView credentials to extract data via websocket')
print(f'Extracted prices and calculated oscillator/momentum indicator values')


top_ten_marketcap = tv.get_hist(
    symbol='CRYPTO10',
    exchange="EIGHTCAP",
    interval=Interval.in_30_minute,n_bars=5000)
top_ten_marketcap.drop(columns=['volume'], inplace=True)
top_ten_marketcap['adjusted'] = ((top_ten_marketcap['open']+top_ten_marketcap['high']+top_ten_marketcap['low']+top_ten_marketcap['close']) / 4)
top_ten_marketcap['adjusted'] = top_ten_marketcap['adjusted'].astype('int64') 
top_ten_marketcap['top_ten_marketcap'] = top_ten_marketcap['adjusted'].pct_change()
top_ten_marketcap['candle'] = (top_ten_marketcap['close'] - top_ten_marketcap['open'])
bbands_df = TA.BBANDS(top_ten_marketcap)
top_ten_marketcap['bb_upper'] = bbands_df['BB_UPPER']
top_ten_marketcap['bb_middle'] = bbands_df['BB_MIDDLE']
top_ten_marketcap['bb_lower'] = bbands_df['BB_LOWER']
top_ten_marketcap['stoch_k'] = TA.STOCH(top_ten_marketcap)
top_ten_marketcap['stoch_d'] = TA.STOCHD(top_ten_marketcap)
top_ten_marketcap['short_ema'] = TA.EMA(top_ten_marketcap, short_window)
top_ten_marketcap['long_ema'] = TA.EMA(top_ten_marketcap, long_window)
top_ten_marketcap.dropna(inplace=True)
top_ten_marketcap.to_csv('Data/CurrentTopTen.csv')