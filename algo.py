# Extract Algo & NLP results from Funcationality folders

from datetime import datetime, timedelta
import os
time = datetime.now()
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
lemmatizer = WordNetLemmatizer()
from runapp import Main
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
import os
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


def clean_text(text):
    regex = re.compile("[^a-zA-Z0-9]")
    re_clean = regex.sub(' ', text)
    words = word_tokenize(re_clean)
    return words

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

#Configure algo strategy
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