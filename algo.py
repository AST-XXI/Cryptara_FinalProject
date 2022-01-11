# Extract Algo & NLP results from Funcationality folders

from datetime import datetime, timedelta
import os
time = datetime.now()
import pandas as pd
from runapp import Main
#Import dependencies
from tradingview_ta import TA_Handler, Interval
from finta import TA
from pyod.models.copod import COPOD
import pandas as pd 
import nltk
import hvplot.pandas
import time
import numpy as np
from pycoingecko import CoinGeckoAPI
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import requests
from requests import Request, Session
from requests.exceptions import ConnectionError, Timeout, TooManyRedirects
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
from datetime import datetime, timedelta
from ibm_watson import ToneAnalyzerV3
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
import os
import json
from pandas import json_normalize
from ibm_watson import ToneAnalyzerV3
import re
import json
from pandas import json_normalize
import os
import ast
import tweepy
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
load_dotenv()
cg = CoinGeckoAPI()
coins = cg.get_coins()
screener = 'CRYPTO'

#Activate environment variables
consumer_key = os.getenv("consumer_key")
consumer_secret = os.getenv("consumer_secret")
access_token = os.getenv("access_token")
access_secret = os.getenv("access_secret")
tone_api = os.getenv("tone_api")
bearer_token = os.getenv("bearer_token")
consumer_key = os.getenv("consumer_key")
consumer_secret_key = os.getenv("consumer_secret")


#Twitter & Reddit credentials
reddit_id = os.getenv("reddit_id")
reddit_key = os.getenv("reddit_key")
data = ast.literal_eval(os.getenv("reddit_data"))
auth = requests.auth.HTTPBasicAuth(reddit_id, reddit_key)
headers = {'User-Agent': 'TaraRedditApi/0.0.1'}

res = requests.post('https://www.reddit.com/api/v1/access_token',
                   auth=auth, data=data, headers=headers)

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

#Grab indicator recommendations from Trading View API
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

#Configure Linear Regression analysis
def linear_regression_analysis(data):
    time_series = data.copy()
    projection = 30
    time_series['prediction'] = time_series[['close']].shift(-projection)
    X = np.array(time_series[['close']])
    X = X[:-projection]
    scaler = StandardScaler()
    y = time_series['prediction'].values
    y = y[:-projection]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20, shuffle=False)
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    lr_confidence = lin_reg.score(X_test, y_test)
    x_projection = np.array(time_series[['close']])[-projection:]
    linear_prediction = lin_reg.predict(x_projection)
    linear_pred_df = pd.DataFrame(linear_prediction)
    forecasted_trend = linear_pred_df.hvplot.line( 
        label=f'Forecasted trend of {tickers}, Confidence Score;{lr_confidence}', rot=90
    ).opts(yformatter="%.0f")
    hvplot.save(forecasted_trend, f"Data/Functionality/Linear_Regression/linear_model_predictions_{tickers}.html")
    return linear_pred_df

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
import logging
from tvDatafeed import TvDatafeed,Interval
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
            prediction = linear_regression_analysis(data)
            predictions[tickers] = prediction
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
            clf = COPOD()
            X = data.iloc[:,:15]
            clf.fit(X)
            y_train_pred = clf.labels_
            data['anomoly'] = y_train_pred
            algo_strategy(data)
            prediction = linear_regression_analysis(data)
            predictions[tickers] = prediction
            data.to_csv(f'Data/Functionality/Algo_Bot/{tickers}.csv')
            crypto_data = crypto_data.append(data)
except:
    print('ERROR: Check for websocket upgrade, ensure dependencies in requirements.txt are installed, or try CONDA UPDATE CONDA')
print(f'Extracted prices and calculated oscillator/momentum indicator values')


#Isolate individual ticker data from main dataframe, visualize sample
dataset = crypto_data.copy()
isolation = dict()
for k, v in dataset.groupby('ticker'):
    isolation[k] = v

single_ticker = isolation[cryptos[0]].copy()
relocation = single_ticker.pop("profit")
ticker = single_ticker.pop("ticker")
single_ticker.insert(0, "Algo_Profit", relocation)
single_ticker.insert(1, "Crypto", ticker)
single_ticker.tail(3)


reddit_path = r"Data/Functionality/Reddit/"
algo_path = r"Data/Functionality/Algo_Bot/"
twitter_sentiment = pd.read_csv('Data/Functionality/Twitter/subjectivity_and_polarity.csv')
extension = '.csv'

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

file_names = []
reddit_data = {}
for root, dirs_list, files_list in os.walk(reddit_path):
    for file_name in files_list:
        if os.path.splitext(file_name)[-1] == extension:
            file_name_path = os.path.join(root, file_name)
            data = pd.read_csv(file_name_path)
            reddit_data[file_name] = data
            file_names.append(file_name)

compound_scores = []
for crypto_sentiment in file_names:
    positive = (reddit_data[crypto_sentiment]['compound'][1])
    compound_scores.append(positive)
compound_sentiment = Average(compound_scores)

#Algo strategy results
algo_crypto = []
investment_algorithm = {}
for root, dirs_list, files_list in os.walk(algo_path):
    for file_name in files_list:
        if os.path.splitext(file_name)[-1] == extension:
            file_name_path = os.path.join(root, file_name)
            data = pd.read_csv(file_name_path)
            investment_algorithm[file_name] = data
            algo_crypto.append(file_name)

#Google Results
google_corr = pd.read_csv('Data/Functionality/Google/Correlation.csv', index_col='Unnamed: 0')
google_sent = pd.read_csv('Data/Functionality/Google/Sentiments.csv')
gcrypto_trends=google_sent['crypto_choice_avg'].mean()
ginflation_trends=google_sent['inflation_headlinese_avg'].mean()
genergy_trends=google_sent['energy_consumption_avg'].mean()

total_profit = []
for algo in algo_crypto:
    total_returns = (investment_algorithm[algo]['profit'].iloc[-1])
    total_profit.append(total_returns)
total_profit = sum(total_profit)

#Twitter results
twitter_sentiment = pd.read_csv('Data/Functionality/Twitter/subjectivity_and_polarity.csv')
sentiment_counts = twitter_sentiment["Analysis"].value_counts('Positive')
subjectivity = format_convert(twitter_sentiment['Subjectivity'].mean())
polarity = format_convert(twitter_sentiment['Polarity'].mean())
try:
    positive_posts = format_convert(sentiment_counts['Positive'])
except:
    pass
try:
    neutral_posts = format_convert(sentiment_counts['Neutral'])
except:
    pass
try:
    negative_posts = format_convert(sentiment_counts['Negative'])
except:
    pass

twitter_market_sentiment = pd.read_csv('Data/Functionality/Twitter/market_sentiment_analysis.csv')
market_sentiment_counts = twitter_market_sentiment["Analysis"].value_counts('Positive')
market_subjectivity = format_convert(twitter_market_sentiment['Subjectivity'].mean())
market_polarity = format_convert(twitter_market_sentiment['Polarity'].mean())

try: 
    market_positive_posts = format_convert(market_sentiment_counts['Positive'])
except:
    pass
try:
    market_neutral_posts = format_convert(market_sentiment_counts['Neutral'])
except:
    pass
try:
    market_negative_posts = format_convert(market_sentiment_counts['Negative'])
except:
    pass
print('')
print('')
print('Import print_results()')

class print_results(Main):
    print('')
    print('')
    print(f'The top picks have a {compound_sentiment} Average Compound Score on Reddit at {time}.') 
    print('')
    print('')
    print('')
    print('')
    print('')
    print(f'Overall Crypto Market Sentiment: Twitter scanned at {time}')
    print('')
    print(f'{market_positive_posts} of their posts have a positive tone.')
    print('')
    print(f'{market_negative_posts} of their posts have a negative tone.')
    print('')
    print(f'The remaining {market_neutral_posts} of their posts have a neutral tone.')
    print('')
    print('')
    print('')
    print('')
    print('')
    print('')
    print(f'Top Influential Icons in the Cryptocurrency market: Twitter scanned at {time}')
    print('')
    print(f'{positive_posts} of their posts have a positive tone.')
    print('')
    print(f'{negative_posts} of their posts have a negative tone.')
    print('')
    print(f'The remaining {neutral_posts} of their posts have a neutral tone.')
    print('')
    print('')
    print('') 
    print('')
    print('')
    print('')
    print('')
