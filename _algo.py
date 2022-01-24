#Import libraries and dependensies
from datetime import datetime, timedelta
import os
from _functions import *
from main import Main
time = datetime.now()
import pandas as pd
import re
import numpy as np
import pandas as pd
import hvplot.pandas
from tensorflow import random
from sklearn.preprocessing import MinMaxScaler
from numpy.random import seed
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
seed(1)
random.set_seed(2)

#Read csv results, these csv files are replenished when 'Algo' pushbutton is activated
algo_path = r"Data/Functionality/TradingView/"
big_movers = pd.read_csv('Data/Functionality/Twitter/big_movers.csv')
twitter_handle = pd.read_csv('Data/Functionality/Twitter/twitter_handle.csv')
twitter_handle = (twitter_handle['0']).tolist()
big_movers_df = big_movers['Tweets']
twitter_market_sentiment = pd.read_csv('Data/Functionality/Twitter/crypto_tweet_sentiment.csv')
influencer_positive = format_convert(twitter_market_sentiment['positive'].mean())
influencer_negative = format_convert(twitter_market_sentiment['negative'].mean())
influencer_neutral = format_convert(twitter_market_sentiment['neutral'].mean())
bitcoin_df = pd.read_csv('Data/Functionality/Twitter/bitcoin.csv')
blockchain_df = pd.read_csv('Data/Functionality/Twitter/blockchain.csv')
nft_df = pd.read_csv('Data/Functionality/Twitter/nft.csv')
tickers = pd.read_csv('Data/ticker_names.csv', index_col='Ticker2')
market_cap = pd.read_csv('Data/CurrentTopTen.csv', index_col="datetime", infer_datetime_format=True, parse_dates=True)
market_cap = market_cap.sort_index()
market_cap = market_cap[['candle','adjusted']]
tickers = tickers.index.values.tolist()
extension = '.csv'

#Run lstm on most recent crypto market-cap data
predictions = run_lstm(market_cap, window_size, close_feature, close_target)
real_price = predictions['Real'][-1]
predicted_price = predictions['Predicted'][-1]
increase = percent_sign((predicted_price-real_price)/real_price)
decrease = percent_sign((real_price-predicted_price)/real_price)
predictions.tail()

#Format NLP data for printed analysis
big_string=''.join([str(item) for item in big_movers_df])
movers = re.findall('([A-Z]+)', big_string)
movers = [val for val in movers if val in tickers]
movers = list( dict.fromkeys(movers) )

recent_btc = bitcoin_df['Tweets'][0]
big_string=''.join([str(item) for item in recent_btc])
bitcoin = re.findall('([0-9]+)', big_string)
btc_price = float(bitcoin[0])
btc_onehr = (float(bitcoin[5])) + (float(bitcoin[6])/100)
btc_fivehr = (float(bitcoin[10])) + (float(bitcoin[11])/100)
btc_24hr = (float(bitcoin[15])) + (float(bitcoin[16])/100)

nft_list = []
regex_data = []
target= nft_df['Tweets'].copy()
for row in target:
    big_string=''.join([str(item) for item in row])
    words = re.findall('[A-Z][a-z]+', big_string)
    regex_data.append(words)

for row in regex_data:
    if row[0] =='Buyer':
        pass
    else:
        nft_list.append(row)

# Read results. 'Algo' push button source code: 
class algo(Main):
    print("")
    print("Big Movers in the Crypto Market:")
    print(movers)
    print("")
    print("")
    print("NFT BOT")
    print('Biggest words in the Non-Fungible Token (NFT) marketplace')
    for row in nft_list:
        print(row)
    print('')
    print('')
    print('BitcoinBot:')
    print(f'BTC Price: {dollar_sign(btc_price)}')
    print(f'1 Hour Movement: {percent_sign(btc_onehr)}')
    print(f'5 Hour Movement: {percent_sign(btc_fivehr)}')
    print(f'24 Hour Movement: {percent_sign(btc_24hr)}')
    print('')
    print('')
    print('According to Lunar Crush,')
    print('The most popular cryptocurrencies are influenced by these Twitter Handles:')
    print(twitter_handle)
    print('')
    print('Here is the sentiment of the latest tweets from these influencers:')
    print(f'Positive: {influencer_positive}')
    print(f'Negative: {influencer_negative}')
    print(f'Neutral: {influencer_neutral}')
    print('')
    print('')
    if predicted_price > real_price:
        print(f'Our Deep Learning Model suggest total crypto market-cap will increase by {increase} within 30 days.')
    else:
        print(f'Our Deep Learning Model suggest total crypto market-cap will decrease by {decrease} within 30 days.')
