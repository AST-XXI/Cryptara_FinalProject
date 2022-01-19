from datetime import datetime, timedelta
import os
from _functions import *
from main import Main
time = datetime.now()
import pandas as pd
import re
algo_path = r"Data/Functionality/TradingView/"
big_movers = pd.read_csv('Data/Functionality/Twitter/big_movers.csv')
big_movers_df = big_movers['Tweets']
bitcoin_df = pd.read_csv('Data/Functionality/Twitter/bitcoin.csv')
blockchain_df = pd.read_csv('Data/Functionality/Twitter/blockchain.csv')
nft_df = pd.read_csv('Data/Functionality/Twitter/nft.csv')
tickers = pd.read_csv('Data/ticker_names.csv', index_col='Ticker2')
tickers = tickers.index.values.tolist()
extension = '.csv'

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


class print_results(Main):
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
