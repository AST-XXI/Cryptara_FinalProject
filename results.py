# Extract Algo & NLP results from Funcationality folders

from datetime import datetime, timedelta
import os
from runapp import Main
time = datetime.now()
import pandas as pd
import re
#Import dependencies
algo_path = r"Data/Functionality/TradingView/"
big_movers = pd.read_csv('Data/Functionality/Twitter/big_movers.csv')
big_movers_df = big_movers['Tweets']
bitcoin_df = pd.read_csv('Data/Functionality/Twitter/bitcoin.csv')
blockchain_df = pd.read_csv('Data/Functionality/Twitter/blockchain.csv')
nft_df = pd.read_csv('Data/Functionality/Twitter/nft.csv')
tickers = pd.read_csv('Data/ticker_names.csv', index_col='Ticker2')
tickers = tickers.index.values.tolist()
extension = '.csv'

def Average(lst):
    return sum(lst) / len(lsbt)

def Sum(lst):
    return sum(lst) / len(lst)

def format_convert(x):
    try:
        return "{:.0%}".format(x)
    except:
        return "{:.0%}".format(float(x))

def dollar_sign(x):
    return "${:,.2f}".format(x)

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


# #Algo strategy results
# algo_crypto = []
# investment_algorithm = {}
# for root, dirs_list, files_list in os.walk(algo_path):
#     for file_name in files_list:
#         if os.path.splitext(file_name)[-1] == extension:
#             file_name_path = os.path.join(root, file_name)
#             data = pd.read_csv(file_name_path)
#             investment_algorithm[file_name] = data
#             algo_crypto.append(file_name)

# total_profit = []
# for algo in algo_crypto:
#     total_returns = (investment_algorithm[algo]['profit'].iloc[-1])
#     total_profit.append(total_returns)
# total_profit = sum(total_profit)

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
    print('Biggest names in the NFT space')
    print('')
    for row in nft_list:
        print(row)
    print('')
    print('')
