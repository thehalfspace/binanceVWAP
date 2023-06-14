"""Trade Analysis Test

Download and use Binance trade data to create analyses and log files. 

Author: Prithvi Thakur
Date: 06/14/2023

"""

# Import modules
import requests
import os
import pandas as pd
from datetime import datetime
import zipfile
import io
import logging
import argparse
import time
import plotly.express as px

# Inputs
# Coin pair format: string ''
#coin_pair = 'BTCUSDT'

# Date format: string 'YYYYMMDD'
#date_var = '20230327'

# Get terminal inputs from arguments parser
parser = argparse.ArgumentParser()
parser.add_argument('coin_pair', type=str)
parser.add_argument('date', type=str)

args = parser.parse_args()
coin_pair = args.coin_pair
date_var = args.date

# Create an output log file
output_log_fname = coin_pair+date_var+'.log'

# If the log file already exists, delete it
if os.path.isfile(output_log_fname):
    os.remove(output_log_fname)

logging.basicConfig(filename= coin_pair+date_var+'.log', 
                    filemode='a', 
                    format='%(asctime)s %(message)s', 
                    level=logging.INFO)

# Create a logger object
logger = logging.getLogger()
logger.info('LOGFILE:\nStarting the script!')


# Data directory to save files
data_dir = os.path.join(os.path.curdir, 'Data')

# Create the data directory if it doesn't exist
if not os.path.exists(data_dir):
    logger.info("\nThe data directory doesn't exist! Creating one now:")
    os.makedirs(data_dir)

# Url to fetch data
trade_url = 'https://data.binance.vision/data/spot/daily/trades'
aggTrade_url = 'https://data.binance.vision/data/spot/daily/aggTrades'

# Column names for trade and aggTrade
trade_columns = ['TradeID', 'Price', 'Quantity', 'QuoteQuantity', 'TradeTime', 'BuyerMarketMaker', 'BestPriceMatch']
aggTrade_columns = ['AggTradeID', 'Price', 'Quantity', 'FirstTradeID', 'LastTradeID', 'Timestamp', 'BuyerMarketMaker', 'BestPriceMatch']


###########################
# FUNCTIONS DEFINITIONS
###########################

# Fetch data from url and download to directory
def getData(url, coin_pair, date_var, data_dir=data_dir):
    """Fetches data from the url and saves data as zip and csv
        Also returns the full url useful for reading the data later
        
        Input: url: can be trade_url or aggTrade_url
               coin_pair: e.g., BTCUSDT
               date_var: date variable as string
    """
    date_str = checkDate(date_var)
    
    # Check if trade or aggTrade and assign appropriate filenames
    fname = ''
    if url.split('/')[-1] == 'trades':
        fname = coin_pair + '-trades-' + date_str + '.zip'
    elif url.split('/')[-1] == 'aggTrades':
        fname = coin_pair + '-aggTrades-' + date_str + '.zip'
    else:
        ValueError('Check input URL:')
    
    # Full url to download data
    full_url = url + '/' + coin_pair + '/' + fname
    full_url_checksum = url + '/' + coin_pair + '/' + fname + '.CHECKSUM'
    
    logger.info('\nFetching data from: '+full_url)
    
    # Get the url contents
    response = requests.get(full_url)
    response_checksum = requests.get(full_url_checksum)
    
    # Extract zipfiles to data directory
    z = zipfile.ZipFile(io.BytesIO(response.content))
    z.extractall(data_dir)
    
    open(os.path.join(data_dir, fname+'.CHECKSUM'), 'wb').write(response_checksum.content)
    
    logger.info('\nDownloaded the data from the above url!')
    return  full_url

# Read the trade data as a pandas dataframe
def readData(url, coin_pair, date_var, columns):
    """This function reads the data as a pandas dataframe
        and returns the dataframe
    """
    full_url = getData(url, coin_pair, date_var)
    response = requests.get(full_url)

    # Read data as pandas dataframe. The raw file doesn't have
    # headers, so assign columns manually
    df = pd.read_csv(io.BytesIO(response.content),
                 header=None,
                 names=columns,
                 compression='zip',
                 sep=',',
                 quotechar='"')
    return df

# Basic check of trade data
def checkTradeData(df):
    logger.info('\n\nTrade data basic description:')
    # Print the datatypes of the trade dataframe to logfile
    logger.info('Trade data types:')
    logger.info(df.dtypes.to_string())
    
    # If duplicate data present, list the duplicate data
    if df.duplicated().sum() > 0:
        logger.info('Duplicate data found. Listing them below:')
        logger.info(df[df.duplicated()].to_string())
        
    else:
        logger.info('Checked for duplicate data. No duplicates found!')

    # Print out the basic statistics
    basic_stats = df[['Price', 'Quantity']].describe()
    logger.info('Total number of trades: %s', str(df.shape[0]))
    logger.info('Mean Price of trades: %s', str(basic_stats.loc['mean', 'Price']))
    logger.info('Mean Quantity of trades: %s', str(basic_stats.loc['mean', 'Quantity']))
    
    logger.info('\nMin. Price of trades: %s', str(basic_stats.loc['min', 'Price']))
    logger.info('Min. Quantity of trades: %s', str(basic_stats.loc['min', 'Quantity']))
    
    logger.info('\nMax Price of trades: %s', str(basic_stats.loc['max', 'Price']))
    logger.info('Max Quantity of trades: %s', str(basic_stats.loc['max', 'Quantity']))

    logger.info('Finished checking the trade data. Looks Good!!')

# Basic check of aggTrade data
def checkAggTradeData(df):
    logger.info('\n\nAggTrade data basic description:')
    # Print the datatypes of the trade dataframe to logfile
    logger.info(df.dtypes)
    
    # If duplicate data present, list the duplicate data
    if df.duplicated().sum() > 0:
        logger.info('Duplicate data found. Listing them below:')
        logger.info(df[df.duplicated()].to_string())
    else:
        logger.info('Checked for duplicate data. No duplicates found!')

    # Print out the basic statistics
    logger.info('Total number of aggTrades: %s', str(df.shape[0]))
    logger.info('Finished checking the aggTrades data. Looks Good!!')

# Generate result_trades.txt file    
def resultTrades_toFile(trade_df, aggTrade_df, data_dir=data_dir):
    # Difference between adjaent trade ids
    trade_id_diff = trade_df['TradeID'].diff()
    
    # If any trade_id_diff is greater than 1, we have missing values
    missing_trade_ids = trade_id_diff[trade_id_diff > 1.0]
    
    # Get existing trade ids as a python set
    tid_in_data = set(trade_df['TradeID'].tolist())
    
    # Get a list of continuous trade IDs from minimum 
    # value of tid_in_data to max. value of tid_in_data
    # Store as set
    cont_tid =  set(range(min(tid_in_data), max(tid_in_data)))
    
    # Get the missing trade ids as list using set operation differenec
    missing_trade_ids = list(cont_tid - tid_in_data)
    missing_trade_ids_sorted = sorted(missing_trade_ids)
    
    
    logger.info('\n\nCreating results_trade.txt file.')
    
    # Open output file for writing
    with open('result_trades.txt', 'w') as f:
    
        # Check if the missing_trade_ids list is empty
        if not missing_trade_ids_sorted:
            logger.info('There are no missing Trade IDs!!!')
            f.write('There are no missing Trade IDs!!\n\n')
        else:
            logger.info('Missing TradeIDs found. Saving to file ..')
            
            # get missing ids as string to write to file
            f.write('Missing Trade IDs:\n')
            f.write('\n'.join([str(i) for i in missing_trade_ids_sorted]))
            
        # List out the trades aggregated by aggTrades
        # If first and last trade id are same, that trade
        # is not aggregated, skip those
        trade_id_aggregated = aggTrade_df['LastTradeID'] - aggTrade_df['FirstTradeID']
        aggregated_trades = aggTrade_df[trade_id_aggregated > 0]
        
        # Loop over aggregated trades and write to file
        if aggregated_trades.empty:
            logger.info('No aggregated trades found. Please check the AggTrades file!!!')    
            f.write('\nNo aggregated trades found. Please check the AggTrades file!')
        else:
            # Set TradeID as index for easier querying
            trade_df_query = trade_df.set_index('TradeID')
            
            f.write('\nListing the Aggregated Trades below:')
            
            # Write Header
            f.write('\n\nHeaders:\nTradeID Price Quantity QuoteQuantity TradeTime BuyerMarketMaker BestPriceMatch')
            
            for idx, row in aggregated_trades.iterrows():
                first_id = row['FirstTradeID']
                last_id = row['LastTradeID']
                trades = trade_df_query.loc[first_id:last_id].reset_index()

                # Write aggtradeID
                f.write('\n\nAggTradeID: '+ str(row['AggTradeID'])+'\n')
                
                # Write the trade details
                f.write(trades.to_string(index=False, header=False))

    f.close()
    
# Generate results_trade_bar.csv file
def resultsTradeBar(trade_df, data_dir=data_dir):
    """This function creates the results trade bar csv file.
        Also returns the results_trade_bar as dataframe used for
        plotting.
    """ 
    columns = ['Time', 'Open', 'High', 'Low', 'Close', 'Qty', 'Number', 'VWAP']

    logger.info('\n\nCreating results_trade_bars file!')
    
    # The trade data is already sorted by Trade Time
    
    # Trade times are in milliseconds from epoch time
    # Map to readable time (YYYY-MM-DD HH:MIN:SEC) and add that as a new column
    trade_df['formattedTradeTime'] = trade_df['TradeTime'].map(lambda x: pd.to_datetime(x, unit='ms'))
    
    # Resample the trade data into 1 minute intervals
    # We are grouping by closed left interval, i.e., tStart <= trade < tEnd
    try:
        resampled_trade = trade_df.resample('min', on='formattedTradeTime', closed='left')
    except:
        logger.info('Unable to group trade data per minute and generate results_trade_bars. Please check the trade data type!')
        ValueError('Unable to group trade data per minute and generate results_trade_bars. Please check the trade data type!')
    
    # Create an output dataframe (result_trade_bars) to save output
    result_trade_bars = pd.DataFrame(index=range(len(resampled_trade)), columns=columns)
    
    # Save the start time of trade
    result_trade_bars['Time'] = resampled_trade.Price.first().index

    # Open = Price of first trade
    result_trade_bars['Open'] = resampled_trade.Price.first().values
    
    # High = Max. Price of trades in bars
    result_trade_bars['High'] = resampled_trade.Price.max().values
    
    # Low = Min. Price of trades in bars
    result_trade_bars['Low'] = resampled_trade.Price.min().values
    
    # Close = Price of last trade by timestamp
    result_trade_bars['Close'] = resampled_trade.Price.last().values
    
    # Qty = Quantity (sum of shares traded)
    result_trade_bars['Qty'] = resampled_trade.Quantity.sum().values
    
    # Number = Number of Trades (count)
    result_trade_bars['Number'] = resampled_trade.Quantity.count().values
    
    # VWAP = Weighted average = [Price x Volume]/Total_Volume
    # The column QuoteQuantity of trade data already gives Price x Quantity
    # So, in this case, VWAP = sum(QuoteQuantity)/sum(Quantity)
    result_trade_bars['VWAP'] = (resampled_trade.QuoteQuantity.sum()/resampled_trade.Quantity.sum()).values
    
    # Save as csv
    result_trade_bars.to_csv('result_trade_bars.csv', sep=',', index=False)

    logger.info('\nFinished writing results_trade_bars.csv file!')
    
    return result_trade_bars

# VWAP history plot: time vs VWAP
def plotVwap(df, coin_pair, date_var):
    """Reads the input dataframe results_trade_bar (df),
    and saves an interactive html plot in the current directory.
    """
    logger.info('Creating VWAP Time-History plot.')
    fig = px.line(df, x='Time', y='VWAP', hover_data=['VWAP'],
        width=1200, height=675,
        labels={
            'Time':'Time',
            'VWAP':'Weighted Average Price (VWAP)',
        })
    fig.update_layout(
        title={
            'text': 'Time-Series of VWAP Trade for '+coin_pair+' on '+checkDate(date_var),
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'})

    fig.write_html('VWAP_vs_Time.html')

# Check input date (date_var) validity
def dateError(date_var):
    print('Please check the date format you entered. \
        \nExpecting: YYYYMMDD as string \
        \nYou Entered: ',date_var, ' as', type(date_var))

# Format date_var to match with the url
def checkDate(date_var):
    """Check validity of datetime here"""
    try:
        dt = datetime.strptime(date_var, '%Y%m%d')
        print(dt)
        return dt.strftime('%Y-%m-%d')
    except:
        raise ValueError(dateError(date_var))

# Extra function: doing nothing for now.
# Can use this to test the input coin_pair
# against all coin pairs listed on Binance.
def checkCoinPair(coin_pair):
    """I do not have a list of coin pairs to check against"""
    return None

# Main code: run the above functions
def main(trade_url, aggTrade_url, coin_pair, date_var,
         trade_columns, aggTrade_columns, data_dir):
    
    # Save the code runtime to log file
    start_time = time.time()
    
    # Get trade data: save to directory and read as dataframe
    trade_df = readData(trade_url, coin_pair, date_var, trade_columns)
    aggTrade_df = readData(aggTrade_url, coin_pair, date_var, aggTrade_columns)
    
    # Basic checks for data
    checkTradeData(trade_df)
    checkAggTradeData(aggTrade_df)
    
    # Generate results_trade.txt file
    resultTrades_toFile(trade_df, aggTrade_df)
    
    # Generate results_trade_bar.csv file
    results_trade_bar = resultsTradeBar(trade_df)
    
    # Plot and save VWAP time series
    plotVwap(results_trade_bar, coin_pair, date_var)
    
    # End time
    end_time = time.time()
    logger.info('\n\nEnd of Script.\nExecution time: %s seconds', str(end_time-start_time))
    
# %%
##########################
# Run the main code here
##########################

# Run the code
if __name__ == "__main__":
    main(trade_url, aggTrade_url, coin_pair, date_var, 
        trade_columns, aggTrade_columns, data_dir)



# %%
