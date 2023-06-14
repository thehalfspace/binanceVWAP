# Python Script to scrape and analyze binance trade data
Scrapes the day trade data from binance for a given coin pair.

### Installation
Make sure to have anaconda and python 3.x in your system.

Create a virtual environment with the package dependencies using: `conda env create -f test_env.yml`

Activate the virtual environment in the termial: `conda activate test_env`

### Usage

Run the script as :
`python binance_fetch_trade_data.py <coin_pair> <date>`.

E.g., for BTCUSDT on 2023/06/14, run the script:
`python binance_fetch_trade_data.py BTCUSDT 20230613`

The script outputs the csv data file and .CHECKSUM file in the data directory, and a volume-weighted average price time-series as html plot.
