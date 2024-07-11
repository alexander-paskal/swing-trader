import yfinance as yf
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("ticker", help="ticker to download")
parser.add_argument("period", help="period of interest i.e. 1d, 5d")
parser.add_argument("-o","--output", help="directory to save to", default="/mnt/c/Users/alexc/Projects/swing-trader/data")
parser.add_argument("-t","--timeframe", help="timeframe to get i.e. 1y", default="1y")

DATA_MAP = {
    "1d": "daily",
    "5d": "weekly"
}


args = parser.parse_args()
print(args.ticker)
print(args.period)
print(args.output)

import os
dirpath = os.path.join(args.output, DATA_MAP[args.period]) 
if not os.path.exists(dirpath):
    os.mkdir(dirpath)

fpath = os.path.join(dirpath, f"{args.ticker}-{DATA_MAP[args.period]}.csv")
print("Acquiring data at", fpath)
ticker = yf.Ticker(args.ticker)
if not os.path.exists(fpath):
    ticker = yf.Ticker(args.ticker)
    history = ticker.history(interval=args.period, period=args.timeframe)
    history.to_csv(fpath)
else:
    df = pd.read_csv(fpath)
    last_date = df["Date"].iloc[-1].split(" ")[0]
    history = ticker.history(interval=args.period, start=last_date)
    big_df = pd.concat([df, history])
    history = big_df.drop_duplicates().dropna()
    history.to_csv(fpath, index=False)
    

