"""
Data Model contains all the data needed for the environment
"""
import pandas as pd
import os
from typing import *
from datetime import datetime
from swing_trader.env.utils import Date, weekdays_after


__all__ = ['DataModel', 'NoDataException']

class NoDataException(Exception):
    pass

class DataModel:
    ticker: str
    daily: pd.DataFrame
    weekly: pd.DataFrame
    monthly: pd.DataFrame

    data_path = "data"  # from root

    def __init__(self, ticker: os.PathLike, freqs: List[str], market: str = None):

        
        if market is not None:
            self.market = self.__class__(ticker=market, freqs=freqs)
        
        self.ticker = ticker
        
        for f in freqs:
            df = pd.read_csv(self._csv_path(ticker, f))

            if df.empty:
                raise NoDataException(f"No data! {ticker} - {f}")
            df = self._clean(df)
            setattr(self, f, df)
        
    
    def _csv_path(self, ticker: str, freq: str) -> os.PathLike:
        return os.path.join(self.data_path, freq, f"{ticker}-{freq}.csv")

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
        df = df.dropna()

        df = df[df["Open"] != 0]
        df = df[df["Close"] != 0]
        
        date_df = df["Date"].str.split(" ", expand=True)[0]
        df["Date_str"] = date_df
        df["Date"] = pd.to_datetime(date_df)
        df = df.set_index("Date")
        return df
    
    def access(self, freq: str, date: Union[Date, datetime, pd.Timestamp, str], attrs: Optional[List[str]] = None, length: Optional[int] = None) -> Tuple[Dict, List[Dict]]:
        """
        Access the latest tick(s) of the frequency data based on date
        """
        date = Date(date)
        df = getattr(self, freq)

        df = df[df.index <= date.as_timestamp]

        if attrs is None:
            attrs = df.columns

        df = df[attrs]

        if length is None:
            length = 1
        
        df = df.iloc[-length:, :]

        return df

    def get_price_on_open(self, date: Date) -> float:
        """
        Get the open price on the next tick following a given tick
        """
        next_tick = weekdays_after(date, 1)
        df = self.access(
            freq="daily",
            attrs=["Open"],
            date=next_tick,
            length=1
        )
        return df.iloc[0, 0]

    def get_price_on_close(self, date: Date) -> float:
        """
        Get the price at the close
        """
        df = self.access(
            freq="daily",
            attrs=["Close"],
            date=date,
            length=1
        )
        return df.iloc[0, 0]


    def get_next_tick(self, freq: str, date: Date) -> Date:
        """Get the next date at a given frequency"""
        return self.get_n_ticks_after(freq, date, 1)
    
    def get_n_ticks_after(self, freq: str, date: Date, n: int) -> Date:
        """Get the date N ticks later"""
        df = getattr(self, freq)
        i1 = list(df.index).index(Date(date).as_timestamp)
        i2 = i1 + n
        ts2 = df.index[i2]
        return Date(ts2)
    
    def get_date_bounds(self, freq: Optional[str] = None) -> Tuple[Date, Date]:

        if freq is None:
            freqs = ['daily', 'weekly', 'monthly']
        else:
            freqs = [freq]

        maxs, mins = [], []
        for freq in freqs:
            if hasattr(self, freq):
                df = getattr(self, freq)
                
                if df.empty:
                    continue
                df_dates = [d for d in df.index]
                maxs.append(max(df_dates))
                mins.append(min(df_dates))
        
        return Date(max(mins)), Date(min(maxs))
    
    def buy_and_hold(self, start: Date, end: Date) -> float:
        return self.get_price_on_close(end) / self.get_price_on_open(start)

    def end_date(self) -> Date:
        return self.get_date_bounds()[1]