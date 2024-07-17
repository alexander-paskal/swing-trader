import pandas as pd


class Indicators:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def sma(self, period: int) -> pd.Series:
        raise NotImplementedError
    
    def ema(self, period: int) -> pd.Series:
        raise NotImplementedError
    
    def macd_histogram(self, a: int, b: int, c: int) -> pd.Series:
        raise NotImplementedError
    