"""
Generates test data
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from swing_trader.env.data.data_model import DataModel







data = DataModel("AAPL", freqs=["daily", "weekly", "monthly"])
data.set_date_bounds("2000-01-01", "2020-12-31")

# pandas convert daily to weekly
# https://stackoverflow.com/questions/34597926/converting-daily-stock-data-to-weekly-based-via-pandas-in-python

# build daily data
PERIOD = 100
AMPLITUDE = 20

points = np.linspace(0, 1, len(data.daily))
vals = AMPLITUDE * np.sin(PERIOD * points )
plt.plot(points, vals)
plt.show()
print()


logic = {'Open'  : 'first',
         'High'  : 'max',
         'Low'   : 'min',
         'Close' : 'last',
         'Volume': 'sum'}

data.daily["Open"] = data.daily["High"] = data.daily["Close"] = data.daily["Low"] = points

from pandas.tseries.frequencies import to_offset
weekly = data.daily.resample("W").apply(logic)
weekly.index -= to_offset("6D")


monthly = data.daily.resample("M").apply(logic)

new_dates = []
for i, date in enumerate(monthly.index):
    new_date = pd.Timestamp(year=date.year, month=date.month, day=1)
    new_dates.append(new_date)
monthly["Date"] = new_dates
monthly = monthly.set_index("Date")

# f = pd.read_clipboard(parse_dates=['Date'], index_col=['Date'])
# data.daily.resample('W', loffset=offset).apply(logic)
print()