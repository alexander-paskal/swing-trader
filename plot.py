"""
A Stock charting animation, with buys and sells
"""
from env import Transaction, Env
from typing import List
import yfinance
transactions: List[Transaction] = []
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd



sns.set()

import json
with open("env.json") as f:
    env = json.load(f)
name = env["name"]
start = env["start_date"].split(" ")[0]
end = env["end_date"].split(" ")[0]
data = yfinance.download(
    tickers=name,
    start=start,
    end=end,
    period="1d"
)
print()
sns.set_style("ticks")
sns.lineplot(data=data,x="Date",y='Open',color='firebrick', alpha=0.3)
sns.despine()
plt.title(name,size='x-large',color='blue')

from dateutil import parser
import numpy as np
for transaction in env["history"]:
    t: Transaction = transaction
    
    d1 = parser.parse(t["buy_date"])
    d2 = parser.parse(t["sell_date"])
    
    trans_data = data[np.logical_or(data.index == d1, data.index ==d2)]
    sns.lineplot(data=trans_data, x="Date", y="Open", color="green")
    print()
# plt.legend(False)

plt.show()