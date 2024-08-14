import os
# os.chdir("..")


from swing_trader.env.data import DataModel
# from swing_trader.env.utils import Date

data = DataModel("AAPL", freqs=["daily", "weekly", "monthly"], market="QQQ")

from swing_trader.env.utils import Date

date = Date("2023-07-19")

from swing_trader.env.states.DWM import DWMState
import random
from datetime import datetime

for _ in range(100):
    y = random.randint(2000, 2010)
    m = random.randint(1, 12)
    d = random.randint(1, 28)
    date = Date(datetime(y, m, d))

    s = DWMState(date, data, cfg={
        "column": "Volume", "log": True, "normalize": 1, "zero_center": True
    })
    print()