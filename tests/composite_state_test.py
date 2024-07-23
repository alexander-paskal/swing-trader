import os
# os.chdir("..")


from swing_trader.env.data import DataModel
# from swing_trader.env.utils import Date

data = DataModel("AAPL", freqs=["daily", "weekly", "monthly"], market="QQQ")

from swing_trader.env.utils import Date

date = Date("2023-07-19")

from swing_trader.env.states.DWM import DWMState
from swing_trader.env.states.composite import CompositeState
import random
from datetime import datetime

for _ in range(100):
    y = random.randint(2000, 2010)
    m = random.randint(1, 12)
    d = random.randint(1, 28)
    date = Date(datetime(y, m, d))
    
    s1 = DWMState(date, data, {"column": "Close", "normalize": 1})
    print(s1.array)
    s2 = DWMState(date, data, {"column": "Volume", "log": True, "normalize": 1})
    s = CompositeState(s1, s2)
    print()