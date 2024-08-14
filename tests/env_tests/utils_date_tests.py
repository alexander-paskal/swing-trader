from swing_trader.env.utils import Date

import pandas as pd
from datetime import datetime

s = "1999-01-01"
d = datetime(1999,1,1)
p = pd.Timestamp(1999, 1, 1)


date1 = Date(s)
date2 = Date(d)
date3 = Date(p)

assert date1 == date2 == date3 == s
assert date1 <= date2 >= date3 == d
assert date1 >= date2 <= date3 == p

s = "1999-01-02"
date4 = Date(s)

assert date4 > date1
assert date2 < date4
assert date3 <= date4
assert date4 >= date1


df = pd.read_csv("data/weekly/QQQ-weekly.csv")
date_df = df["Date"].str.split(" ", expand=True)[0]
df["Date_str"] = date_df
df["Date"] = pd.to_datetime(date_df)
df = df.set_index("Date")

d = Date("1999-03-10")
r = df.loc[d.as_timestamp]
print("Tests Successful!")

