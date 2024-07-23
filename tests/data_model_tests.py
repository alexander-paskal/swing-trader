import os
# os.chdir("..")


from swing_trader.env.data import DataModel
# from swing_trader.env.utils import Date

data = DataModel("AAPL", freqs=["daily", "weekly", "monthly"], market="QQQ")

# df1 = data.access("daily", "1999-03-15")
# df2 = data.access("weekly", "1999-03-15")
df3 = data.access("weekly", "1999-03-15", attrs=['Open'], length=5)
open_price = data.get_price_on_open("1996-03-16")

d = data.get_next_tick("daily", "1999-03-18")
d2 = data.get_next_tick("monthly", "2024-06-01")  # 7 01
d3 = data.get_n_ticks_after("weekly", "2024-05-30", 3)  # 6 24
print()

