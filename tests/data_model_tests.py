import os
# os.chdir("..")


from swing_trader.env.data import DataModel
# from swing_trader.env.utils import Date

data = DataModel("AAPL", freqs=["daily", "weekly", "monthly"], market="QQQ")

# df1 = data.access("daily", "1999-03-15")
# df2 = data.access("weekly", "1999-03-15")
df3 = data.access("weekly", "1999-03-15", attrs=['Open'], length=5)
print()

