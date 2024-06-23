# Introduction

Dis be a swing tradin bot

# Gameplan

- Data format
- Time series data
  - Daily
  - Weekly
  - Monthly
- Gonna use pandas dataframes for everything

I need datastreams

I want an environment that generates those data streams for each t increment (whether its daily, weekly, etc.)
It will return a dictionary of values {
  daily_hloc: 4-float
  weekly_hloc: 4-float # shows the last one
  market_daily_hloc
  volume_daily: float
  volume_weekly: 4-float
  position_owned: bool
  ticks_bought: int # number of ticks since bought
}

Env:
- returns last 50 of each value as a time series

Action Space:
- Just buy to start

Reward:
- percentage gain - market percentage gain over next 20 ticks
- minus 2 percent to discourage frivolous buying

Model:
- MLP to flush out pipes
- Switch to GRU or attention based 

Policy:
- PPO to start

Train/Test
- Pick random 800 stocks to train and sample
- Pick 200 for Val
- 200 for test