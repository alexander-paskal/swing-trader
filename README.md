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

# Ideas:
 - use of GNNS - find ways to build a heterograph of stocks in different sectors, or other form of connectivity. Then I can construct a dynamic graph of their price movements
 

 # Architectural Vision

There are 4 main components to this part of the application:

  - the env
  - the data model
  - the UI
  - The training pipeline

the intent is for these 4 things to be sufficiently factored out such that I can build apply different front ends.

A short term vision is as such:
1. Get an agent training on my simple environment in ray
2. Build a matplotlib interactive backtrading simulation with the following functionality:
  1. Pick a random stock
  2. Send the stock forward tick by tick using the keypad
  3. Place buy and sell orders using the B and S keys
  4. Annotate those using scatters
  4. Track your return over time compared to the market
3. Use the UI to visualize the Agent's results as I experiment with different models
4. Once the AI is trained in any capacity (even if it sucks), story board what a website might look like
  1. Define the backend, which will leverage yfinance to build the state
  2. Figure out how to perform inference using Ray
  3. Pick a web stack

Longer term visions:
1. Expand the UI to include comprehensive portfolio management for backtesting as well as show indicators and support algo trading
2. Expand the agent to day trading
3. Build a highly parallelized way of pulling new data for a library of stocks and storing it in a database
