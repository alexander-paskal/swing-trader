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
}

I
