"""
Implements a core UI based on matplotlib and seaborn
"""
from env import Env, Action
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
sns.set_theme()
from datetime import datetime
from typing import *


class MPLCore:

    env: Env
    to_buy: bool
    to_sell: bool
    buy_points: List[datetime]
    sell_points: List[datetime]

    def __init__(self, env: Env):
        self.env = env
        self.to_buy = False
        self.to_sell = False
        self.buy_points = [False for _ in self.env.raw_records]
        self.sell_points = [False for _ in self.env.raw_records]

    def show(self):
        plt.cla()
        start_ind = max([self.env.raw_index+self.env.ind-self.env.state_history_length, 0])
        end_ind = self.env.raw_index+self.env.ind


        windowed_df = self.env.raw_df[start_ind: end_ind]
        
        if windowed_df.empty:
            raise ValueError("Windowed DF empty!")
        title = f"{self.env.ics['name']}\nYour Performance:{round(self.env.multiplier * self.env.hold_multiplier, 2)}"
        title += f"\nMarket Performance:{round(self.env.market_multiplier(), 2)}"
        
        sns.lineplot(data=windowed_df,x="Date",y='Open',color='firebrick', alpha=0.3)
        sns.despine()
        plt.title(title,size='x-large',color='blue')
        self.ax = plt.gca()
        print("Date: ", windowed_df.head(1))

        windowed_buys = self.buy_points[start_ind: end_ind]
        windowed_sells = self.sell_points[start_ind: end_ind]

        buy_df = windowed_df[windowed_buys]
        if not buy_df.empty:
            sns.scatterplot(buy_df, x="Date", y="Open", color="teal")
        sell_df = windowed_df[windowed_sells]
        if not sell_df.empty:
            sns.scatterplot(sell_df, x="Date", y="Open", color="orange")
        

        plt.draw()
        plt.pause(0.01)



    def step(self):
        """
        step the chart forward one tick
        """
        action = Action(
            buy=False,
            sell=False
        )
        if self.to_buy:
            action['buy'] = True
            self.to_buy = False
        
        if self.to_sell:
            action['sell'] = True
            self.to_sell = False
        
        results = self.env.step(Action.serialize(action))
        self.env.print_summary()
        self.show()
        return results
    
    def buy(self):
        """
        record a buy at the current point
        """
        print("Bought!")
        self.to_buy = True
        self.buy_points[
            self.env.raw_index + self.env.ind + 1
        ] = True
    
    def sell(self):
        """
        record a sell at the current point
        """
        print("Sold!")
        self.to_sell = True
        self.sell_points[
            self.env.raw_index + self.env.ind + 1
        ] = True
    
    
