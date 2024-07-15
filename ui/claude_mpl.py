import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
import pandas as pd
import mplfinance as mpf

class OHLCPlotter:
    def __init__(self, df):
        self.df = df
        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        self.current_start = 0
        self.ticks = 50
        self.buy_events = []
        self.sell_events = []

        # Create UI elements
        self.next_button_ax = plt.axes([0.81, 0.05, 0.1, 0.075])
        self.next_button = Button(self.next_button_ax, 'Next')
        self.next_button.on_clicked(self.next_data)

        self.prev_button_ax = plt.axes([0.7, 0.05, 0.1, 0.075])
        self.prev_button = Button(self.prev_button_ax, 'Previous')
        self.prev_button.on_clicked(self.prev_data)

        self.date_input_ax = plt.axes([0.1, 0.05, 0.2, 0.075])
        self.date_input = TextBox(self.date_input_ax, 'Start Date:', initial='YYYY-MM-DD')
        self.date_input.on_submit(self.update_date)

        # Connect keyboard events
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

        self.plot_data()

    def plot_data(self):
        self.ax.clear()
        data = self.df.iloc[self.current_start:self.current_start+self.ticks]
        mpf.plot(data, type='candle', ax=self.ax, style='charles')
        self.ax.set_title(f'OHLC Data from {data.index[0]} to {data.index[-1]}')
        
        # Plot buy and sell events
        for date, price in self.buy_events:
            if date in data.index:
                self.ax.plot(date, price, 'bo', markersize=10)
        
        for date, price in self.sell_events:
            if date in data.index:
                self.ax.plot(date, price, 'yo', markersize=10)
        
        plt.draw()

    def next_data(self, event):
        if self.current_start + self.ticks < len(self.df):
            self.current_start += self.ticks
            self.plot_data()

    def prev_data(self, event):
        if self.current_start - self.ticks >= 0:
            self.current_start -= self.ticks
            self.plot_data()

    def update_date(self, date):
        try:
            start_date = pd.to_datetime(date)
            self.current_start = self.df.index.get_loc(start_date, method='nearest')
            self.plot_data()
        except ValueError:
            print("Invalid date format. Please use YYYY-MM-DD.")

    def on_key_press(self, event):
        if event.key == 'right':
            self.move_by_tick(1)
        elif event.key == 'left':
            self.move_by_tick(-1)
        elif event.key == 'b':
            self.register_event('buy')
        elif event.key == 's':
            self.register_event('sell')

    def move_by_tick(self, direction):
        new_start = self.current_start + direction
        if 0 <= new_start < len(self.df) - self.ticks:
            self.current_start = new_start
            self.plot_data()

    def register_event(self, event_type):
        current_date = self.df.index[self.current_start + self.ticks // 2]
        current_price = self.df.loc[current_date, 'Close']
        
        if event_type == 'buy':
            self.buy_events.append((current_date, current_price))
            print(f"Buy event registered at {current_date} with price {current_price}")
        elif event_type == 'sell':
            self.sell_events.append((current_date, current_price))
            print(f"Sell event registered at {current_date} with price {current_price}")
        
        self.plot_data()

    def show(self):
        plt.show()

# Example usage:
# df = yf.download('AAPL', start='2023-01-01', end='2023-12-31')
# plotter = OHLCPlotter(df)
# plotter.show()
# Example usage:
if __name__ == "__main__":
    import yfinance as yf
    df = yf.download('AAPL', start='2023-01-01', end='2023-12-31')
    plotter = OHLCPlotter(df)
    plotter.show()