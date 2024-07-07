from env import Env, ICs, Config
from ui.mpl import MPLCore
from datetime import datetime
import time




ics = ICs(
    name="AAPL",
    date=datetime(2013, 1, 1)
)

config = Config(
    state_history_length=49
)

env = Env(ics=ics, config=config)
ui = MPLCore(env)
ui.show()
while True:
    inp = input("Click X to end program, B to buy, S to sell, anything else to continue")
    if inp == "X":
        import sys
        sys.exit()
    
    elif inp == "B":
        ui.buy()
    
    elif inp == "S":
        ui.sell()
        
    ui.step()