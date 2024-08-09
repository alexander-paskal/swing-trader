from swing_trader.env.states import DWMState, CompositeState
from swing_trader.env.utils import Date
from swing_trader.env.data.data_model import DataModel


class CloseVolumeState(CompositeState):

    timeframe = "daily"

    def __init__(self, date: Date, data: DataModel, ticks: int):

        s1 = DWMState(date, data, {"column": "Close", "normalize": True, "ticks": ticks})
        s2 = DWMState(date, data, {"column": "Volume", "log": True, "divide": 10, "ticks": ticks})
        
        super().__init__(s1, s2)