import datetime
from typing import Any
import pandas as pd
from typing import *
from datetime import datetime, timedelta
from typing_extensions import Self
from pandas.tseries.holiday import USFederalHolidayCalendar

HOLIDAYS = USFederalHolidayCalendar().holidays(start=datetime(1970,1,1), end=datetime.today())

__all__ = ["Date", "weekdays_after"]

class Date:
    """
    Utility class for representing date information. Simplifies indexing and equality 
    Expects 
    """
    year: int
    month: int
    day: int

    format_string: str = "%Y-%m-%d"
    
    _supported_types = {
        "datetime": datetime,
        "timestamp": pd.Timestamp,
        "string": str
    }

    def __init__(self, arg: Union[Self,datetime, pd.Timestamp, str], *, format_string: str = "%Y-%m-%d"):
        
        self.month = None
        self.year = None
        self.day = None
        # self.format_string = format_string

        if isinstance(arg, self.__class__):
            self.__dict__.update(arg.__dict__)

        elif isinstance(arg, datetime):
            self._parse_datetime(arg)

        elif isinstance(arg, pd.Timestamp):
            self._parse_timestamp(arg)
        
        elif isinstance(arg, str):
            self._parse_string(arg, format_string)

        else:
            raise TypeError(f"Cannot parse date from type {type(arg)}")

    def __str__(self): return self.as_string

    def __repr__(self): return f"Date({self.as_string})"

    def __eq__(self, other: Union[Self, datetime, pd.Timestamp, str]) -> bool:

        if not isinstance(other, self.__class__):
            other = self.__class__(other)
        
        return self.as_datetime == self.__class__(other).as_datetime
    
    def __req__(self, other: Union[Self, datetime, pd.Timestamp, str]) -> bool:
        return self.__eq__(other)
    
    def __lt__(self, other: Union[Self, datetime, pd.Timestamp, str]) -> bool:

        if not isinstance(other, self.__class__):
            other = self.__class__(other)
        
        return self.as_datetime < other.as_datetime
    
    def __rlt__(self, other: Union[Self, datetime, pd.Timestamp, str]) -> bool:
        return self.__lt__(other)
    
    def __le__(self, other: Union[Self, datetime, pd.Timestamp, str]) -> bool:

        if not isinstance(other, self.__class__):
            other = self.__class__(other)
        return self.as_datetime <= other.as_datetime
    
    def __rle__(self, other: Union[Self, datetime, pd.Timestamp, str]) -> bool:
        return self.__le__(other)
    
    def __ge__(self, other: Union[Self, datetime, pd.Timestamp, str]) -> bool:

        if not isinstance(other, self.__class__):
            other = self.__class__(other)
        
        return self.as_datetime >= other.as_datetime
    
    def __rge__(self, other: Union[Self, datetime, pd.Timestamp, str]) -> bool:
        return self.__ge__(other)

    def __gt__(self, other: Union[Self, datetime, pd.Timestamp, str]) -> bool:

        if not isinstance(other, self.__class__):
            other = self.__class__(other)
        
        return self.as_datetime > other.as_datetime
    
    def __rgt__(self, other: Union[Self, datetime, pd.Timestamp, str]) -> bool:
        return self.__gt__(other)

    @property
    def as_datetime(self) -> datetime: return datetime(self.year, self.month, self.day)

    @property
    def as_timestamp(self) -> pd.Timestamp: return pd.Timestamp(self.year, self.month, self.day)

    @property
    def as_string(self) -> str: return datetime.strftime(self.as_datetime, self.format_string)

    def _parse_datetime(self, arg: datetime):
        self.month = arg.month
        self.day = arg.day
        self.year = arg.year
    
    def _parse_timestamp(self, arg: pd.Timestamp):
        self.month = arg.month
        self.day = arg.day
        self.year = arg.year
    
    def _parse_string(self, arg: str, format_string: str):
        dt = datetime.strptime(arg, format_string)
        self._parse_datetime(dt)
    
    def tomorrow(self) -> Self:
        """Computes tomorrows day"""
        return self.__class__(self.as_datetime + timedelta(days=1))

    def is_weekday(self) -> bool:
        return self.as_datetime.weekday() < 5 and self.as_datetime not in HOLIDAYS
    
def weekdays_after(start_date: Date, num_days: int):

    count = 0
    day = Date(start_date)
    while count < num_days:
        day = day.tomorrow()
        if day.is_weekday() and not day.as_datetime in HOLIDAYS:
            count += 1
    
    return day


def log_env(env):
    print(f"""
    cur_date:  {env.cur_date}
    end_date:  {env.end_date}
    n_steps:   {env.n_steps}
    history:   {env.history}
    reward:    {env.reward}
    next_open: {env.data.get_price_on_close(env.cur_date)}
    is_holding:{env.is_holding}
    state:     {env.state}
    """)