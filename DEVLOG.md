### 7-7-24

Looked into inference with RLLib, looks annoyingly difficult to do. Need to consult the docs
- It might make more sense to take a stab at registering a custom model first - then I can drop a breakpoint in and see the call stack

Got a very simple UI going - want to add a keyboard interface to it to make it smoother, as well as to get OHLC bars
from the yfinance plotting library that I saw last time


I could stand to clean up the environment class a little bit.
Things I want to add to it in terms of customization:

    - Different Time Scales (Weekly, Daily, Monthly)
    - Properties for pulling views of the data currently being accessed
    - Terminate early if you lose 50% of your money

### 7-3-24

Got it training! some absolutely bonkers rewards out there, i definitely need to reign in my environment a little
bit by restricting the types of trades its allowed to make:
    1. volume restrictions
    2. price restrictions

I also need to implement reward clipping asap, since some rewards are insane

Apart from that, two things I want to figure out in ray are:
    1. Inference
    2. Custom Model Selection

I need to fix the reward clipping cause i'm getting weird values