### 7-14-24

Got inference running with RLLib and populating the same UI plot taht I made. I'm on the cusp of getting the custom model to work - i was getting shockingly good results with the single layer MLP, but frankly i'm at the point where I don't trust it yet. I need to develop some methods of evaluation that give me more confidence in my model. I probably could stand to develop a few tests where I run it and check against hand calculated data

I need to figure out some error with nans right now

I think that some idea for modularizing might be:

    Imports:
        - I can register all my custom models as imports
    
    State Variations:
        - I can set State, Action objects on my environment and encapsulate all the logic for building/validating/serializing, etc.
        - I can have an underlying DataModel object that just controls the data access logic, as opposed to overloading it
            - This same data model can be used for computing indicators, including:
                - simple moving average
                - MACD histogram

I also need to start thinking about what an MVP for a website might look like


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