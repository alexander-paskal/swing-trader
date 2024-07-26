
### 7-25
I should definitely just cut off invalid environments for now, as they won't have enough data. I suppose i can pass that into the initial conditions - make sure there's enough data left over

Ok almost have it training right now - i need to figure out why the state setting for stock env isn't working properly - kinda annoying

It's traiing! Got a full training run overnight, results are underwhelming, which is to be expected. I think I'm good with the current state space representation - it should have enough information to at least do decently well

TODO for next time:
    - Figure out the state setting issue
    - Make it so that train and inference pull from the same config file
    - Make a composite reward class
    - Move Action, Reward, CloseVolumeState out of the env
    - Fix ticks held
    - Model Iterations:
        - I need a simple attention model as a baseline
    - Reward Shaping
        - reinstitute clipping
        - Minimize number of trades
        - Encourage longer holds
        - Buy&HoldBaseline
        - Definitely starting with shorter rollouts (curriculum learning reference https://docs.ray.io/en/latest/rllib/rllib-advanced-api.html)
    - Action Space
        - ??? Maybe a single bit action space
    - Saving off Runs
        - I could join (model)-(action)-(state)-(reward) clases together to name it and save to a csv, along with the checkpoint path
    

### 7-23

Need to start running tests for the env and see what I'm missing. Should have close to evertyhing, just need to formalize Actions and Rewards



### 7-22

Finished an initial thing for the State, just need to check to make sure that the env is working. I need to rework:

    Initial Conditions
    LoadData
    randomize
    Action
    Reward

The last two should be pretty quick. As for randomizing initial conditions, i could probably add something to specify start and end date. Can just load the data model again to simplify things



### 7-17

Last two sessions have been spent refactoring. I think it was time well spent, but i'm starting to dilly dally. I have one more to get my env working and tested, and to get a training run going. Next couple sessions are gonna be about testing different models. Here's what I want to be able to run:

    State: N Daily, N Weekly and N Monthly open ticks, along with their associated volumes, for a 6N + 1 input vector. Same normalizations as before
    Reward: A comparison with the buy & hold performance of the 

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