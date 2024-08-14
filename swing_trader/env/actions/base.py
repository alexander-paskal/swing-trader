import numpy as np
from typing_extensions import Self

class Action:
    
    def serialize(self) -> np.array:
        raise NotImplementedError

    @classmethod
    def from_array(cls, arr: np.array) -> Self:
        raise NotImplementedError