from typing_extensions import TypedDict
from ray.rllib.utils.typing import ModelConfigDict, ModelInputDict, TensorType

from typing import *

__all__ = [
    'ModelV2InputDict'
]

class ModelV2InputDict(TypedDict):
    obs: TensorType
    obs_flat: TensorType
    prev_action: TensorType
    prev_reward: TensorType
    is_training: TensorType
    eps_id: TensorType
    agent_id: TensorType
    infos: TensorType
    t: TensorType

