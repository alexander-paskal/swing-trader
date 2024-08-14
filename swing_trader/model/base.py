import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.utils.annotations import override
import numpy as np



class Model(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        
       raise NotImplementedError

    @override(TorchModelV2)
    def value_function(self):
        raise NotImplementedError