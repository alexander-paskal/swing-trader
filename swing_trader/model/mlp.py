import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.utils.annotations import override
import numpy as np
from ray.rllib.models import ModelCatalog


class CustomTorchModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.torch_sub_model = nn.Sequential(
            SlimFC(int(np.product(obs_space.shape)), 256, activation_fn=nn.ReLU),
            SlimFC(256, 256, activation_fn=nn.ReLU),
            SlimFC(256, num_outputs, activation_fn=None)
        )

        self.value_branch = nn.Sequential(
            SlimFC(int(np.product(obs_space.shape)), 256, activation_fn=nn.ReLU),
            SlimFC(256, 1, activation_fn=None)
        )

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs_flat"].float()
        self._last_flat_in = obs.reshape(obs.shape[0], -1)
        self._last_out = self.torch_sub_model(self._last_flat_in)
        return self._last_out, state

    @override(TorchModelV2)
    def value_function(self):
        assert self._last_flat_in is not None, "must call forward() first"
        return self.value_branch(self._last_flat_in).squeeze(1)
# ModelCatalog.register_custom_model("custom_torch_model", CustomTorchModel)