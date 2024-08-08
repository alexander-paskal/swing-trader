from ray.rllib.models.torch.attention_net import AttentionWrapper
import gymnasium as gym

# model = AttentionWrapper(
#     obs_space=gym.spaces.Box(0, 1, (100,)),
#     action_space=gym.spaces.Box(0, 1, (2,)),
#     num_outputs=20,
#     model_config={
#         "attention_use_n_prev_actions": 0,
#         "attention_use_n_prev_rewards": 0,
#         "attention_dim": 50,
#         "_disable_action_flattening": False,
#         "attention_num_transformer_units": 50,
#         "attention_num_heads": 2,
#         "attention_head_dim": 20,
#         "attention_memory_inference": 0,
#         "attention_memory_training": 0,
#         "attention_position_wise_mlp_dim": 256,
#         "attention_init_gru_gate_bias": 0.02,
#         "max_seq_len": 256
#     },
#     name="MyModel"
# )


import torch
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from swing_trader.model.self_attention import AttentionNetwork



model = AttentionNetwork(
    obs_space=gym.spaces.Box(0, 1, (10,)),
    action_space=gym.spaces.Box(0, 1, (20,)),
    num_outputs=20,
    model_config = {
        "num_layers": 5,
        "embed_dim": 1024,
        "num_heads": 4
    },
    name="AttentionV1"
)



# model = TorchModelV2()

t_in = torch.rand((256, 10))
input_dict = {
    "obs": t_in,
    "obs_flat": t_in,
}
t_out = model(input_dict)[0]
print()

from ray.rllib.models import ModelCatalog
ModelCatalog.register_custom_model("AttentionNetwork", AttentionNetwork)
