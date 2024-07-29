from swing_trader.model.base import Model
import torch.nn as nn



class MultiHeadModule(nn.MultiheadAttention):
    def __init__(self, embed_dim, num_heads, *args, **kwargs):
        super().__init__(embed_dim, num_heads, *args, **kwargs)
        self.proj_qkv = nn.Linear(embed_dim, 3*embed_dim)
        self._embed_dim = embed_dim

    def forward(self, x):
        qkv = self.proj_qkv(x)
        ind = self._embed_dim
        q = qkv[:, ind*0:ind*1]
        k = qkv[:, ind*1:ind*2] 
        v = qkv[:, ind*2:ind*3]     
        output = super().forward(q, k, v)
        return output[0]

class GELU(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.gelu(x)

class AttentionNetwork(Model):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        

        print(model_config)
        # num_layers = model_config["num_layers"]
        # embed_dim = model_config["embed_dim"]
        # num_heads = model_config["num_heads"]
        num_layers = 5
        embed_dim = 1024
        num_heads = 4

        _attention_modules = []
        for _ in range(num_layers - 2):
            attention = MultiHeadModule(embed_dim, num_heads)
            activation = nn.LeakyReLU()
            _attention_modules.append(nn.Sequential(attention, activation))
        self.attention_module = nn.Sequential(*_attention_modules)
        self.encoder = nn.Sequential(
            nn.Linear(obs_space.shape[0], embed_dim),
            nn.LeakyReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, action_space.shape[0])
        )
        self.value_decoder = nn.Sequential(
            # nn.Linear(embed_dim, action_space.shape[0]),
            # nn.LeakyReLU(),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, input_dict, state, seq_lens):
        
        h = self.encoder(input_dict['obs_flat'])
        h = self.attention_module(h)
        out = self.decoder(h)
        self.embedding = h
        return out, state
    
    def value_function(self):
        val =  self.value_decoder(self.embedding)
        # print(f"\n\n\nTHIS IS THE VALUE: {val}\n\n\n")
        return val.squeeze(1)