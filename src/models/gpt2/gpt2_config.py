# Reference:
# https://github.com/graykode/gpt-2-Pytorch

import json

class GPT2Config(object):
    def __init__(
            self,
            vocab_size_or_config_json_file=50257,
            n_positions=1024,
            n_ctx=1024,
            n_embd=768,
            n_layer=12,
            n_head=12,
            layer_norm_epsilon=1e-5,
            initializer_range=0.02,
    ):
        self.n_vocab = vocab_size_or_config_json_file
        self.n_ctx = n_ctx
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range

    def load_from_json(self, config_file):
        with open(config_file, 'r') as f:
            configs = json.load(f)

        for key in configs:
            setattr(self, key, configs[key])
        
