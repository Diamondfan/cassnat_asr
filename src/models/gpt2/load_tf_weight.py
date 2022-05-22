# reference:
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py

import os
import re
import torch
import tensorflow as tf

def load_tf_weights_in_gpt2(model, gpt2_checkpoint_path):
    """Load tf checkpoints in a pytorch model"""
        
    tf_path = os.path.abspath(gpt2_checkpoint_path)
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array.squeeze())

    for name, array in zip(names, arrays):
        name = name[6:]  # skip "model/"
        name = name.split("/")
        pointer = model
        for m_name in name:
            if re.fullmatch(r"[A-Za-z]+\d+", m_name):
                scope_names = re.split(r"(\d+)", m_name)
            else:
                scope_names = [m_name]
            if scope_names[0] == "w" or scope_names[0] == "g":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "b":
                pointer = getattr(pointer, "bias")
            elif scope_names[0] == "wpe" or scope_names[0] == "wte":
                pointer = getattr(pointer, scope_names[0])
                pointer = getattr(pointer, "weight")
            else:
                pointer = getattr(pointer, scope_names[0])
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]

        assert (
            pointer.shape == array.shape
        ), f"Pointer shape {pointer.shape} and array shape {array.shape} mismatched"
        
        pointer.data = torch.from_numpy(array)

    return model


