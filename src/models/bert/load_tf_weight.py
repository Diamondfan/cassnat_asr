# reference:
# https://github.com/maknotavailable/pytorch-pretrained-BERT/blob/master/pytorch_pretrained_bert/convert_tf_checkpoint_to_pytorch.py

import os
import re
import torch
import numpy as np
import tensorflow as tf

def load_tf_weights_in_bert(model, bert_checkpoint_path):
    tf_path = os.path.abspath(bert_checkpoint_path)

    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []

    for name, shape in init_vars:
        #print("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)

    # Initialise PyTorch model
    for name, array in zip(names, arrays):
        no_param = False
        name = name.split('/')
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if name[-1] in ["adam_v", "adam_m"]:
            #print("Skipping {}".format("/".join(name)))
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch(r'[A-Za-z]+_\d+', m_name):
                l = re.split(r'_(\d+)', m_name)
            else:
                l = [m_name]
            if l[0] == 'kernel':
                pointer = getattr(pointer, 'weight')
            elif l[0] == 'output_bias':
                pointer = getattr(pointer, 'bias')
            elif l[0] == 'output_weights':
                pointer = getattr(pointer, 'weight')
            elif l[0] == 'cls':
                no_param = True
                break
            else:
                pointer = getattr(pointer, l[0])
            if len(l) >= 2:
                num = int(l[1])
                pointer = pointer[num]
        
        if no_param:
            continue

        if m_name[-11:] == '_embeddings':
            pointer = getattr(pointer, 'weight')
        elif m_name == 'kernel':
            array = np.transpose(array)
        try:
            assert pointer.shape == array.shape
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        #print("Initialize PyTorch weight {}".format(name))
        pointer.data = torch.from_numpy(array)

    return model

