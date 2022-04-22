# 2022 Ruchao Fan
# SPAPL

import argparse

class BaseParser(object):
    def __init__(self, description="Basic argument parser"):
        parser = argparse.ArgumentParser(description=description)
        # for general settings
        parser.add_argument("--exp_dir")
        parser.add_argument("--train_config")
        parser.add_argument("--data_config")
        parser.add_argument("--load_data_workers", default=1, type=int, help="Number of parallel data loaders")
        
        parser.add_argument("--optim_type", default='normal', type=str, help="Type of optimizer, normal, norm, multistep")
        parser.add_argument("--epochs", default=100, type=int, help="Number of training epochs")
        parser.add_argument("--start_saving_epoch", default=20, type=int, help="Starting to save the model")  
        parser.add_argument("--end_patience", default=2, type=int, help="Number of epochs without improvements for early stop")
        
        # training settings
        parser.add_argument("--task", default="art", type=str, help="the task for training, art, cassnat, ctc")
        parser.add_argument("--print_freq", default=100, type=int, help="Number of iter to print")
        parser.add_argument("--use_slurm", action='store_true', help="use slurm")
        parser.add_argument("--port", default=1001, type=int, help="port for multi-gpu training")
        parser.add_argument("--seed", default=1, type=int, help="Random number seed")

        self.parser = parser
        
    def get_args(self):
        return self.parser.parse_args()

