#!/usr/bin/env python3
# 2022 Ruchao Fan

import os
import sys
import yaml
import json
import torch
import numpy as np

sys.path.append(os.environ['E2EASR']+'/src')
from tasks import CTCTask, ArtTask, CassNATTask, CASSNAT2Task, HubertArtTask, HubertCASSNATTask
from utils.parser import DecodeParser

class Config():
    name = 'config'

def main():
    args = DecodeParser().get_args()
   
    #os.environ['CUDA_VISIBLE_DEVICES'] = str(int(os.environ['CUDA_VISIBLE_DEVICES']) % 2)
    with open(args.test_config) as f:
        config = yaml.safe_load(f)
    
    if args.text_label and args.utt2num_frames:
        config['test_paths'] = [{'name': 'test', 'scp_path': args.data_path, 'text_label': args.text_label, 'utt2num_frames': args.utt2num_frames}]
    elif hasattr(args, 'text_label') and args.text_label:
        config['test_paths'] = [{'name': 'test', 'scp_path': args.data_path, 'text_label': args.text_label}]
    elif not args.text_label and hasattr(args, 'utt2num_frames') and args.utt2num_frames:
        config['test_paths'] = [{'name': 'test', 'scp_path': args.data_path, 'utt2num_frames': args.utt2num_frames} ]
    else:
        config['test_paths'] = [{'name': 'test', 'scp_path': args.data_path} ]

    for key, val in config.items():
        setattr(args, key, val)
    for var in vars(args):
        config[var] = getattr(args, var)
    print("Experiment starts with config {}".format(json.dumps(config, sort_keys=True, indent=4)))

    use_cuda = args.use_gpu
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)
    args.rank = 0

    task_dict = {"art": ArtTask, "cassnat": CassNATTask, "ctc": CTCTask, "cassnat2": CASSNAT2Task, "hubert_cassnat": HubertCASSNATTask,
                    "hubert_art": HubertArtTask}
    if args.task in task_dict:
        task = task_dict[args.task]("test", args)
    else:
        raise NotImplementedError
    
    task.load_lm_model(args)
    task.decode(args)

if __name__ == '__main__':
    main()


