#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import torch
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="average models")
    parser.add_argument("--exp_dir", required=True, type=str)
    parser.add_argument("--out_name", required=True, type=str)
    parser.add_argument("--num", default=12, type=int)
    parser.add_argument("--last_epoch", type=int)
    
    args = parser.parse_args()

    average_epochs = range(args.last_epoch - args.num + 1, args.last_epoch + 1, 1)
    print("average over", average_epochs)
    avg = None

    # sum
    for epoch in average_epochs:
        path = os.path.join(args.exp_dir, "model.{}.mdl".format(epoch))
        states = torch.load(path, map_location=torch.device("cpu"))["model_state"]
        if avg is None:
            avg = states
        else:
            for k in avg.keys():
                avg[k] += states[k]

    # average
    for k in avg.keys():
        if avg[k] is not None:
            avg[k] /= args.num

    out_path = os.path.join(args.exp_dir, args.out_name)
        
    save_model = {}
    save_model['model_state'] = avg
    torch.save(save_model, out_path)


if __name__ == "__main__":
    main()
