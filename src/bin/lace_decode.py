#!/usr/bin/env python3
# 2020 Ruchao Fan

import os
import sys
import time
import yaml
import json
import torch
import argparse
import numpy as np
import torch.nn.functional as F

sys.path.append(os.environ['E2EASR']+'/src')
import utils.util as util
from data.vocab import Vocab
from models.lace_nat import make_model
from data.speech_loader import SpeechDataset, SpeechDataLoader
from utils.beam_decode import ctc_beam_decode


class Config():
    name = 'config'

def main():
    parser = argparse.ArgumentParser(description="Configuration for transformer testing")
   
    parser.add_argument("--test_config")
    parser.add_argument("--data_path")
    parser.add_argument("--use_cmvn", default=False, action='store_true', help="Use global cmvn or not")
    parsed.add_argument("--global_cmvn", type=str, help="Cmvn file to load")
    parser.add_argument("--batch_size", default=32, type=int, help="Training minibatch size")
    parser.add_argument("--load_data_workers", default=1, type=int, help="Number of parallel data loaders")
    parser.add_argument("--resume_model", default='', type=str, help="Model to do evaluation")
    parser.add_argument("--result_file", default='', type=str, help="File to save the results")
    parser.add_argument("--print_freq", default=100, type=int, help="Number of iter to print")
    parser.add_argument("--decode_type", default='att', type=str, help="CTC, ATT or ctc-att hybrid decoding")
    parser.add_argument("--ctc_weight", type=float, default=0.0, help="CTC weight in joint decoding")
    parser.add_argument("--rnnlm", type=str, default=None, help="RNNLM model file to read")
    parser.add_argument("--lm_weight", type=float, default=0.1, help="RNNLM weight")
    parser.add_argument("--max_decode_ratio", type=float, default=0, help='Decoding step to length ratio')
    parser.add_argument("--seed", default=1, type=int, help="random number seed")

    args = parser.parse_args()
    with open(args.test_config) as f:
        config = yaml.safe_load(f)

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

    vocab = Vocab(args.vocab_file)
    args.vocab_size = vocab.n_words
    args.rank = 0
    assert args.input_size == (args.left_ctx + args.right_ctx + 1) // args.skip_frame * args.n_features
    model = make_model(args.input_size, args)

    if args.resume_model:
        print("Loading model from {}".format(args.resume_model))
        checkpoint = torch.load(args.resume_model, map_location='cpu')
        model.load_state_dict(checkpoint["state_dict"])
        start_epoch = checkpoint['epoch']+1

    num_params = 0
    for name, param in model.named_parameters():
        num_params += param.numel()
    print("Number of parameters: {}".format(num_params))
    
    if use_cuda:
        model = model.cuda()

    testset = SpeechDataset(vocab, args.test_paths, args)
    if args.use_cmvn:
        testset._load_cmvn(args.global_cmvn)
    test_loader = SpeechDataLoader(testset, args.batch_size, args.padding_idx, num_workers=args.load_data_workers, shuffle=False)
    print("Finish Loading test files. Number batches: {}".format(len(test_loader)))
    
    batch_time = util.AverageMeter('Time', ':6.3f')
    progress = util.ProgressMeter(len(test_loader), batch_time)
    end = time.time()
    
    out_file = open(args.result_file, 'w')
    with torch.no_grad():
        model.eval()
        for i, data in enumerate(test_loader):
            utt_list, feats, _, feat_sizes, _ = data
            src, src_mask = feats, (feats[:,:,0] != args.padding_idx).unsqueeze(1)
        
            if args.use_gpu:
                src, src_mask = src.cuda(), src_mask.cuda()
                feat_sizes = feat_sizes.cuda()

            if args.decode_type == 'ctc_only':
                recog_results = ctc_beam_decode(model, src, src_mask, feat_sizes, vocab, args)
            else:
                recog_results = model.beam_decode(src, src_mask, feat_sizes, vocab, args)
            
            for j in range(len(utt_list)):
                hyp = []
                for idx in recog_results[j][0]['hyp']:
                    if idx == vocab.word2index['sos'] or idx == args.padding_idx:
                        continue
                    if idx == vocab.word2index['eos']:
                        break
                    hyp.append(vocab.index2word[idx])
                #print(utt_list[j]+' '+' '.join(hyp))
                print(utt_list[j]+' '+' '.join(hyp), flush=True, file=out_file)
    
            batch_time.update(time.time() - end)
            if i % args.print_freq == 0:
                progress.print(i)
        progress.print(i)
    
if __name__ == '__main__':
    main()

