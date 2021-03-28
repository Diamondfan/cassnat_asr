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
from models import make_fanat, make_fanat_conformer
from data.speech_loader import SpeechDataset, SpeechDataLoader
from utils.beam_decode import ctc_beam_decode


class Config():
    name = 'config'

def main():
    parser = argparse.ArgumentParser(description="Configuration for transformer testing")
   
    parser.add_argument("--test_config")
    parser.add_argument("--lm_config")
    parser.add_argument("--data_path")
    parser.add_argument("--text_label")
    parser.add_argument("--use_cmvn", default=False, action='store_true', help="Use global cmvn or not")
    parser.add_argument("--global_cmvn", type=str, help="Cmvn file to load")
    parser.add_argument("--batch_size", default=32, type=int, help="Training minibatch size")
    parser.add_argument("--load_data_workers", default=1, type=int, help="Number of parallel data loaders")
    parser.add_argument("--resume_model", default='', type=str, help="Model to do evaluation")
    parser.add_argument("--result_file", default='', type=str, help="File to save the results")
    parser.add_argument("--print_freq", default=100, type=int, help="Number of iter to print")
    parser.add_argument("--rnnlm", type=str, default=None, help="RNNLM model file to read")
    parser.add_argument("--rank_model", type=str, default='lm', help="model type to Rank ESA")
    parser.add_argument("--lm_weight", type=float, default=0.1, help="RNNLM weight")
    parser.add_argument("--word_embed", default='', type=str, help='Ground truth of word embedding')
    parser.add_argument("--save_embedding", default=False, action='store_true', help="save embedding for analysis")
    parser.add_argument("--max_decode_ratio", type=float, default=0, help='Decoding step to length ratio')
    parser.add_argument("--seed", default=1, type=int, help="random number seed")

    os.environ['CUDA_VISIBLE_DEVICES'] = str(int(os.environ['CUDA_VISIBLE_DEVICES']) % 4)

    args = parser.parse_args()
    with open(args.test_config) as f:
        config = yaml.safe_load(f)
    
    if args.text_label:
        config['test_paths'] = [{'name': 'test', 'scp_path': args.data_path, 'text_label': args.text_label} ]
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

    vocab = Vocab(args.vocab_file)
    args.vocab_size = vocab.n_words
    args.interctc_alpha = 0
    args.interce_alpha = 0
    args.ctc_alpha = 1
    args.interce_location='other'

    args.rank = 0
    assert args.input_size == (args.left_ctx + args.right_ctx + 1) // args.skip_frame * args.n_features
    if args.model_type == "transformer":
        model = make_fanat(args.input_size, args)
    elif args.model_type == "conformer":
        model = make_fanat_conformer(args.input_size, args)

    if args.resume_model:
        print("Loading model from {}".format(args.resume_model))
        checkpoint = torch.load(args.resume_model, map_location='cpu')
        model_state = checkpoint["state_dict"]
        for name, param in model.named_parameters():
            if name not in model_state:
                name = "module." + name
            param.data.copy_(model_state[name])

    if args.lm_weight > 0 or args.ctc_lm_weight > 0:
        with open(args.lm_config) as f:
            lm_config = yaml.safe_load(f)
        lm_args = Config()
        for key, val in lm_config.items():
            setattr(lm_args, key, val)
        
        lm_args.vocab_size = vocab.n_words
        if args.rank_model == 'lm':
            from models.lm import make_model as make_lm_model
            lm_model = make_lm_model(lm_args)
        
        if args.rank_model == 'at_baseline':
            if lm_args.model_type == 'transformer':
                from models.transformer import make_model as make_lm_model
            if lm_args.model_type == 'conformer':
                from models.conformer import make_model as make_lm_model
            lm_args.interctc_alpha = 0
            lm_model = make_lm_model(args.input_size, lm_args)

        print("Loading language model from {}".format(args.rnnlm))
        checkpoint_lm = torch.load(args.rnnlm, map_location='cpu')
        model_state = checkpoint_lm["state_dict"]
        for name, param in lm_model.named_parameters():
            if name not in model_state:
                name = "module." + name
            param.data.copy_(model_state[name])
        if use_cuda:
            lm_model.cuda()
    else:
        lm_model = None

    if args.use_unimask:
        checkpoint = torch.load(args.word_embed, map_location='cpu')['state_dict']
        from models.modules.embedding import TextEmbedding
        word_embed = TextEmbedding(args.d_model, args.vocab_size)
        for name, param in word_embed.named_parameters():
            param.data.copy_(checkpoint['module.tgt_embed.0.lut.weight'])
            param.requires_grad = False
        if use_cuda:
            torch.cuda.set_device(args.rank)
            word_embed = word_embed.cuda()
        args.word_embed = word_embed

    num_params = 0
    for name, param in model.named_parameters():
        num_params += param.numel()
    print("Number of parameters: {}".format(num_params))
    
    if use_cuda:
        model = model.cuda()

    args.use_specaug = False
    args.specaug_conf = None
    testset = SpeechDataset(vocab, args.test_paths, args)
    if args.use_cmvn:
        testset._load_cmvn(args.global_cmvn)
    test_loader = SpeechDataLoader(testset, args.batch_size, args.padding_idx, num_workers=args.load_data_workers, shuffle=False)
    print("Finish Loading test files. Number batches: {}".format(len(test_loader)))
    
    batch_time = util.AverageMeter('Time', ':6.3f')
    progress = util.ProgressMeter(len(test_loader), batch_time)
    end = time.time()
    
    args.decode_type = "att_only"
    out_file = open(args.result_file, 'w')
    args.num_correct, args.total = 0, 0
    args.length_correct, args.length_total = 0, 0
    if args.save_embedding:
        ac_embeds, pred_embeds = [], []

    with torch.no_grad():
        model.eval()
        if lm_model is not None:
            lm_model.eval()

        for i, data in enumerate(test_loader):
            utt_list, feats, labels, feat_sizes, label_sizes = data
            src, src_mask = feats, (feats[:,:,0] != args.padding_idx).unsqueeze(1)
            tgt_label = labels[:,1:]
            tgt = labels[:,:-1]
            
            if args.use_gpu:
                src, src_mask = src.cuda(), src_mask.cuda()
                tgt_label = tgt_label.cuda()
                tgt = tgt.cuda()
                feat_sizes = feat_sizes.cuda()
                labels, label_sizes = labels.cuda(), label_sizes.cuda()
            
            #recog_results, args = model.beam_decode(src, src_mask, feat_sizes, vocab, args, lm_model, labels=labels, label_sizes=label_sizes)

            ctc_out, att_out, pred_embed, tgt_mask_pred, interctc_out, interce_out = model(src, src_mask, feat_sizes, tgt_label, label_sizes, args)

            if args.save_embedding:
                ac_embeds.append((args.ac_embed[0], tgt_label.cpu()[0]))
                pred_embeds.append((args.pred_embed[0], tgt_label.cpu()[0]))
            
            #for j in range(len(utt_list)):
            #    hyp = []
            #    for idx in recog_results[j][0]['hyp']:
            #        if idx == vocab.word2index['sos'] or idx == args.padding_idx:
            #            continue
            #        if idx == vocab.word2index['eos']:
            #            break
            #        hyp.append(vocab.index2word[idx])
            #    #print(utt_list[j]+' '+' '.join(hyp))
            #    #print(recog_results[j][0]['score_ctc'], recog_results[j][0]['score_lm'])
            #    print(utt_list[j]+' '+' '.join(hyp), flush=True, file=out_file)
            
            batch_time.update(time.time() - end)
            if i % args.print_freq == 0:
                progress.print(i)
            if i == 3000:
                break

        progress.print(i)
        if args.test_hitrate:
            print(args.num_correct, args.total, args.num_correct / args.total)
            print(args.length_correct, args.length_total, args.length_correct / args.length_total)
        if args.save_embedding:
            import pickle
            pickle.dump(ac_embeds, open('ac_embeddings', 'wb'))
            pickle.dump(pred_embeds, open('pred_embeddings', 'wb'))


if __name__ == '__main__':
    main()


