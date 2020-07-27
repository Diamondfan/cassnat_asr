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
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.distributed import ReduceOp

sys.path.append(os.environ['E2EASR']+'/src')
import utils.util as util
from utils.wer import ctc_greedy_wer, att_greedy_wer
from data.vocab import Vocab
from utils.optimizer import get_opt
from models.transformer import make_model
from utils.loss import LabelSmoothing
from data.speech_loader import SpeechDataset, SpeechDataLoader

class Config():
    name = 'config'

def main():
    parser = argparse.ArgumentParser(description="Configuration for training ctc-attention system")
   
    parser.add_argument("--exp_dir")
    parser.add_argument("--train_config")
    parser.add_argument("--data_config")
    parser.add_argument("--use_cmvn", default=False, action='store_true', help="Use cmvn or not")
    parser.add_argument("--batch_size", default=32, type=int, help="Training minibatch size")
    parser.add_argument("--epochs", default=30, type=int, help="Number of training epochs")
    parser.add_argument("--save_epoch", default=20, type=int, help="Starting to save the model")
    parser.add_argument("--learning_rate", default=2e-4, type=float, help="Initial learning rate")
    parser.add_argument("--min_lr", default=1e-6, type=float, help="Minimal learning rate")
    parser.add_argument("--patience", default=2, type=int, help="Number of epochs without improvements")
    parser.add_argument("--end_patience", default=2, type=int, help="Number of epochs without improvements for early stop")
    parser.add_argument("--opt_type", default='normal', type=str, help="Type of optimizer, normal or noam")
    parser.add_argument("--anneal_lr_ratio", default=0.5, type=float, help="Learning rate decay ratio, used when opt_type='normal'")
    parser.add_argument("--weight_decay", default=0.00001, type=float, help="Weight decay in optimizer")
    parser.add_argument("--label_smooth", default=0.1, type=float, help="Label smoothing for CE loss")
    parser.add_argument("--load_data_workers", default=1, type=int, help="Number of parallel data loaders")
    parser.add_argument("--ctc_alpha", default=0, type=float, help="Task ratio of CTC")
    parser.add_argument("--resume_model", default='', type=str, help="The model path to resume")
    parser.add_argument("--print_freq", default=100, type=int, help="Number of iter to print")
    parser.add_argument("--seed", default=1, type=int, help="Random number seed")

    ## 1. Parse and print config Main process
    args = parser.parse_args()
    with open(args.train_config) as f:
        config = yaml.safe_load(f)

    with open(args.data_config) as f:
        data = yaml.safe_load(f)
        config['train_paths'] = [j for i, j in data['train_data_path'].items()]
        config['dev_paths'] = [j for i, j in data['dev_data_path'].items()]
        config['global_cmvn'] = data['global_cmvn']
        config['vocab_file'] = data['vocab_file']

    for key, val in config.items():
        setattr(args, key, val)
    for var in vars(args):
        config[var] = getattr(args, var)
    print("Experiment starts with config {}".format(json.dumps(config, sort_keys=True, indent=4)))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.deterministic = True
    if not os.path.isdir(args.exp_dir):
        os.makedirs(args.exp_dir)

    num_gpu = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    args.distributed = True if num_gpu > 1 else False
    if args.distributed:
        import torch.multiprocessing as mp
        mp.spawn(main_worker, nprocs=num_gpu, args=(num_gpu, args))
    else:
        main_worker(0, 1, args)
        

def main_worker(rank, world_size, args, backend='nccl'):
    args.rank, args.world_size = rank, world_size
    if args.distributed:
        dist.init_process_group(backend=backend, init_method='tcp://localhost:21980',
                                    world_size=world_size, rank=rank)

    ## 2. Define model and optimizer
    use_cuda = args.use_gpu
    if use_cuda:
        torch.cuda.manual_seed(args.seed)

    vocab = Vocab(args.vocab_file, args.rank)
    args.vocab_size = vocab.n_words
    assert args.input_size == (args.left_ctx + args.right_ctx + 1) // args.skip_frame * args.n_features
    model = make_model(args.input_size, args)
    optimizer = get_opt(args.opt_type, model, args) 
    
    if args.resume_model:
        if rank == 0:
            print("Loading model from {}".format(args.resume_model))
        checkpoint = torch.load(args.resume_model, map_location='cpu')
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if use_cuda:
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()   
        start_epoch = checkpoint['epoch'] + 1
    else:
        start_epoch = 0
    
    num_params = 0
    for name, param in model.named_parameters():
        num_params += param.numel()
    if args.rank == 0:
        print("Number of parameters: {}".format(num_params))
    if use_cuda:
        torch.cuda.set_device(args.rank)
        model = model.cuda(args.rank)
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.rank])
    
    ## 3. Define vocabulary and data loader
    trainset = SpeechDataset(vocab, args.train_paths, args)
    if args.use_cmvn:
        trainset._load_cmvn(args.global_cmvn)
    train_loader = SpeechDataLoader(trainset, args.batch_size, args.padding_idx, num_workers=args.load_data_workers, 
                                       distributed=args.distributed, shuffle=True)
    if args.rank == 0:
        print("Finish Loading training files. Number batches: {}".format(len(train_loader)))

    validset = SpeechDataset(vocab, args.dev_paths, args)
    if args.use_cmvn:
        validset._load_cmvn(args.global_cmvn)
    valid_loader = SpeechDataLoader(validset, args.batch_size, args.padding_idx, num_workers=args.load_data_workers, 
                                        distributed=False, shuffle=False)
    if args.rank == 0:
        print("Finish Loading dev files. Number batches: {}".format(len(valid_loader)))
    
    criterion_ctc = torch.nn.CTCLoss(reduction='mean')
    criterion_att = LabelSmoothing(args.vocab_size, args.padding_idx, args.label_smooth)
    criterion = (criterion_ctc, criterion_att)
    
    ## 4. Start training iteratively
    best_wer = 100
    # This is used for noam early stop
    early_stop_patience = args.end_patience
    best_epoch = 0
    # This is used for noraml adam control
    if args.opt_type == 'normal':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=args.anneal_lr_ratio, 
                            patience=args.patience, min_lr=args.min_lr)
    
    for epoch in range(start_epoch, args.epochs):
        if args.distributed:
            train_loader.set_epoch(epoch)
        model.train()
        train_loss, train_wer, train_ctc_wer = run_epoch(epoch, train_loader, model, criterion, args, optimizer, is_train=True)
        model.eval()
        with torch.no_grad():
            valid_loss, valid_wer, valid_ctc_wer = run_epoch(epoch, valid_loader, model, criterion, args, is_train=False)
        
        temp_lr = optimizer.param_groups[0]['lr'] if args.opt_type == "normal" else optimizer.optimizer.param_groups[0]['lr']
        if args.distributed:
            average_number = torch.Tensor([train_loss, train_wer, valid_loss, valid_wer]).float().cuda(args.rank)
            torch.distributed.all_reduce(average_number, op=ReduceOp.SUM)
            train_loss, train_wer, valid_loss, valid_wer = (average_number / args.world_size).cpu().numpy()
        if args.rank == 0:
            print("Epoch {} done, Train Loss: {:.4f}, Train WER: {:.4f} Train ctc WER: {:.4f} Valid Loss: {:.4f} Valid WER: {:.4f} Valid ctc WER: {:.4f} Current LR: {:4e}".format(
                        epoch, train_loss, train_wer, train_ctc_wer, valid_loss, valid_wer, valid_ctc_wer, temp_lr), flush=True)
        
        if args.opt_type == 'normal':
            scheduler.step(valid_wer)

        if epoch > args.save_epoch and args.rank == 0:
            output_file=args.exp_dir + '/model.' + str(epoch) + '.mdl'
            checkpoint = {'epoch': epoch, 'optimizer': optimizer.state_dict(),
                            'state_dict': model.state_dict()}
            torch.save(checkpoint, output_file)
        
        if valid_wer < best_wer:
            best_wer = valid_wer
            best_epoch = epoch
            if args.rank == 0:
                output_file=args.exp_dir + '/best_model.mdl'
                checkpoint = {'epoch': epoch, 'optimizer': optimizer.state_dict(),
                                'state_dict': model.state_dict()}
                torch.save(checkpoint, output_file)
        
        if epoch - best_epoch > early_stop_patience:
            if args.rank == 0:
                print("Early stop since valid_wer doesn't decrease")
            break

def subsequent_mask(size):
    ret = torch.ones(size, size, dtype=torch.uint8)
    return torch.tril(ret, out=ret).unsqueeze(0)

def run_epoch(epoch, dataloader, model, criterion, args, optimizer=None, is_train=True):
    batch_time = util.AverageMeter('Time', ':6.3f')
    losses = util.AverageMeter('Loss', ':.4e')
    ctc_losses = util.AverageMeter('CtcLoss', ":.4e")
    att_losses = util.AverageMeter('AttLoss', ":.4e")
    ctc_wers = util.AverageMeter('CtcWer', ':.4f')
    att_wers = util.AverageMeter('AttWer', ':.4f')
    token_speed = util.AverageMeter('TokenSpeed', ":.2f")
    progress = util.ProgressMeter(len(dataloader), batch_time, losses, ctc_losses, att_losses, ctc_wers, att_wers, token_speed, prefix="Epoch: [{}]".format(epoch))
    
    end = time.time()
    
    for i, data in enumerate(dataloader):
        start = time.time()
        utt_list, feats, labels, feat_sizes, label_sizes = data
        src, src_mask = feats, (feats[:,:,0] != args.padding_idx).unsqueeze(1)
        tgt, tgt_label = labels[:,:-1], labels[:,1:]
        tgt_mask = (tgt != args.padding_idx).unsqueeze(1) 
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask)
        tokens = (tgt_label != args.padding_idx).sum().item()
        
        if args.use_gpu:
            src, src_mask, tgt, tgt_mask = src.cuda(), src_mask.cuda(), tgt.cuda(), tgt_mask.cuda()
            tgt_label = tgt_label.cuda()
            feat_sizes = feat_sizes.cuda()
            label_sizes = label_sizes.cuda()
        
        ctc_out, att_out, enc_h = model(src, tgt, src_mask, tgt_mask, args.ctc_alpha)
        bs, max_feat_size, _ = enc_h.size()

        # loss computation
        att_loss = criterion[1](att_out.view(-1, att_out.size(-1)), tgt_label.view(-1))
        if args.ctc_alpha > 0:
            feat_sizes = (feat_sizes * max_feat_size).long()
            ctc_loss = criterion[0](ctc_out.transpose(0,1), tgt_label, feat_sizes, label_sizes)
            loss = args.ctc_alpha * ctc_loss + att_loss #(1 - args.ctc_alpha) * att_loss
        else:
            ctc_loss = torch.Tensor([0])
            loss = att_loss

        # unit error rate computation
        if args.ctc_alpha > 0:
            ctc_errs, all_tokens = ctc_greedy_wer(ctc_out, tgt_label.cpu().numpy(), feat_sizes.cpu().numpy(), args.padding_idx)
        else:
            ctc_errs, all_tokens = 1, 1
        ctc_wers.update(ctc_errs/all_tokens, all_tokens)
        att_errs, all_tokens = att_greedy_wer(att_out, tgt_label.cpu().numpy(), args.padding_idx)
        att_wers.update(att_errs/all_tokens, all_tokens)
        
        losses.update(loss.item(), tokens)
        ctc_losses.update(ctc_loss.item(), tokens)
        att_losses.update(att_loss.item(), tokens)
                
        if is_train:
            loss = loss / args.accum_grad
            loss.backward()
            if i % args.accum_grad == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                optimizer.zero_grad()
        
        batch_time.update(time.time() - end)
        token_speed.update(tokens/(time.time()-start))

        if i % args.print_freq == 0 and args.rank == 0:
            progress.print(i)
    return losses.avg, att_wers.avg, ctc_wers.avg
    
if __name__ == '__main__':
    main()


