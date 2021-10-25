#!/usr/bin/env python3
# 2021 Ruchao Fan
# SPAPL
# For self-supervised learning training

import os
import sys
import copy
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
from utils.optimizer import get_opt
from models.ssl_model import make_model
from utils.loss import ContrastiveLoss, HubertLoss, PredictLoss
from data.ssl_loader import SSLDataset, DynamicDataset, SSLDataLoader

class Config():
    name = 'config'

def main():
    parser = argparse.ArgumentParser(description="Configuration for training ctc-attention system")
   
    parser.add_argument("--exp_dir")
    parser.add_argument("--train_config")
    parser.add_argument("--data_config")
    parser.add_argument("--use_cmvn", default=False, action='store_true', help="Use cmvn or not")
    parser.add_argument("--epochs", default=30, type=int, help="Number of training epochs")
    parser.add_argument("--save_epoch", default=20, type=int, help="Starting to save the model")
    parser.add_argument("--learning_rate", default=2e-4, type=float, help="Initial learning rate")
    parser.add_argument("--min_lr", default=1e-6, type=float, help="Minimal learning rate")
    parser.add_argument("--patience", default=2, type=int, help="Number of epochs without improvements")
    parser.add_argument("--end_patience", default=2, type=int, help="Number of epochs without improvements for early stop")
    parser.add_argument("--opt_type", default='normal', type=str, help="Type of optimizer, normal or noam")
    parser.add_argument("--anneal_lr_ratio", default=0.5, type=float, help="Learning rate decay ratio, used when opt_type='normal'")
    parser.add_argument("--weight_decay", default=0.00001, type=float, help="Weight decay in optimizer")
    parser.add_argument("--load_data_workers", default=1, type=int, help="Number of parallel data loaders")
    parser.add_argument("--resume_model", default='', type=str, help="The model path to resume")
    parser.add_argument("--out_alpha", default=1, type=float, help="Task ratio of ssl loss")
    parser.add_argument("--inter_alpha", default=0, type=float, help="Task ratio of intermediate ssl loss")
    parser.add_argument("--inter_layer", default=3, type=float, help="Layer to add intermediate ssl loss")
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

    if not os.path.isdir(args.exp_dir):
        os.makedirs(args.exp_dir)
    
    for key, val in config.items():
        setattr(args, key, val)
    for var in vars(args):
        config[var] = getattr(args, var)
    print("Experiment starts with config {}".format(json.dumps(config, sort_keys=True, indent=4)))
    json.dump(config, open(os.path.join(args.exp_dir, "config.yaml"), 'w'), sort_keys=True, indent=4)

    specaug_conf = Config()
    for key, val in config["spec_aug"].items():
        setattr(specaug_conf, key, val)
    args.specaug_conf = specaug_conf

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
        dist.init_process_group(backend=backend, init_method='tcp://localhost:47829',
                                    world_size=world_size, rank=rank)

    ## 2. Define model and optimizer
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.deterministic = True
    use_cuda = args.use_gpu
    if use_cuda:
        torch.cuda.manual_seed(args.seed)

    assert args.input_size == (args.left_ctx + args.right_ctx + 1) // args.skip_frame * args.n_features
    model = make_model(args.input_size, args)
    optimizer = get_opt(args.opt_type, model, args) 
    
    if args.resume_model:
        if rank == 0:
            print("Loading model from {}".format(args.resume_model))
        checkpoint = torch.load(args.resume_model, map_location='cpu')
        model_state = checkpoint["state_dict"]
        for name, param in model.named_parameters():
            if name not in model_state:
                name = "module." + name
            param.data.copy_(model_state[name])
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
    dataset_types = {"SSLDataset": (SSLDataset, args.batch_size), "DynamicDataset": (DynamicDataset, 1)}
    Dataset, actual_bs = dataset_types[args.dataset_type]

    trainset = Dataset(None, args.train_paths, args)
    if args.use_cmvn:
        trainset._load_cmvn(args.global_cmvn)
    train_loader = SSLDataLoader(trainset, actual_bs, args.padding_idx, num_workers=args.load_data_workers, 
                                       distributed=args.distributed, shuffle=True)
    if args.rank == 0:
        print("Finish Loading training files. Number batches: {}".format(len(train_loader)))

    args.use_specaug = False  # specaug cannot be applied to valid
    validset = Dataset(None, args.dev_paths, args)
    if args.use_cmvn:
        validset._load_cmvn(args.global_cmvn)
    valid_loader = SSLDataLoader(validset, actual_bs, args.padding_idx, num_workers=args.load_data_workers, 
                                        distributed=False, shuffle=False)
    if args.rank == 0:
        print("Finish Loading dev files. Number batches: {}".format(len(valid_loader)))
    
    if args.loss_type in ['l1', 'l2']:
        criterion = PredictLoss(args.k, args.n_generator, args.loss_type)
    elif args.loss_type == 'contrastive':
        criterion = ContrastiveLoss()
    elif args.loss_type == 'hubert':
        criterion = HubertLoss()
    else:
        raise NotImplementedError

    if args.inter_alpha > 0:
        criterion = [criterion, copy.deepcopy(criterion)]
    else:
        criterion = [criterion]

    ## 4. Start training iteratively
    best_loss = 100
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
        train_loss = run_epoch(epoch, train_loader, model, criterion, args, optimizer, is_train=True)
        #train_loss = 0
        model.eval()
        with torch.no_grad():
            valid_loss = run_epoch(epoch, valid_loader, model, criterion, args, is_train=False)
        
        temp_lr = optimizer.param_groups[0]['lr'] if args.opt_type == "normal" else optimizer.optimizer.param_groups[0]['lr']
        if args.distributed:
            average_number = torch.Tensor([train_loss, valid_loss]).float().cuda(args.rank)
            torch.distributed.all_reduce(average_number, op=ReduceOp.SUM)
            train_loss, valid_loss= (average_number / args.world_size).cpu().numpy()
        if args.rank == 0:
            print("Epoch {} done, Train Loss: {:.4f}, Valid Loss: {:.4f} Current LR: {:4e}".format(
                        epoch, train_loss, valid_loss, temp_lr), flush=True)
        
        if args.opt_type == 'normal':
            scheduler.step(valid_wer)

        if epoch > args.save_epoch and args.rank == 0:
            output_file=args.exp_dir + '/model.' + str(epoch) + '.mdl'
            checkpoint = {'epoch': epoch, 'optimizer': optimizer.state_dict(),
                            'state_dict': model.state_dict()}
            torch.save(checkpoint, output_file)
        
        if valid_loss < best_loss:
            best_loss = valid_loss
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
    ssl_losses = util.AverageMeter('SSLLoss', ':.4e')
    inter_losses = util.AverageMeter('InterLoss', ':.4e')
    frame_speed = util.AverageMeter('FrameSpeed', ":.2f")
    progress = util.ProgressMeter(len(dataloader), batch_time, losses, ssl_losses, inter_losses, frame_speed, prefix="Epoch: [{}]".format(epoch))
    
    end = time.time()

    for i, data in enumerate(dataloader):
        start = time.time()
        utt_list, feats, labels, feat_sizes, label_sizes = data
        src, src_mask = feats, (feats[:,:,0] != args.padding_idx).unsqueeze(1)
        tgt_label = labels
        
        if args.use_gpu:
            src, src_mask = src.cuda(), src_mask.cuda()
            tgt_label = tgt_label.cuda()
            feat_sizes = feat_sizes.cuda()
            label_sizes = label_sizes.cuda()
        
        output, inter_out, enc_h, encoded_out = model(src, src_mask, args.out_alpha, args.inter_alpha, args.inter_layer, args)
        bs, max_feat_size, _ = enc_h.size()

        # loss computation
        assert args.out_alpha > 0
        feat_sizes = (feat_sizes * max_feat_size).long()
        frames = feat_sizes.sum().item()
        if args.loss_type in ["l1", "l2"]:
            ssl_loss = criterion[0](output, tgt_label)
        elif args.loss_type in ["hubert", "contrastive"]:
            ssl_loss = criterion[0](output, encoded_out)
        else:
            raise NotImplementedError
        loss = args.out_alpha * ssl_loss
            
        if args.inter_alpha > 0:
            if args.loss_type in ["l1", "l2"]:
                inter_loss = criterion[1](inter_out, tgt_label)
            else:
                inter_loss = criterion[1](inter_out, encoded_out)
            loss += args.inter_alpha * inter_loss
        else:
            inter_loss = torch.Tensor([0])

        losses.update(loss.item(), 1)
        ssl_losses.update(ssl_loss.item(), 1)
        inter_losses.update(inter_loss.item(), 1)
                
        if is_train:
            loss = loss / args.accum_grad
            loss.backward()
            if i % args.accum_grad == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                optimizer.zero_grad()
    
        batch_time.update(time.time() - end)
        frame_speed.update(frames/(time.time()-start))

        if i % args.print_freq == 0 and args.rank == 0:
            progress.print(i)
    return losses.avg
    
if __name__ == '__main__':
    main()


