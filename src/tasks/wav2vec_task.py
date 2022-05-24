# 2022 Ruchao Fan
# SPAPL

import os
import time
import math
import yaml
import torch
from torch.distributed import ReduceOp

import utils.util as util
from tasks import BaseTask
from utils.optimizer import get_optim
from models.wav2vec_model import make_model
from data.speech_loader import SpeechDataset, DynamicDataset, SSLDataLoader
from utils.loss import Wav2vecLoss

class Config():
    name = 'config'

class Wav2vecTask(BaseTask):
    def __init__(self, mode, args):
        super(Wav2vecTask, self).__init__(args)
        self.vocab = None
        
        self._num_updates = 0
        self.set_model(args)
        self.set_optimizer(args)
        self.load_model(args)
        self.set_dataloader(args)
        self.loss = Wav2vecLoss(args.infonce, args.loss_weights, args.log_keys)
        
    def set_model(self, args):
        assert args.input_size == args.n_features
        self.model = make_model(args.input_size, args)

    def load_model(self, args):
        last_checkpoint = os.path.join(args.exp_dir, 'model.last.mdl')
        if os.path.exists(last_checkpoint):
            self.load_checkpoint(last_checkpoint, args.rank, args.use_gpu)
        else:
            self.load_pretrained_model(args.resume_model, args.rank)
    
        self.model_stats(args.rank, args.use_slurm, args.distributed)
            
    def load_pretrained_model(self, resume_model, rank): 
        if resume_model:
            if rank == 0:
                print("Loading model from {}".format(resume_model))
            checkpoint = torch.load(resume_model, map_location='cpu')
            model_state = checkpoint["model_state"]
            for name, param in self.model.named_parameters():
                if name not in model_state:
                    name = "module." + name
                param.data.copy_(model_state[name])

        self.start_epoch = 0

    def set_dataloader(self, args):
        dataset_types = {"SpeechDataset": (SpeechDataset, args.batch_size), "DynamicDataset": (DynamicDataset, 1)}
        Dataset, actual_bs = dataset_types[args.dataset_type]

        trainset = Dataset(self.vocab, args.train_paths, args)
        if args.use_cmvn:
            trainset._load_cmvn(args.global_cmvn)
        train_loader = SSLDataLoader(trainset, actual_bs, args.padding_idx, num_workers=args.load_data_workers, 
                                       distributed=args.distributed, shuffle=True)
        if args.rank == 0:
            print("Finish Loading training files. Number batches: {}".format(len(train_loader)))

        args.use_specaug = False  # specaug cannot be applied to valid
        validset = Dataset(self.vocab, args.dev_paths, args)
        if args.use_cmvn:
            validset._load_cmvn(args.global_cmvn)
        valid_loader = SSLDataLoader(validset, actual_bs, args.padding_idx, num_workers=args.load_data_workers, 
                                        distributed=False, shuffle=False)
        if args.rank == 0:
            print("Finish Loading dev files. Number batches: {}".format(len(valid_loader)))

        self.train_loader = train_loader
        self.valid_loader = valid_loader
    
    def set_optimizer(self, args):
        self.optimizer = get_optim(args.optim_type, self.model, args) 
        
    def run(self, args):
        best_epoch = 0
        best_code_acc = 0
        early_stop_patience = args.end_patience

        if args.optim_type == 'normal':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=args.anneal_lr_ratio, 
                                                                    patience=args.patience, min_lr=args.min_lr)
    
        for epoch in range(self.start_epoch, args.epochs):
            if args.distributed:
                self.train_loader.set_epoch(epoch)
            
            self.model.train()
            train_logs = self.run_one_epoch(epoch, args, is_train=True)

            self.model.eval()

            with torch.no_grad():
                valid_logs = self.run_one_epoch(epoch, args, is_train=False)
            
            temp_lr = self.optimizer.param_groups[0]['lr'] if args.optim_type == "normal" else self.optimizer.optimizer.param_groups[0]['lr']

            average_list = []
            for key in train_logs:
                average_list.append(train_logs[key])
            for key in valid_logs:
                average_list.append(valid_logs[key])
            
            if args.distributed:    
                average_number = torch.Tensor(average_list).float().cuda(args.rank)
                torch.distributed.all_reduce(average_number, op=ReduceOp.SUM)
                average_list = (average_number / args.world_size).cpu().numpy()
            
            if args.rank == 0:
                strings = "Epoch {} done. Train Stats ".format(epoch)
                idx = 0
                for key in train_logs:
                    strings += "{}: {:.4f} ".format(key, average_list[idx])
                    idx += 1

                strings += "Valid Stats:"
                for key in valid_logs:
                    strings += "{}: {:.4f} ".format(key, average_list[idx])
                    idx += 1

                strings += "Current LR: {:4e}".format(temp_lr)
                print(strings, flush=True)
            
            valid_code_acc = valid_logs['code_acc']
            if args.optim_type == 'normal':
                scheduler.step(valid_code_acc)
            
            if args.rank == 0:
                checkpoint = {'epoch': epoch, 
                              'optimizer': self.optimizer.state_dict(),
                              'model_state': self.model.state_dict(),
                            }                
                last_output_file = args.exp_dir + '/model.last.mdl'
                torch.save(checkpoint, last_output_file)

                if epoch > args.start_saving_epoch:
                    output_file = args.exp_dir + '/model.' + str(epoch) + '.mdl'   
                    torch.save(checkpoint, output_file)

            if valid_code_acc > best_code_acc:
                best_code_acc = valid_code_acc
                best_epoch = epoch
                if args.rank == 0:
                    output_file = args.exp_dir + '/best_model.mdl'
                    checkpoint = {'epoch': epoch, 'optimizer': self.optimizer.state_dict(),
                                    'model_state': self.model.state_dict()}
                    torch.save(checkpoint, output_file)
            
            if epoch + 1 - best_epoch >= early_stop_patience:
                if args.rank == 0:
                    print("Early stop since valid_wer doesn't decrease")
                break
        
    def run_one_epoch(self, epoch, args, is_train=True):
        dataloader = self.train_loader if is_train else self.valid_loader
    
        # define logging meters
        batch_time = util.AverageMeter('Time', ':6.3f')
        loss_total = util.AverageMeter('loss', ':.4e')
        code_acc = util.AverageMeter('code_acc', ':.4f')
        log_meters = [batch_time, loss_total, code_acc]

        for idx in range(len(args.loss_weights) + 1):
            log_meters.append(util.AverageMeter('loss_{}'.format(idx), ':.4e'))
        
        for key in args.log_keys:
            meter = util.AverageMeter(key, ':.4f')
            log_meters.append(meter)

        ntokens = util.AverageMeter('nTokens', ":.2f")
        log_meters.append(ntokens)
        nutts = util.AverageMeter('nUtts', ":.2f")
        log_meters.append(nutts)
        if is_train:
            num_updates = math.ceil(len(dataloader) / args.accum_grad)
        else:
            num_updates = len(dataloader)

        progress = util.ProgressMeter(num_updates, *log_meters, prefix="Epoch: [{}]".format(epoch))
     
        # startt training
        end = time.time()      

        updates = -1
        for i, data in enumerate(dataloader):
            
            start = time.time()
            utt_list, feats = data
            src, src_mask = feats, (feats[:,:,0] != args.padding_idx).unsqueeze(1)
            
            if args.use_gpu:
                src, src_mask = src.cuda(), src_mask.cuda()

            net_output = self.model(src, src_mask, self._num_updates, mask=True)
            if hasattr(self.model, 'module'):
                loss, sample_size, log_output = self.loss(self.model.module, net_output, reduce=True)
            else:
                loss, sample_size, log_output = self.loss(self.model, net_output, reduce=True)
 
            # log printitng
            bs = src.size(0)
            batch_time.update(time.time() - end)
            loss_total.update(log_output["loss"]/math.log(2), 1)
            code_acc.update(log_output['correct']/log_output['count'], log_output['count'])
            
            for idx in range(len(args.loss_weights)+1):
                log_meters[3+idx].update(log_output['loss_{}'.format(idx)]/math.log(2), 1)

            idx += 1
            for key in args.log_keys:
                log_meters[3+idx].update(log_output[key], 1)
                idx += 1
                        
            ntokens.update(sample_size, 1)
            nutts.update(src.size(0), 1)
            
            if is_train:
                loss = loss / args.accum_grad
                loss.backward()
                if (i+1) % args.accum_grad == 0 or (i == (len(dataloader)-1)):
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.grad_clip)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    updates += 1
                    self._num_updates += 1
                    
                    if updates % args.print_freq == 0 and args.rank == 0:
                        progress.print(updates)
            else:
                updates += 1
                if updates % args.print_freq == 0 and args.rank == 0:
                    progress.print(updates)
            
        logging_outputs = {
            "loss": loss_total.avg,
            "code_acc": code_acc.avg,
            "code_ppl": log_meters[-4].avg,
            "prob_ppl": log_meters[-5].avg,
        }
        return logging_outputs

