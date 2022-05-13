# 2022 Ruchao Fan
# SPAPL

import os
import time
import torch
from torch.distributed import ReduceOp

import utils.util as util
from tasks import BaseTask
from data.vocab import Vocab
from utils.wer import ctc_greedy_wer
from utils.optimizer import get_optim
from models import make_ctc_model
from data.speech_loader import SpeechDataset, DynamicDataset, SpeechDataLoader

class Config():
    name = 'config'

class CTCTask(BaseTask):
    def __init__(self, mode, args):
        super(CTCTask, self).__init__(args)
        self.vocab = Vocab(args.vocab_file, args.rank)
        args.vocab_size = self.vocab.n_words

        if mode == "train":
            self.set_model(args)
            self.set_optimizer(args)
            self.load_model(args)
            self.set_dataloader(args)
        if mode == "test":
            args.ctc_alpha = 0
            args.interctc_alpha = 0
            args.interctc_layer = 0
            self.set_model(args)
            self.set_test_dataloader(args)
            self.load_test_model(args.resume_model)
            self.model_stats(0, False, False)

    def set_model(self, args):
        assert args.input_size == (args.left_ctx + args.right_ctx + 1) // args.skip_frame * args.n_features
        self.model = make_ctc_model(args.input_size, args)

    def load_model(self, args):
        last_checkpoint = os.path.join(args.exp_dir, 'model.last.mdl')
        if os.path.exists(last_checkpoint):
            self.load_checkpoint(last_checkpoint, args.rank)
        else:
            self.load_pretrained_model(args.resume_model, args.rank)
    
        self.model_stats(args.rank, args.use_slurm, args.distributed)
            
    def load_pretrained_model(self, resume_model, rank): 
        if resume_model:
            for name, param in self.model.named_parameters():
                if name.split('.')[0] in ["ctc_generator", "interctc_generator"]:
                    print("no need for init for {}, random init".format(name)) 
                    continue
                if name not in model_state:
                    name = "module." + name
            
                if name not in model_state:
                    print("no param init for {}".format(name))
                    continue
                else:
                    param.data.copy_(model_state[name])

        self.start_epoch = 0

    def load_lm_model(self, args):
        if args.lm_weight > 0:
            with open(args.lm_config) as f:
                lm_config = yaml.safe_load(f)
            lm_args = Config()
            for key, val in lm_config.items():
                setattr(lm_args, key, val)

            from models.lm import make_model as make_lm_model
            lm_args.vocab_size = self.vocab.n_words
            lm_model = make_lm_model(lm_args)
            print("Loading language model from {}".format(args.rnnlm))
            checkpoint_lm = torch.load(args.rnnlm, map_location='cpu')
            model_state = checkpoint_lm["state_dict"]
            for name, param in lm_model.named_parameters():
                if name not in model_state:
                    name = "module." + name
                param.data.copy_(model_state[name])
            if args.use_gpu:
                lm_model.cuda()
        else:
            lm_model = None
        self.lm_model = lm_model

    def set_dataloader(self, args):
        dataset_types = {"SpeechDataset": (SpeechDataset, args.batch_size), "DynamicDataset": (DynamicDataset, 1)}
        Dataset, actual_bs = dataset_types[args.dataset_type]

        trainset = Dataset(self.vocab, args.train_paths, args)
        if args.use_cmvn:
            trainset._load_cmvn(args.global_cmvn)
        train_loader = SpeechDataLoader(trainset, actual_bs, args.padding_idx, num_workers=args.load_data_workers, 
                                       distributed=args.distributed, shuffle=True)
        if args.rank == 0:
            print("Finish Loading training files. Number batches: {}".format(len(train_loader)))

        args.use_specaug = False  # specaug cannot be applied to valid
        validset = Dataset(self.vocab, args.dev_paths, args)
        if args.use_cmvn:
            validset._load_cmvn(args.global_cmvn)
        valid_loader = SpeechDataLoader(validset, actual_bs, args.padding_idx, num_workers=args.load_data_workers, 
                                        distributed=False, shuffle=False)
        if args.rank == 0:
            print("Finish Loading dev files. Number batches: {}".format(len(valid_loader)))

        self.train_loader = train_loader
        self.valid_loader = valid_loader

    def set_test_dataloader(self, args):
        args.use_specaug = False
        args.specaug_conf = None
        testset = SpeechDataset(self.vocab, args.test_paths, args)
        if args.use_cmvn:
            testset._load_cmvn(args.global_cmvn)
        test_loader = SpeechDataLoader(testset, args.batch_size, args.padding_idx, num_workers=args.load_data_workers, shuffle=False)
        print("Finish Loading test files. Number batches: {}".format(len(test_loader)))
        self.test_loader = test_loader

    def set_optimizer(self, args):
        self.optimizer = get_optim(args.optim_type, self.model, args) 
        
    def run(self, args):
        best_epoch = 0
        best_wer = 100
        early_stop_patience = args.end_patience

        if args.optim_type == 'normal':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=args.anneal_lr_ratio, 
                                                                    patience=args.patience, min_lr=args.min_lr)
    
        for epoch in range(self.start_epoch, args.epochs):
            if args.distributed:
                self.train_loader.set_epoch(epoch)
            
            self.model.train()
            train_loss, train_ctc_wer, train_interctc_wer = self.run_one_epoch(epoch, args, is_train=True)
            
            self.model.eval()
            with torch.no_grad():
                valid_loss, valid_ctc_wer, valid_interctc_wer = self.run_one_epoch(epoch, args, is_train=False)
            
            temp_lr = self.optimizer.param_groups[0]['lr'] if args.optim_type == "normal" else self.optimizer.optimizer.param_groups[0]['lr']
            lr_random = self.optimizer.optimizer.param_groups[-1]['lr']
            if args.distributed:
                average_number = torch.Tensor([train_loss, train_ctc_wer, train_interctc_wer, valid_loss, valid_ctc_wer, valid_interctc_wer]).float().cuda(args.rank)
                torch.distributed.all_reduce(average_number, op=ReduceOp.SUM)
                train_loss, train_ctc_wer, train_interctc_wer, valid_loss, valid_ctc_wer, valid_interctc_wer = (average_number / args.world_size).cpu().numpy()
            
            if args.rank == 0:
                print("Epoch {} done, Train: Loss={:.4f}, ctc WER={:.4f} interctc WER={:.4f} Valid: Loss={:.4f} ctc WER={:.4f} interctc WER={:.4f} LR_pretrained: {:.4e}, LR_random: {:4e}".format(
                            epoch, train_loss, train_ctc_wer, train_interctc_wer, valid_loss, valid_ctc_wer, valid_interctc_wer, temp_lr, lr_random), flush=True)

            if args.optim_type == 'normal':
                scheduler.step(valid_wer)
            
            if epoch > args.start_saving_epoch and args.rank == 0:
                output_file = args.exp_dir + '/model.' + str(epoch) + '.mdl'
                checkpoint = {'epoch': epoch, 
                              'optimizer': self.optimizer.state_dict(),
                              'model_state': self.model.state_dict(),
                            }
                torch.save(checkpoint, output_file)
                last_output_file = args.exp_dir + '/model.last.mdl'
                torch.save(checkpoint, last_output_file)
            
            if valid_ctc_wer < best_wer:
                best_wer = valid_ctc_wer
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
        batch_time = util.AverageMeter('Time', ':6.3f')
        losses = util.AverageMeter('Loss', ':.4e')
        ctc_losses = util.AverageMeter('CtcLoss', ":.4e")
        interctc_losses = util.AverageMeter('InterctcLoss', ":.4e")
        ctc_wers = util.AverageMeter('CtcWer', ':.4f')
        interctc_wers = util.AverageMeter('InterCtcWer', ':.4f')
        token_speed = util.AverageMeter('TokenSpeed', ":.2f")
        progress = util.ProgressMeter(len(dataloader), batch_time, losses, ctc_losses, interctc_losses, ctc_wers, interctc_wers, token_speed, prefix="Epoch: [{}]".format(epoch))
        
        end = time.time()

        for i, data in enumerate(dataloader):
            start = time.time()
            utt_list, feats, labels, feat_sizes, label_sizes = data
            src, src_mask = feats, (feats[:,:,0] != args.padding_idx).unsqueeze(1)       
            tgt_label = labels[:,1:-1]
            tokens = (tgt_label != args.padding_idx).sum().item()
            
            if args.use_gpu:
                src, src_mask = src.cuda(), src_mask.cuda()
                tgt_label = tgt_label.cuda()
                feat_sizes = feat_sizes.cuda()
                label_sizes = label_sizes.cuda()
            
            ctc_out, inter_out, loss, ctc_loss, interctc_loss, feat_sizes = self.model(src, src_mask, feat_sizes, tgt_label, label_sizes)

            ctc_errs, all_tokens = ctc_greedy_wer(ctc_out, tgt_label.cpu().numpy(), feat_sizes.cpu().numpy(), args.padding_idx)
            ctc_wers.update(ctc_errs/all_tokens, all_tokens)

            if args.interctc_alpha > 0:
                interctc_errs, all_tokens = ctc_greedy_wer(inter_out, tgt_label.cpu().numpy(), feat_sizes.cpu().numpy(), args.padding_idx)
                interctc_wers.update(interctc_errs/all_tokens, all_tokens)
            else:
                interctc_wers.update(0, all_tokens)

            losses.update(loss.item(), tokens)
            ctc_losses.update(ctc_loss.item(), tokens)
            interctc_losses.update(interctc_loss.item(), tokens)
                    
            if is_train:
                loss = loss / args.accum_grad
                loss.backward()
                if (i+1) % args.accum_grad == 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.grad_clip)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
        
            batch_time.update(time.time() - end)
            token_speed.update(tokens/(time.time()-start))
            if i % args.print_freq == 0 and args.rank == 0:
                progress.print(i)

        return losses.avg, ctc_wers.avg, interctc_wers.avg

    def decode(self, args):    
        batch_time = util.AverageMeter('Time', ':6.3f')
        progress = util.ProgressMeter(len(self.test_loader), batch_time)
        end = time.time()
    
        out_file = open(args.result_file, 'w')
        with torch.no_grad():
            self.model.eval()
            if self.lm_model is not None:
                self.lm_model.eval()

            for i, data in enumerate(self.test_loader):
                utt_list, feats, _, feat_sizes, _ = data
                src, src_mask = feats, (feats[:,:,0] != args.padding_idx).unsqueeze(1)
        
                if args.use_gpu:
                    src, src_mask = src.cuda(), src_mask.cuda()
                    feat_sizes = feat_sizes.cuda()

                if args.decode_type == "greedy":
                    recog_results = self.model.greedy_decode(src, src_mask, feat_sizes, self.vocab, args)
                elif args.decode_type == "beam":
                    recog_results = self.model.beam_decode(src, src_mask, feat_sizes, self.vocab, args, lm_model)
                else:
                    raise NotImplementedError

                for j in range(len(utt_list)):
                    hyp = []
                    for idx in recog_results[j][0]['hyp']:
                        if idx == self.vocab.word2index['sos'] or idx == args.padding_idx:
                            continue
                        if idx == self.vocab.word2index['eos']:
                            break
                        hyp.append(self.vocab.index2word[idx])
                    #print(utt_list[j]+' '+' '.join(hyp))
                    print(utt_list[j]+' '+' '.join(hyp), flush=True, file=out_file)
    
                batch_time.update(time.time() - end)
                if i % args.print_freq == 0:
                    progress.print(i)
            progress.print(i)
        return 0

    
