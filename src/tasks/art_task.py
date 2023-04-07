# 2022 Ruchao Fan
# SPAPL

import os
import math
import time
import yaml
import torch
from torch.distributed import ReduceOp

import utils.util as util
from tasks import BaseTask
from data.tokenizer import SPTokenizer, CharTokenizer
from utils.optimizer import get_optim, get_mul_optim
from models import make_art_model
from utils.wer import ctc_greedy_wer, att_greedy_wer
from utils.beam_decode import ctc_beam_decode

class Config():
    name = 'config'

class ArtTask(BaseTask):
    def __init__(self, mode, args):
        self.model_type = args.model_type
        if args.use_BERT_tokenizer:
            from models.bert.tokenization import get_tokenizer
            self.tokenizer = get_tokenizer(args.bert_name, args.bert_path)
        elif args.tokenizer:
            self.tokenizer = SPTokenizer(args.tokenizer, args.vocab_file)
        else:
            self.tokenizer = CharTokenizer(args.vocab_file)
        args.vocab_size = len(self.tokenizer.vocab)

        super(ArtTask, self).__init__(args)

        if mode == "train":
            self.set_model(args)
            self.set_optimizer(args)
            args.find_unused_parameters = False if self.model_type != "hubert" else True
            self.load_model(args)
            self.set_dataloader(args)
        if mode == "test":
            args.rank = 0
            args.ctc_alpha = 0
            args.interctc_alpha = 0
            args.interctc_layer = 0
            args.label_smooth = 0
            self.set_model(args)
            self.set_test_dataloader(args)
            self.load_test_model(args.resume_model)
            self.model_stats(0, False, False)
        
    def set_model(self, args):
        assert args.input_size == (args.left_ctx + args.right_ctx + 1) // args.skip_frame * args.n_features
        if args.model_type in ["transformer", "conformer", "hubert"]:
            self.model = make_art_model(args.input_size, args)
        else:
            raise NotImplementedError

    def load_model(self, args):
        #Either load from checkpoint, or load model defined in config file
        last_checkpoint = os.path.join(args.exp_dir, 'model.last.mdl')
        if os.path.exists(last_checkpoint):
            self.load_checkpoint(last_checkpoint, args.rank, args.use_gpu) #load checkpoint from baseclass
        else:
            self.load_pretrained_model(args.resume_model, args.rank, args.init_encoder, args.fix_encoder)
    
        self.model_stats(args.rank, args.use_slurm, args.distributed, args.find_unused_parameters)

    def load_pretrained_model(self, resume_model, rank, init_encoder, fix_encoder): 
        if init_encoder and resume_model:
            if rank == 0:
                print("Loading model from {}".format(resume_model))
            checkpoint = torch.load(resume_model, map_location='cpu')

            if self.model_type != "hubert":
                model_state = checkpoint["model_state"]

                for name, param in self.model.named_parameters():
                    if name not in model_state:
                        name = "module." + name
                    param.data.copy_(model_state[name])

            else:
                model_state = checkpoint["model"]
                for name, param in self.model.named_parameters(): #decoder is trained from scratch
                    if name.split('.')[0] == "module":
                        name = '.'.join(name.split('.')[1:]).strip()
                    
                    if name.split('.')[0] == "encoder":
                        name_sp = '.'.join(name.split('.')[1:]).strip()
                        try:
                            param.data.copy_(model_state[name_sp])
                        except:
                            if rank == 0:
                                print("No param of {} in resume model".format(name))
                        
                        if fix_encoder:
                            param.requires_grad = False

        self.start_epoch = 0

    def load_lm_model(self, args):
        if args.lm_weight > 0 or args.ctc_lm_weight > 0:
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

    def set_optimizer(self, args):
        #if not multi optimiser, we directly set optimiser, else 
        if not args.multi_optim:
            self.optimizer = get_optim(args.optim_type, self.model, args)
        else:
            #obtain all params that need updating (requires_grad) and pass them to multi_opt
            def func(module):
                return filter(lambda p: p.requires_grad, module.parameters())
           
            pretrained_encoders = [list(func(self.model.encoder))]
            generators = list(func(self.model.ctc_generator)) + list(func(self.model.projection_layer))
            if args.interctc_alpha > 0:
                generators += list(func(self.model.interctc_generator)) + list(func(self.model.interctc_projection_layer))

            decoder_params = list(func(self.model.tgt_embed)) + list(func(self.model.decoder)) + list(func(self.model.att_generator))

            pretrained_encoders.extend([generators])
            pretrained_encoders.extend([decoder_params])
            update_param_groups = pretrained_encoders
            self.optimizer = get_mul_optim(args.optim_type, update_param_groups, args)

    def run(self, args):
        best_epoch = 0
        best_wer = 100
        early_stop_patience = args.end_patience

        if args.optim_type == 'normal':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=args.anneal_lr_ratio, 
                                                                        patience=args.patience, min_lr=args.min_lr)

        if args.use_fp16:
            scaler = torch.cuda.amp.GradScaler()
        else:
            scaler = None                                                         
    
        mask_prob = args.mask_prob
        for epoch in range(self.start_epoch, args.epochs):
            if args.distributed:
                self.train_loader.set_epoch(epoch)
            
            self.model.train()
            args.mask_prob = mask_prob
            train_loss, train_wer, train_ctc_wer = self.run_one_epoch(epoch, scaler, args, is_train=True)
            #train_loss, train_wer, train_ctc_wer = 0, 0, 0
            
            self.model.eval()
            with torch.no_grad():
                args.mask_prob = 0
                valid_loss, valid_wer, valid_ctc_wer = self.run_one_epoch(epoch, scaler, args, is_train=False)
                #valid_loss, valid_wer, valid_ctc_wer = 0, 0, 0
            temp_lr = self.optimizer.param_groups[0]['lr'] if args.optim_type == "normal" else self.optimizer.optimizer.param_groups[0]['lr']
            if args.distributed:
                average_number = torch.Tensor([train_loss, train_wer, train_ctc_wer, valid_loss, valid_wer, valid_ctc_wer]).float().cuda(args.rank)
                torch.distributed.all_reduce(average_number, op=ReduceOp.SUM)
                train_loss, train_wer, train_ctc_wer, valid_loss, valid_wer, valid_ctc_wer = (average_number / args.world_size).cpu().numpy()
            
            if args.rank == 0:
                print("Epoch {} done, Train Loss: {:.4f}, Train WER: {:.4f} Train ctc WER: {:.4f} Valid Loss: {:.4f} Valid WER: {:.4f} Valid ctc WER: {:.4f} Current LR: {:4e}".format(
                            epoch, train_loss, train_wer, train_ctc_wer, valid_loss, valid_wer, valid_ctc_wer, temp_lr), flush=True)
            
            if args.optim_type == 'normal':
                scheduler.step(valid_wer)
            
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

            if valid_wer < best_wer:
                best_wer = valid_wer
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
        
    def subsequent_mask(self, size):
        ret = torch.ones(size, size, dtype=torch.uint8)
        return torch.tril(ret, out=ret).unsqueeze(0)

    def run_one_epoch(self, epoch, scaler, args, is_train=True):
        dataloader = self.train_loader if is_train else self.valid_loader
    
        batch_time = util.AverageMeter('Time', ':6.3f')
        losses = util.AverageMeter('Loss', ':.4e')
        ctc_losses = util.AverageMeter('CtcLoss', ":.4e")
        att_losses = util.AverageMeter('AttLoss', ":.4e")
        ctc_wers = util.AverageMeter('CtcWer', ':.4f')
        att_wers = util.AverageMeter('AttWer', ':.4f')
        token_speed = util.AverageMeter('TokenSpeed', ":.2f")
        
        if is_train:
            num_updates = math.ceil(len(dataloader) / args.accum_grad)
        else:
            num_updates = len(dataloader)

        progress = util.ProgressMeter(num_updates, batch_time, losses, ctc_losses, att_losses, ctc_wers, att_wers, token_speed, prefix="Epoch: [{}]".format(epoch))
     
        end = time.time()
        updates = -1       
        for i, data in enumerate(dataloader):
            start = time.time()
            
            if args.model_type != "hubert":
                utt_list, feats, labels, feat_sizes, label_sizes = data
                src, src_mask = feats, (feats[:,:,0] != args.text_padding_idx).unsqueeze(1)
            else:
                utt_list, feats, labels, feat_sizes, label_sizes = data
                src, src_mask = feats, (feats == args.padding_idx) #all audio except files which are masked
                src_mask = src_mask.sum(-1) / src.size(-1)

            tgt, tgt_label = labels[:,:-1], labels[:,1:]
            tgt_mask = (tgt != args.text_padding_idx).unsqueeze(1) 
            tgt_mask = tgt_mask & self.subsequent_mask(tgt.size(-1)).type_as(tgt_mask)
            tokens = (tgt_label != args.padding_idx).sum().item()
            
            if args.use_gpu:
                src, src_mask, tgt, tgt_mask = src.cuda(), src_mask.cuda(), tgt.cuda(), tgt_mask.cuda()
                tgt_label = tgt_label.cuda()
                feat_sizes = feat_sizes.cuda()
                label_sizes = label_sizes.cuda()
            
            try:
                if scaler is not None:
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        ctc_out, att_out, loss, att_loss, ctc_loss, feat_sizes = self.model(src, tgt, src_mask, tgt_mask, feat_sizes, label_sizes, tgt_label, args.mask_prob)
                else:
                    ctc_out, att_out, loss, att_loss, ctc_loss, feat_sizes = self.model(src, tgt, src_mask, tgt_mask, feat_sizes, label_sizes, tgt_label, args.mask_prob)

                if is_train:
                    if scaler is not None:
                        with torch.cuda.amp.autocast(dtype=torch.float16):
                            loss = loss / args.accum_grad
                        scaler.scale(loss).backward()
                        if (i+1) % args.accum_grad == 0 or i == (len(dataloader) - 1):
                            scaler.unscale_(self.optimizer.optimizer)
                            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.grad_clip)
                            self.optimizer.step(step_optimizer=False)
                            scaler.step(self.optimizer.optimizer)
                            scaler.update()
                            if args.disable_ls and self.optimizer._step == self.optimizer.s_decay:
                                if args.rank == 0:
                                    print("Disable label smoothing from here.")
                                self.model.att_loss.set_smoothing(0.0)
                            self.optimizer.zero_grad()
                            updates += 1
                            if updates % args.print_freq == 0 and args.rank == 0:
                                progress.print(updates)
                    else:
                        loss = loss / args.accum_grad
                        loss.backward()
                        if (i+1) % args.accum_grad == 0 or i == (len(dataloader) - 1):
                            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.grad_clip)
                            self.optimizer.step()
                            if args.disable_ls and self.optimizer._step == self.optimizer.s_decay:
                                if args.rank == 0:
                                    print("Disable label smoothing from here.")
                                self.model.att_loss.set_smoothing(0.0)
                            self.optimizer.zero_grad()
                            updates += 1
                            
                            if updates % args.print_freq == 0 and args.rank == 0:
                                progress.print(updates)
                else:
                    updates += 1
                    if updates % args.print_freq == 0 and args.rank == 0:
                        progress.print(updates)
            except RuntimeError as err:
                print("{}!, Skip batch, cuda out of memory".format(err))
                continue

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
            batch_time.update(time.time() - end)
            token_speed.update(tokens/(time.time()-start))  

        return losses.avg, att_wers.avg, ctc_wers.avg

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
                if self.model_type != "hubert":
                    utt_list, feats, _, feat_sizes, _ = data
                    src, src_mask = feats, (feats[:,:,0] != args.text_padding_idx).unsqueeze(1)
                else:
                    utt_list, feats, _, feat_sizes, _ = data
                    src, src_mask = feats, (feats == args.padding_idx)
                    src_mask = src_mask.sum(-1) / src.size(-1)

                if args.use_gpu:
                    src, src_mask = src.cuda(), src_mask.cuda()
                    feat_sizes = feat_sizes.cuda()

                if args.decode_type == 'ctc_only':
                    recog_results = ctc_beam_decode(self.model, src, src_mask, feat_sizes, self.tokenizer.vocab, args, self.lm_model)
                elif args.decode_type == 'ctc_correct':
                    recog_results = self.model.fast_decode_with_ctc(src, src_mask, self.tokenizer, args, self.lm_model)
                elif args.decode_type == "ctc_att":
                    recog_results = self.model.beam_decode(src, src_mask, self.tokenizer, args, self.lm_model)
                else:
                    raise NotImplementedError

                for j in range(len(utt_list)):
                    hyp = []
                    for idx in recog_results[j][0]['hyp']:
                        if idx == self.tokenizer.vocab['sos'] or idx == args.padding_idx:
                            continue
                        if idx == self.tokenizer.vocab['eos']:
                            break
                        hyp.append(self.tokenizer.ids_to_tokens[idx])
                    #print(utt_list[j]+' '+' '.join(hyp))
                    print(utt_list[j]+' '+' '.join(hyp), flush=True, file=out_file)
    
                batch_time.update(time.time() - end)
                if i % args.print_freq == 0:
                    progress.print(i)
            progress.print(i)
        return 0

