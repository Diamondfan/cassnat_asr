# 2023 Ruchao Fan
# SPAPL, UCLA

import os
import yaml
import time
import math
import torch
from torch.distributed import ReduceOp

import utils.util as util
from tasks import BaseTask
from data.tokenizer import SPTokenizer, CharTokenizer
from utils.optimizer import get_optim, get_mul_optim
from models import make_unienc_cassnat_model
from utils.beam_decode import ctc_beam_decode
from utils.wer import ctc_greedy_wer, att_greedy_wer


class Config():
    name = 'config'

class UECassNATTask(BaseTask):
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

        super(UECassNATTask, self).__init__(args)

        if mode == "train":
            self._num_updates = 0
            self.set_model(args)
            self.set_optimizer(args)
            args.find_unused_parameters = False if self.model_type != "hubert" else True
            self.load_model(args)
            self.set_dataloader(args)

        if mode == "test":
            args.interctc_alpha = 0
            args.interctc_layer = 0
            args.interce_alpha = 0
            args.interce_layer = 0
            args.label_smooth = 0
            self.set_model(args)
            self.load_test_model(args.resume_model)
            self.set_test_dataloader(args)
            self.model_stats(args.rank, False, False)
            #import deepspeed
            #from fairseq.models.wav2vec.wav2vec2 import TransformerSentenceEncoderLayer
            #self.model = deepspeed.init_inference(self.model,
            #                                        mp_size=2,
            #                                        dtype=torch.float,
            #                                        injection_policy={TransformerSentenceEncoderLayer: ('self_attn.out_proj','fc2')},
            #                                        replace_with_kernel_inject=False)
        
    def set_model(self, args):
        assert args.input_size == (args.left_ctx + args.right_ctx + 1) // args.skip_frame * args.n_features
        if args.model_type in ["transformer", "conformer", "hubert"]:
            self.model = make_unienc_cassnat_model(args.input_size, args)
        else:
            raise NotImplementedError
    
    def load_model(self, args):
        last_checkpoint = os.path.join(args.exp_dir, 'model.last.mdl')
        if os.path.exists(last_checkpoint):
            self.load_checkpoint(last_checkpoint, args.rank, args.use_gpu)
        else:
            model_path = os.path.join(args.resume_model)
            self.load_pretrained_model(model_path, args.rank, args.init_encoder, args.fix_encoder)
    
        self.model_stats(args.rank, args.use_slurm, args.distributed, find_unused_parameters=args.find_unused_parameters)
            
    def load_pretrained_model(self, resume_model, rank, init_encoder, fix_encoder):
        if init_encoder and resume_model:
            if rank == 0:
                print("Loading model from {}".format(resume_model))
            checkpoint = torch.load(resume_model, map_location='cpu')
            
            if self.model_type != "hubert":
                model_state = checkpoint["model_state"]

                for name, param in self.model.named_parameters():
                    if name.split('.')[0] in ['src_embed', 'encoder', 'ctc_generator', 'interctc_generator']:
                        try:
                            if name in model_state:
                                param.data.copy_(model_state[name])
                            else:
                                param.data.copy_(model_state['module.'+name])
                        except:
                            if rank == 0:
                                print("No param of {} in resume model".format(name))

                        if fix_encoder:
                            param.requires_grad = False

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
            if args.rank_model == "n-gram":
                import kenlm
                lm_model = kenlm.Model(args.rnnlm)
            else:
                with open(args.lm_config) as f:
                    lm_config = yaml.safe_load(f)
                lm_args = Config()
                for key, val in lm_config.items():
                    setattr(lm_args, key, val)
                
                lm_args.vocab_size = len(self.tokenizer.vocab)
                if args.rank_model == 'lm':
                    from models.lm import make_model as make_lm_model
                    lm_model = make_lm_model(lm_args)
                
                if args.rank_model == 'at_baseline':
                    from models import make_art_model as make_lm_model
                    lm_args.ctc_alpha = 1
                    lm_args.interctc_alpha = 0
                    lm_args.interctc_layer = 0
                    lm_args.label_smooth = 0
                    lm_model = make_lm_model(args.input_size, lm_args)

                print("Loading language model from {}".format(args.rnnlm))
                checkpoint_lm = torch.load(args.rnnlm, map_location='cpu')
                model_state = checkpoint_lm["model_state"]

                for name, param in lm_model.named_parameters():
                    if name not in model_state and not name.startswith('module'):
                        name = "module." + name

                    if name not in model_state:
                        # to be compatible with model trained with a legacy version
                        name = name.replace('encoder', 'hub_base', 1)
                    
                    param.data.copy_(model_state[name])

                if args.use_gpu:
                    lm_model.cuda()
        else:
            lm_model = None

        self.lm_model = lm_model

    def set_optimizer(self, args):
        def func(module):
            return filter(lambda p: p.requires_grad, module.parameters())
    
        if args.model_type != "hubert":
            pretrained_encoders = [list(func(self.model.src_embed)) + list(func(self.model.encoder))]
            generators = list(func(self.model.ctc_generator))
            if args.interctc_alpha > 0:
                generators += list(func(self.model.interctc_generator))
        else:
            pretrained_encoders = [list(func(self.model.encoder))]  
            generators = list(func(self.model.ctc_generator)) + list(func(self.model.projection_layer)) + list(func(self.model.back_projection_layer))
            if args.interctc_alpha > 0:
                generators += list(func(self.model.interctc_generator)) + list(func(self.model.inter_projection_layer))
            
        decoder_params = list(func(self.model.acembed_extractor)) + list(func(self.model.att_generator)) 
        if args.interce_alpha > 0:
            decoder_params += list(func(self.model.interce_generator))
        if args.use_mlm:
            decoder_params += list(func(self.model.mlm_generator))

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
        
        sample_topk = args.sample_topk
        mask_prob = args.mask_prob
        apply_mask = args.apply_mask

        for epoch in range(self.start_epoch, args.epochs):
            if args.distributed:
                self.train_loader.set_epoch(epoch)
        
            self.train_loader.dataset.use_specaug = (epoch >= args.specaug_start_epoch)    
            self.model.train()
            args.sample_topk = sample_topk
            args.mask_prob = mask_prob
            args.apply_mask = apply_mask
            train_loss, train_wer, train_ctc_wer, train_ctc_wer2 = self.run_one_epoch(epoch, args, is_train=True)
            #train_loss, train_wer, train_ctc_wer, train_ctc_wer2 = 0, 0, 0, 0
            
            self.model.eval()
            with torch.no_grad():
                args.sample_topk = 0
                args.mask_prob = 0
                args.apply_mask = False
                valid_loss, valid_wer, valid_ctc_wer, valid_ctc_wer2 = self.run_one_epoch(epoch, args, is_train=False)
            
            temp_lr = self.optimizer.param_groups[0]['lr'] if args.optim_type == "normal" else self.optimizer.optimizer.param_groups[0]['lr']
            if args.distributed:
                average_number = torch.Tensor([train_loss, train_wer, train_ctc_wer, train_ctc_wer2, valid_loss, valid_wer, valid_ctc_wer, valid_ctc_wer2]).float().cuda(args.rank)
                torch.distributed.all_reduce(average_number, op=ReduceOp.SUM)
                train_loss, train_wer, train_ctc_wer, train_ctc_wer2, valid_loss, valid_wer, valid_ctc_wer, valid_ctc_wer2 = (average_number / args.world_size).cpu().numpy()
    
            if args.rank == 0:
                print("Epoch {} done, Train Loss: {:.4f}, Train WER: {:.4f} Train ctc WER: {:.4f} ctc WER2: {:.4f} Valid Loss: {:.4f} Valid WER: {:.4f} Valid ctc WER: {:.4f} ctc WER2: {:.4f} Current LR: {:4e}".format(
                            epoch, train_loss, train_wer, train_ctc_wer, train_ctc_wer2, valid_loss, valid_wer, valid_ctc_wer, valid_ctc_wer2, temp_lr), flush=True)
             
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
                    checkpoint = {'epoch': epoch, 
                                    'optimizer': self.optimizer.state_dict(),
                                    'model_state': self.model.state_dict(),
                                }
                    torch.save(checkpoint, output_file)
            
            if epoch + 1 - best_epoch >= early_stop_patience:
                if args.rank == 0:
                    print("Early stop since valid_wer doesn't decrease")
                break
        
    def subsequent_mask(self, size):
        ret = torch.ones(size, size, dtype=torch.uint8)
        return torch.tril(ret, out=ret).unsqueeze(0)

    def run_one_epoch(self, epoch, args, is_train=True):
        dataloader = self.train_loader if is_train else self.valid_loader

        batch_time = util.AverageMeter('Time', ':6.3f')
        losses = util.AverageMeter('Loss', ':.4e')
        ctc_losses = util.AverageMeter('CtcLoss', ":.4e")
        ctc_losses2 = util.AverageMeter('CtcLoss2', ":.4e")
        att_losses = util.AverageMeter('AttLoss', ":.4e")
        ctc_wers = util.AverageMeter('CtcWer', ':.4f')
        ctc_wers2 = util.AverageMeter('CtcWer2', ':.4f')
        att_wers = util.AverageMeter('AttWer', ':.4f')
        token_speed = util.AverageMeter('TokenSpeed', ":.2f")
        
        if is_train:
            num_updates = math.ceil(len(dataloader) / args.accum_grad)
            scaler = torch.cuda.amp.GradScaler() if args.use_fp16 else None
        else:
            num_updates = len(dataloader)
            scaler = None

        progress = util.ProgressMeter(num_updates, batch_time, losses, ctc_losses, ctc_losses2, att_losses, ctc_wers, ctc_wers2, \
                                        att_wers, token_speed, prefix="Epoch: [{}]".format(epoch))
        
        end = time.time()
        updates = -1
        
        for i, data in enumerate(dataloader):
            start = time.time()
            if args.model_type != "hubert":
                utt_list, feats, labels, feat_sizes, label_sizes = data
                src, src_mask = feats, (feats[:,:,0] != args.padding_idx).unsqueeze(1)
            else:
                utt_list, feats, labels, feat_sizes, label_sizes = data
                src, src_mask = feats, (feats == args.padding_idx)                                                                   #check first col to see if file is masked
                #src_mask: ratios of unmasked length
                src_mask = src_mask.sum(-1) / src.size(-1)
                      
            tgt_label = labels[:,1:] #exclude sos
            tokens = (tgt_label != args.text_padding_idx).sum().item()
            
            if args.use_gpu:
                src, src_mask = src.cuda(), src_mask.cuda()
                tgt_label = tgt_label.cuda()
                feat_sizes = feat_sizes.cuda()
                label_sizes = label_sizes.cuda()
            
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(scaler is not None)):
                ctc_out, ctc_out2, att_out, loss, ctc_loss, ctc_loss2, att_loss, src_size = self.model(src, src_mask, feat_sizes, tgt_label, label_sizes, args) 
                loss = loss / args.accum_grad

            # unit error rate computation
            ctc_errs, all_tokens = ctc_greedy_wer(ctc_out, tgt_label.cpu().numpy(), src_size.cpu().numpy(), args.text_padding_idx)
            ctc_wers.update(ctc_errs/all_tokens, all_tokens)
            if ctc_out2 is not None:
                ctc_errs2, all_tokens = ctc_greedy_wer(ctc_out2, tgt_label.cpu().numpy(), src_size.cpu().numpy(), args.text_padding_idx)
                ctc_wers2.update(ctc_errs2/all_tokens, all_tokens)
            else:
                ctc_wers2.update(0, 1)
            att_errs, all_tokens = att_greedy_wer(att_out, tgt_label.cpu().numpy(), args.text_padding_idx)
            att_wers.update(att_errs/all_tokens, all_tokens)

            losses.update(loss.item(), tokens)
            ctc_losses.update(ctc_loss.item(), tokens)
            ctc_losses2.update(ctc_loss2.item(), tokens)
            att_losses.update(att_loss.item(), tokens)
            batch_time.update(time.time() - end)
            token_speed.update(tokens/(time.time()-start))

            if is_train:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    if i % args.accum_grad == 0 or i == (len(dataloader) - 1):
                        scaler.unscale_(self.optimizer.optimizer)
                        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.grad_clip)
                        self.optimizer.step(step_optimizer=False)
                        scaler.step(self.optimizer.optimizer)
                        scaler.update()
                        self.optimizer.zero_grad()
                        updates += 1
                        if updates % args.print_freq == 0 and args.rank == 0:
                            progress.print(updates)
                else:
                    loss.backward()
                    if i % args.accum_grad == 0 or i == (len(dataloader) - 1):
                        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.grad_clip)
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        updates += 1
                        
                        if updates % args.print_freq == 0 and args.rank == 0:
                            progress.print(updates)
            else:
                updates += 1
                if updates % args.print_freq == 0 and args.rank == 0:
                    progress.print(updates)

        return losses.avg, att_wers.avg, ctc_wers.avg, ctc_wers2.avg

    def decode(self, args):

        batch_time = util.AverageMeter('Time', ':6.3f')
        progress = util.ProgressMeter(len(self.test_loader), batch_time)
        end = time.time()
        
        out_file = open(args.result_file, 'w')
        if args.print_utt2diff:
            utt2diff = open(os.path.join(os.path.dirname(args.result_file), 'utt2diff'), 'w')

        args.num_correct, args.total = 0, 0
        args.length_correct, args.length_total = 0, 0
        with torch.no_grad():
            self.model.eval()

            if self.lm_model is not None and args.rank_model != "n-gram":
                self.lm_model.eval()

            for i, data in enumerate(self.test_loader):
                if args.model_type != "hubert":
                    utt_list, feats, labels, feat_sizes, label_sizes = data
                    src, src_mask = feats, (feats[:,:,0] != args.padding_idx).unsqueeze(1)
                else:
                    utt_list, feats, labels, feat_sizes, label_sizes = data
                    src, src_mask = feats, (feats == args.padding_idx)
                    #src_mask: ratios of unmasked length
                    src_mask = src_mask.sum(-1) / src.size(-1)
              
                if args.use_gpu:
                    src, src_mask = src.cuda(), src_mask.cuda()
                    feat_sizes = feat_sizes.cuda()
                    labels, label_sizes = labels.cuda(), label_sizes.cuda()

                if args.decode_type == 'ctc_only':
                    recog_results = ctc_beam_decode(self.model, src, src_mask, feat_sizes, self.tokenizer, args, self.lm_model)
                
                elif args.decode_type == 'ctc_att':
                    batch_top_seqs = ctc_beam_decode(self.model, src, src_mask, feat_sizes, self.tokenizer, args, self.lm_model)
                    recog_results, args = self.model.beam_decode(src, src_mask, feat_sizes, self.tokenizer, args, self.lm_model, batch_top_seqs, labels=labels, label_sizes=label_sizes)
                else:
                    recog_results, args = self.model.beam_decode(src, src_mask, feat_sizes, self.tokenizer, args, self.lm_model, labels=labels, label_sizes=label_sizes)
                
                for j in range(len(utt_list)):
                    hyp = []
                    for idx in recog_results[j][0]['hyp']:
                        if idx == self.tokenizer.vocab['sos'] or idx == args.text_padding_idx:
                            continue
                        if idx == self.tokenizer.vocab['eos']:
                            break
                        hyp.append(self.tokenizer.ids_to_tokens[idx])

                    print(utt_list[j]+' '+' '.join(hyp), flush=True, file=out_file)

                    if args.print_utt2diff:  
                        diff = len(recog_results[j][0]['hyp']) - len(labels[j])
                        if diff > 2:
                            diff = 3
                        elif diff < -2:
                            diff = -3
                        else:
                            diff = diff
                        print(utt_list[j] + ' ' + str(diff), flush=True, file=utt2diff)
                
                batch_time.update(time.time() - end)
                if i % args.print_freq == 0:
                    progress.print(i)

            progress.print(i)
            if args.test_hitrate:
                print(args.num_correct, args.total, 1 - args.num_correct / args.total)
                print(args.length_correct, args.length_total, 1 - args.length_correct / args.length_total)
        
        return 0

