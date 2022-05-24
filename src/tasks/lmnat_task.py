# 2022 Ruchao Fan
# SPAPL

import os
import yaml
import time
import math
import torch
from torch.distributed import ReduceOp

import utils.util as util
from tasks import BaseTask
from data.vocab import Vocab
from data.tokenizer import SPTokenizer
from utils.optimizer import get_optim, get_mul_optim
from models import make_lmnat_model
from utils.wer import ctc_greedy_wer, att_greedy_wer

class Config():
    name = 'config'

class LMNATTask(BaseTask):
    def __init__(self, mode, args):
        super(LMNATTask, self).__init__(args)
        self.vocab = Vocab(args.vocab_file, args.rank)
        args.vocab_size = self.vocab.n_words
        self.tokenizer = SPTokenizer(args.tokenizer, self.vocab)

        if args.text_encoder_type == "lm":
            self.text_encoder_vocab = Vocab(args.text_encoder_vocab, args.rank)
            args.text_encoder_vocab_size = self.text_encoder_vocab.n_words
            self.text_encoder_tokenizer = SPTokenizer(args.text_encoder_tokenizer, self.text_encoder_vocab)

        elif args.text_encoder_type == "gpt2":
            from models.gpt2.encoder import get_encoder
            model_name = args.gpt2_name
            models_dir = args.text_encoder_path
            self.text_encoder_tokenizer = get_encoder(model_name, models_dir)

        if mode == "train":
            self._num_updates = 0
            self.set_model(args)
            self.set_optimizer(args)
            self.set_dataloader(args)
            self.load_model(args)

        if mode == "test":
            args.rank = 0
            args.interctc_alpha = 0
            args.interctc_layer = 0
            args.interce_alpha = 0
            args.interce_layer = 0
            args.label_smooth = 0
            self.set_model(args)
            self.load_test_model(args.resume_model)
            self.set_test_dataloader(args)
            self.model_stats(0, False, False)
        
    def set_model(self, args):
        assert args.input_size == (args.left_ctx + args.right_ctx + 1) // args.skip_frame * args.n_features
        self.model = make_lmnat_model(args.input_size, args)
    
    def load_model(self, args):
        last_checkpoint = os.path.join(args.exp_dir, 'model.last.mdl')
        if os.path.exists(last_checkpoint):
            self.load_checkpoint(last_checkpoint, args.rank, args.use_gpu)
        else:
            self.load_pretrained_model(args)
    
        self.model_stats(args.rank, args.use_slurm, args.distributed)
            
    def load_pretrained_model(self, args):
        if args.init_encoder and args.resume_model:
            if args.rank == 0:
                print("Loading pretrained encoder from {}".format(args.resume_model))
            checkpoint = torch.load(args.resume_model, map_location='cpu')['model_state']
            
            for name, param in self.model.named_parameters():
                if name.split('.')[0] in ['src_embed', 'encoder', 'ctc_generator', 'interctc_generator']:
                    try:
                        if name in checkpoint:
                            param.data.copy_(checkpoint[name])
                        else:
                            param.data.copy_(checkpoint['module.'+name])
                    except:
                        if rank == 0:
                            print("No param of {} in resume model".format(name))

                    if args.fix_encoder:
                        param.requires_grad = False

        if args.init_text_encoder and args.text_encoder_path:
            if args.rank == 0:
                print("Loading pretrained text encoder from {}".format(args.text_encoder_path))
            
            if args.text_encoder_type == "lm":
                checkpoint = torch.load(args.text_encoder_path, map_location='cpu')['state_dict']

                for name, param in self.model.text_encoder.named_parameters():
                    if name in checkpoint:
                        para.data.copy_(checkpoint[name])
                    else:
                        param.data.copy_(checkpoint['module.'+name])

                    if args.freeze_text_encoder:
                        param.requires_grad = False

            elif args.text_encoder_type == "gpt2":
                from models.gpt2.load_tf_weight import load_tf_weights_in_gpt2
                gpt2_checkpoint = os.path.join(args.text_encoder_path, args.gpt2_name)
                self.model.text_encoder.transformer = load_tf_weights_in_gpt2(self.model.text_encoder.transformer, gpt2_checkpoint)
                #self.model.text_encoder.set_tied()

                for name, param in self.model.text_encoder.named_parameters():
                    if args.freeze_text_encoder:
                        param.requires_grad = False   
        
        self.start_epoch = 0

    def load_test_model(self, resume_model):
        if resume_model:
            print("Loading model from {}".format(resume_model))
            checkpoint = torch.load(resume_model, map_location='cpu')
            model_state = checkpoint["model_state"]
            for name, param in self.model.named_parameters():
                if name not in model_state:
                    name = "module." + name
                param.data.copy_(model_state[name])

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
                
                lm_args.vocab_size = self.vocab.n_words
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
                model_state = checkpoint_lm["model_state"]
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
        def func(module):
            return filter(lambda p: p.requires_grad, module.parameters())
    
        pretrained_encoders = [list(func(self.model.src_embed)), list(func(self.model.encoder)),
            list(func(self.model.ctc_generator))]
        if args.interctc_alpha > 0:
            pretrained_encoders[-1] += list(func(self.model.interctc_generator))

        decoder_params = list(func(self.model.taee)) + list(func(self.model.sad)) \
                            + list(func(self.model.mad)) + list(func(self.model.att_generator)) \
                            + list(func(self.model.dim_map))
        if args.interce_alpha > 0:
            decoder_params += list(func(self.model.interce_generator))

        pretrained_encoders.extend([decoder_params, list(func(self.model.text_encoder))])
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

        for epoch in range(self.start_epoch, args.epochs):
            if args.distributed:
                self.train_loader.set_epoch(epoch)
        
            self.train_loader.dataset.use_specaug = (epoch >= args.specaug_start_epoch)    
            self.model.train()
            args.sample_topk = sample_topk
            train_loss, train_wer, train_ctc_wer = self.run_one_epoch(epoch, args, is_train=True)
            
            self.model.eval()
            with torch.no_grad():
                args.sample_topk = 0
                valid_loss, valid_wer, valid_ctc_wer = self.run_one_epoch(epoch, args, is_train=False)
            
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
        att_losses = util.AverageMeter('AttLoss', ":.4e")
        ctc_wers = util.AverageMeter('CtcWer', ':.4f')
        att_wers = util.AverageMeter('AttWer', ':.4f')
        token_speed = util.AverageMeter('TokenSpeed', ":.2f")
        
        if is_train:
            num_updates = math.ceil(len(dataloader) / args.accum_grad)
        else:
            num_updates = len(dataloader)

        progress = util.ProgressMeter(num_updates, batch_time, losses, ctc_losses, att_losses, ctc_wers, \
                                        att_wers, token_speed, prefix="Epoch: [{}]".format(epoch))
        
        end = time.time()
        updates = -1
        
        for i, data in enumerate(dataloader):
            start = time.time()
            utt_list, feats, labels, feat_sizes, label_sizes = data
            src, src_mask = feats, (feats[:,:,0] != args.padding_idx).unsqueeze(1)
            tgt_label = labels[:,1:]
            tokens = (tgt_label != args.padding_idx).sum().item()
            
            if args.use_gpu:
                src, src_mask = src.cuda(), src_mask.cuda()
                tgt_label = tgt_label.cuda()
                feat_sizes, label_sizes = feat_sizes.cuda(), label_sizes.cuda()

            args.mix_gt_prob = args.mix_gt_prob_max - self._num_updates * (args.mix_gt_prob_max - args.mix_gt_prob_min) / args.mix_gt_steps 
            
            ctc_out, att_out, loss, ctc_loss, att_loss = self.model(src, src_mask, feat_sizes, tgt_label, label_sizes, self.tokenizer, self.text_encoder_tokenizer, args)
            bs, max_feat_size, _ = ctc_out.size()
            feat_sizes = (feat_sizes * max_feat_size).long()
            
            # unit error rate computation
            ctc_errs, all_tokens = ctc_greedy_wer(ctc_out, tgt_label.cpu().numpy(), feat_sizes.cpu().numpy(), args.padding_idx)
            ctc_wers.update(ctc_errs/all_tokens, all_tokens)
            att_errs, all_tokens = att_greedy_wer(att_out, tgt_label.cpu().numpy(), args.padding_idx)
            att_wers.update(att_errs/all_tokens, all_tokens)
            
            losses.update(loss.item(), tokens)
            ctc_losses.update(ctc_loss.item(), tokens)
            att_losses.update(att_loss.item(), tokens)
            batch_time.update(time.time() - end)
            token_speed.update(tokens/(time.time()-start))

            if is_train:
                loss = loss / args.accum_grad
                loss.backward()
                if i % args.accum_grad == 0 or i == (len(dataloader) - 1):
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

        return losses.avg, att_wers.avg, ctc_wers.avg

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
                utt_list, feats, labels, feat_sizes, label_sizes = data
                src, src_mask = feats, (feats[:,:,0] != args.padding_idx).unsqueeze(1)
            
                if args.use_gpu:
                    src, src_mask = src.cuda(), src_mask.cuda()
                    feat_sizes = feat_sizes.cuda()
                    labels, label_sizes = labels.cuda(), label_sizes.cuda()

                if args.decode_type == 'ctc_only':
                    recog_results = ctc_beam_decode(self.model, src, src_mask, feat_sizes, self.vocab, args, self.lm_model)
                else:
                    recog_results, args = self.model.beam_decode(src, src_mask, feat_sizes, args, self.lm_model, self.tokenizer, self.text_encoder_tokenizer, labels, label_sizes)
                
                for j in range(len(utt_list)):
                    hyp = []
                    for idx in recog_results[j][0]['hyp']:
                        if idx == self.vocab.word2index['sos'] or idx == args.padding_idx:
                            continue
                        if idx == self.vocab.word2index['eos']:
                            break
                        hyp.append(self.vocab.index2word[idx])

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

