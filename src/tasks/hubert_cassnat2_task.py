
# 2022 Ruchao Fan
# SPAPL

import os
import yaml
import math
import time
import torch
from torch.distributed import ReduceOp

import utils.util as util
from tasks import BaseTask
from data.tokenizer import SPTokenizer
from utils.optimizer import get_mul_optim, get_optim
from models import make_hubert_cassnat2_model
from utils.wer import ctc_greedy_wer, att_greedy_wer
from utils.beam_decode import ctc_beam_decode
from data.audio_loader import HubertDataset, HubertLoader

class Config():
    name = 'config'

class HubertCASSNAT2Task(BaseTask):
    def __init__(self, mode, args):
        if args.use_BERT_tokenizer:
            from models.bert.tokenization import get_tokenizer
            self.tokenizer = get_tokenizer(args.bert_name, args.bert_path)
        else:
            self.tokenizer = SPTokenizer(args.tokenizer, args.vocab_file)
        args.vocab_size = len(self.tokenizer.vocab)

        super(HubertCASSNAT2Task, self).__init__(args)

        if args.text_encoder_type == "lm":
            self.text_encoder_vocab = Vocab(args.text_encoder_vocab, args.rank)
            args.text_encoder_vocab_size = self.text_encoder_vocab.n_words
            self.text_encoder_tokenizer = SPTokenizer(args.text_encoder_tokenizer, self.text_encoder_vocab)

        elif args.text_encoder_type == "gpt2":
            from models.gpt2.encoder import get_encoder
            model_name = args.gpt2_name
            models_dir = args.text_encoder_path
            self.text_encoder_tokenizer = get_encoder(model_name, models_dir)

        elif args.text_encoder_type == "bert":
            from models.bert.tokenization import get_tokenizer
            self.text_encoder_tokenizer = get_tokenizer(args.bert_name, args.text_encoder_path)

        if mode == "train":
            self._num_updates = 0
            self.set_model(args)
            print("done setting model")
            self.set_optimizer(args) 
            print("done setting opt")
            args.find_unused_parameters = True
            self.load_model(args)
            print("done loading model")
            self.set_dataloader(args)
            print("done setting dataloader")
            
        if mode == "test":
            args.rank = 0
            args.interctc_alpha = 0
            args.interctc_layer = 0
            args.interce_alpha = 0.1
            args.interce_layer = 5
            args.label_smooth = 0
            self.set_model(args)
            self.set_test_dataloader(args)
            self.load_test_model(args.resume_model)
            self.model_stats(0, False, False)
        
    def set_model(self, args):
        self.model = make_hubert_cassnat2_model(args)

    def load_model(self, args):
        #Either load from checkpoint, or load model defined in config file
        last_checkpoint = os.path.join(args.exp_dir, 'model.last.mdl')
        if os.path.exists(last_checkpoint):
            self.load_checkpoint(last_checkpoint, args.rank, args.use_gpu, args.freeze_text_encoder) #load checkpoint from baseclass
        else:
            model_path = os.path.join(args.exp_dir, args.resume_model)
            self.load_pretrained_model(model_path, args.rank, args.init_encoder, args.fix_encoder, args)
    
        self.model_stats(args.rank, args.use_slurm, args.distributed, args.find_unused_parameters)

    def load_checkpoint(self, checkpoint, rank, use_cuda, freeze_text_encoder):
        if rank == 0:
            print("Loading checkpoint from {}".format(checkpoint))
        checkpoint = torch.load(checkpoint, map_location='cpu')
        model_state = checkpoint["model_state"]
        
        for name, param in self.model.named_parameters():
            if name not in model_state:
                name = "module." + name
          
            if "module.text_encoder.bert.pooler.dense" in  name:
                continue 
            param.data.copy_(model_state[name])

        if freeze_text_encoder:
            for name, param in self.model.text_encoder.named_parameters():
                param.requires_grad = False
        
        self.optimizer.load_state_dict(checkpoint['optimizer'], rank, use_cuda)
        self._num_updates = self.optimizer._step
        self.start_epoch = checkpoint['epoch'] + 1

    def load_pretrained_model(self, resume_model, rank, init_encoder, fix_encoder, args): 
        #first load and obtain model param names, then extract params, and then set epoch
        #resume_model refers to model_path
        if init_encoder and resume_model:
            if rank == 0:
                print("Loading model from {}".format(resume_model))
            # checkpoint = torch.load(resume_model, map_location='cpu')['model_state']
            model_state = torch.load(resume_model, map_location='cpu')
            model = model_state["model"]
            #parse named_parameters for values in dict and then copy appropriate params
            # for param_tensor in model:
            #     print(param_tensor, "\t", model[param_tensor].size())

            for name, param in self.model.named_parameters(): #decoder for cassnat is trained from scratch
                #if name.split('.')[0] in ['src_embed', 'encoder', 'ctc_generator', 'interctc_generator']:
                if name.split('.')[0] == "module":

                    n_name = '.'.join(name.split('.')[1:]).strip()

                    if n_name.split('.')[0] == "hub_base":
                        
                        name_sp = '.'.join(n_name.split('.')[1:]).strip()
                        try:
                            param.data.copy_(model[name_sp])
                        
                        except:
                            if rank == 0:
                                print("No param of {} in resume model".format(name))

                if name.split('.')[0] == "hub_base":
                    try:
                        name_sp = '.'.join(name.split('.')[1:]).strip()
                        param.data.copy_(model[name_sp])
                        # param.requires_grad = False

                    except:
                        if rank == 0:
                            print("No param of {} in resume model".format(name))

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

            elif args.text_encoder_type == "gpt2":
                from models.gpt2.load_tf_weight import load_tf_weights_in_gpt2
                gpt2_checkpoint = os.path.join(args.text_encoder_path, args.gpt2_name)
                self.model.text_encoder.transformer = load_tf_weights_in_gpt2(self.model.text_encoder.transformer, gpt2_checkpoint)
                #self.model.text_encoder.set_tied()

            elif args.text_encoder_type == "bert":
                from models.bert.load_tf_weight import load_tf_weights_in_bert
                bert_checkpoint = os.path.join(args.text_encoder_path, args.bert_name, "bert_model.ckpt")
                self.model.text_encoder = load_tf_weights_in_bert(self.model.text_encoder, bert_checkpoint)
                self.model.text_encoder.remove_unused_module()
           
            for name, param in self.model.text_encoder.named_parameters():
                if args.freeze_text_encoder:
                    param.requires_grad = False   

            self.model.text_encoder.remove_unused_module()

        self.start_epoch = 0

    def load_test_model(self, resume_model):
        if resume_model:
            print("Loading model from {}".format(resume_model))
            checkpoint = torch.load(resume_model, map_location='cpu')
            model_state = checkpoint["model_state"]
            for name, param in self.model.named_parameters():
                if name not in model_state:
                    name = "module." + name
                if name not in model_state:
                    continue
                param.data.copy_(model_state[name])

    def load_lm_model(self, args):
        if args.lm_weight > 0 or args.ctc_lm_weight > 0:
            #either load ngram model from kenlm, or load from yaml file
            if args.rank_model == "n-gram":
                import kenlm
                lm_model = kenlm.Model(args.rnnlm)

            else:
                with open(args.lm_config) as f:
                    lm_config = yaml.safe_load(f)

                #using args from yml file, we intiliase lm model, and then load pretrained model in

                lm_args = Config()

                for key, val in lm_config.items():
                    setattr(lm_args, key, val)
                
                lm_args.vocab_size = len(self.tokenizer.vocab)

                if args.rank_model == 'lm':
                    from models.lm import make_model as make_lm_model
                    lm_model = make_lm_model(lm_args)
                
                if args.rank_model == 'at_baseline':
                    from models.hubert.hubert_art_model import make_model as make_lm_model
                    lm_args.interctc_alpha = 0
                    lm_args.interctc_layer = 0
                    lm_args.ctc_alpha = 0
                    lm_args.label_smooth = 0
                    lm_model = make_lm_model(lm_args)
                
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
        #if not multi optimiser, we directly set optimiser, else 
        if not args.multi_optim:
            self.optimizer = get_optim(args.optim_type, self.model, args)
        else:
            #obtain all params that need updating (requires_grad) and pass them to multi_opt
            def func(module):
                return filter(lambda p: p.requires_grad, module.parameters())
           
            pretrained_encoders = [list(func(self.model.hub_base))]
            generators = list(func(self.model.ctc_generator)) + list(func(self.model.projection_layer))
            if args.interctc_alpha > 0:
                generators += list(func(self.model.interctc_generator)) + list(func(self.model.interctc_projection_layer))

            decoder_params = list(func(self.model.acembed_extractor)) + list(func(self.model.embed_mapper)) \
                                + list(func(self.model.decoder)) + list(func(self.model.att_generator))
            if args.interce_alpha > 0:
                decoder_params += list(func(self.model.interce_generator))

            pretrained_encoders.extend([generators])
            pretrained_encoders.extend([decoder_params])
            pretrained_encoders.extend([list(func(self.model.text_encoder))])
            update_param_groups = pretrained_encoders
            self.optimizer = get_mul_optim(args.optim_type, update_param_groups, args) 

    def set_dataloader(self, args):
        trainset = HubertDataset(self.tokenizer, args.train_paths,args)
        train_loader = HubertLoader(trainset, 1, args.text_padding_idx, args.padding_idx, num_workers=args.load_data_workers, 
                                    distributed=args.distributed, shuffle=True)
        if args.rank == 0:
            print("Finish Loading training files. Number batches: {}".format(len(train_loader)))

        validset = HubertDataset(self.tokenizer, args.dev_paths, args)

        valid_loader = HubertLoader(validset, 1, args.text_padding_idx, args.padding_idx, num_workers=args.load_data_workers, 
                                        distributed=False, shuffle=False)
        if args.rank == 0:
            print("Finish Loading dev files. Number batches: {}".format(len(valid_loader)))

        self.train_loader = train_loader
        self.valid_loader = valid_loader
    
    def set_test_dataloader(self, args):
        #load test set
        args.use_specaug = False
        args.specaug_conf = None

        testset = HubertDataset(self.tokenizer, args.test_paths, args)
        test_loader = HubertLoader(testset, 1, args.text_padding_idx, args.padding_idx, num_workers=args.load_data_workers, shuffle=False)

        self.test_loader = test_loader
        
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
        
        sample_topk = args.sample_topk
        mask_prob = args.mask_prob
        for epoch in range(self.start_epoch, args.epochs):
            #if distributed, set sampler using distributed sampler
            if args.distributed:
                self.train_loader.set_epoch(epoch)
                # self.model = DDP(self.model, device_ids=[args.rank], output_device=args.rank, find_unused_parameters=True)

            #set model to train mode, and train for one epoch
            self.model.train()
            args.sample_topk = sample_topk
            args.mask_prob = mask_prob
            train_loss, train_wer, train_ctc_wer = self.run_one_epoch(epoch, scaler, args, is_train=True)
            #train_loss, train_wer, train_ctc_wer = 0, 0, 0
            #set to eval mode and eval same epoch
            self.model.eval()
            with torch.no_grad():
                args.sample_topk = 0
                args.mask_prob = 0
                valid_loss, valid_wer, valid_ctc_wer = self.run_one_epoch(epoch, scaler, args, is_train=False)
            
            temp_lr = self.optimizer.param_groups[0]['lr'] if args.optim_type == "normal" else self.optimizer.optimizer.param_groups[0]['lr']
            #if distributed we need to average stats
            if args.distributed:
                average_number = torch.Tensor([train_loss, train_wer, train_ctc_wer, valid_loss, valid_wer, valid_ctc_wer]).float().cuda(args.rank)
                torch.distributed.all_reduce(average_number, op=ReduceOp.SUM)
                train_loss, train_wer, train_ctc_wer, valid_loss, valid_wer, valid_ctc_wer = (average_number / args.world_size).cpu().numpy()
    
            if args.rank == 0:
                print("Epoch {} done, Train Loss: {:.4f}, Train WER: {:.4f} Train ctc WER: {:.4f} Valid Loss: {:.4f} Valid WER: {:.4f} Valid ctc WER: {:.4f} Current LR: {:4e}".format(
                            epoch, train_loss, train_wer, train_ctc_wer, valid_loss, valid_wer, valid_ctc_wer, temp_lr), flush=True)
             
            #set optimiser to reduce lr based on wer
            if args.optim_type == 'normal':
                scheduler.step(valid_wer)

            #save intermediary op based on args
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

            #save best model
            if valid_wer < best_wer:
                best_wer = valid_wer
                best_epoch = epoch
                if args.rank == 0:
                    output_file = args.exp_dir + '/best_model.mdl'
                    checkpoint = {'epoch': epoch, 'optimizer': self.optimizer.state_dict(),
                                    'model_state': self.model.state_dict()}
                    torch.save(checkpoint, output_file)
            
            #determine whether we need early stopping
            if epoch + 1 - best_epoch >= early_stop_patience:
                if args.rank == 0:
                    print("Early stop since valid_wer doesn't decrease")
                break
       
    def subsequent_mask(self, size):
        #masking future ip using lower tri mask
        ret = torch.ones(size, size, dtype=torch.uint8)
        return torch.tril(ret, out=ret).unsqueeze(0)

    def run_one_epoch(self, epoch, scaler, args, is_train=True):
        #set dataloader and args
        dataloader = self.train_loader if is_train else self.valid_loader

        batch_time = util.AverageMeter('Time', ':6.3f')
        losses = util.AverageMeter('Loss', ':.4e')
        ctc_losses = util.AverageMeter('CtcLoss', ":.4e")
        att_losses = util.AverageMeter('AttLoss', ":.4e")
        ctc_wers = util.AverageMeter('CtcWer', ':.4f')
        att_wers = util.AverageMeter('AttWer', ':.4f')
        interce_wers = util.AverageMeter('InterCEWER', ":.2f")
        token_speed = util.AverageMeter('TokenSpeed', ":.2f")
        
        if is_train:
            num_updates = math.ceil(len(dataloader) / args.accum_grad)
        else:
            num_updates = len(dataloader)

        progress = util.ProgressMeter(num_updates, batch_time, losses, ctc_losses, att_losses, ctc_wers, \
                                        att_wers, interce_wers, token_speed, prefix="Epoch: [{}]".format(epoch))
        
        end = time.time()
        updates = -1
        
        for i, data in enumerate(dataloader):

            start = time.time()
            utt_list, audios, labels, audio_sizes, label_sizes = data
            src, src_mask = audios, (audios == args.padding_idx) #all audio except files which are masked
                                                                                    #check first col to see if file is masked
            #src_mask: ratios of unmasked length
            src_mask = src_mask.sum(-1) / src.size(-1)           

            tgt_label = labels[:,1:] #exclude sos
            tgt = labels[:,:-1] #exclude eos
            tokens = (tgt_label != args.text_padding_idx).sum().item()
            
            if args.use_gpu:
                src, src_mask = src.cuda(), src_mask.cuda()
                tgt_label = tgt_label.cuda()
                tgt = tgt.cuda()
                audio_sizes = audio_sizes.cuda()
                label_sizes = label_sizes.cuda()
            
            if scaler is not None:
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    ctc_out, att_out, loss, ctc_loss, att_loss, interce_out, src_size = self.model(src, src_mask, audio_sizes, tgt_label, label_sizes, self.tokenizer, self.text_encoder_tokenizer, self._num_updates, args)
            else:
                ctc_out, att_out, loss, ctc_loss, att_loss, interce_out, src_size = self.model(src, src_mask, audio_sizes, tgt_label, label_sizes, self.tokenizer, self.text_encoder_tokenizer, self._num_updates, args)

            #///OG unit error rate computation
            ctc_errs, all_tokens = ctc_greedy_wer(ctc_out, tgt_label.cpu().numpy(), src_size.cpu().numpy(), args.text_padding_idx)
            ctc_wers.update(ctc_errs/all_tokens, all_tokens)
            att_errs, all_tokens = att_greedy_wer(att_out, tgt_label.cpu().numpy(), args.text_padding_idx)
            att_wers.update(att_errs/all_tokens, all_tokens)
            interce_errs, all_tokens = att_greedy_wer(interce_out, tgt_label.cpu().numpy(), args.text_padding_idx)
            interce_wers.update(interce_errs/all_tokens, all_tokens)
            
            losses.update(loss.item(), tokens)
            ctc_losses.update(ctc_loss.item(), tokens)
            att_losses.update(att_loss.item(), tokens)
            batch_time.update(time.time() - end)
            token_speed.update(tokens/(time.time()-start))

            #updates weights in model
            if is_train:
                if scaler is not None:
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        loss = loss / args.accum_grad
                    scaler.scale(loss).backward()
                    if i % args.accum_grad == 0 or i == (len(dataloader) - 1):
                        scaler.unscale_(self.optimizer.optimizer)
                        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.grad_clip)
                        self.optimizer.step(step_optimizer=False)
                        scaler.step(self.optimizer.optimizer)
                        scaler.update()
                        self.optimizer.zero_grad()
                        updates += 1
                        self._num_updates += 1
                        if updates % args.print_freq == 0 and args.rank == 0:
                            progress.print(updates)
                else:
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
        
        #open file to store results in
        out_file = open(args.result_file, 'w')
        if args.print_utt2diff:
            utt2diff = open(os.path.join(os.path.dirname(args.result_file), 'utt2diff'), 'w')

        args.num_correct, args.total = 0, 0
        args.length_correct, args.length_total = 0, 0
        with torch.no_grad():
            #set models ot eval mode
            self.model.eval()
            if self.lm_model is not None and args.rank_model != "n-gram":
                self.lm_model.eval()

            for i, data in enumerate(self.test_loader):
                utt_list, audios, labels, audio_sizes, label_sizes = data
                src, src_mask = audios, (audios == args.padding_idx) #all audio except files which are masked
                                                                                    #check first col to see if file is masked
                #src_mask: ratios of unmasked length
                src_mask = src_mask.sum(-1) / src.size(-1)
                #obtain results by performing beam decoding
                if args.use_gpu:
                    src, src_mask = src.cuda(), src_mask.cuda()
                    audio_sizes = audio_sizes.cuda()
                    labels, label_sizes = labels.cuda(), label_sizes.cuda()

                if args.decode_type == 'ctc_only':
                    recog_results = ctc_beam_decode(self.model, src, src_mask, audio_sizes, self.tokenizer, args, self.lm_model)
                
                elif args.decode_type == 'ctc_att':
                    batch_top_seqs = ctc_beam_decode(self.model, src, src_mask, audio_sizes, self.tokenizer, args, self.lm_model)
                    recog_results, args = self.model.beam_decode(src, src_mask, audio_sizes, self.tokenizer, args, self.lm_model, batch_top_seqs, labels=labels, label_sizes=label_sizes)
                else:
                    recog_results, args = self.model.beam_decode(src, src_mask, audio_sizes, self.tokenizer, args, self.lm_model, labels=labels, label_sizes=label_sizes)
                
                #obtian hyp utt for seq
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
            #print stats
            if args.test_hitrate:
                print(args.num_correct, args.total, 1 - args.num_correct / args.total)
                print(args.length_correct, args.length_total, 1 - args.length_correct / args.length_total)
        
        return 0
