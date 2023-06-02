#!/usr/bin/env python3

import math
import torch
import kaldiio
import numpy as np
import soundfile as sf
import torchaudio

import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader, BatchSampler
from data.feat_op import skip_feat, context_feat
from data.spec_augment import spec_aug
from data.speech_loader import SpeechDataset, SpeechDataLoader, DynamicDataset

#added normalize, sample_rate args; removed frame related (min len, max len, left ctx, right ctx, skip)

class SingleWavSet(object):
    def __init__(self, tokenizer, data_path, rank, filter_max, filter_min):

        #set paths to data and vocab files
        self.name = data_path['name']
        self.tokenizer = tokenizer
        self.rank = rank
        wav_path = data_path['scp_path']
        file_dict = self._load_audiopath(wav_path)
        
        if 'text_label' in data_path:
            text_dict = self._load_label(data_path['text_label'])
            assert (len(file_dict)-len(text_dict))<5, "label and sample size mismatch"

        self.items = []

        #append text and num_frames for each utt along with path to items list
        for i in range(len(file_dict)):
            utt, path, samples = file_dict[i]
            if 'text_label' in data_path:
                text = text_dict[utt]
            else:
                text = [1]

            if samples > filter_max or samples < filter_min:
                continue
            self.items.append((utt, path, samples, text))
        
    def get_len(self):
        return len(self.items)
    
    def _load_audiopath(self, wav_path):
        
        file_dict = []

        with open(wav_path, 'r') as fin:
            lines = fin.readlines()
            for line in lines:
                cont = line.strip().split(' ')
                utt = cont[0]
                path = cont[-3]
                samples = int(cont[-1])
                # audio, sr = torchaudio.load(path)
                file_dict.append((utt,path,samples))

        if self.rank == 0:
            print("Reading %d lines from %s" % (len(file_dict), wav_path))

        return file_dict
    
    def _load_label(self, lab_path, is_text=True):
        label_dict = dict()
        with open(lab_path, 'r') as fin:
            line = fin.readline()

            #for each utt create entry in dict with list consisting of tokens
            while line:
                utt, label = line.strip().split(' ', 1)

                #tokenise words and use utt as key for dict
                if is_text:
                    label = self.tokenizer.tokenize(label)
                    label_dict[utt] = [self.tokenizer.vocab[word] if word in self.tokenizer.vocab else
                                        self.tokenizer.vocab['unk'] for word in label]
                    label_dict[utt].insert(0, self.tokenizer.vocab['sos'])
                    label_dict[utt].append(self.tokenizer.vocab['eos'])
                else:
                    label_dict[utt] = [int(l) for l in label.split(' ')]
                    
                line = fin.readline()
        if self.rank == 0:
            print("Reading %d lines from %s" % (len(label_dict), lab_path))
        return label_dict


class HubertDataset(Dataset):
    def __init__(self, tokenizer, data_paths, args):
        self.tokenizer = tokenizer
        self.seed = args.seed
        self.rank = args.rank
        self.max_len = int(args.max_len / 100 * args.sample_rate)
        self.batch_size = args.batch_size
        self.max_frmlen = args.max_frmlen
        self.max_lablen = args.max_lablen
        self.filter_max = int(args.filter_max / 100 * args.sample_rate) 
        self.filter_min = int(args.filter_min / 100 * args.sample_rate)
        self.sample_rate = args.sample_rate
        self.normalize = args.normalize
        
        #for hubert we only filter based on len of samples
        self.max_samplen = args.max_samplen

        #similar to impl in speechdataset class
        self.data_streams = self._load_streams(data_paths)
        self.data_stream_sizes = [i.get_len() for i in self.data_streams]
        self.data_stream_cum_sizes = [self.data_stream_sizes[0]]
        for i in range(1, len(self.data_stream_sizes)):
            self.data_stream_cum_sizes.append(self.data_stream_cum_sizes[-1] + self.data_stream_sizes[i])
        
        if args.batch_type == "utterance":
            self.batched_data = self.make_batch_data_by_utt()
        elif args.batch_type == 'samples':
            self.batched_data = self.make_batch_data_by_samples()
        else:
            raise NotImplementedError

    def get_audio(self, path):
        audio, sr = sf.read(path)

        # audio = self.postprocess(audio,sr)
        if self.normalize:
            audio = torch.Tensor(audio)
            with torch.no_grad():
                audio = F.layer_norm(audio, audio.shape)
            audio = audio.numpy()
        samples = audio.size

        return audio, samples
    
    def postprocess(self, wav, cur_sample_rate):

        if wav.dim() == 2:
            wav = wav.mean(-1)
        assert wav.dim() == 1, wav.size()

        if cur_sample_rate != self.sample_rate:
            raise Exception(f"sr {cur_sample_rate} != {self.sample_rate}")

        if self.normalize:
            with torch.no_grad():
                wav = F.layer_norm(wav, wav.shape)
        return wav
        
    def _load_streams(self, data_paths):
        data_streams = []
        for i in range(len(data_paths)):
            stream = SingleWavSet(self.tokenizer, data_paths[i], self.rank, self.filter_max, self.filter_min)
            data_streams.append(stream)
        return data_streams
           
    def set_epoch(self, epoch):
        self.epoch = epoch

    def make_batch_data_by_utt(self):
        all_data = []
        #append items from data stream in single list
        for stream in self.data_streams:
            all_data.extend(stream.items)
        
        all_data = sorted(all_data, key=lambda x: x[-2], reverse=True)

        batches = []
        start = 0

        while True:
            frmlen = all_data[start][-2]
            lablen = len(all_data[start][-1])
            
            factor = max(int(frmlen / self.max_frmlen), int(lablen / self.max_lablen))
            bs = max(1, int(self.batch_size / (1 + factor)))
            end = min(len(all_data), start + bs)

            #slice all_data and append to batches
            batch = all_data[start:end]
            batch.reverse() #keep longest frame length first; doesn't matter for transformer, but matters for lstm
            batches.append(batch)

            if end == len(all_data):
                break
            start = end
        return batches
    
    def make_batch_data_by_samples(self):
        all_data = []
        #append items from data stream in single list
        for stream in self.data_streams:
            all_data.extend(stream.items)
        
        all_data = sorted(all_data, key=lambda x: x[-2], reverse=True)

        batches = []
        
        this_batch = []
        idx = 0
        samples_in_batch = 0
        while True:
            
            samples = all_data[idx][-2]
            samples_in_batch += samples
            
            if samples_in_batch > self.max_samplen:
                if len(this_batch) > 0:
                    this_batch.reverse()
                    batches.append(this_batch)

                this_batch = [all_data[idx]]
                idx += 1
                samples_in_batch = samples
            else:
                this_batch.append(all_data[idx])
                idx += 1
        
            if idx == len(all_data):
                this_batch.reverse()
                batches.append(this_batch)
                break
                
        return batches
            
    def __getitem__(self, idx):
        #similar to impl in dataset class, but here we don't need to compute the indices
        batch = self.batched_data[idx]
        torch_data = []
        for i in range(len(batch)):
            utt, path, samples, text = batch[i]

            audio, _  = self.get_audio(path)
            
            torch_data.append((utt, audio, samples, text))
        return torch_data

    def __len__(self):
        return len(self.batched_data)

class WavLoader(DataLoader):
    def __init__(self, dataset, **kwargs):
        self.padding_idx = kwargs.pop("padding_idx")
        self.text_padding_idx = kwargs.pop("text_padding_idx")
        kwargs["collate_fn"] = self.collate_fn
        super(WavLoader, self).__init__(dataset, **kwargs)

    def collate_fn(self, batch):
        #list list of batches and create torch tensor with all batches

        #if we have outer list encapsulating elements, extract
        if isinstance(batch[0], list):
            batch = batch[0]

        #obtain max feat and text length for each batch 
        audio_max_length = max(int(x[2]) for x in batch)
        # audio_size = batch[0][1].shape[1]
        text_max_length = max(len(x[3]) for x in batch)
        batch_size = len(batch)
        #dimension changed from 80 x 40 (feats x mfcc) to 1-D (no of samples in audio file)
        audios = torch.full([batch_size, audio_max_length], float(self.padding_idx))
        texts = torch.full([batch_size, text_max_length], int(self.text_padding_idx))
        utt_list = []
        audio_sizes = torch.zeros(batch_size)
        text_sizes = torch.zeros(batch_size)
        #convert from batch list to torch tensor
        for x in range(batch_size):
            utt, audio, samples, text = batch[x]
            audio_length = int(samples)
            text_length = len(text)
            #use torch narrow to trim along dimension
            audios[x].narrow(0, 0, audio_length).copy_(torch.Tensor(audio))
            texts[x].narrow(0, 0, text_length).copy_(torch.Tensor(text))
            utt_list.append(utt)
            audio_sizes[x] = audio_length / audio_max_length
            text_sizes[x] = text_length - 2 #substract sos and eos
        return utt_list, audios.float(), texts.long(), audio_sizes.float(), text_sizes.long()

class HubertLoader(WavLoader):
    #implements base class, along with a sampling function and setting epoch
    def __init__(self, dataset, batch_size, text_padding_idx=0, padding_idx=-1, distributed=False, shuffle=False, num_workers=0, timeout=1000):
        self.dataset = dataset
        if distributed:
            base_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        elif shuffle:
            base_sampler = torch.utils.data.RandomSampler(dataset)
        else:
            base_sampler = torch.utils.data.SequentialSampler(dataset)
        
        self.base_sampler = base_sampler
        
        sampler = torch.utils.data.BatchSampler(base_sampler, batch_size, False)
        kwargs = {"batch_sampler": sampler, "padding_idx": padding_idx, "text_padding_idx": text_padding_idx, "num_workers": num_workers}
        super(HubertLoader, self).__init__(dataset, **kwargs)

    def set_epoch(self, epoch):
        try:
            self.base_sampler.set_epoch(epoch)
        except:
            pass