"""
Copyright 2021 Ruchao Fan SPAPL
"""

import copy
import random
import math
import torch
import kaldiio
import numpy as np

from torch.utils.data import Dataset, DataLoader, BatchSampler

class SingleSet(object):
    def __init__(self, data_path, rank):
        self.name = data_path['name']
        self.rank = rank
        scp_path = data_path['scp_path']
        ark_dict = self._load_feature(scp_path)
        
        if 'utt2num_frames' in data_path:
            nframes_dict = self._load_label(data_path['utt2num_frames'], is_text=False)

        self.items = []
        for i in range(len(ark_dict)):
            utt, ark_path = ark_dict[i]

            if 'utt2num_frames' in data_path:
                num_frames = nframes_dict[utt][0]
            else:
                num_frames = None

            self.items.append((utt, ark_path, num_frames))
        
    def get_len(self):
        return len(self.items)

    def _load_feature(self, scp_path):
        ark_dict = []
        with open(scp_path, 'r') as fin:
            line = fin.readline()
            while line:
                utt, path = line.strip().split(' ')
                ark_dict.append((utt, path))
                line = fin.readline()
        if self.rank == 0:
            print("Reading %d lines from %s" % (len(ark_dict), scp_path))
        return ark_dict

    def _load_label(self, lab_path, is_text=True):
        label_dict = dict()
        with open(lab_path, 'r') as fin:
            line = fin.readline()
            while line:
                utt, label = line.strip().split(' ', 1)
                if is_text:
                    label_dict[utt] = [self.vocab.word2index[word] if word in self.vocab.word2index else
                                        self.vocab.word2index['unk'] for word in label.split(' ')]
                    label_dict[utt].insert(0, self.vocab.word2index['sos'])
                    label_dict[utt].append(self.vocab.word2index['eos'])
                else:
                    label_dict[utt] = [int(l) for l in label.split(' ')]
                line = fin.readline()
        if self.rank == 0:
            print("Reading %d lines from %s" % (len(label_dict), lab_path))
        return label_dict

class SSLDataset(Dataset):
    def __init__(self, data_paths, args):
        self.rank = args.rank
        self.filter_max = args.filter_max
        self.filter_min = args.filter_min
        self.use_specaug = args.use_specaug
        self.specaug_conf = args.specaug_conf 
        self.use_cmvn = False
        self.data_streams = self._load_streams(data_paths)
        self.data_stream_sizes = [i.get_len() for i in self.data_streams]
        self.data_stream_cum_sizes = [self.data_stream_sizes[0]]
        for i in range(1, len(self.data_stream_sizes)):
            self.data_stream_cum_sizes.append(self.data_stream_cum_sizes[-1] + self.data_stream_sizes[i])

    def _load_cmvn(self, cmvn_file):
        self.use_cmvn = True
        cmvn = kaldiio.load_mat(cmvn_file)
        self.mean = cmvn[0,:-1] / cmvn[0,-1]
        square = cmvn[1,:-1] / cmvn[0,-1]
        self.std = np.sqrt(square - np.power(self.mean, 2))
        return 0
        
    def _load_streams(self, data_paths):
        data_streams = []
        for i in range(len(data_paths)):
            stream = SingleSet(data_paths[i], self.rank)
            data_streams.append(stream)
        return data_streams

    def __getitem__(self, idx):
        stream_idx = -1
        for i in range(len(self.data_stream_cum_sizes)):
            if idx < self.data_stream_cum_sizes[i]:
                stream_idx = i
                break
        if stream_idx == -1:
            raise Exception('index exceed.')
        if stream_idx == 0:
            internal_idx = idx
        else:
            internal_idx = idx - self.data_stream_cum_sizes[stream_idx-1]
        
        utt, ark_path, num_frames = self.data_streams[stream_idx].items[internal_idx]
        feat = kaldiio.load_mat(ark_path)

        if self.use_cmvn:
            assert feat.shape[1] == self.mean.shape[0]
            feat = (feat - self.mean) / self.std

        # suppose specaug is not used
        assert self.use_specaug == False
        if self.use_specaug:
            feat = spec_aug(feat, self.specaug_conf)

        return (utt, feat, labels)

    def specaug_mask(self, feat):
        x = spec_aug(feat, self.specaug_conf)
        labels = (feat - x)
        return x, labels

    def random_mask(self, fbank):
        fbank_input, fbank_label = [], []
        seq_len = fbank.shape[0]
        for i, frame in enumerate(fbank):
            prob = random.random()
            if prob < 0.15:
                prob /= 0.15

                # 80% randomly change token to mask token
                if prob < 0.8:
                    fbank_input.append(np.zeros(frame.shape))

                # 10% randomly change token to random token
                elif prob < 0.9:
                    all_index = list(range(seq_len))
                    all_index.remove(i)
                    index = random.choice(all_index)
                    fbank_input.append(fbank[index])

                # 10% randomly change token to current token
                else:
                    fbank_input.append(fbank[i])

                fbank_label.append(fbank[i])

            else:
                fbank_input.append(fbank[i])
                fbank_label.append(np.zeros(frame.shape))

        return np.asarray(fbank_input), np.asarray(fbank_label)

    def __len__(self):
        return sum(self.data_stream_sizes)

class DynamicDataset(Dataset):
    def __init__(self, vocab, data_paths, args):
        self.vocab = vocab
        self.seed = args.seed
        self.rank = args.rank
        self.max_len = args.max_len
        self.max_frmlen = args.max_frmlen
        self.batch_size = args.batch_size
        self.ssl_type = args.ssl_type

        self.left_context = args.left_ctx
        self.right_context = args.right_ctx
        self.skip_frame = args.skip_frame  
        self.use_specaug = args.use_specaug
        self.specaug_conf = args.specaug_conf 
        self.use_cmvn = False
        self.data_streams = self._load_streams(data_paths)
        self.data_stream_sizes = [i.get_len() for i in self.data_streams]
        self.data_stream_cum_sizes = [self.data_stream_sizes[0]]
        for i in range(1, len(self.data_stream_sizes)):
            self.data_stream_cum_sizes.append(self.data_stream_cum_sizes[-1] + self.data_stream_sizes[i])
        
        if args.batch_type == "utterance":
            self.batched_data = self.make_batch_data_by_utt()
        elif args.batch_type == "frame":
            self.batched_data = self.make_batch_data_by_frame()
        else:
            raise NotImplementedError

    def _load_cmvn(self, cmvn_file):
        self.use_cmvn = True
        cmvn = kaldiio.load_mat(cmvn_file)
        self.mean = cmvn[0,:-1] / cmvn[0,-1]
        square = cmvn[1,:-1] / cmvn[0,-1]
        self.std = np.sqrt(square - np.power(self.mean, 2))
        return 0
        
    def _load_streams(self, data_paths):
        data_streams = []
        for i in range(len(data_paths)):
            stream = SingleSet(data_paths[i], self.rank)
            data_streams.append(stream)
        return data_streams
           
    def set_epoch(self, epoch):
        self.epoch = epoch

    def make_batch_data_by_utt(self):
        all_data = []
        for stream in self.data_streams:
            all_data.extend(stream.items)
        all_data = sorted(all_data, key=lambda x: x[-1], reverse=True)
        
        batches = []
        start = 0
        while True:
            frmlen = all_data[start][-1]
            if frmlen > self.max_len:
                start += 1 
                continue
            factor = int(frmlen / self.max_frmlen)
            bs = max(1, int(self.batch_size / (1 + factor)))
            end = min(len(all_data), start + bs)
            batch = all_data[start:end]
            batch.reverse()
            batches.append(batch)

            if end == len(all_data):
                break
            start = end
        return batches

    def make_batch_data_by_frame(self):
        all_data = []
        for stream in self.data_streams:
            all_data.extend(stream.items)
        all_data = sorted(all_data, key=lambda x: x[-1], reverse=True)
        
        batches = []
        start = 0
        while True:
            bs = 0
            frmlen_tot = 0
            while start + bs < len(all_data):
                frmlen = all_data[start+bs][-1]
                if frmlen > self.batch_size:
                    bs = 1
                    break

                frmlen_tot += frmlen
                if frmlen_tot <= self.batch_size:
                    bs += 1
                else:
                    break

            end = min(len(all_data), start + bs)
            batch = all_data[start:end]
            batch.reverse()
            batches.append(batch)
        
            if end == len(all_data):
                break
            start = end
        return batches
            
    def __getitem__(self, idx):
        batch = self.batched_data[idx]
        torch_data = []
        for i in range(len(batch)):
            utt, ark_path, utt2num_frames = batch[i]
            feat = kaldiio.load_mat(ark_path)
            
            if self.use_cmvn:
                assert feat.shape[1] == self.mean.shape[0]
                feat = (feat - self.mean) / self.std

            if self.ssl_type in ["apc", "biapc", "cpc", "hubert"]:
                labels = feat
            elif self.ssl_type == "mpc":
                feat, labels = self.random_mask(feat)
            elif self.ssl_type == "mrc":
                feat, labels = self.specaug_mask(feat)
            
            if self.use_specaug:
                feat = spec_aug(feat, self.specaug_conf)

            seq_len, dim = feat.shape
            if seq_len % self.skip_frame != 0:
                pad_len = self.skip_frame - seq_len % self.skip_frame
                feat = np.vstack([feat,np.zeros((pad_len, dim))])
                labels = np.vstack([labels,np.zeros((pad_len, dim))])

            feat = skip_feat(context_feat(feat, self.left_context, self.right_context), self.skip_frame)
            labels = skip_feat(context_feat(labels, self.left_context, self.right_context), self.skip_frame)
            torch_data.append((utt, feat, labels))
        return torch_data

    def __len__(self):
        return len(self.batched_data)

class MyDataLoader(DataLoader):
    def __init__(self, dataset, **kwargs):
        self.padding_idx = kwargs.pop("padding_idx")
        kwargs["collate_fn"] = self.collate_fn
        super(MyDataLoader, self).__init__(dataset, **kwargs)

    def collate_fn(self, batch):
        if isinstance(batch[0], list):
            batch = batch[0]
        feats_max_length = max(x[1].shape[0] for x in batch)
        feat_size = batch[0][1].shape[1]
        labels_max_length = max(x[2].shape[0] for x in batch)
        batch_size = len(batch)
        
        feats = torch.full([batch_size, feats_max_length, feat_size], float(self.padding_idx))
        labels = torch.full([batch_size, labels_max_length, feat_size], float(self.padding_idx))
        utt_list = []
        feat_sizes = torch.zeros(batch_size)
        label_sizes = torch.zeros(batch_size)

        for x in range(batch_size):
            utt, feature, label = batch[x]
            feature_length = feature.shape[0]
            label_length = label.shape[0]
            
            feats[x].narrow(0, 0, feature_length).copy_(torch.Tensor(feature))
            labels[x].narrow(0, 0, label_length).copy_(torch.Tensor(label))
            utt_list.append(utt)
            feat_sizes[x] = feature_length / feats_max_length
            label_sizes[x] = label_length / labels_max_length
        return utt_list, feats.float(), labels.float(), feat_sizes.float(), label_sizes.float()

class SSLDataLoader(MyDataLoader):
    def __init__(self, dataset, batch_size, padding_idx=-1, distributed=False, shuffle=False, num_workers=0, timeout=1000):
        self.dataset = dataset
        if distributed:
            base_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        elif shuffle:
            base_sampler = torch.utils.data.RandomSampler(dataset)
        else:
            base_sampler = torch.utils.data.SequentialSampler(dataset)
        
        self.base_sampler = base_sampler
        sampler = torch.utils.data.BatchSampler(base_sampler, batch_size, False)
        kwargs = {"batch_sampler": sampler, "padding_idx": padding_idx, "num_workers": num_workers}
        super(SSLDataLoader, self).__init__(dataset, **kwargs)

    def set_epoch(self, epoch):
        try:
            self.base_sampler.set_epoch(epoch)
        except:
            pass

