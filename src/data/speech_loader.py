#!/usr/bin/env python3
# 2020 Ruchao Fan

import torch
import kaldiio
import numpy as np
from torch.utils.data import Dataset, DataLoader
from data.feat_op import skip_feat, context_feat

class SingleSet(object):
    def __init__(self, vocab, data_path):
        self.name = data_path['name']
        self.vocab = vocab
        scp_path = data_path['scp_path']
        ark_dict = self._load_feature(scp_path)
        
        if 'text_label' in data_path:
            text_dict = self._load_label(data_path['text_label'])
            assert (len(ark_dict)-len(text_dict))<5, "label and sample size mismatch"

        self.items = []
        for i in range(len(ark_dict)):
            utt, ark_path = ark_dict[i]
            if 'text_label' in data_path:
                text = text_dict[utt]
            else:
                text = [1]
            self.items.append((utt, ark_path, text))
        
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
        print("Reading %d lines from %s" % (len(ark_dict), scp_path))
        return ark_dict
    
    def _load_label(self, lab_path):
        label_dict = dict()
        with open(lab_path, 'r') as fin:
            line = fin.readline()
            while line:
                utt, label = line.strip().split(' ', 1)
                label_dict[utt] = [self.vocab.word2index[word] if word in self.vocab.word2index else
                        self.vocab.word2index['unk'] for word in label.split(' ')]
                label_dict[utt].insert(0, self.vocab.word2index['sos'])
                label_dict[utt].append(self.vocab.word2index['eos'])
                line = fin.readline()
        print("Reading %d lines from %s" % (len(label_dict), lab_path))
        return label_dict

class SpeechDataset(Dataset):
    def __init__(self, vocab, data_paths, left_context=0, right_context=0, skip_frame=1):
        self.vocab = vocab
        self.left_context = left_context
        self.right_context = right_context
        self.skip_frame = skip_frame     
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
            stream = SingleSet(self.vocab, data_paths[i])
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
        
        utt, ark_path, text = self.data_streams[stream_idx].items[internal_idx]
        feat = kaldiio.load_mat(ark_path)
        if self.use_cmvn:
            assert feat.shape[1] == self.mean.shape[0]
            feat = (feat - self.mean) / self.std
        seq_len, dim = feat.shape
        if seq_len % self.skip_frame != 0:
            pad_len = self.skip_frame - seq_len % self.skip_frame
            feat = np.vstack([feat,np.zeros((pad_len, dim))])
        feat = skip_feat(context_feat(feat, self.left_context, self.right_context), self.skip_frame)
        return (utt, feat, text)

    def __len__(self):
        return sum(self.data_stream_sizes)

class SpeechDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, padding_idx=-1, shuffle=False, num_workers=0, timeout=1000):
        self.padding_idx = padding_idx
        if shuffle:
            base_sampler = torch.utils.data.RandomSampler(dataset)
        else:
            base_sampler = torch.utils.data.SequentialSampler(dataset)
        
        sampler = torch.utils.data.BatchSampler(base_sampler, batch_size, False)
        super(SpeechDataLoader, self).__init__(dataset, 
                                                batch_sampler=sampler,
                                                num_workers=num_workers,
                                                collate_fn=self.collate_fn)

    def collate_fn(self, batch):
        feats_max_length = max(x[1].shape[0] for x in batch)
        feat_size = batch[0][1].shape[1]
        text_max_length = max(len(x[2]) for x in batch)
        batch_size = len(batch)
        
        feats = torch.full([batch_size, feats_max_length, feat_size], self.padding_idx)
        texts = torch.full([batch_size, text_max_length], self.padding_idx)
        utt_list = []
        feat_sizes = torch.zeros(batch_size)
        text_sizes = torch.zeros(batch_size)

        for x in range(batch_size):
            utt, feature, text = batch[x]
            feature_length = feature.shape[0]
            text_length = len(text)
            
            feats[x].narrow(0, 0, feature_length).copy_(torch.Tensor(feature))
            texts[x].narrow(0, 0, text_length).copy_(torch.Tensor(text))
            utt_list.append(utt)
            feat_sizes[x] = feature_length / feats_max_length
            text_sizes[x] = text_length - 2 #substract sos and eos
        return utt_list, feats.float(), texts.long(), feat_sizes.float(), text_sizes.long()

if __name__ == "__main__":
    import yaml
    data_paths = yaml.safe_load(open('config/data.yaml', 'r'))
    train_dataset = SpeechDataset(data_paths['train_data_path'])
    train_dataset._load_cmvn('data/fbank/train_global_cmvn.txt')
    train_loader = SpeechDataLoader(train_dataset, batch_size=10, shuffle=False)
    for i, data in enumerate(train_loader):
        utt_list, feats, text, phrase, speaker = data
        print(speaker.size())
        break


