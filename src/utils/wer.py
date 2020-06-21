#!/usr/bin/env python3
# 2020 Ruchao Fan

import torch
import editdistance as ed

def ctc_greedy_wer(ctc_out, label, feat_size, pad=0):
    batch_errs, batch_tokens = 0, 0
    pred = torch.max(ctc_out, dim=-1)[1].cpu().numpy()
    bs, length = pred.shape
    for i in range(bs):
        p_seq = []
        h_seq = []
        for j in range(feat_size[i]):
            if pred[i][j] != 0:
                if j != 0 and pred[i][j] == pred[i][j-1]:
                    continue
                else:
                    p_seq.append(pred[i][j])
        for j in range(len(label[i])):
            if label[i][j] != pad and label[i][j] != 2:
                h_seq.append(label[i][j])

        batch_errs += ed.eval(p_seq, h_seq)
        batch_tokens += len(h_seq)
    return batch_errs, batch_tokens


def att_greedy_wer(att_out, label, pad=0):
    batch_errs, batch_tokens = 0, 0
    pred = torch.max(att_out, dim=-1)[1].cpu().numpy()
    bs, length = pred.shape
    for i in range(bs):
        p_seq = []
        h_seq = []
        for j in range(length):
            if pred[i][j] == pad and label[i][j] == 1:
                continue
            if pred[i][j] == 2:
                break
            p_seq.append(pred[i][j])

        for j in range(len(label[i])):
            if label[i][j] != pad and label[i][j] != 2:
                h_seq.append(label[i][j])

        batch_errs += ed.eval(p_seq, h_seq)
        batch_tokens += len(h_seq)
    return batch_errs, batch_tokens



