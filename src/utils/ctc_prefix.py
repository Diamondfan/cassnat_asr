#!/usr/bin/env python3
# 2020 Ruchao Fan

# This code is revised based on
# https://github.com/espnet/espnet/blob/master/espnet/nets/ctc_prefix_score.py and
# https://github.com/hirofumi0810/neural_sp/blob/master/neural_sp/models/seq2seq/decoders/ctc.py

import torch
import numpy

logone = 0
logzero = -1e10

class CTCPrefixScore(object):
    """Compute CTC label sequence scores
    which is based on Algorithm 2 in WATANABE et al.
    "HYBRID CTC/ATTENTION ARCHITECTURE FOR END-TO-END SPEECH RECOGNITION,"
    """

    def __init__(self, input_length, blank, eos):
        self.logzero = logzero
        self.blank = blank
        self.eos = eos
        self.input_length = input_length
        
    def initial_state(self, x):
        """Obtain an initial CTC state
        :return: CTC state
        """
        # initial CTC state is made of a frame x 2 tensor that corresponds to
        # r_t^n(<sos>) and r_t^b(<sos>), where 0 and 1 of axis=1 represent
        # superscripts n and b (non-blank and blank), respectively.
        bs = x.size(0)
        r = torch.full((bs, self.input_length, 2), self.logzero).type_as(x)
        r[:, :, 1] = torch.cumsum(x[:, :, self.blank], dim=1)
        return r
    
    def logaddexp(self, a, b):
        """torch.logaddexp is not implemented in torch1.2"""
        """This is quite slow and not used. Using torch.logsumexp instead."""
        # make sure a is always smaller than b
        mask = (a > b)
        big_a = torch.masked_select(a, mask)
        small_b = torch.masked_select(b, mask)
        new_a = a.masked_scatter(mask, small_b)
        new_b = b.masked_scatter(mask, big_a)
        result = new_b + torch.log(1 + torch.exp(new_a-new_b))
        return result

    def __call__(self, y, cs, x, r_prev):
        """Compute CTC prefix scores for next labels
        :param y     : prefix label sequence
        :param cs    : array of next labels
        :param r_prev: previous CTC state
        :return ctc_scores, ctc_states
        """
        # initialize CTC states
        output_length = y.size(1) - 1  # ignore sos
        bs = y.size(0)
        beam = cs.size(1)
        # new CTC states are prepared as a frame x (n or b) x n_labels tensor
        # that corresponds to r_t^n(h) and r_t^b(h).
        r = torch.full((bs, self.input_length, 2, beam), self.logzero).type_as(x)
        
        x_select = torch.gather(x, 2, cs.unsqueeze(1).repeat(1, x.size(1),1).long()) #bs, T, beam
        if output_length == 0: #if g == <sos>
            r[:, 0, 0, :] = x_select[:, 0]
            r[:, 0, 1, :] = self.logzero
        else:
            r[:, output_length - 1, :, :] = self.logzero

        # prepare forward probabilities for the last label
        r_sum = torch.logsumexp(r_prev, 2) #bs, T
        log_phi = r_sum.unsqueeze(2).repeat(1,1,beam) #bs, T, beam
        for b in range(bs):
            mask = (y[b,-1] == cs[b,:])  # if last(g) == c
            log_phi[b,:,:].masked_scatter_(mask, r_prev[b,:,1])

        # compute forward probabilities log(r_t^n(h)), log(r_t^b(h)),
        # and log prefix probabilites log(psi)
        start = max(output_length, 1)
        end = self.input_length
        x_ = torch.stack([x_select, x[:,:,self.blank:self.blank+1].repeat(1,1,beam)], dim=2)
        # start and self.input_length can be restricted with attention weight for faster computation
        for t in range(start, end):
            rp = r[:, t-1,:,:]
            rr = torch.stack([rp[:, 0,:], log_phi[:, t - 1, :], rp[:,0,:], rp[:,1,:]], dim=1).view(bs, 2, 2, beam)
            r[:, t, :, :] = torch.logsumexp(rr, 2) + x_[:,t,:,:]
        
        log_phi_x = torch.cat([log_phi[:,0:1,:], log_phi[:,:-1,:]], dim=1) + x_select
        log_psi = torch.logsumexp(torch.cat((r[:,start - 1, 0,:].unsqueeze(1), log_phi_x[:,start:end,:]), \
                                    dim=1), dim=1)  #bs, beam

        # get P(...eos|X) that ends with the prefix itself
        eos_pos = torch.where(cs == self.eos)
        if len(eos_pos) > 0:
            log_psi[eos_pos] = r_sum[eos_pos[0], -1]  # log(r_T^n(g) + r_T^b(g))

        # exclude blank probs
        blank_pos = torch.where(cs == self.blank)
        if len(blank_pos) > 0:
            log_psi[blank_pos] = self.logzero

        # return the log prefix probability and CTC states, where the label axis
        # of the CTC states is moved to the first axis to slice it easily
        return log_psi, r.transpose(2,3).transpose(1,2)



