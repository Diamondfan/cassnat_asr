# 2022 Ruchao Fan
# SPAPL

import os
import torch

class BaseTask(object):
    def __init__(self, args):
        self.use_cuda = args.use_gpu

    def load_checkpoint(self, checkpoint, rank, use_cuda):
        if rank == 0:
            print("Loading checkpoint from {}".format(checkpoint))
        checkpoint = torch.load(checkpoint, map_location='cpu')
        model_state = checkpoint["model_state"]
        
        for name, param in self.model.named_parameters():
            if name not in model_state:
                name = "module." + name
            param.data.copy_(model_state[name])
        
        self.optimizer.load_state_dict(checkpoint['optimizer'], rank, use_cuda)
        self.start_epoch = checkpoint['epoch'] + 1

    def load_test_model(self, resume_model):
        if resume_model:
            print("Loading model from {}".format(resume_model))
            checkpoint = torch.load(resume_model, map_location='cpu')
            model_state = checkpoint["model_state"]
            for name, param in self.model.named_parameters():
                if name not in model_state:
                    name = "module." + name
                param.data.copy_(model_state[name])
    
    def model_stats(self, rank, use_slurm, distributed):
        if rank == 0:
            print(self.model)
            num_params, updated_params = 0, 0

            for name, param in self.model.named_parameters():
                num_params += param.numel()
                if param.requires_grad == True:
                    updated_params += param.numel()

            print("Number of parameters: {}, updated params: {}".format(num_params, updated_params))

            self.model_params = num_params
            self.updated_params = updated_params

        if use_slurm:
            local_rank = rank % torch.cuda.device_count()
        else:
            local_rank = rank

        if self.use_cuda:
            torch.cuda.set_device(local_rank)
            self.model = self.model.cuda(local_rank)

        if distributed:        
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[local_rank])

       
    def load_model(self, checkpoint, rank):
        raise NotImplementedError

    def set_model(self, args):
        raise NotImplementedError    

    def set_optimizer(self, args):
        raise NotImplementedError
    
    def set_dataloader(self, args):
        raise NotImplementedError
     
