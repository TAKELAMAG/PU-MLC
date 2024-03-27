from torch import nn
import torch
import math
from torch.cuda.amp import autocast


class vpuLoss(nn.Module):

    def __init__(self, gamma, alpha):
        super(vpuLoss, self).__init__()
        self.positive = 1
        self.negative = -1
        self.unlabeled = 0
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inp, target):
        output_mean = torch.mean(inp, dim=0)
        std = self.alpha * ((torch.sum((inp - output_mean).pow(2), dim=0) / inp.size(0)).pow(0.5))
        std = torch.where(std > 1.0, 1.0, std)
        std = std.detach()
        
        # calculate the variational loss
        positive = target == self.positive
        unlabeled = torch.ones_like(positive)
        positive, unlabeled = positive.type(torch.float), unlabeled.type(torch.float)

        num_p_rec = torch.sum(positive, dim=0, dtype=torch.float).pow(-1)
        num_x_rec = torch.sum(unlabeled, dim=0, dtype=torch.float).pow(-1)
        num_p_rec = torch.where(torch.isinf(num_p_rec), torch.full_like(num_p_rec, 0), num_p_rec)
        num_x_rec = torch.where(torch.isinf(num_x_rec), torch.full_like(num_x_rec, 0), num_x_rec)

        # sigmoid_log
        output_all_log = torch.log(torch.sigmoid(inp * std.pow(-1)) + 1e-10)
        output_p_log = output_all_log * positive
        output_x_log = output_all_log * unlabeled
        
        output_all_sig = torch.sigmoid(inp * std.pow(-1))
        with torch.no_grad():
            pt1 = torch.sum(((1 - output_all_sig) * unlabeled), dim=0) * num_x_rec
            one_sided_w_n = torch.pow(1 - pt1, self.gamma)
        
        p_loss = (torch.log(torch.exp(output_p_log) + 1e-10) * num_p_rec).sum()
        u_loss = (torch.log((torch.sum(torch.exp(output_x_log), dim=0) * num_x_rec) + 1e-10) * one_sided_w_n).sum()
        var_loss = u_loss - p_loss
        
        return var_loss
    
    
class mixup(nn.Module):

    def __init__(self, mix_alpha):
        super(mixup, self).__init__()
        self.mix_alpha = mix_alpha

    def forward(self, model, inputData, target, output):
        data_x1 = inputData[:int(inputData.size(0) / 2), :, :, :]
        data_x2 = inputData[math.ceil(inputData.size(0) / 2):, :, :, :]
        target_x1 = target[:int(inputData.size(0) / 2), :]
        target_x2 = target[math.ceil(inputData.size(0) / 2):, :]
        output_all_log = torch.log(torch.sigmoid(output) + 1e-10)
        output_x1_log = output_all_log[:int(inputData.size(0) / 2), :]
        output_x2_log = output_all_log[math.ceil(inputData.size(0) / 2):, :]

        if data_x2.size(0) == 0:
            data_x2 = data_x1
            target_x2 = target_x1
            output_x2_log = output_x1_log

        positive_x1 = target_x1 == 1
        unlabeled_x1_1 = target_x1 == 0
        unlabeled_x1_2 = target_x1 == -1
        unlabeled_x1 = unlabeled_x1_1 + unlabeled_x1_2
        positive_x1, unlabeled_x1 = positive_x1.type(torch.float), unlabeled_x1.type(torch.float)
        output_x1 = positive_x1 + output_x1_log.exp() * unlabeled_x1
        
        rand_perm = torch.randperm(positive_x1.size(0))
        data_x1_perm, output_x1_perm = data_x1[rand_perm], output_x1[rand_perm]

        positive_x2 = target_x2 == 1
        unlabeled_x2_1 = target_x2 == 0
        unlabeled_x2_2 = target_x2 == -1
        unlabeled_x2 = unlabeled_x2_1 + unlabeled_x2_2
        positive_x2, unlabeled_x2 = positive_x2.type(torch.float), unlabeled_x2.type(torch.float)
        output_x2 = positive_x2 + output_x2_log.exp() * unlabeled_x2
        
        m = torch.distributions.beta.Beta(self.mix_alpha, self.mix_alpha)
        lam = m.sample()
        output_x1_perm_lam = output_x1_perm * lam
        data_x1_perm_lam = data_x1_perm * lam
        output_x2_lam = output_x2 * (1 - lam)
        data_x2_lam = data_x2 * (1 - lam)
        
        data = (data_x1_perm_lam + data_x2_lam).cuda()
        output_mix = (output_x1_perm_lam + output_x2_lam).cuda()
        
        with autocast():  # mixed precision
            out_all = model(input=data)
            out_all = out_all.float()
        out_all_log = torch.log(torch.sigmoid(out_all) + 1e-10)

        reg_loss = (torch.sum((torch.log(output_mix + 1e-10) - out_all_log).pow(2), dim=0) * (1/data_x1.size(0))).sum()
    
        return reg_loss
    