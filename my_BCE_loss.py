import torch
import torch.nn as nn
import numpy as np
import pdb

class My_BCELoss(nn.Module):

    def __init__(self):
        super(My_BCELoss, self).__init__()

    def forward(self, Y_pred, Y):
        Y_pred_flat = Y_pred.contiguous().view(-1).float()
        Y_flat = Y.contiguous().view(-1).float()

        bceloss = (torch.sum(Y_flat * torch.log(1 + torch.exp(-Y_pred_flat)) \
                + (1-Y_flat) * torch.log(1 + torch.exp(Y_pred_flat))))\
                     / (Y_flat.size()[0])

        return bceloss
