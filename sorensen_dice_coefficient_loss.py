import torch
import torch.nn as nn

class SorensenDiceCoefficientLoss(nn.Module):

    def __init__(self):
        super(SorensenDiceCoefficientLoss, self).__init__()

    def forward(self, Y_pred, Y):
        smooth = 1
        Y_pred_flat = Y_pred.contiguous().view(-1).float()
        Y_flat = Y.contiguous().view(-1).float()

        intersection = Y_pred_flat * Y_flat

        A = Y_pred_flat ** 2
        B = Y_flat ** 2

        return 1 - (2 * torch.sum(intersection) + smooth)\
                /(torch.sum(A) + torch.sum(B) + smooth)
