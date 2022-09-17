import torch.nn as nn
import torch

class SoftCrossEntropy(nn.Module):
    def __init__(self):
        super(SoftCrossEntropy, self).__init__()
        self.cel = nn.CrossEntropyLoss()
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, labels):
        loss_tra_cel = self.cel(inputs, labels)

        ls = (-1) * self.logsoftmax(inputs)
        ls_sum = torch.sum(ls, dim=1) / inputs.shape[1]

        loss_soft_cel = torch.sum(ls_sum) / inputs.shape[0]

        loss = loss_tra_cel + loss_soft_cel
        return loss