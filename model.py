import torch
import torch.nn as nn
import torch.nn.functional as F


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.e_l1 = nn.Linear(4, 512)
        self.e_l2 = nn.Linear(512, 512)
        self.e_l3 = nn.Linear(512, 3)

        self.d_l1 = nn.Linear(3, 512)
        self.d_l2 = nn.Linear(512, 512)
        self.d_l3 = nn.Linear(512, 4)

        self.tahn = nn.functional.tanh

    def encode(self, data):
        res = data
        res = F.tahn(self.e_l1(res))
        res = F.tahn(self.e_l2(res))
        res = self.e_l3(res)
        return res

    def decode(self, data):
        res = data
        res = F.tahn(self.d_l1(res))
        res = F.tahn(self.d_l2(res))
        res = self.d_l3(res)
        return res

    def forward(self, data):
        encoded = self.encode(data)
        decoded = self.decode(encoded)
        return decoded
