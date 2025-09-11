#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import copy

import torch.nn as nn


class DQN(nn.Module):
    def __init__(self, in_channels, num_actions=3):
        super(DQN, self).__init__()
        c, h, w = in_channels, 100, 100
        self.input_dims = in_channels
        self.n_actions = num_actions

        n1_chan = 16
        n2_chan = 8
        n3_chan = 4
        n1_kern = (3, 3)
        n2_kern = (3, 3)
        n3_kern = (2, 2)
        n1_stride = (2, 2)
        n2_stride = (2, 2)
        n3_stride = (1, 1)
        height1 = int((h - (n1_kern[0] - 1) - 1) / n1_stride[0]) + 1
        height2 = int((height1 - (n2_kern[0] - 1) - 1) / n2_stride[0]) + 1
        height3 = int((height2 - (n3_kern[0] - 1) - 1) / n3_stride[0]) + 1
        width1 = int((w - (n1_kern[1] - 1) - 1) / n1_stride[1]) + 1
        width2 = int((width1 - (n2_kern[1] - 1) - 1) / n2_stride[1]) + 1
        width3 = int((width2 - (n3_kern[1] - 1) - 1) / n3_stride[1]) + 1

        self.online = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(5184, 512),
            nn.ReLU(),
            nn.Linear(512, self.n_actions),
        )

        self.target = copy.deepcopy(self.online)

        # Q_target parameters are frozen.
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, x, model):
        # x = x.permute(0, 3, 1, 2)
        x = x.float()
        if model == "online":
            return self.online(x)
        elif model == "target":
            return self.target(x)
        return x
