from torch import nn
import torch.nn.functional as F
import torch


class UP(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, data):
        return F.interpolate(data, scale_factor=self.scale, mode='bicubic')


class ResBlock(nn.Module):
    def __init__(self, channel, hidden_channel):
        super().__init__()
        self.net = nn.Sequential(
            nn.utils.weight_norm(nn.Conv2d(channel, hidden_channel, 3, 1, 1)),
            nn.ReLU6(inplace=True),
            nn.utils.weight_norm(nn.Conv2d(hidden_channel, hidden_channel, 3, 1, 1)),
            nn.ReLU6(inplace=True),
            nn.utils.weight_norm(nn.Conv2d(hidden_channel, channel, 3, 1, 1)),
        )
        self.final_relu = nn.ReLU6(inplace=True)

    def forward(self, x):
        x_net = self.net(x)
        return self.final_relu(x_net + x)


class SR_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
            ResBlock(64, 64),
            ResBlock(64, 64),
            ResBlock(64, 32),
            ResBlock(64, 32),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
            ResBlock(64, 64),
            ResBlock(64, 64),
            ResBlock(64, 32),
            ResBlock(64, 32),
            UP((3, 3)),
        )
        self.net_h = nn.Sequential(
            nn.Conv2d(67, 64, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
            ResBlock(64, 32),
            ResBlock(64, 32),
            ResBlock(64, 64),
            ResBlock(64, 64),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
            ResBlock(64, 32),
            ResBlock(64, 32),
            ResBlock(64, 64),
            ResBlock(64, 64),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
            ResBlock(64, 32),
            ResBlock(64, 32),
            ResBlock(64, 64),
            ResBlock(64, 64),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
            ResBlock(64, 32),
            ResBlock(64, 32),
            ResBlock(64, 64),
            ResBlock(64, 64),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
            ResBlock(64, 32),
            ResBlock(64, 32),
            ResBlock(64, 64),
            ResBlock(64, 64),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
            ResBlock(64, 32),
            ResBlock(64, 32),
            ResBlock(64, 64),
            ResBlock(64, 64),
        )
        self.delta = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 3, 3, 1, 1),
            nn.Tanh(),
        )

    def forward(self, data):
        f = self.net(data)
        up_data = F.interpolate(data, scale_factor=(3, 3), mode='bicubic')
        f = self.net_h(torch.cat([f, up_data], dim=1))
        f = self.delta(f)
        return f + up_data


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 128, 6, 2, 0),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(128, 128, 4, 2, 0),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(128, 256, 4, 2, 0),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(256, 512, 4, 2, 0),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(inplace=True),
            nn.AdaptiveAvgPool2d((16, 16)),
            nn.Conv2d(512, 512, 4, 2, 0),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(512, 512, 4, 1, 0),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(512, 1024, 3, 1, 0),
            nn.LeakyReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(4096, 1024),
            nn.LeakyReLU(inplace=True),
            nn.Linear(1024, 1024),
        )

    def forward(self, data):
        f = self.net(data)
        return f
