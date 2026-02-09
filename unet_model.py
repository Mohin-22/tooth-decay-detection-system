import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()
        self.down1 = nn.Sequential(nn.Conv2d(in_channels, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())
        self.maxpool1 = nn.MaxPool2d(2)
        self.down2 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.Conv2d(128, 128, 3, padding=1), nn.ReLU())
        self.maxpool2 = nn.MaxPool2d(2)
        self.middle = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.Conv2d(256, 256, 3, padding=1), nn.ReLU())
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.upconv2 = nn.Sequential(nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(), nn.Conv2d(128, 128, 3, padding=1), nn.ReLU())
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.upconv1 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())
        self.final = nn.Conv2d(64, out_channels, 1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        d1 = self.down1(x)
        p1 = self.maxpool1(d1)
        d2 = self.down2(p1)
        p2 = self.maxpool2(d2)
        m = self.middle(p2)
        u2 = self.up2(m)
        cat2 = torch.cat([u2, d2], dim=1)
        uc2 = self.upconv2(cat2)
        u1 = self.up1(uc2)
        cat1 = torch.cat([u1, d1], dim=1)
        uc1 = self.upconv1(cat1)
        out = self.final(uc1)
        return self.activation(out)
