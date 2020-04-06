import torch
import torchvision
import torch.nn as nn
import torch.utils.data as Data
import matplotlib.pyplot as plt
from torchsummary import summary


class UNet(nn.Module):

    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()
        self.downconv1 = self.build_conv(in_channels, 64)
        self.downconv2 = nn.Sequential(nn.MaxPool2d(2, stride=2), self.build_conv(64, 128))
        self.downconv3 = nn.Sequential(nn.MaxPool2d(2, stride=2), self.build_conv(128, 256))
        self.downconv4 = nn.Sequential(nn.MaxPool2d(2, stride=2), self.build_conv(256, 512))
        self.midconv = nn.Sequential(nn.MaxPool2d(2, stride=2), self.build_conv(512, 1024),
                                     nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2),
                                     )

        self.upconv1 = nn.Sequential(self.build_conv(1024, 512),
                                     nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2),
                                     )
        self.upconv2 = nn.Sequential(self.build_conv(512, 256),
                                     nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2),
                                     )
        self.upconv3 = nn.Sequential(self.build_conv(256, 128),
                                     nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2),
                                     )
        self.upconv4 = nn.Sequential(self.build_conv(128, 64),
                                     nn.Conv2d(in_channels=64,
                                            out_channels=out_channels,
                                            kernel_size=1)
                                    )
        self.final = nn.Sigmoid()


    def forward(self, x):
        self.h1 = self.downconv1(x)
        self.h2 = self.downconv2(self.h1)
        self.h3 = self.downconv3(self.h2)
        self.h4 = self.downconv4(self.h3)
        self.h5 = self.midconv(self.h4)
        self.h6 = self.upconv1(torch.cat([self.h5, self.h4], 1))
        self.h7 = self.upconv2(torch.cat([self.h6, self.h3], 1))
        self.h8 = self.upconv3(torch.cat([self.h7, self.h2], 1))
        self.h9 = self.upconv4(torch.cat([self.h8, self.h1], 1))
        self.out = self.final(self.h9)
        return self.out

    def build_conv(self, in_channels, out_channels):
        return nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=3, padding=1),
                      nn.ReLU(),
                      nn.BatchNorm2d(out_channels),
                      nn.Conv2d(in_channels=out_channels,
                                out_channels=out_channels,
                                kernel_size=3, padding=1),
                      nn.ReLU(),
                      nn.BatchNorm2d(out_channels)
                      )

if __name__ == "__main__":
    unet = UNet(in_channels=3,out_channels=1)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    unet = unet.to(device)
    summary(unet, input_size=(3, 128, 128))

