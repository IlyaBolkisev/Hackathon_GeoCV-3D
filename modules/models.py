import torch
import torch.nn as nn


# ResBlock
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.res_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        residual = x
        x = self.res_block(x)
        x += residual
        return x


# DispNet-B
class DispNetB(nn.Module):
    def __init__(self):
        super(DispNetB, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(6, 48, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(48, 48, kernel_size=3, padding=1, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(48, 48, kernel_size=3, padding=1),
            nn.LeakyReLU()
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(48, 96, kernel_size=3, padding=1, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(96, 96, kernel_size=3, padding=1),
            nn.LeakyReLU()
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(96, 128, kernel_size=3, padding=1, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.LeakyReLU()
        )

        self.res_block = nn.Sequential(
            ResBlock(128, 128),
            ResBlock(128, 128),
            ResBlock(128, 128)
        )

        self.deconv_block1 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.LeakyReLU()
        )

        self.deconv_block2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 96, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU()
        )

        self.deconv_block3 = nn.Sequential(
            nn.Conv2d(96, 96, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(96, 96, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(96, 48, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU()
        )

        self.deconv_block4 = nn.Sequential(
            nn.Conv2d(48, 48, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(48, 48, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(48, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU()
        )

        self.deconv_block5 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.LeakyReLU()
        )

        self.decode = nn.Conv2d(32, 2, kernel_size=3, padding=1)

    def forward(self, left, right):
        x = torch.cat((left, right), dim=1)

        x1 = self.conv_block1(x)
        x2 = self.conv_block2(x1)
        x3 = self.conv_block3(x2)

        x4 = self.res_block(x3)

        x5 = self.deconv_block1(x4) + x3
        x6 = self.deconv_block2(x5) + x2
        x7 = self.deconv_block3(x6)
        x8 = self.deconv_block4(x7)
        x9 = self.deconv_block5(x8)

        out = self.decode(x9)
        out = torch.split(out.clamp(min=0), 1, dim=1)
        return out[0], out[1]


# Encoder
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(4, 96, kernel_size=7, padding=3),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(),
            nn.Conv2d(96, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(96, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU()
        )

        self.conv_block2_1 = nn.Conv2d(96, 128, kernel_size=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU()
        )

        self.conv_block3_1 = nn.Conv2d(128, 256, kernel_size=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.conv_block4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv_block5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU()
        )

        self.conv_block5_1 = nn.Conv2d(256, 256, kernel_size=1)
        self.pool5 = nn.MaxPool2d(kernel_size=2)

        self.conv_block6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.fc7 = nn.Linear(1024, 8192)

    def forward(self, rgbd_images):
        x = self.conv_block1(rgbd_images)
        x = self.pool2(self.conv_block2(x) + self.conv_block2_1(x))
        x_corr = self.conv_block3(x) + self.conv_block3_1(x)
        x = self.pool3(x_corr)
        x = self.conv_block4(x)
        x = self.pool5(self.conv_block5(x) + self.conv_block5_1(x))
        x = self.conv_block6(x)
        x = self.fc7(x.view(-1, 1024))
        return x, x_corr


# Corr
class CorrNet(torch.nn.Module):
    def __init__(self, device, disp_channels=6):
        super(CorrNet, self).__init__()
        self.disp_channels = disp_channels
        self.device = device

        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv3d(512, 128, kernel_size=1),
            torch.nn.BatchNorm3d(128),
            torch.nn.LeakyReLU()
        )

        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv3d(128, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(128),
            torch.nn.LeakyReLU()
        )
        self.conv2_1 = torch.nn.Sequential(
            torch.nn.Conv3d(128, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(128)
        )

        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv3d(128, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(128),
            torch.nn.LeakyReLU()
        )
        self.conv3_1 = torch.nn.Sequential(
            torch.nn.Conv3d(128, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(128)
        )

        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv3d(128, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(128),
            torch.nn.LeakyReLU()
        )
        self.conv4_1 = torch.nn.Sequential(
            torch.nn.Conv3d(128, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(128)
        )

        self.conv5 = torch.nn.Sequential(
            torch.nn.Conv3d(128, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(128),
            torch.nn.LeakyReLU()
        )
        self.conv5_1 = torch.nn.Sequential(
            torch.nn.Conv3d(128, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(128)
        )

        self.conv6 = torch.nn.Sequential(
            torch.nn.Conv3d(128, 32, kernel_size=1),
            torch.nn.BatchNorm3d(32),
            torch.nn.LeakyReLU(),
            torch.nn.Conv3d(32, 1, kernel_size=1),
            torch.nn.BatchNorm3d(1)
        )

        self.conv7 = torch.nn.Sequential(
            torch.nn.Conv2d(6, 1, kernel_size=1),
            torch.nn.BatchNorm2d(1),
            torch.nn.LeakyReLU()
        )
        self.fc8 = torch.nn.Linear(1156, 4096)

    def forward(self, left, right):
        n, c, h, w = left.size()

        cost_volume = torch.zeros((n, 2 * c, self.disp_channels, h, w)).to(self.device)

        for i in range(self.disp_channels):
            if i == 0:
                cost_volume[:, :c, i, :, i:] = left
                cost_volume[:, c:, i, :, i:] = right
            else:
                cost_volume[:, :c, i, :, i:] = left[:, :, :, i:]
                cost_volume[:, c:, i, :, i:] = right[:, :, :, :-i]
        cost_volume = cost_volume.contiguous()

        x = self.conv1(cost_volume)
        x = self.conv2_1(self.conv2(x)) + x
        x = self.conv3_1(self.conv3(x)) + x
        x = self.conv4_1(self.conv4(x)) + x
        x = self.conv5_1(self.conv5(x)) + x
        x = self.conv6(x).squeeze(dim=1)
        x = self.conv7(x)
        x = self.fc8(x.view(-1, 1156))

        return x


# Decoder
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.deconv1 = nn.Sequential(
            nn.ConvTranspose3d(320, 160, kernel_size=3, stride=2, bias=False, padding=1, output_padding=1),
            nn.BatchNorm3d(160),
            nn.LeakyReLU()
        )
        self.deconv1_1 = nn.ConvTranspose3d(320, 160, kernel_size=1, stride=2, bias=False, output_padding=1)
        self.deconv1_2 = nn.Sequential(
            nn.ConvTranspose3d(160, 160, kernel_size=3, stride=1, bias=False, padding=1),
            nn.BatchNorm3d(160),
            nn.LeakyReLU()
        )

        self.deconv2 = nn.Sequential(
            nn.ConvTranspose3d(160, 80, kernel_size=3, stride=2, bias=False, padding=1, output_padding=1),
            nn.BatchNorm3d(80),
            nn.LeakyReLU()
        )
        self.deconv2_1 = nn.ConvTranspose3d(160, 80, kernel_size=1, stride=2, bias=False, output_padding=1)
        self.deconv2_2 = nn.Sequential(
            nn.ConvTranspose3d(80, 80, kernel_size=3, stride=1, bias=False, padding=1),
            nn.BatchNorm3d(80),
            nn.LeakyReLU()
        )

        self.deconv3 = nn.Sequential(
            nn.ConvTranspose3d(80, 40, kernel_size=3, stride=2, bias=False, padding=1, output_padding=1),
            nn.BatchNorm3d(40),
            nn.LeakyReLU()
        )
        self.deconv3_1 = nn.ConvTranspose3d(80, 40, kernel_size=1, stride=2, bias=False, output_padding=1)
        self.deconv3_2 = nn.Sequential(
            nn.ConvTranspose3d(40, 40, kernel_size=3, stride=1, bias=False, padding=1),
            nn.BatchNorm3d(40),
            nn.LeakyReLU()
        )

        self.deconv4 = nn.Sequential(
            nn.ConvTranspose3d(40, 20, kernel_size=3, stride=1, bias=False, padding=1),
            nn.BatchNorm3d(20),
            nn.LeakyReLU()
        )
        self.deconv4_1 = nn.ConvTranspose3d(40, 20, kernel_size=1, stride=1, bias=False)
        self.deconv4_2 = nn.Sequential(
            nn.ConvTranspose3d(20, 20, kernel_size=3, stride=1, bias=False, padding=1),
            nn.BatchNorm3d(20),
            nn.LeakyReLU()
        )

        self.deconv5 = nn.Sequential(
            nn.ConvTranspose3d(20, 1, kernel_size=3, stride=1, bias=False, padding=1),
            nn.Sigmoid()
        )

    def forward(self, left, right, corr):
        left = left.view(-1, 128, 4, 4, 4)
        right = right.view(-1, 128, 4, 4, 4)
        corr = corr.view(-1, 64, 4, 4, 4)

        x = torch.cat((left, right, corr), dim=1)
        x = self.deconv1_2(self.deconv1(x)) + self.deconv1_1(x)
        x = self.deconv2_2(self.deconv2(x)) + self.deconv2_1(x)
        x = self.deconv3_2(self.deconv3(x)) + self.deconv3_1(x)
        x = self.deconv4_2(self.deconv4(x)) + self.deconv4_1(x)
        x = self.deconv5(x)
        return x.squeeze(dim=1)
