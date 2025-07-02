import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        conv_block = [
            nn.ReflectionPad3d(1),
            nn.Conv3d(in_features, in_features, kernel_size=3, padding=0),
            nn.InstanceNorm3d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad3d(1),
            nn.Conv3d(in_features, in_features, kernel_size=3, padding=0),
            nn.InstanceNorm3d(in_features)
        ]
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(Generator, self).__init__()
        # Initial convolution block
        model_head = [
            nn.ReflectionPad3d(3),
            nn.Conv3d(input_nc, 64, kernel_size=7, padding=0),
            nn.InstanceNorm3d(64),
            nn.ReLU(inplace=True)
        ]

        # Downsampling
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model_head += [
                nn.Conv3d(in_features, out_features, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm3d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features * 2

        # Residual blocks
        model_body = []
        for _ in range(n_residual_blocks):
            model_body.append(ResidualBlock(in_features))

        # Upsampling
        model_tail = []
        out_features = in_features // 2
        for _ in range(2):
            model_tail += [
                nn.ConvTranspose3d(in_features, out_features, kernel_size=3, stride=2,
                                   padding=1, output_padding=1),
                nn.InstanceNorm3d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features // 2

        # Output layer
        model_tail += [
            nn.ReflectionPad3d(3),
            nn.Conv3d(64, output_nc, kernel_size=7, padding=0),
            nn.Tanh()
        ]

        self.model_head = nn.Sequential(*model_head)
        self.model_body = nn.Sequential(*model_body)
        self.model_tail = nn.Sequential(*model_tail)

    def forward(self, x):
        x = self.model_head(x)
        x = self.model_body(x)
        x = self.model_tail(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()
        model = [
            nn.Conv3d(input_nc, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm3d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm3d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(256, 512, kernel_size=4, padding=1),
            nn.InstanceNorm3d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(512, 1, kernel_size=4, padding=1)
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        # Average pooling and flatten
        return F.avg_pool3d(x, x.size()[2:]).view(x.size(0), -1)
