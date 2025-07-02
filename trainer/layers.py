from functools import partial

import torch
import torch.nn.functional as F
from torch import nn

# Configuration for 3D operations
scale_eval = False
alpha = 0.02
beta = 0.00002

resnet_n_blocks = 1

# Use 3D instance normalization by default
norm_layer = partial(nn.InstanceNorm3d, affine=False, track_running_stats=False)
align_corners = False
up_sample_mode = 'trilinear'


def get_init_function(activation, init_function, **kwargs):
    """Get the initialization function from the given name."""
    a = 0.0
    if activation == 'leaky_relu':
        a = 0.2 if 'negative_slope' not in kwargs else kwargs['negative_slope']

    gain = 0.02 if 'gain' not in kwargs else kwargs['gain']
    if isinstance(init_function, str):
        if init_function == 'kaiming':
            activation = 'relu' if activation is None else activation
            return partial(torch.nn.init.kaiming_normal_, a=a, nonlinearity=activation, mode='fan_in')
        elif init_function == 'dirac':
            return torch.nn.init.dirac_
        elif init_function == 'xavier':
            activation = 'relu' if activation is None else activation
            gain = torch.nn.init.calculate_gain(nonlinearity=activation, param=a)
            return partial(torch.nn.init.xavier_normal_, gain=gain)
        elif init_function == 'normal':
            return partial(torch.nn.init.normal_, mean=0.0, std=gain)
        elif init_function == 'orthogonal':
            return partial(torch.nn.init.orthogonal_, gain=gain)
        elif init_function == 'zeros':
            return partial(torch.nn.init.normal_, mean=0.0, std=1e-5)
    elif init_function is None:
        if activation in ['relu', 'leaky_relu']:
            return partial(torch.nn.init.kaiming_normal_, a=a, nonlinearity=activation)
        if activation in ['tanh', 'sigmoid']:
            gain = torch.nn.init.calculate_gain(nonlinearity=activation, param=a)
            return partial(torch.nn.init.xavier_normal_, gain=gain)
    else:
        return init_function


def get_activation(activation, **kwargs):
    """Get the appropriate activation from the given name"""
    if activation == 'relu':
        return nn.ReLU(inplace=False)
    elif activation == 'leaky_relu':
        negative_slope = 0.2 if 'negative_slope' not in kwargs else kwargs['negative_slope']
        return nn.LeakyReLU(negative_slope=negative_slope, inplace=False)
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'sigmoid':
        return nn.Sigmoid()
    else:
        return None


class Conv(nn.Module):
    """Basic 3D convolution -> Norm -> Activation -> (optional ResNet)"""
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 bias=True, activation='relu', init_func='kaiming', use_norm=False,
                 use_resnet=False, **kwargs):
        super(Conv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=bias)
        self.norm = norm_layer(out_channels) if use_norm else None
        self.activation = get_activation(activation, **kwargs)
        self.resnet_block = ResnetTransformer(out_channels, resnet_n_blocks, init_func) if use_resnet else None

        # Initialize weights
        init_ = get_init_function(activation, init_func)
        init_(self.conv.weight)
        if self.conv.bias is not None:
            nn.init.constant_(self.conv.bias, 0)
        if self.norm is not None:
            # InstanceNorm3d has weight and bias
            nn.init.normal_(self.norm.weight, 1.0, 0.02)
            nn.init.constant_(self.norm.bias, 0.0)

    def forward(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        if self.resnet_block is not None:
            x = self.resnet_block(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, nc_down, nc_skip, nc_out, kernel_size, stride, padding,
                 bias=True, activation='relu', init_func='kaiming', use_norm=False,
                 refine=False, use_resnet=False, use_add=False, use_attention=False, **kwargs):
        super(UpBlock, self).__init__()
        nc_inner = kwargs.get('nc_inner', nc_out)

        self.conv0 = Conv(nc_down + nc_skip, nc_inner,
                          kernel_size=kernel_size, stride=stride, padding=padding,
                          bias=bias, activation=activation,
                          init_func=init_func, use_norm=use_norm,
                          use_resnet=use_resnet, **kwargs)
        self.conv1 = Conv(nc_inner, nc_inner,
                          kernel_size=kernel_size, stride=stride, padding=padding,
                          bias=bias, activation=activation,
                          init_func=init_func, use_norm=use_norm,
                          use_resnet=use_resnet, **kwargs) if refine else None
        self.use_attention = use_attention
        if use_attention:
            self.attention_gate = AttentionGate(nc_down, nc_skip, nc_inner,
                                                use_norm=use_norm, init_func=init_func)
        self.up_conv = Conv(nc_inner, nc_out,
                            kernel_size=kernel_size, stride=stride, padding=padding,
                            bias=bias, activation=activation,
                            init_func=init_func, use_norm=use_norm, **kwargs)
        self.use_add = use_add
        if use_add:
            self.output = Conv(nc_out, nc_out,
                               kernel_size=1, stride=1, padding=0,
                               bias=bias, activation=None,
                               init_func='zeros', use_norm=False)

    def forward(self, down, skip):
        if self.use_attention:
            skip = self.attention_gate(down, skip)
        if down.shape[2:] != skip.shape[2:]:
            down = F.interpolate(down, size=skip.shape[2:],
                                 mode=up_sample_mode, align_corners=align_corners)
        x = torch.cat([down, skip], dim=1)
        x = self.conv0(x)
        if self.conv1 is not None:
            x = self.conv1(x)
        if self.use_add:
            x = self.output(x) + down
        else:
            x = self.up_conv(x)
        return x


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 bias=False, activation='relu', init_func='kaiming',
                 use_norm=False, use_resnet=False, skip=True,
                 refine=False, pool=True, pool_size=2, **kwargs):
        super(DownBlock, self).__init__()
        self.conv0 = Conv(in_channels, out_channels,
                          kernel_size=kernel_size, stride=stride, padding=padding,
                          bias=bias, activation=activation,
                          init_func=init_func, use_norm=use_norm,
                          use_resnet=use_resnet, **kwargs)
        self.conv1 = Conv(out_channels, out_channels,
                          kernel_size=kernel_size, stride=stride, padding=padding,
                          bias=bias, activation=activation,
                          init_func=init_func, use_norm=use_norm,
                          use_resnet=use_resnet, **kwargs) if refine else None
        self.skip = skip
        self.pool = nn.MaxPool3d(pool_size) if pool else None

    def forward(self, x):
        x = self.conv0(x)
        skip = x
        if self.conv1 is not None:
            x = self.conv1(x)
            skip = x
        if self.pool is not None:
            x = self.pool(x)
        return (x, skip) if self.skip else x


class AttentionGate(nn.Module):
    def __init__(self, nc_g, nc_x, nc_inner, use_norm=False,
                 init_func='kaiming', mask_channel_wise=False):
        super(AttentionGate, self).__init__()
        self.conv_g = Conv(nc_g, nc_inner,
                           kernel_size=1, stride=1, padding=0,
                           bias=True, activation=None,
                           init_func=init_func, use_norm=use_norm)
        self.conv_x = Conv(nc_x, nc_inner,
                           kernel_size=1, stride=1, padding=0,
                           bias=False, activation=None,
                           init_func=init_func, use_norm=use_norm)
        self.residual = nn.ReLU(inplace=True)
        self.mask_channel_wise = mask_channel_wise
        output_ch = nc_x if mask_channel_wise else 1
        self.att_map = Conv(nc_inner, output_ch,
                            kernel_size=1, stride=1, padding=0,
                            bias=True, activation='sigmoid',
                            init_func=init_func, use_norm=use_norm)

    def forward(self, g, x):
        g_c = self.conv_g(g)
        x_c = self.conv_x(x)
        if g_c.shape[2:] != x_c.shape[2:]:
            x_c = F.interpolate(x_c, size=g_c.shape[2:],
                                 mode=up_sample_mode, align_corners=align_corners)
        combined = self.residual(g_c + x_c)
        alpha = self.att_map(combined)
        if not self.mask_channel_wise:
            alpha = alpha.expand(-1, x.shape[1], -1, -1, -1)
        if alpha.shape[2:] != x.shape[2:]:
            alpha = F.interpolate(alpha, size=x.shape[2:],
                                  mode=up_sample_mode, align_corners=align_corners)
        return alpha * x


class ResnetTransformer(nn.Module):
    def __init__(self, dim, n_blocks, init_func):
        super(ResnetTransformer, self).__init__()
        layers = []
        for _ in range(n_blocks):
            layers.append(ResnetBlock3D(dim, padding_type='reflect',
                                        norm_layer=norm_layer, use_dropout=False,
                                        use_bias=True))
        self.model = nn.Sequential(*layers)
        init_ = get_init_function('relu', init_func)

        def init_weights(m):
            if isinstance(m, nn.Conv3d):
                init_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.InstanceNorm3d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.constant_(m.bias, 0.0)

        self.model.apply(init_weights)

    def forward(self, x):
        return self.model(x)


class ResnetBlock3D(nn.Module):
    """Define a 3D Resnet block"""
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock3D, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type,
                                                norm_layer, use_dropout,
                                                use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer,
                         use_dropout, use_bias):
        conv_block = []
        # padding for 3D
        if padding_type == 'reflect':
            conv_block.append(nn.ReflectionPad3d(1))
        elif padding_type == 'replicate':
            conv_block.append(nn.ReplicationPad3d(1))
        elif padding_type == 'zero':
            pad = 1
        else:
            raise NotImplementedError(f'padding [{padding_type}] not implemented')
        # first conv
        conv_block += [nn.Conv3d(dim, dim, kernel_size=3,
                                 padding=0 if padding_type != 'zero' else pad,
                                 bias=use_bias), norm_layer(dim), nn.ReLU(inplace=True)]
        if use_dropout:
            conv_block.append(nn.Dropout3d(0.5))
        # second conv
        if padding_type == 'reflect':
            conv_block.append(nn.ReflectionPad3d(1))
        elif padding_type == 'replicate':
            conv_block.append(nn.ReplicationPad3d(1))
        elif padding_type == 'zero':
            pad = 1
        conv_block += [nn.Conv3d(dim, dim, kernel_size=3,
                                 padding=0 if padding_type != 'zero' else pad,
                                 bias=use_bias), norm_layer(dim)]
        return nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)  # skip connection
