import torch
import torch.nn as nn
import torch.nn.functional as F

# Local imports: you will need 3D variants of your blocks or adjust these accordingly
from .layers3d import DownBlock3D, Conv3D, ResnetTransformer3D

sampling_align_corners = False

# Number of filters per block for 3D
ndf = {'A': [32, 64, 64, 64, 64, 64, 64]}
nuf = {'A': [64, 64, 64, 64, 64, 64, 32]}
use_down_resblocks = {'A': True}
resnet_nblocks = {'A': 3}
refine_output = {'A': True}
down_activation = {'A': 'leaky_relu'}
up_activation = {'A': 'leaky_relu'}

class ResUnet3D(nn.Module):
    def __init__(self, nc_a, nc_b, cfg, init_func, init_to_identity):
        super(ResUnet3D, self).__init__()
        act_down = down_activation[cfg]

        # Down-sampling path
        self.ndown = len(ndf[cfg])
        self.nup = len(nuf[cfg])
        assert self.ndown >= self.nup

        in_ch = nc_a + nc_b
        skip_channels = {}
        for i, out_ch in enumerate(ndf[cfg], 1):
            block = DownBlock3D(
                in_ch, out_ch,
                kernel_size=3, stride=1, padding=1,
                activation=act_down, init_func=init_func,
                use_resnet=use_down_resblocks[cfg], use_norm=False
            )
            setattr(self, f"down_{i}", block)
            skip_channels[f"down_{i}"] = out_ch
            in_ch = out_ch

        # Bottleneck ResNet transformer if configured
        if use_down_resblocks[cfg]:
            self.c1 = Conv3D(in_ch, 2*in_ch, 1, 1, 0,
                              activation=act_down, init_func=init_func, use_norm=False)
            self.transformer = (
                (lambda x: x) if resnet_nblocks[cfg]==0
                else ResnetTransformer3D(2*in_ch, resnet_nblocks[cfg], init_func)
            )
            self.c2 = Conv3D(2*in_ch, in_ch, 1, 1, 0,
                              activation=act_down, init_func=init_func, use_norm=False)
        else:
            self.transformer = None

        # Up-sampling path
        act_up = up_activation[cfg]
        curr_in = in_ch
        for idx, out_ch in enumerate(nuf[cfg], 1):
            level = self.ndown - idx + 1
            block = Conv3D(
                curr_in + skip_channels[f"down_{level}"],
                out_ch,
                kernel_size=3, stride=1, padding=1,
                activation=act_up, init_func=init_func, use_norm=False, use_resnet=False
            )
            setattr(self, f"up_{level}", block)
            curr_in = out_ch

        # Refinement block
        if refine_output[cfg]:
            self.refine = nn.Sequential(
                ResnetTransformer3D(curr_in, 1, init_func),
                Conv3D(curr_in, curr_in, 1, 1, 0,
                       activation=act_up, init_func=init_func, use_norm=False)
            )
        else:
            self.refine = nn.Identity()

        # Final offset field generator: output 3 channels (dx, dy, dz)
        final_init = 'zeros' if init_to_identity else init_func
        self.output = Conv3D(curr_in, 3, 3, 1, 1,
                             init_func=final_init, activation=None, use_norm=False)

    def forward(self, a, b):
        x = torch.cat([a, b], dim=1)
        skips = {}
        # Down
        for i in range(1, self.ndown+1):
            x, skip = getattr(self, f"down_{i}")(x)
            skips[f"down_{i}"] = skip

        # Bottleneck transform
        if self.transformer is not None:
            x = self.c1(x)
            x = self.transformer(x)
            x = self.c2(x)

        # Up
        for j in range(self.ndown, self.ndown - self.nup, -1):
            s = skips[f"down_{j}"]
            x = F.interpolate(x, size=s.shape[2:], mode='trilinear', align_corners=sampling_align_corners)
            x = torch.cat([x, s], dim=1)
            x = getattr(self, f"up_{j}")(x)

        # Refinement & output
        x = self.refine(x)
        disp = self.output(x)
        return disp

class Reg3D(nn.Module):
    def __init__(self, depth, height, width, in_ch_a, in_ch_b):
        super(Reg3D, self).__init__()
        init_func = 'kaiming'
        init_to_identity = True

        self.depth = depth
        self.height = height
        self.width = width
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.offset_net = ResUnet3D(
            in_ch_a, in_ch_b, cfg='A', init_func=init_func, init_to_identity=init_to_identity
        ).to(self.device)
        self.register_buffer('identity_grid', self._make_identity_grid())

    def _make_identity_grid(self):
        # Coordinates in range [-1,1]
        z = torch.linspace(-1.0, 1.0, self.depth)
        y = torch.linspace(-1.0, 1.0, self.height)
        x = torch.linspace(-1.0, 1.0, self.width)
        zz, yy, xx = torch.meshgrid(z, y, x, indexing='ij')
        grid = torch.stack((xx, yy, zz), dim=-1)  # shape (D,H,W,3)
        grid = grid.unsqueeze(0)  # add batch dim: (1,D,H,W,3)
        return grid

    def forward(self, a, b, apply_on=None):
        # a, b: shape (B, C, D, H, W)
        disp = self.offset_net(a, b)
        # Optionally apply disp to a or b using grid_sample:
        # new_grid = self.identity_grid + disp.permute(0,2,3,4,1)
        # warped = F.grid_sample(a, new_grid, mode='bilinear', padding_mode='border', align_corners=sampling_align_corners)
        return disp
