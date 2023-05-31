# copied from https://github.com/mie-lab/traffic4cast/blob/aea6f90e8884c01689c84255c99e96d2b58dc470/models/unet.py with permission

"""
UNet model architecture with original formatting and args as in t4c challenge.
"""

from typing import Optional
from typing import Tuple

import numpy as np
import torch
from torch import nn


class UNet(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=2,
        depth=5,
        layer_out_pow2=6,
        padding=False,
        batch_norm=False,
        up_mode="upconv",
    ):
        """
        Implementation of
        U-Net: Convolutional Networks for Biomedical Image Segmentation
        (Ronneberger et al., 2015)
        https://arxiv.org/abs/1505.04597
        Using the default arguments will yield the exact version used
        in the original paper
        Args:
            in_channels (int): number of input channels
            out_channels (int): number of output channels
            depth (int): depth of the network
            layer_out_pow2 (int): number of filters in the first layer is 2**layer_out_pow2
            padding (bool): if True, apply padding such that the input shape
                            is the same as the output.
                            This may introduce artifacts
            batch_norm (bool): Use BatchNorm after layers with an
                               activation function
            up_mode (str): one of 'upconv' or 'upsample'.
                           'upconv' will use transposed convolutions for
                           learned upsampling.
                           'upsample' will use bilinear upsampling.
        """
        super(UNet, self).__init__()
        assert up_mode in ("upconv", "upsample")
        self.padding = padding
        self.depth = depth
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(UNetConvBlock(prev_channels, 2 ** (layer_out_pow2 + i), padding, batch_norm))
            prev_channels = 2 ** (layer_out_pow2 + i)

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(UNetUpBlock(prev_channels, 2 ** (layer_out_pow2 + i), up_mode, padding, batch_norm))
            prev_channels = 2 ** (layer_out_pow2 + i)

        self.final = nn.Conv2d(prev_channels, out_channels, kernel_size=1)

    def forward(self, x, *args, **kwargs):
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path) - 1:
                blocks.append(x)
                x = torch.nn.functional.max_pool2d(x, 2)

        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 1])

        return self.final(x)


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, padding, batch_norm):
        super(UNetConvBlock, self).__init__()
        block = []

        block.append(nn.Conv2d(in_size, out_size, kernel_size=3, padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        block.append(nn.Conv2d(out_size, out_size, kernel_size=3, padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        self.block = nn.Sequential(*block)

    def forward(self, x):  # noqa
        out = self.block(x)
        return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, up_mode, padding, batch_norm):
        super(UNetUpBlock, self).__init__()
        if up_mode == "upconv":
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        elif up_mode == "upsample":
            self.up = nn.Sequential(
                nn.Upsample(mode="bilinear", scale_factor=2),
                nn.Conv2d(in_size, out_size, kernel_size=1),
            )

        self.conv_block = UNetConvBlock(in_size, out_size, padding, batch_norm)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        diff_y_target_size_ = diff_y + target_size[0]
        diff_x_target_size_ = diff_x + target_size[1]
        return layer[:, :, diff_y:diff_y_target_size_, diff_x:diff_x_target_size_]

    def forward(self, x, bridge):  # noqa
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        out = torch.cat([up, crop1], 1)
        out = self.conv_block(out)

        return out


class UNetTransformer:
    """Transformer `T4CDataset` <-> `UNet`.

    zeropad2d only works with
    """

    @staticmethod
    def unet_pre_transform(
        data: np.ndarray,
        zeropad2d: Optional[Tuple[int, int, int, int]] = None,
        stack_channels_on_time: bool = False,
        batch_dim: bool = False,
        from_numpy: bool = False,
        **kwargs
    ) -> torch.Tensor:
        """Transform data from `T4CDataset` be used by UNet:

        - put time and channels into one dimension
        - padding
        """
        if from_numpy:
            data = torch.from_numpy(data).float()

        if not batch_dim:
            data = torch.unsqueeze(data, 0)

        if stack_channels_on_time:
            data = UNetTransformer.transform_stack_channels_on_time(data, batch_dim=True)
        if zeropad2d is not None:
            zeropad2d = torch.nn.ZeroPad2d(zeropad2d)
            data = zeropad2d(data)
        if not batch_dim:
            data = torch.squeeze(data, 0)
        return data

    @staticmethod
    def unet_post_transform(
        data: torch.Tensor, crop: Optional[Tuple[int, int, int, int]] = None, unstack_channels_on_time: bool = False, batch_dim: bool = False, **kwargs
    ) -> torch.Tensor:
        """Bring data from UNet back to `T4CDataset` format:

        - separats common dimension for time and channels
        - cropping
        """
        if not batch_dim:
            data = torch.unsqueeze(data, 0)

        if crop is not None:
            _, _, height, width = data.shape
            left, right, top, bottom = crop
            right = width - right
            bottom = height - bottom
            data = data[:, :, top:bottom, left:right]
        if unstack_channels_on_time:
            data = UNetTransformer.transform_unstack_channels_on_time(data, batch_dim=True)
        if not batch_dim:
            data = torch.squeeze(data, 0)
        return data

    @staticmethod
    def transform_stack_channels_on_time(data: torch.Tensor, batch_dim: bool = False):
        """
        `(k, 12, 495, 436, 8) -> (k, 12 * 8, 495, 436)`
        """

        if not batch_dim:
            # `(12, 495, 436, 8) -> (1, 12, 495, 436, 8)`
            data = torch.unsqueeze(data, 0)
        num_time_steps = data.shape[1]
        num_channels = data.shape[4]

        # (k, 12, 495, 436, 8) -> (k, 12, 8, 495, 436)
        data = torch.movedim(data, 4, 2)

        # (k, 12, 8, 495, 436) -> (k, 12 * 8, 495, 436)
        data = torch.reshape(data, (data.shape[0], num_time_steps * num_channels, 495, 436))

        if not batch_dim:
            # `(1, 12, 495, 436, 8) -> (12, 495, 436, 8)`
            data = torch.squeeze(data, 0)
        return data

    @staticmethod
    def transform_unstack_channels_on_time(data: torch.Tensor, num_channels=8, batch_dim: bool = False):
        """
        `(k, 12 * 8, 495, 436) -> (k, 12, 495, 436, 8)`
        """
        if not batch_dim:
            # `(12, 495, 436, 8) -> (1, 12, 495, 436, 8)`
            data = torch.unsqueeze(data, 0)

        num_time_steps = int(data.shape[1] / num_channels)
        # (k, 12 * 8, 495, 436) -> (k, 12, 8, 495, 436)
        data = torch.reshape(data, (data.shape[0], num_time_steps, num_channels, 495, 436))

        # (k, 12, 8, 495, 436) -> (k, 12, 495, 436, 8)
        data = torch.movedim(data, 2, 4)

        if not batch_dim:
            # `(1, 12, 495, 436, 8) -> (12, 495, 436, 8)`
            data = torch.squeeze(data, 0)
        return data
