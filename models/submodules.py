import math

import torch
import torch.nn as nn

import utils.quantization as quant


class ConvLayer(nn.Module):
    """
    Convolutional layer.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        activation="relu",
        norm=None,
        BN_momentum=0.1,
        w_scale=None,
        padding=None,
        bias=None,
        weights_bits=None,
    ):
        super(ConvLayer, self).__init__()

        if padding is None:
            padding = kernel_size // 2
        if bias is None:
            bias = False if norm == "BN" else True
        if weights_bits is not None:
            assert weights_bits in [1, 2, 4, 5, 6, 8]
        if w_scale is None:
            w_scale = math.sqrt(1 / in_channels)

        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        nn.init.uniform_(self.conv2d.weight, -w_scale, w_scale)
        if bias:
            nn.init.zeros_(self.conv2d.bias)

        if weights_bits is not None:
            quant_dw = 2 ** (8 - (weights_bits - 1))
            quant.quantize_param(
                self.conv2d,
                "weight",
                -w_scale,
                w_scale - w_scale * quant_dw / 256,
                w_scale * quant_dw / 256,
            )

        if activation is not None:
            if hasattr(torch, activation):
                self.activation = getattr(torch, activation)
        else:
            self.activation = None

        self.norm = norm
        if norm == "BN":
            self.norm_layer = nn.BatchNorm2d(out_channels, momentum=BN_momentum)
        elif norm == "IN":
            self.norm_layer = nn.InstanceNorm2d(out_channels, track_running_stats=True)

    def forward(self, x):
        out = self.conv2d(x)

        if self.norm in ["BN", "IN"]:
            out = self.norm_layer(out)

        if self.activation is not None:
            out = self.activation(out)

        return out


class RecurrentConvLayer(nn.Module):
    """
    Layer comprised of a convolution followed by a recurrent convolutional block.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        recurrent_block_type="convgru",
        activation_ff="relu",
        activation_rec=None,
        norm=None,
        BN_momentum=0.1,
    ):
        super(RecurrentConvLayer, self).__init__()

        assert recurrent_block_type in ["convgru"]
        self.recurrent_block_type = recurrent_block_type
        if recurrent_block_type == "convgru":
            RecurrentBlock = ConvGRU
        else:
            raise NotImplementedError

        self.conv = ConvLayer(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            activation_ff,
            norm,
            BN_momentum=BN_momentum,
        )
        self.recurrent_block = RecurrentBlock(
            input_size=out_channels, hidden_size=out_channels, kernel_size=3, activation=activation_rec
        )

    def forward(self, x, prev_state):
        x = self.conv(x)
        x, state = self.recurrent_block(x, prev_state)
        return x, state


class ConvGRU(nn.Module):
    """
    Convolutional GRU cell.
    Adapted from https://github.com/jacobkimmel/pytorch_convgru/blob/master/convgru.py
    """

    def __init__(self, input_size, hidden_size, kernel_size, activation=None):
        super().__init__()
        padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.reset_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.update_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.out_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        assert activation is None, "ConvGRU activation cannot be set (just for compatibility)"

        nn.init.orthogonal_(self.reset_gate.weight)
        nn.init.orthogonal_(self.update_gate.weight)
        nn.init.orthogonal_(self.out_gate.weight)
        nn.init.constant_(self.reset_gate.bias, 0.0)
        nn.init.constant_(self.update_gate.bias, 0.0)
        nn.init.constant_(self.out_gate.bias, 0.0)

    def forward(self, input_, prev_state):
        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            prev_state = torch.zeros(state_size, dtype=input_.dtype).to(input_.device)

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat([input_, prev_state], dim=1)
        update = torch.sigmoid(self.update_gate(stacked_inputs))
        reset = torch.sigmoid(self.reset_gate(stacked_inputs))
        out_inputs = torch.tanh(self.out_gate(torch.cat([input_, prev_state * reset], dim=1)))
        new_state = prev_state * (1 - update) + out_inputs * update

        return new_state, new_state
