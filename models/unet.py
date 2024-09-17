import os
import sys

import torch
import torch.nn as nn

from .submodules import ConvLayer, RecurrentConvLayer
from .spiking_submodules import (
    ConvLIF,
    ConvLoihiLIF,
    SpikingRecurrentConvLayer,
)

parent_dir_name = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir_name)

from utils.utils import batch_to_channel


class BaseUNet(nn.Module):
    """
    Base class for conventional UNet architecture and networks derived from it.
    """

    ff_type = ConvLayer
    pred_type = ConvLayer

    def __init__(
        self,
        base_channels,
        num_encoders,
        num_output_channels,
        crop,
        norm,
        num_bins,
        recurrent_block_type=None,
        kernel_size=5,
        channel_multiplier=2,
        activations=["relu", None],
        final_activation=None,
        final_bias=True,
        final_w_scale=0.01,
        spiking_neuron=None,
    ):
        super(BaseUNet, self).__init__()
        self._store_activity = False

        self.base_channels = base_channels
        self.num_encoders = num_encoders
        self.num_output_channels = num_output_channels
        self.kernel_size = kernel_size
        self.crop = crop
        self.norm = norm
        self.num_bins = num_bins
        self.recurrent_block_type = recurrent_block_type
        self.channel_multiplier = channel_multiplier
        self.ff_act, self.rec_act = activations
        self.final_activation = final_activation
        self.final_bias = final_bias
        self.final_w_scale = final_w_scale

        self.final_weights_bits = None
        if spiking_neuron is not None and "weights_bits" in spiking_neuron.keys():
            self.final_weights_bits = spiking_neuron["weights_bits"]

        self.spiking_kwargs = {}
        if type(spiking_neuron) is dict:
            self.spiking_kwargs.update(spiking_neuron)

        assert self.num_output_channels > 0

        self.encoder_input_sizes = [
            int(self.base_channels * pow(self.channel_multiplier, i - 1)) for i in range(self.num_encoders)
        ]
        self.encoder_output_sizes = [
            int(self.base_channels * pow(self.channel_multiplier, i)) for i in range(self.num_encoders)
        ]

        self.max_num_channels = self.encoder_output_sizes[-1]

    def get_axonal_delays(self):
        self.delays = 0

    def store_activity(self):
        self._store_activity = True
        padding = self.kernel_size // 2
        self.pad = nn.ZeroPad2d((padding, padding, padding, padding))  # padding needed for direct comparison with loihi

    def build_encoders(self):
        encoders = nn.ModuleList()
        for i, (input_size, output_size) in enumerate(zip(self.encoder_input_sizes, self.encoder_output_sizes)):
            if i == 0:
                input_size = self.num_bins
            encoders.append(
                self.ff_type(
                    input_size,
                    output_size,
                    kernel_size=self.kernel_size,
                    stride=2,
                    activation=self.ff_act,
                    norm=self.norm,
                )
            )
        return encoders

    def build_recurrent_encoders(self):
        encoders = nn.ModuleList()
        for i, (input_size, output_size) in enumerate(zip(self.encoder_input_sizes, self.encoder_output_sizes)):
            if i == 0:
                input_size = self.num_bins

            if len(self.spiking_kwargs.keys()) == 0:
                encoders.append(
                    self.rec_type(
                        input_size,
                        output_size,
                        kernel_size=self.kernel_size,
                        stride=2,
                        recurrent_block_type=self.recurrent_block_type,
                        activation_ff=self.ff_act,
                        activation_rec=self.rec_act,
                        norm=self.norm,
                        **self.spiking_kwargs,
                    )
                )

            else:
                encoders.append(
                    self.rec_type(
                        input_size,
                        output_size,
                        kernel_size=self.kernel_size,
                        stride=2,
                        recurrent_block_type=self.recurrent_block_type,
                        activation=self.rec_act,
                        norm=self.norm,
                        **self.spiking_kwargs,
                    )
                )
        return encoders

    def build_prediction_layer(self, num_output_channels, norm=None):
        base_channels = self.base_channels
        if num_output_channels == 8:
            base_channels *= 4

        return self.pred_type(
            base_channels,
            num_output_channels,
            kernel_size=1,
            activation=self.final_activation,
            norm=norm,
            w_scale=self.final_w_scale,
            bias=self.final_bias,
            weights_bits=self.final_weights_bits,
        )

    def build_pooling(self, shape):
        assert shape[2] == shape[3]  # only works with squared kernels
        base_channels = self.base_channels
        if self.num_output_channels == 8:
            base_channels *= 4

        return self.ff_type(
            shape[1],
            base_channels,
            kernel_size=shape[2],
            padding=0,
            activation=self.ff_act,
            norm=None,
            **self.spiking_kwargs,
        )

    def unstack_corners(self, x):
        batch_size = x.shape[0] // 4
        row1 = [x[batch_size * 0 : batch_size * 1, :, :, :], x[batch_size * 1 : batch_size * 2, :, :, :]]
        row2 = [x[batch_size * 3 : batch_size * 4, :, :, :], x[batch_size * 2 : batch_size * 3, :, :, :]]

        return torch.cat([torch.cat(row1, dim=3), torch.cat(row2, dim=3)], dim=2)


class EncoderNet(BaseUNet):
    """
    Conventional convolutional encoder architecture.
    Series of strided convolutions followed by convolutional pooling layer.
    If crop is not None, corner information is unstacked and arranged spatially
    before the pooling layer (i.e., params in the encoders are shared).
    """

    def __init__(self, unet_kwargs):
        super().__init__(**unet_kwargs)

        self.encoders = self.build_encoders()
        self.pred = self.build_prediction_layer(self.num_output_channels, self.norm)
        self.pooling = None

    def forward(self, x):
        # encoders
        for encoder in self.encoders:
            x = encoder(x)

        # unstack corner information
        if self.crop is not None:
            x = self.unstack_corners(x)

        # pooling convolution
        if self.pooling is None:
            self.pooling = self.build_pooling(x.shape).to(x.device)
        x = self.pooling(x)

        # prediction
        x = self.pred(x)

        return x, None, None


class SplitEncoderNet(EncoderNet):
    """
    Conventional convolutional encoder architecture.
    Series of strided convolutions followed by convolutional pooling layer.
    Parameter sharing in all layers (i.e., encoders, pooling, pred).
    """

    def __init__(self, unet_kwargs):
        super().__init__(unet_kwargs)

    def forward(self, x):
        # encoders
        for encoder in self.encoders:
            x = encoder(x)

        # pooling convolution
        if self.pooling is None:
            self.pooling = self.build_pooling(x.shape).to(x.device)
        x = self.pooling(x)

        # prediction
        x = self.pred(x)

        # unstack corner information
        if self.crop is not None:
            batch_size = x.shape[0] // 4
            x = batch_to_channel(x, batch_size, splits=4)

        return x, None, None


class SpikingEncoderNet(EncoderNet):
    """
    Spiking version of EncoderNet.
    """

    ff_type = ConvLIF

    def __init__(self, unet_kwargs):
        super().__init__(unet_kwargs)

        self.num_states = self.num_encoders + 1
        self.states = [None] * self.num_states

    def get_axonal_delays(self):
        self.delays = 0
        for module in self.modules():
            self.delays += getattr(module, "delay", False)

    def forward(self, x):
        activity = {}
        if self._store_activity:
            activity["input"] = self.pad(x)

        # encoders (weight sharing among corners)
        for i, encoder in enumerate(self.encoders):
            x, self.states[i] = encoder(x, self.states[i])
            if self._store_activity:
                activity["encoder_" + str(i + 1)] = self.pad(x)
        offset = self.num_encoders

        # unstack corner information
        if self.crop is not None:
            x = self.unstack_corners(x)

        # pooling convolution
        if self.pooling is None:
            self.pooling = self.build_pooling(x.shape).to(x.device)
            self.get_axonal_delays()

        x, self.states[offset] = self.pooling(x, self.states[offset])
        y = x.detach().clone()
        if self._store_activity:
            activity["pooling"] = x

        # prediction
        x = self.pred(x)

        return x, y, activity


class LoihiEncoderNet(SpikingEncoderNet):
    """
    Loihi-compatible (i.e, quantized states and weights) version of SpikingEncoderNet.
    """

    ff_type = ConvLoihiLIF


class SplitSpikingEncoderNet(EncoderNet):
    """
    Spiking version of SplitEncoderNet.
    """

    ff_type = ConvLIF

    def __init__(self, unet_kwargs):
        super().__init__(unet_kwargs)

        self.num_states = self.num_encoders + 1
        self.states = [None] * self.num_states

    def get_axonal_delays(self):
        self.delays = 0
        for module in self.modules():
            self.delays += getattr(module, "delay", False)

    def forward(self, x):
        activity = {}
        if self._store_activity:
            activity["input"] = self.pad(x)

        # encoders (weight sharing among corners)
        for i, encoder in enumerate(self.encoders):
            x, self.states[i] = encoder(x, self.states[i])
            c, v, z, _ = self.states[i]
            if self._store_activity:
                if i == len(self.encoders) - 1:
                    activity["voltages_encoder_" + str(i + 1)] = v
                    activity["currents_encoder_" + str(i + 1)] = c
                    activity["spikes_encoder_" + str(i + 1)] = z
                else:
                    activity["voltages_encoder_" + str(i + 1)] = self.pad(v)
                    activity["currents_encoder_" + str(i + 1)] = self.pad(c)
                    activity["spikes_encoder_" + str(i + 1)] = self.pad(z)
        offset = self.num_encoders

        # pooling convolution
        if self.pooling is None:
            self.pooling = self.build_pooling(x.shape).to(x.device)
            self.get_axonal_delays()

        x, self.states[offset] = self.pooling(x, self.states[offset])
        c, v, z, _ = self.states[offset]
        y = x.detach().clone()
        if self._store_activity:
            activity["voltages_pooling"] = v
            activity["currents_pooling"] = c
            activity["spikes_pooling"] = z

        # prediction
        x = self.pred(x)

        # unstack corner information
        if self.crop is not None:
            batch_size = x.shape[0] // 4
            x = batch_to_channel(x, batch_size, splits=4)
            y = batch_to_channel(y, batch_size, splits=4)

        return x, y, activity


class SplitLoihiEncoderNet(SplitSpikingEncoderNet):
    """
    Loihi-compatible (i.e, quantized states and weights) version of SplitSpikingEncoderNet.
    """

    ff_type = ConvLoihiLIF


class RecEncoderNet(BaseUNet):
    """
    Conventional recurrent convolutional encoder architecture.
    Series of strided convolutions and recurrent blocks followed by convolutional pooling layer.
    If crop is not None, corner information is unstacked and arranged spatially
    before the pooling layer (i.e., params in the encoders are shared).
    """

    rec_type = RecurrentConvLayer

    def __init__(self, unet_kwargs):
        super().__init__(**unet_kwargs)

        self.encoders = self.build_recurrent_encoders()
        self.pred = self.build_prediction_layer(self.num_output_channels, self.norm)
        self.pooling = None

        self.num_states = self.num_encoders
        self.states = [None] * self.num_states

    def forward(self, x):
        # encoders
        for i, encoder in enumerate(self.encoders):
            x, self.states[i] = encoder(x, self.states[i])

        # unstack corner information
        if self.crop is not None:
            x = self.unstack_corners(x)

        # pooling convolution
        if self.pooling is None:
            self.pooling = self.build_pooling(x.shape).to(x.device)
        x = self.pooling(x)

        # prediction
        x = self.pred(x)

        return x, None, None


class SplitRecEncoderNet(RecEncoderNet):
    """
    Conventional convolutional encoder architecture.
    Series of strided convolutions and recurrent blocks followed by convolutional pooling layer.
    Parameter sharing in all layers (i.e., encoders, pooling, pred).
    """

    def __init__(self, unet_kwargs):
        super().__init__(unet_kwargs)

    def forward(self, x):
        # encoders
        for i, encoder in enumerate(self.encoders):
            x, self.states[i] = encoder(x, self.states[i])

        # pooling convolution
        if self.pooling is None:
            self.pooling = self.build_pooling(x.shape).to(x.device)
        x = self.pooling(x)

        # prediction
        x = self.pred(x)

        # unstack corner information
        if self.crop is not None:
            batch_size = x.shape[0] // 4
            x = batch_to_channel(x, batch_size, splits=4)

        return x, None, None


class SpikingRecEncoderNet(RecEncoderNet):
    """
    Spiking version of RecEncoderNet.
    """

    ff_type = ConvLIF
    rec_type = SpikingRecurrentConvLayer

    def __init__(self, unet_kwargs):
        super().__init__(unet_kwargs)

        self.num_states = self.num_encoders + 1
        self.states = [None] * self.num_states

    def get_axonal_delays(self):
        self.delays = 0
        for module in self.modules():
            self.delays += getattr(module, "delay", False)

    def forward(self, x):
        activity = {}
        if self._store_activity:
            activity["input"] = self.pad(x)

        # encoders (weight sharing among corners)
        for i, encoder in enumerate(self.encoders):
            x, self.states[i] = encoder(x, self.states[i])
            if self._store_activity:
                activity["encoder_" + str(i + 1)] = self.pad(x)
        offset = self.num_encoders

        # unstack corner information
        if self.crop is not None:
            x = self.unstack_corners(x)

        # pooling convolution
        if self.pooling is None:
            self.pooling = self.build_pooling(x.shape).to(x.device)
            self.get_axonal_delays()

        x, self.states[offset] = self.pooling(x, self.states[offset])
        y = x.detach().clone()
        if self._store_activity:
            activity["pooling"] = x

        # prediction
        x = self.pred(x)

        return x, y, activity


class LoihiRecEncoderNet(SpikingRecEncoderNet):
    """
    Loihi-compatible (i.e, quantized states and weights) version of SpikingRecEncoderNet.
    """

    ff_type = ConvLoihiLIF
    rec_type = SpikingRecurrentConvLayer


class SplitSpikingRecEncoderNet(RecEncoderNet):
    """
    Spiking version of SplitRecEncoderNet.
    """

    ff_type = ConvLIF
    rec_type = SpikingRecurrentConvLayer

    def __init__(self, unet_kwargs):
        super().__init__(unet_kwargs)

        self.num_states = self.num_encoders + 1
        self.states = [None] * self.num_states

    def get_axonal_delays(self):
        self.delays = 0
        for module in self.modules():
            self.delays += getattr(module, "delay", False)

    def forward(self, x):
        activity = {}
        if self._store_activity:
            activity["input"] = self.pad(x)

        # encoders (weight sharing among corners)
        for i, encoder in enumerate(self.encoders):
            x, self.states[i] = encoder(x, self.states[i])
            c, v, z, _ = self.states[i]
            if self._store_activity:
                if i == len(self.encoders) - 1:
                    activity["voltages_encoder_" + str(i + 1)] = v
                    activity["currents_encoder_" + str(i + 1)] = c
                    activity["spikes_encoder_" + str(i + 1)] = z
                else:
                    activity["voltages_encoder_" + str(i + 1)] = self.pad(v)
                    activity["currents_encoder_" + str(i + 1)] = self.pad(c)
                    activity["spikes_encoder_" + str(i + 1)] = self.pad(z)
        offset = self.num_encoders

        # pooling convolution
        if self.pooling is None:
            self.pooling = self.build_pooling(x.shape).to(x.device)
            self.get_axonal_delays()

        x, self.states[offset] = self.pooling(x, self.states[offset])
        c, v, z, _ = self.states[offset]
        y = x.detach().clone()
        if self._store_activity:
            activity["voltages_pooling"] = v
            activity["currents_pooling"] = c
            activity["spikes_pooling"] = z

        # prediction
        x = self.pred(x)

        # unstack corner information
        if self.crop is not None:
            batch_size = x.shape[0] // 4
            x = batch_to_channel(x, batch_size, splits=4)
            y = batch_to_channel(y, batch_size, splits=4)

        return x, y, activity


class SplitLoihiRecEncoderNet(SplitSpikingRecEncoderNet):
    """
    Loihi-compatible (i.e, quantized states and weights) version of SplitSpikingRecEncoderNet.
    """

    ff_type = ConvLoihiLIF
    rec_type = SpikingRecurrentConvLayer
