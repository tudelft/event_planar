import math

import torch
import torch.nn as nn

import models.spiking_util as spiking
import utils.quantization as quant


"""
Relevant literature:
- Zenke et al. 2018: "SuperSpike: Supervised Learning in Multilayer Spiking Neural Networks"
- Bellec et al. 2020: "A solution to the learning dilemma for recurrent networks of spiking neurons"
- Fang et al. 2020: "Incorporating Learnable Membrane Time Constant to Enhance Learning of Spiking Neural Networks"
- Ledinauskas et al. 2020: "Training Deep Spiking Neural Networks"
- Perez-Nieves et al. 2021: "Neural heterogeneity promotes robust learning"
- Yin et al. 2021: "Accurate and efficient time-domain classification with adaptive spiking recurrent neural networks"
- Zenke et al. 2021: "The Remarkable Robustness of Surrogate Gradient Learning for Instilling Complex Function in Spiking Neural Networks"
- Paredes-Valles et al. 2020: "Unsupervised Learning of a Hierarchical Spiking Neural Network for Optical Flow Estimation: From Events to Global Motion Perception"
- Hagenaars and Paredes-Valles et al. 2021: "Self-Supervised Learning of Event-Based Optical Flow with Spiking Neural Networks"
"""


class ConvLIF(nn.Module):
    """
    Convolutional spiking LIF cell.
    Design choices:
    + No delays between current, voltage and spike
    + Reset with spikes from last timestep
    + Stateful current and voltage (e.g. Zenke et al., 2021)
    - Arctan surrogate grad (Fang et al. 2021)
    - Hard reset (Ledinauskas et al. 2020)
    - Detach reset (Zenke et al. 2021)
    - Make leaks numerically stable with sigmoid (Fang et al. 2020)
    - Learnable threshold instead of bias
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        kernel_size,
        stride=1,
        activation="arctanspike",
        act_width=10.0,
        leak_i=(-4.0, 0.1),
        leak_v=(-4.0, 0.1),
        thresh=(1.0, 0.0),
        learn_leak=True,
        learn_thresh=True,
        hard_reset=True,
        detach=True,
        norm=None,
        padding=None,
        share_params=False,
        **kargs,
    ):
        super().__init__()

        # shapes
        if padding is None:
            padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size

        # weights
        self.ff = nn.Conv2d(input_size, hidden_size, kernel_size, stride=stride, padding=padding, bias=False)
        w_scale = math.sqrt(1 / input_size)
        nn.init.uniform_(self.ff.weight, -w_scale, w_scale)

        # parameters
        if share_params:
            leak_i = torch.randn(1, 1, 1) * leak_i[1] + leak_i[0]
            leak_v = torch.randn(1, 1, 1) * leak_v[1] + leak_v[0]
        else:
            leak_i = torch.randn(hidden_size, 1, 1) * leak_i[1] + leak_i[0]
            leak_v = torch.randn(hidden_size, 1, 1) * leak_v[1] + leak_v[0]
        if learn_leak:
            self.leak_i = nn.Parameter(leak_i)
            self.leak_v = nn.Parameter(leak_v)
        else:
            self.register_buffer("leak_i", leak_i)
            self.register_buffer("leak_v", leak_v)

        if share_params:
            thresh = torch.randn(1, 1, 1) * thresh[1] + thresh[0]
        else:
            thresh = torch.randn(hidden_size, 1, 1) * thresh[1] + thresh[0]
        if learn_thresh:
            self.thresh = nn.Parameter(thresh)
        else:
            self.register_buffer("thresh", thresh)

        # spiking and reset mechanics
        assert isinstance(
            activation, str
        ), "Spiking neurons need a valid activation, see models/spiking_util.py for choices"
        self.spike_fn = getattr(spiking, activation)
        self.register_buffer("act_width", torch.tensor(act_width))
        self.hard_reset = hard_reset
        self.detach = detach

        # norm
        if norm == "weight":
            self.ff = nn.utils.weight_norm(self.ff)
            self.norm = None
        elif norm == "group":
            groups = min(1, input_size // 4)  # at least instance norm
            self.norm = nn.GroupNorm(groups, input_size)
        else:
            self.norm = None

    def forward(self, input_, prev_state):
        # input current
        if self.norm is not None:
            input_ = self.norm(input_)
        ff = self.ff(input_)

        # generate empty prev_state, if None is provided
        if prev_state is None:
            prev_state = torch.zeros(4, *ff.shape, dtype=ff.dtype, device=ff.device)
        i, v, z, _ = prev_state  # unbind op, removes dimension

        # clamp thresh
        thresh = self.thresh.clamp_min(0.01)

        # get leaks
        leak_i = torch.sigmoid(self.leak_i)
        leak_v = torch.sigmoid(self.leak_v)

        # detach reset
        if self.detach:
            z = z.detach()

        # current update: decay, add
        i_out = i * leak_i + ff

        # voltage update: decay, reset, add
        if self.hard_reset:
            v_out = v * leak_v * (1 - z) + i_out
        else:
            v_out = v * leak_v + i_out - z * thresh

        # spike
        z_out = self.spike_fn(v_out, thresh, self.act_width)

        return z_out, torch.stack([i_out, v_out, z_out, ff])


class ConvLoihiLIF(nn.Module):
    """
    Loihi-compatible version of ConvLIF.

    Loihi has:
    - quantized weights in [-256..2..254]
    - quantized leaks in [0..4096] (comparmentCurrentDecay, compartmentVoltageDecay)
    - quantized threshold in [0..131071] (vthMant)
    - factor of 2**6 in current and in threshold
    - voltage reset with new spike afterwards
    - a delay of one timestep between layers, but not at either end of the network

    Because we have spiking neurons with a threshold, we don't actually need to
    scale the parameters to these large ranges. This is nice, because then we don't
    have to scale things like learning rates, surrogate gradients, etc. So:
    - quantized weights in [-1, 1 - 2/256] in steps of 2/256
    - quantized leaks in [0, 1] in steps of 1/4096
    - quantized threshold in [0, 2] in steps of 1/256 (threshold of 1 should be 256)
    - left out factor of 2**6 in current and in threshold
    - voltage reset with new spike afterwards, but can't do detach in this case (blocks all gradients)
    - a delay buffer of length one before each layer, emulated by updating voltage before current
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        kernel_size,
        stride=1,
        activation="arctanspike",
        act_width=10.0,
        leak_i=(-4.0, 0.1),
        leak_v=(-4.0, 0.1),
        thresh=(1.0, 0.0),
        learn_leak=True,
        learn_thresh=True,
        hard_reset=True,
        detach=False,
        norm=None,
        padding=None,
        delay=False,
        share_params=True,
        weights_bits=6,
        **kargs,
    ):
        super().__init__()
        assert weights_bits in [1, 2, 4, 5, 6, 8]

        # axonal delay
        self.delay = delay

        # shapes
        if padding is None:
            padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size

        # parameters

        # weights
        self.ff = nn.Conv2d(input_size, hidden_size, kernel_size, stride=stride, padding=padding, bias=False)
        w_scale = math.sqrt(1 / input_size)
        nn.init.uniform_(self.ff.weight, -w_scale, w_scale)
        quant_dw = 2 ** (8 - (weights_bits - 1))
        quant.quantize_param(self.ff, "weight", -1, 1 - quant_dw / 256, quant_dw / 256)

        # leaks: [0..1..4096] = 12 bits
        if share_params:
            leak_i = 1 - torch.sigmoid(torch.randn(1, 1, 1) * leak_i[1] + leak_i[0])
            leak_v = 1 - torch.sigmoid(torch.randn(1, 1, 1) * leak_v[1] + leak_v[0])
        else:
            leak_i = 1 - torch.sigmoid(torch.randn(hidden_size, 1, 1) * leak_i[1] + leak_i[0])
            leak_v = 1 - torch.sigmoid(torch.randn(hidden_size, 1, 1) * leak_v[1] + leak_v[0])
        if learn_leak:
            self.leak_i = nn.Parameter(leak_i)
            self.leak_v = nn.Parameter(leak_v)
            quant.quantize_param(self, "leak_i", 0, 1, 1 / 4096)
            quant.quantize_param(self, "leak_v", 0, 1, 1 / 4096)
        else:
            self.register_buffer("leak_i", leak_i)
            self.register_buffer("leak_v", leak_v)
            quant.quantize_buffer(self, "leak_i", 0, 1, 1 / 4096)
            quant.quantize_buffer(self, "leak_v", 0, 1, 1 / 4096)

        # thresh: [0..1..131071] = almost 17 bits, but scale with same factor as weights
        if share_params:
            thresh = torch.randn(1, 1, 1) * thresh[1] + thresh[0]
        else:
            thresh = torch.randn(hidden_size, 1, 1) * thresh[1] + thresh[0]
        if learn_thresh:
            self.thresh = nn.Parameter(thresh)
            quant.quantize_param(self, "thresh", 0, 2, 1 / 256)
        else:
            self.register_buffer("thresh", thresh)
            quant.quantize_buffer(self, "thresh", 0, 2, 1 / 256)

        # spiking and reset mechanics
        assert isinstance(
            activation, str
        ), "Spiking neurons need a valid activation, see models/spiking_util.py for choices"
        self.spike_fn = getattr(spiking, activation)
        self.register_buffer("act_width", torch.tensor(act_width))
        self.hard_reset = hard_reset
        assert hard_reset, "ConvLoihiLIF does not support soft reset"
        assert not detach, "For ConvLoihiLIF, detaching reset would block all grads"

        # norm
        assert norm is None, "ConvLoihiLIF does not support norm"

    def forward(self, input_, prev_state):
        # input current
        ff_out = self.ff(input_)

        # generate empty prev_state, if None is provided
        if prev_state is None:
            prev_state = torch.zeros(4, *ff_out.shape, dtype=ff_out.dtype, device=ff_out.device)
        i, v, _, ff = prev_state  # unbind op, removes dimension

        # get parameters
        leak_i = 1 - self.leak_i
        leak_v = 1 - self.leak_v
        thresh = self.thresh

        # current update: decay
        if not self.training:
            i_out = i.double() * leak_i.double()
            i_out = quant.round_buffer_twd_zero(i_out, 2**14).float()
        else:
            i_out = i * leak_i
            i_out = quant.round_buffer_twd_zero(i_out, 2**14)

        # current update: add
        if self.delay:
            i_out = i_out + ff
        else:
            i_out = i_out + ff_out
        # i_out = torch.clamp(i_out * 2 ** 14, min=-2 ** 23, max=2 ** 23 - 1) / 2 ** 14  # TODO: hardcoded

        # voltage update: decay
        if not self.training:
            v_out = v.double() * leak_v.double()
            v_out = quant.round_buffer_twd_zero(v_out, 2**14).float()
        else:
            v_out = v * leak_v
            v_out = quant.round_buffer_twd_zero(v_out, 2**14)

        # voltage update: add
        v_out = v_out + i_out
        # v_out = torch.clamp(v_out * 2 ** 14, min=-2 ** 23, max=2 ** 23 - 1) / 2 ** 14  # TODO: hardcoded

        # spike
        z_out = self.spike_fn(v_out, thresh, self.act_width)

        # reset voltage with new spike
        v_out = v_out * (1 - z_out)

        return z_out, torch.stack([i_out, v_out, z_out, ff_out])


class ConvLIFRecurrent(nn.Module):
    """
    Convolutional recurrent spiking LIF cell.
    Design choices:
    + No delays between current, voltage and spike
    + Reset with spikes from last timestep
    + Stateful current and voltage (e.g. Zenke et al., 2021)
    - Arctan surrogate grad (Fang et al. 2021)
    - Hard reset (Ledinauskas et al. 2020)
    - Detach reset (Zenke et al. 2021)
    - Make leaks numerically stable with sigmoid (Fang et al. 2020)
    - Learnable threshold instead of bias
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        kernel_size,
        stride=1,
        activation="arctanspike",
        act_width=10.0,
        leak_i=(-4.0, 0.1),
        leak_v=(-4.0, 0.1),
        thresh=(1.0, 0.0),
        learn_leak=True,
        learn_thresh=True,
        hard_reset=True,
        detach=True,
        norm=None,
        share_params=False,
        self_rnn=True,
        **kargs,
    ):
        super().__init__()

        # shapes
        padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size

        # weights
        self.ff = nn.Conv2d(input_size, hidden_size, kernel_size, stride=stride, padding=padding, bias=False)
        if self_rnn:
            self.rec = nn.Conv2d(hidden_size, hidden_size, 1, padding=0, bias=False, groups=hidden_size)
        else:
            self.rec = nn.Conv2d(hidden_size, hidden_size, kernel_size, padding=padding, bias=False)
        w_scale_ff = math.sqrt(1 / input_size)
        w_scale_rec = math.sqrt(1 / hidden_size)
        nn.init.uniform_(self.ff.weight, -w_scale_ff, w_scale_ff)
        nn.init.uniform_(self.rec.weight, -w_scale_rec, w_scale_rec)

        # parameters
        if share_params:
            leak_i = torch.randn(1, 1, 1) * leak_i[1] + leak_i[0]
            leak_v = torch.randn(1, 1, 1) * leak_v[1] + leak_v[0]
        else:
            leak_i = torch.randn(hidden_size, 1, 1) * leak_i[1] + leak_i[0]
            leak_v = torch.randn(hidden_size, 1, 1) * leak_v[1] + leak_v[0]
        if learn_leak:
            self.leak_i = nn.Parameter(leak_i)
            self.leak_v = nn.Parameter(leak_v)
        else:
            self.register_buffer("leak_i", leak_i)
            self.register_buffer("leak_v", leak_v)

        if share_params:
            thresh = torch.randn(1, 1, 1) * thresh[1] + thresh[0]
        else:
            thresh = torch.randn(hidden_size, 1, 1) * thresh[1] + thresh[0]
        if learn_thresh:
            self.thresh = nn.Parameter(thresh)
        else:
            self.register_buffer("thresh", thresh)

        # spiking and reset mechanics
        assert isinstance(
            activation, str
        ), "Spiking neurons need a valid activation, see models/spiking_util.py for choices"
        self.spike_fn = getattr(spiking, activation)
        self.register_buffer("act_width", torch.tensor(act_width))
        self.hard_reset = hard_reset
        self.detach = detach

        # norm
        if norm == "weight":
            self.ff = nn.utils.weight_norm(self.ff)
            self.rec = nn.utils.weight_norm(self.rec)
            self.norm_ff = None
            self.norm_rec = None
        elif norm == "group":
            groups_ff = min(1, input_size // 4)  # at least instance norm
            groups_rec = min(1, hidden_size // 4)  # at least instance norm
            self.norm_ff = nn.GroupNorm(groups_ff, input_size)
            self.norm_rec = nn.GroupNorm(groups_rec, hidden_size)
        else:
            self.norm_ff = None
            self.norm_rec = None

    def forward(self, input_, prev_state):
        # input current
        if self.norm_ff is not None:
            input_ = self.norm_ff(input_)
        ff = self.ff(input_)

        # generate empty prev_state, if None is provided
        if prev_state is None:
            prev_state = torch.zeros(4, *ff.shape, dtype=ff.dtype, device=ff.device)
        i, v, z, _ = prev_state  # unbind op, removes dimension

        # recurrent current
        if self.norm_rec is not None:
            z = self.norm_rec(z)
        rec = self.rec(z)

        # clamp thresh
        thresh = self.thresh.clamp_min(0.01)

        # get leaks
        leak_i = torch.sigmoid(self.leak_i)
        leak_v = torch.sigmoid(self.leak_v)

        # detach reset
        if self.detach:
            z = z.detach()

        # current update: decay, add
        i_out = i * leak_i + (ff + rec)

        # voltage update: decay, reset, add
        if self.hard_reset:
            v_out = v * leak_v * (1 - z) + i_out
        else:
            v_out = v * leak_v + i_out - z * thresh

        # spike
        z_out = self.spike_fn(v_out, thresh, self.act_width)

        return z_out, torch.stack([i_out, v_out, z_out, ff])


class ConvLoihiLIFRecurrent(nn.Module):
    """
    Loihi-compatible version of ConvLIFRecurrent.

    Loihi has:
    - quantized weights in [-256..2..254]
    - quantized leaks in [0..4096] (comparmentCurrentDecay, compartmentVoltageDecay)
    - quantized threshold in [0..131071] (vthMant)
    - factor of 2**6 in current and in threshold
    - voltage reset with new spike afterwards
    - a delay of one timestep between layers, but not at either end of the network

    Because we have spiking neurons with a threshold, we don't actually need to
    scale the parameters to these large ranges. This is nice, because then we don't
    have to scale things like learning rates, surrogate gradients, etc. So:
    - quantized weights in [-1, 1 - 2/256] in steps of 2/256
    - quantized leaks in [0, 1] in steps of 1/4096
    - quantized threshold in [0, 2] in steps of 1/256 (threshold of 1 should be 256)
    - left out factor of 2**6 in current and in threshold
    - voltage reset with new spike afterwards, but can't do detach in this case (blocks all gradients)
    - a delay buffer of length one before each layer, emulated by updating voltage before current
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        kernel_size,
        stride=1,
        activation="arctanspike",
        act_width=10.0,
        leak_i=(-4.0, 0.1),
        leak_v=(-4.0, 0.1),
        thresh=(1.0, 0.0),
        learn_leak=True,
        learn_thresh=True,
        hard_reset=True,
        detach=False,
        norm=None,
        delay=False,
        share_params=True,
        self_rnn=True,
        weights_bits=6,
    ):
        super().__init__()
        assert weights_bits in [1, 2, 4, 5, 6, 8]

        # axonal delay
        self.delay = delay

        # shapes
        padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size

        # parameters

        # weights
        self.ff = nn.Conv2d(input_size, hidden_size, kernel_size, stride=stride, padding=padding, bias=False)
        if self_rnn:
            self.rec = nn.Conv2d(hidden_size, hidden_size, 1, padding=0, bias=False, groups=hidden_size)
        else:
            self.rec = nn.Conv2d(hidden_size, hidden_size, kernel_size, padding=padding, bias=False)
        w_scale_ff = math.sqrt(1 / input_size)
        w_scale_rec = math.sqrt(1 / hidden_size)
        nn.init.uniform_(self.ff.weight, -w_scale_ff, w_scale_ff)
        nn.init.uniform_(self.rec.weight, -w_scale_rec, w_scale_rec)
        quant_dw = 2 ** (8 - (weights_bits - 1))
        quant.quantize_param(self.ff, "weight", -1, 1 - quant_dw / 256, quant_dw / 256)
        quant.quantize_param(self.rec, "weight", -1, 1 - quant_dw / 256, quant_dw / 256)

        # leaks: [0..1..4096] = 12 bits
        if share_params:
            leak_i = 1 - torch.sigmoid(torch.randn(1, 1, 1) * leak_i[1] + leak_i[0])
            leak_v = 1 - torch.sigmoid(torch.randn(1, 1, 1) * leak_v[1] + leak_v[0])
        else:
            leak_i = 1 - torch.sigmoid(torch.randn(hidden_size, 1, 1) * leak_i[1] + leak_i[0])
            leak_v = 1 - torch.sigmoid(torch.randn(hidden_size, 1, 1) * leak_v[1] + leak_v[0])
        if learn_leak:
            self.leak_i = nn.Parameter(leak_i)
            self.leak_v = nn.Parameter(leak_v)
            quant.quantize_param(self, "leak_i", 0, 1, 1 / 4096)
            quant.quantize_param(self, "leak_v", 0, 1, 1 / 4096)
        else:
            self.register_buffer("leak_i", leak_i)
            self.register_buffer("leak_v", leak_v)
            quant.quantize_buffer(self, "leak_i", 0, 1, 1 / 4096)
            quant.quantize_buffer(self, "leak_v", 0, 1, 1 / 4096)

        # thresh: [0..1..131071] = almost 17 bits, but scale with same factor as weights
        if share_params:
            thresh = torch.randn(1, 1, 1) * thresh[1] + thresh[0]
        else:
            thresh = torch.randn(hidden_size, 1, 1) * thresh[1] + thresh[0]
        if learn_thresh:
            self.thresh = nn.Parameter(thresh)
            quant.quantize_param(self, "thresh", 0, 2, 1 / 256)
        else:
            self.register_buffer("thresh", thresh)
            quant.quantize_buffer(self, "thresh", 0, 2, 1 / 256)

        # spiking and reset mechanics
        assert isinstance(
            activation, str
        ), "Spiking neurons need a valid activation, see models/spiking_util.py for choices"
        self.spike_fn = getattr(spiking, activation)
        self.register_buffer("act_width", torch.tensor(act_width))
        assert hard_reset, "ConvLoihiLIFRecurrent does not support soft reset"
        self.hard_reset = hard_reset
        assert not detach, "For ConvLoihiLIFRecurrent, detaching reset would block all grads"

        # norm
        assert norm is None, "ConvLoihiLIFRecurrent does not support norm"

    def forward(self, input_, prev_state):
        # input current
        ff_out = self.ff(input_)

        # generate empty prev_state, if None is provided
        if prev_state is None:
            prev_state = torch.zeros(4, *ff_out.shape, dtype=ff_out.dtype, device=ff_out.device)
        i, v, z, ff = prev_state  # unbind op, removes dimension

        # recurrent current
        rec = self.rec(z)

        # get parameters
        leak_i = 1 - self.leak_i
        leak_v = 1 - self.leak_v
        thresh = self.thresh

        # current update: decay
        if not self.training:
            i_out = i.double() * leak_i.double()
            i_out = quant.round_buffer_twd_zero(i_out, 2**14).float()
        else:
            i_out = i * leak_i
            i_out = quant.round_buffer_twd_zero(i_out, 2**14)

        # current update: decay
        if self.delay:
            i_out = i_out + (ff + rec)
        else:
            i_out = i_out + (ff_out + rec)
        # i_out = torch.clamp(i_out * 2 ** 14, min=-2 ** 23, max=2 ** 23 - 1) / 2 ** 14  # TODO: hardcoded

        # voltage update: decay
        if not self.training:
            v_out = v.double() * leak_v.double()
            v_out = quant.round_buffer_twd_zero(v_out, 2**14).float()
        else:
            v_out = v * leak_v
            v_out = quant.round_buffer_twd_zero(v_out, 2**14)

        # voltage update: add
        v_out = v_out + i_out
        # v_out = torch.clamp(v_out * 2 ** 14, min=-2 ** 23, max=2 ** 23 - 1) / 2 ** 14  # TODO: hardcoded

        # spike
        z_out = self.spike_fn(v_out, thresh, self.act_width)

        # reset voltage with new spike
        v_out = v_out * (1 - z_out)

        return z_out, torch.stack([i_out, v_out, z_out, ff_out])


class SpikingRecurrentConvLayer(nn.Module):
    """
    Layer comprised of a recurrent and spiking convolutional block.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        recurrent_block_type="lif",
        activation="arctanspike",
        **kwargs,
    ):
        super().__init__()

        assert recurrent_block_type in ["lif", "loihi"]
        if recurrent_block_type == "lif":
            RecurrentBlock = ConvLIFRecurrent
        elif recurrent_block_type == "loihi":
            RecurrentBlock = ConvLoihiLIFRecurrent

        self.recurrent_block = RecurrentBlock(
            in_channels, out_channels, kernel_size, stride, activation=activation, **kwargs
        )

    def forward(self, x, prev_state):
        return self.recurrent_block(x, prev_state)
