from .base import BaseModel
from .model_util import copy_states
from .unet import *


class EV4ptNet(BaseModel):
    """
    Set of convolutional enconders blocks that estimate the optical flow of the four
    corners of the input's image space from a set of input events.
    """

    unet_type = EncoderNet
    num_output_channels = 8

    def __init__(self, kwargs, crop=None, num_bins=2):
        super().__init__()

        norm = None
        kernel_size = 3
        num_encoders = 3
        base_channels = 32
        final_activation = None

        if "crop" in kwargs.keys():
            crop = kwargs["crop"]
        if "norm" in kwargs.keys():
            norm = kwargs["norm"]
        if "kernel_size" in kwargs.keys():
            kernel_size = kwargs["kernel_size"]
        if "num_encoders" in kwargs.keys():
            num_encoders = kwargs["num_encoders"]
        if "base_channels" in kwargs.keys():
            base_channels = kwargs["base_channels"]
        if "num_output_channels" in kwargs.keys():
            self.num_output_channels = kwargs["num_output_channels"]
        if "final_activation" in kwargs.keys():
            final_activation = kwargs["final_activation"]

        EncoderNet_kwargs = {
            "num_bins": num_bins,
            "base_channels": base_channels,
            "num_encoders": num_encoders,
            "num_output_channels": self.num_output_channels,
            "crop": crop,
            "norm": norm,
            "kernel_size": kernel_size,
            "channel_multiplier": 2,
            "final_activation": final_activation,
            "spiking_neuron": kwargs["spiking_neuron"],
        }

        self.crop = crop

        kwargs.update(EncoderNet_kwargs)
        kwargs.pop("name", None)
        kwargs.pop("quantization", None)

        self.encoder_unet = self.unet_type(kwargs)

    def detach_states(self):
        pass

    def reset_states(self):
        pass

    def store_activity(self):
        self.encoder_unet.store_activity()

    def forward(self, event_cnt):
        """
        :param event_cnt: N x 2 x H x W
        :return: output dict with [N x num_output_channels X 1 X 1] encoded visual information.
        """

        # input encoding
        x = event_cnt.clone()

        # input cropping
        if self.crop is not None:
            x = self.input_cropping(x, self.crop)

        # forward pass
        flow_vectors, _, _ = self.encoder_unet.forward(x)

        return {"flow_vectors": flow_vectors}


class SplitEV4ptNet(EV4ptNet):
    """
    Split version of EV4ptNet.
    """

    unet_type = SplitEncoderNet
    num_output_channels = 2


class SpikingEV4ptNet(EV4ptNet):
    """
    Spiking version of EV4ptNet.
    """

    unet_type = SpikingEncoderNet

    def detach_states(self):
        detached_states = []
        for state in self.encoder_unet.states:
            if type(state) is tuple:
                tmp = []
                for hidden in state:
                    tmp.append(hidden.detach())
                detached_states.append(tuple(tmp))
            else:
                detached_states.append(state.detach())
        self.encoder_unet.states = detached_states

    def reset_states(self):
        self.encoder_unet.states = [None] * self.encoder_unet.num_states


class LoihiEV4ptNet(SpikingEV4ptNet):
    """
    Loihi-compatible version of SpikingEV4ptNet.
    """

    unet_type = LoihiEncoderNet


class SplitSpikingEV4ptNet(SpikingEV4ptNet):
    """
    Spiking version of SplitEV4ptNet.
    """

    unet_type = SplitSpikingEncoderNet
    num_output_channels = 2


class SplitLoihiEV4ptNet(SpikingEV4ptNet):
    """
    Loihi-compatible version of SplitSpikingEV4ptNet.
    """

    unet_type = SplitLoihiEncoderNet
    num_output_channels = 2


class RecEV4ptNet(BaseModel):
    """
    Set of recurrent convolutional enconders blocks that estimate the optical flow of the four
    corners of the input's image space from a set of input events.
    """

    unet_type = RecEncoderNet
    recurrent_block_type = "convgru"
    num_output_channels = 8

    def __init__(self, kwargs, crop=None, num_bins=2):
        super().__init__()

        norm = None
        kernel_size = 3
        num_encoders = 3
        base_channels = 32
        final_activation = None

        if "crop" in kwargs.keys():
            crop = kwargs["crop"]
        if "norm" in kwargs.keys():
            norm = kwargs["norm"]
        if "kernel_size" in kwargs.keys():
            kernel_size = kwargs["kernel_size"]
        if "num_encoders" in kwargs.keys():
            num_encoders = kwargs["num_encoders"]
        if "base_channels" in kwargs.keys():
            base_channels = kwargs["base_channels"]
        if "num_output_channels" in kwargs.keys():
            self.num_output_channels = kwargs["num_output_channels"]
        if "final_activation" in kwargs.keys():
            final_activation = kwargs["final_activation"]

        EncoderNet_kwargs = {
            "num_bins": num_bins,
            "base_channels": base_channels,
            "num_encoders": num_encoders,
            "num_output_channels": self.num_output_channels,
            "crop": crop,
            "norm": norm,
            "kernel_size": kernel_size,
            "channel_multiplier": 2,
            "recurrent_block_type": self.recurrent_block_type,
            "final_activation": final_activation,
            "spiking_neuron": kwargs["spiking_neuron"],
        }

        self.crop = crop

        kwargs.update(EncoderNet_kwargs)
        kwargs.pop("name", None)
        kwargs.pop("quantization", None)

        self.encoder_unet = self.unet_type(kwargs)

    @property
    def states(self):
        return copy_states(self.encoder_unet.states)

    @states.setter
    def states(self, states):
        self.encoder_unet.states = states

    def detach_states(self):
        detached_states = []
        for state in self.encoder_unet.states:
            if type(state) is tuple:
                tmp = []
                for hidden in state:
                    tmp.append(hidden.detach())
                detached_states.append(tuple(tmp))
            else:
                detached_states.append(state.detach())
        self.encoder_unet.states = detached_states

    def reset_states(self):
        self.encoder_unet.states = [None] * self.encoder_unet.num_states

    def store_activity(self):
        self.encoder_unet.store_activity()

    def forward(self, event_cnt):
        """
        :param event_cnt: N x 2 x H x W
        :return: output dict with [N x num_output_channels X 1 X 1] encoded visual information.
        """

        # input encoding
        x = event_cnt.clone()

        # input cropping
        if self.crop is not None:
            x = self.input_cropping(x, self.crop)

        # forward pass
        flow_vectors, spikes, activity = self.encoder_unet.forward(x)

        return {"flow_vectors": flow_vectors, "spikes": spikes, "activity": activity}


class SplitRecEV4ptNet(RecEV4ptNet):
    """
    Split version of RecEV4ptNet.
    """

    unet_type = SplitRecEncoderNet
    num_output_channels = 2


class SpikingRec4ptNet(RecEV4ptNet):
    """
    Spiking version of RecEV4ptNet.
    """

    unet_type = SpikingRecEncoderNet
    recurrent_block_type = "lif"


class SplitSpikingRec4ptNet(RecEV4ptNet):
    """
    Spiking version of RecEV4ptNet.
    """

    unet_type = SplitSpikingRecEncoderNet
    recurrent_block_type = "lif"
    num_output_channels = 2


class LoihiRec4ptNet(RecEV4ptNet):
    """
    Loihi-compatible version of SpikingRec4ptNet.
    """

    unet_type = LoihiRecEncoderNet
    recurrent_block_type = "loihi"


class SplitLoihiRec4ptNet(RecEV4ptNet):
    """
    Loihi-compatible version of SpikingRec4ptNet.
    """

    unet_type = SplitLoihiRecEncoderNet
    recurrent_block_type = "loihi"
    num_output_channels = 2
