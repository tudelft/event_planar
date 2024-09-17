"""
Structure based on torch.nn.utils.weight_norm: https://github.com/pytorch/pytorch/blob/master/torch/nn/utils/weight_norm.py.
"""

import torch
import torch.nn as nn


class Quantize:
    """
    Quantization object that keeps track of full-precision parameters
    and computes quantized parameters before the forward pass.
    """

    name: str

    def __init__(self, name, quant_min, quant_max, step):
        # name of parameter
        self.name = name

        # quantization range
        self.quant_min = quant_min
        self.quant_max = quant_max

        # quantization step
        self.step = step

    def compute_quantized(self, module):
        """
        Computes the quantized parameter from the stored full-precision parameter.
        """
        # get full-precision
        full = getattr(module, self.name + "_f")
        return _quantize(full, self.quant_min, self.quant_max, self.step)

    @staticmethod
    def apply(module, name, quant_min, quant_max, step):
        """
        Apply quantization to a parameter of a certain module as a pre-forward hook.
        """
        # function to call in hook
        fn = Quantize(name, quant_min, quant_max, step)

        # get full-precision, remove from parameter list
        full = getattr(module, name)
        del module._parameters[name]

        # register full-precision under different name
        module.register_parameter(name + "_f", nn.Parameter(full.data))
        setattr(module, name, fn.compute_quantized(module))

        # register quantization to occur before forward pass
        module.register_forward_pre_hook(fn)

        return fn

    def remove(self, module):
        """
        Remove full-precision, keep only quantized parameter.
        """
        quantized = self.compute_quantized(module)
        delattr(module, self.name)
        del module._parameters[self.name + "_f"]
        setattr(module, self.name, nn.Parameter(quantized.data))

    def __call__(self, module, inputs):
        """
        Called by hook: re-computes the quantized parameter.
        """
        setattr(module, self.name, self.compute_quantized(module))


class QuantizeFn(torch.autograd.Function):
    """
    Quantizes parameter with a certain step size and range
    while propagating gradients.
    """

    @staticmethod
    def forward(ctx, x, quant_min, quant_max, step):
        x_quant = ((x / step).round() * step).clamp(quant_min, quant_max)
        return x_quant

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None, None, None


_quantize = QuantizeFn.apply


def quantize(x, quant_min, quant_max, step):
    """
    Apply quantization to a certain parameter.
    """

    return _quantize(x, quant_min, quant_max, step)


def quantize_param(module, name, quant_min, quant_max, step):
    """
    Apply quantization to a certain parameter of a certain module.
    """
    Quantize.apply(module, name, quant_min, quant_max, step)
    return module


def quantize_buffer(module, name, quant_min, quant_max, step):
    """
    Apply quantization to a certain parameter that is not going to be learned of a certain module.
    """
    full = getattr(module, name)
    setattr(module, name, _quantize(full, quant_min, quant_max, step))
    return module


def remove_quantization(module, name):
    """
    Remove quantization from a certain module.
    """
    for k, hook in module._forward_pre_hooks.items():
        if isinstance(hook, Quantize) and hook.name == name:
            hook.remove(module)
            del module._forward_pre_hooks[k]
            print(f"Quantization of {name} removed from {module}")
            return module

    print(f"Quantization of {name} not found in {module}")


class RoundTwdZeroFn(torch.autograd.Function):
    """
    Rounds tensor towards zero.
    """

    @staticmethod
    def forward(ctx, x, scale):
        x_round = x * scale
        x_round[x_round > 0] = x_round[x_round > 0].floor()
        x_round[x_round < 0] = x_round[x_round < 0].ceil()
        x_round = x_round / scale
        return x_round

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None


_round_twd_zero = RoundTwdZeroFn.apply


def round_buffer_twd_zero(x, scale):
    """
    Apply rounding towards zero to a tensor.
    """

    return _round_twd_zero(x, scale)
