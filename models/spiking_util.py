import torch


class BaseSpike(torch.autograd.Function):
    """
    Baseline spiking function.
    """

    @staticmethod
    def forward(ctx, x, width):
        ctx.save_for_backward(x, width)
        return x.gt(0).float()

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError


class ArctanSpike(BaseSpike):
    """
    Spike function with derivative of arctan surrogate gradient.
    Featured in Fang et al. 2020/2021.
    """

    @staticmethod
    def backward(ctx, grad_output):
        x, width = ctx.saved_tensors
        grad_input = grad_output.clone()
        sg = 1 / (1 + width * x * x)
        return grad_input * sg, None


def arctanspike(x, thresh=torch.tensor(1.0), width=torch.tensor(10.0)):
    return ArctanSpike.apply(x - thresh, width)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    x = torch.linspace(-5, 5, 1001)
    arctanspike_ = 1 / (1 + 10 * x * x)

    plt.plot(x.numpy(), arctanspike_.numpy(), label="arctanspike")
    plt.xlabel("v - thresh")
    plt.ylabel("grad")
    plt.grid()
    plt.legend()
    plt.show()
