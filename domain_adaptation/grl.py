import math
import torch


class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.alpha * grad, None


def grad_reverse(x, alpha=1.0):
    return _GradReverse.apply(x, alpha)


def dann_alpha(progress, gamma=10.0):
    """Ganin & Lempitsky schedule: 0 -> 1 over training (progress in [0, 1])."""
    return 2.0 / (1.0 + math.exp(-gamma * progress)) - 1.0
