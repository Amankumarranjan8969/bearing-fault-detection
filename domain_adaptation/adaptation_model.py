import torch.nn as nn
from .grl import grad_reverse


class DomainHead(nn.Module):
    """Domain classifier behind a gradient-reversal layer: the encoder learns to FOOL it, which removes
    dataset-specific information from the shared representation while the fault head keeps the fault information."""

    def __init__(self, dim, n_domains=2, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, n_domains))

    def forward(self, z, alpha=1.0):
        return self.net(grad_reverse(z, alpha))
