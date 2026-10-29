import torch
import torch.nn as nn
from .cnn import Encoder1D


class MultimodalCNN(nn.Module):
    """One CNN per signal (vibration, current1, current2) + small dense branch for RPM/torque/force/temperature,
    fused into a shared representation (experiment 7). n_channels=1 gives a vibration-only model."""

    def __init__(self, n_channels, n_classes, aux_dim=4, emb=64, fusion=128, dropout=0.3):
        super().__init__()
        self.aux_dim = aux_dim
        self.vib_enc = Encoder1D(1, emb=emb)
        self.cur_encs = nn.ModuleList([Encoder1D(1, emb=emb) for _ in range(n_channels - 1)])
        self.aux_net = nn.Sequential(nn.Linear(aux_dim, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU()) if aux_dim else None
        self.fuse = nn.Sequential(nn.Linear(emb * n_channels + (16 if aux_dim else 0), fusion), nn.ReLU(), nn.Dropout(dropout))
        self.head = nn.Linear(fusion, n_classes)
        self.emb_dim = fusion

    def embed(self, x, aux=None):
        z = [self.vib_enc(x[:, :1])]
        for i, enc in enumerate(self.cur_encs):
            z.append(enc(x[:, i + 1:i + 2]))
        if self.aux_net is not None:
            z.append(self.aux_net(aux if aux is not None else x.new_zeros(len(x), self.aux_dim)))
        return self.fuse(torch.cat(z, 1))

    def forward(self, x, aux=None, alpha=1.0):
        z = self.embed(x, aux)
        return {"logits": self.head(z), "embedding": z}
