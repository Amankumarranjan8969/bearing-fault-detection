import torch.nn as nn
from .cnn import Encoder1D


class CNNRecurrent(nn.Module):
    """CNN local features -> GRU/LSTM temporal representation (experiment 6)."""

    def __init__(self, in_ch, n_classes, cell="gru", hidden=64, dropout=0.3):
        super().__init__()
        self.encoder = Encoder1D(in_ch)
        self.rnn = (nn.GRU if cell == "gru" else nn.LSTM)(self.encoder.out_ch, hidden, batch_first=True)
        self.head = nn.Sequential(nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, n_classes))
        self.emb_dim = hidden

    def embed(self, x, aux=None):
        f = self.encoder.feature_map(x).transpose(1, 2)        # (B, T, C)
        out, _ = self.rnn(f)
        return out.mean(1)

    def forward(self, x, aux=None, alpha=1.0):
        z = self.embed(x, aux)
        return {"logits": self.head(z), "embedding": z}
