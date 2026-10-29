import torch.nn as nn


def conv_block(cin, cout, k=9, pool=4):
    return nn.Sequential(nn.Conv1d(cin, cout, k, padding=k // 2), nn.BatchNorm1d(cout), nn.ReLU(), nn.MaxPool1d(pool))


class Encoder1D(nn.Module):
    """Raw window (B, C, L) -> embedding (B, emb). Aggressive early pooling keeps long windows cheap."""

    def __init__(self, in_ch, widths=(16, 32, 64, 64), k=9, emb=64):
        super().__init__()
        layers, c = [], in_ch
        for i, w in enumerate(widths):
            layers.append(conv_block(c, w, k, pool=4 if i < 2 else 2))
            c = w
        self.blocks = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(c, emb)
        self.out_ch = c

    def feature_map(self, x):
        return self.blocks(x)

    def forward(self, x):
        return self.proj(self.gap(self.blocks(x)).flatten(1))


class CNN1D(nn.Module):
    """Baseline (experiment 5). All DL models share: forward(x, aux=None, alpha=1.0) -> dict."""

    def __init__(self, in_ch, n_classes, emb=64, dropout=0.3):
        super().__init__()
        self.encoder = Encoder1D(in_ch, emb=emb)
        self.head = nn.Sequential(nn.ReLU(), nn.Dropout(dropout), nn.Linear(emb, n_classes))
        self.emb_dim = emb

    def embed(self, x, aux=None):
        return self.encoder(x)

    def forward(self, x, aux=None, alpha=1.0):
        z = self.embed(x, aux)
        return {"logits": self.head(z), "embedding": z}
