import torch.nn as nn
from domain_adaptation.adaptation_model import DomainHead


class PhysicsInformedNet(nn.Module):
    """backbone (MultimodalCNN) -> shared representation -> FAULT head | PHYSICS head | DOMAIN head (optional)."""

    def __init__(self, backbone, n_classes, n_phys=4, use_domain=False, n_domains=2):
        super().__init__()
        self.backbone = backbone
        d = backbone.emb_dim
        self.fault_head = nn.Linear(d, n_classes)
        self.physics_head = nn.Sequential(nn.Linear(d, 32), nn.ReLU(), nn.Linear(32, n_phys), nn.Sigmoid())
        self.domain_head = DomainHead(d, n_domains) if use_domain else None

    def forward(self, x, aux=None, alpha=1.0):
        z = self.backbone.embed(x, aux)
        out = {"logits": self.fault_head(z), "physics": self.physics_head(z), "embedding": z}
        if self.domain_head is not None:
            out["domain_logits"] = self.domain_head(z, alpha)
        return out
