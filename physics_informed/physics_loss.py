"""Physics loss built on MEASURED spectral evidence (envelope-spectrum energy at BPFO/BPFI/BSF/FTF), not an
arbitrary penalty.  target: (B, 4) in [0, 1] = energy ratios [BPFO, BPFI, BSF, FTF] scaled by a training percentile.
  L_align : the physics head must reproduce the measured characteristic-frequency energies.
  L_cons  : probability mass on a fault class is penalised when that fault's frequency shows no energy.
Both are unsupervised w.r.t. labels, so they can also be applied to unlabeled target-domain windows.
Ablate w_consistency first: if it hurts validation F1, set it to 0."""
import torch
import torch.nn.functional as F

FAULT_TO_PHYS_IDX = {"outer_race": 0, "inner_race": 1, "ball": 2}


def spectral_alignment_loss(pred, target):
    return F.mse_loss(pred, target)


def evidence_consistency_loss(logits, target, class_names):
    p = F.softmax(logits, dim=1)
    terms = [(p[:, i] * (1.0 - target[:, FAULT_TO_PHYS_IDX[n]])).mean()
             for i, n in enumerate(class_names) if n in FAULT_TO_PHYS_IDX]
    return sum(terms) / max(len(terms), 1) if terms else logits.new_zeros(())


def physics_loss(out, target, class_names, w_align=1.0, w_cons=0.5):
    return (w_align * spectral_alignment_loss(out["physics"], target)
            + w_cons * evidence_consistency_loss(out["logits"], target, class_names))
