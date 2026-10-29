"""Train DL models on the .npz windows produced by scripts.prepare_data.
   kinds: cnn | cnn_gru | cnn_lstm | multimodal | pi (physics) | da (domain adaptation) | pi_da (physics + DA)
   python -m deep_learning.train_dl --model pi_da --source data/processed/synthetic_windows.npz \
                                    --target data/processed/synthetic_target_windows.npz
Model selection uses SOURCE validation macro-F1 only. Target labels are used for reporting, never for training."""
import argparse
import copy
import json
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from configs.config import load_config
from domain_adaptation.grl import dann_alpha
from evaluation.metrics import classification_metrics
from models import registry
from physics_informed.physics_loss import physics_loss
from physics_informed.pinn_model import PhysicsInformedNet
from .cnn import CNN1D
from .cnn_gru import CNNRecurrent
from .multimodal import MultimodalCNN

KINDS = ("cnn", "cnn_gru", "cnn_lstm", "multimodal", "pi", "da", "pi_da")


def infinite(loader):
    """Restart the loader every pass (itertools.cycle would replay the first shuffle forever)."""
    while True:
        yield from loader


def load_npz(path):
    d = np.load(path, allow_pickle=False)
    return {k: d[k] for k in d.files}


def build_model(kind, n_ch, n_classes, aux_dim=4):
    if kind == "cnn":
        return CNN1D(n_ch, n_classes)
    if kind in ("cnn_gru", "cnn_lstm"):
        return CNNRecurrent(n_ch, n_classes, cell=kind.split("_")[1])
    backbone = MultimodalCNN(n_ch, n_classes, aux_dim)
    if kind == "multimodal":
        return backbone
    return PhysicsInformedNet(backbone, n_classes, use_domain=kind in ("da", "pi_da"))


def norm_stats(d, mask):
    """Per-domain statistics (unsupervised, so it is legitimate to compute them on target windows too)."""
    import warnings
    X, aux, ph = d["X"][mask], d["aux"][mask], d["phys"][mask]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mu = np.nan_to_num(np.nanmean(aux, 0), nan=0.0)
        sd = np.nan_to_num(np.nanstd(aux, 0), nan=1.0)
    sd = np.where(sd < 1e-6, 1.0, sd)                       # constant column -> centred, not exploded
    return {"ch_std": X.std(axis=(0, 2), keepdims=True) + 1e-8, "aux_mu": mu, "aux_sd": sd,
            "phys_scale": np.percentile(ph, 95, axis=0) + 1e-6}


def to_dataset(d, mask, st, vib_only=False):
    X = d["X"][mask]
    X = X[:, :1] if vib_only else X
    ch = st["ch_std"][:, :1] if vib_only else st["ch_std"]
    aux = np.nan_to_num((d["aux"][mask] - st["aux_mu"]) / st["aux_sd"], nan=0.0)
    ph = np.clip(d["phys"][mask] / st["phys_scale"], 0, 1)
    return TensorDataset(torch.tensor(X / ch, dtype=torch.float32), torch.tensor(aux, dtype=torch.float32),
                         torch.tensor(d["y"][mask], dtype=torch.long), torch.tensor(ph, dtype=torch.float32))


@torch.no_grad()
def predict(model, loader, dev):
    model.eval()
    P, Y = [], []
    for xb, ab, yb, _ in loader:
        P.append(F.softmax(model(xb.to(dev), ab.to(dev))["logits"], 1).cpu().numpy())
        Y.append(yb.numpy())
    P, Y = np.concatenate(P), np.concatenate(Y)
    return Y, P.argmax(1), P


def evaluate(model, ds, dev, labels):
    ok = ds.tensors[2] >= 0
    ds = TensorDataset(*[t[ok] for t in ds.tensors])
    if len(ds) == 0:
        return None
    y, yp, pr = predict(model, DataLoader(ds, batch_size=64), dev)
    return classification_metrics(y, yp, pr, labels=labels)


def train(kind, source, target=None, epochs=None, vib_only=False, out_dir="models/dl", cfg=None, verbose=True):
    assert kind in KINDS, f"unknown model kind {kind}"
    cfg = cfg or load_config()
    dl = cfg["dl"]
    epochs = epochs or dl["epochs"]
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    S = load_npz(source)
    classes = [str(c) for c in S["classes"]]
    labels = list(range(len(classes)))
    tr, va, te = ((S["split"] == s) & (S["y"] >= 0) for s in ("train", "val", "test"))
    use_dom = kind in ("da", "pi_da")
    lam_p = dl["lambda_physics"] if kind in ("pi", "pi_da") else 0.0
    lam_d = dl["lambda_domain"] if use_dom else 0.0
    if use_dom and target is None:
        raise ValueError(f"'{kind}' needs --target (unlabeled target-domain windows)")

    st = norm_stats(S, tr)
    ds_tr, ds_va, ds_te = (to_dataset(S, m, st, vib_only) for m in (tr, va, te))
    n_ch = ds_tr.tensors[0].shape[1]
    bs = max(2, min(dl["batch_size"], len(ds_tr) // 2))
    loader = DataLoader(ds_tr, batch_size=bs, shuffle=True, drop_last=True)
    T = tgt_train = tgt_test = None
    if target:
        T = load_npz(target)
        t_tr = np.isin(T["split"], ["train", "val"])
        t_te = T["split"] == "test"
        t_te = t_te if t_te.any() else np.ones(len(T["y"]), bool)
        t_st = norm_stats(T, t_tr)
        tgt_train, tgt_test = to_dataset(T, t_tr, t_st, vib_only), to_dataset(T, t_te, t_st, vib_only)
    tgt_iter = infinite(DataLoader(tgt_train, batch_size=bs, shuffle=True, drop_last=True)) if use_dom else None

    model = build_model(kind, n_ch, len(classes)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=dl["lr"], weight_decay=dl["weight_decay"])
    cw = torch.tensor(len(S["y"][tr]) / (len(classes) * np.maximum(np.bincount(S["y"][tr], minlength=len(classes)), 1)),
                      dtype=torch.float32, device=dev)
    total, step, best, bad, best_state = epochs * len(loader), 0, -1.0, 0, None
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        for xb, ab, yb, pb in loader:
            xb, ab, yb, pb = xb.to(dev), ab.to(dev), yb.to(dev), pb.to(dev)
            alpha = dann_alpha(step / max(total, 1))
            step += 1
            out = model(xb, ab, alpha)
            loss = F.cross_entropy(out["logits"], yb, weight=cw)
            if lam_p:
                loss = loss + lam_p * physics_loss(out, pb, classes, dl["w_align"], dl["w_consistency"])
            if use_dom:
                xt, at, _, pt = (t.to(dev) for t in next(tgt_iter))
                out_t = model(xt, at, alpha)
                dom_logits = torch.cat([out["domain_logits"], out_t["domain_logits"]])
                dom_y = torch.cat([torch.zeros(len(xb)), torch.ones(len(xt))]).long().to(dev)
                loss = loss + lam_d * F.cross_entropy(dom_logits, dom_y)
                if lam_p:                                      # physics terms need no labels -> also on target
                    loss = loss + lam_p * physics_loss(out_t, pt, classes, dl["w_align"], dl["w_consistency"])
            opt.zero_grad()
            loss.backward()
            opt.step()
        m = evaluate(model, ds_va, dev, labels)
        score = m["f1_macro"] if m else 0.0
        if verbose:
            print(f"epoch {ep + 1:3d}/{epochs} loss {loss.item():.4f} val macro-F1 {score:.3f}")
        if score > best:
            best, bad, best_state = score, 0, copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= dl["patience"]:
                break
    model.load_state_dict(best_state)
    res = {"kind": kind, "source_val_f1": best, "source_test": evaluate(model, ds_te, dev, labels),
           "target": evaluate(model, tgt_test, dev, labels) if tgt_test is not None else None,
           "epochs_run": ep + 1, "train_seconds": round(time.time() - t0, 1), "vib_only": vib_only}
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = Path(out_dir) / f"{kind}.pt"
    torch.save({"state_dict": model.state_dict(), "kind": kind, "n_ch": n_ch, "classes": classes,
                "stats": {k: np.asarray(v).tolist() for k, v in st.items()}, "vib_only": vib_only}, path)
    (Path(out_dir) / f"{kind}_metrics.json").write_text(json.dumps(res, indent=2, default=str))
    registry.register(f"dl-{kind}", "dl", Path(source).stem, None, {"lambda_physics": lam_p, "lambda_domain": lam_d},
                      best, (res["source_test"] or {}).get("f1_macro"), path)
    if verbose:
        print(f"source test macro-F1: {(res['source_test'] or {}).get('f1_macro')}", "| target macro-F1:", (res["target"] or {}).get("f1_macro"))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=KINDS, default="multimodal")
    ap.add_argument("--source", default="data/processed/synthetic_windows.npz")
    ap.add_argument("--target", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--vib-only", action="store_true", help="recommended for domain adaptation across datasets without current")
    a = ap.parse_args()
    train(a.model, a.source, a.target, a.epochs, a.vib_only)


if __name__ == "__main__":
    main()
