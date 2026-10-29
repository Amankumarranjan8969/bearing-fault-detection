"""1-D Grad-CAM and saliency for the CNN models (needs PyTorch)."""
import torch
import torch.nn.functional as F


def _default_layer(model):
    root = None
    for path in ("backbone.vib_enc", "vib_enc", "encoder"):
        obj = model
        try:
            for p in path.split("."):
                obj = getattr(obj, p)
            root = obj
            break
        except AttributeError:
            continue
    convs = [m for m in (root or model).modules() if isinstance(m, torch.nn.Conv1d)]
    return convs[-1]


def gradcam_1d(model, x, aux=None, target_class=None, layer=None):
    """x: (B, C, L). Returns (cam (B, L) in [0, 1], predicted class (B,)). High values = influential samples."""
    model.eval()
    layer = layer or _default_layer(model)
    store = {}

    def fwd(_, __, out):
        store["a"] = out
        out.register_hook(lambda g: store.__setitem__("g", g))

    h = layer.register_forward_hook(fwd)
    try:
        logits = model(x, aux)["logits"]
        cls = logits.argmax(1) if target_class is None else torch.full((len(x),), target_class, device=x.device)
        model.zero_grad()
        logits.gather(1, cls[:, None]).sum().backward()
    finally:
        h.remove()
    w = store["g"].mean(dim=2, keepdim=True)
    cam = F.relu((w * store["a"]).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-1], mode="linear", align_corners=False)[:, 0]
    cam = cam - cam.amin(1, keepdim=True)
    cam = cam / (cam.amax(1, keepdim=True) + 1e-9)
    return cam.detach().cpu(), cls.detach().cpu()


def saliency_1d(model, x, aux=None, target_class=None):
    model.eval()
    x = x.clone().requires_grad_(True)
    logits = model(x, aux)["logits"]
    cls = logits.argmax(1) if target_class is None else torch.full((len(x),), target_class, device=x.device)
    logits.gather(1, cls[:, None]).sum().backward()
    return x.grad.abs().detach().cpu()
