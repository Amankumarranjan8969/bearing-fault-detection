"""Needs PyTorch. 2-epoch smoke run of every model kind on real CWRU & Paderborn windows."""
import pytest

torch = pytest.importorskip("torch")


def test_all_models_train_two_epochs(tmp_path):
    from deep_learning.train_dl import train
    src = "data/processed/cwru_windows.npz"
    tgt = "data/processed/paderborn_windows.npz"
    for kind in ("cnn", "cnn_gru", "multimodal", "pi"):
        assert train(kind, src, None, epochs=2, out_dir=str(tmp_path), verbose=False)["source_test"] is not None
    for kind in ("da", "pi_da"):
        assert train(kind, src, tgt, epochs=2, vib_only=True, out_dir=str(tmp_path), verbose=False)["target"] is not None
