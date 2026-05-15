"""
Inference helpers. Used by the external-validation notebook and (eventually)
any final inference pipeline.

PERSON C: extend this with ensembling and TTA.
"""
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.eval import qwk, round_preds


def predict(model, dataset, device, batch_size=16, num_workers=0):
    """Run a single model on a Dataset and return raw (un-rounded) predictions
    as a 1-D float array, in the same order as the Dataset."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in loader:
            xb = batch[0].to(device, non_blocking=True)
            out = model(xb).cpu().numpy()
            preds.append(out)
    return np.concatenate(preds)


def ensemble_predict(models, dataset, device, batch_size=16):
    """Average predictions across N trained models. Same model architecture
    expected for all of them."""
    all_preds = [predict(m, dataset, device, batch_size) for m in models]
    return np.mean(all_preds, axis=0)


def score(preds, targets):
    """Wrap qwk for convenience — round, clip, compute kappa."""
    return qwk(preds, targets)
