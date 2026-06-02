"""
Inference helpers. Used by the external-validation notebook and (eventually)
any final inference pipeline.

PERSON C: extend this with ensembling and TTA.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.dataset import PandaDataset
from src.eval import qwk, outputs_to_pred_values
from src.model import ConcatTilePoolingModel, EfficientNetBaseline


def _pin_memory_for(device):
    return str(device).startswith('cuda')


def infer_out_dim(state_dict):
    """Infer whether saved weights are regression (1) or ordinal (5)."""
    for key, value in state_dict.items():
        if (
            key == 'head.1.weight'
            or key.endswith('.head.1.weight')
            or key == 'head.weight'
            or key.endswith('.head.weight')
        ):
            return int(value.shape[0])
    raise KeyError('Could not infer output dimension from saved weights')


def load_model(weight_path, backbone='efficientnet-b0', device='cpu',
               model_kind='baseline'):
    """Instantiate a model and load saved weights from disk.

    ``model_kind`` is ``baseline`` for single-image models and ``tiles`` for
    concat-tile-pooling models.
    """
    state = torch.load(weight_path, map_location=device)
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    out_dim = infer_out_dim(state)
    model_cls = EfficientNetBaseline if model_kind == 'baseline' else ConcatTilePoolingModel
    model = model_cls(backbone, pretrained=False, out_dim=out_dim).to(device).eval()
    model.load_state_dict(state)
    return model


def predict_loader(model, loader, device, ordinal_mode='threshold'):
    """Run a model on a pre-built loader and return one float per slide."""
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in loader:
            xb = batch[0].to(device, non_blocking=True)
            out = model(xb).cpu().numpy()
            preds.append(outputs_to_pred_values(out, ordinal_mode=ordinal_mode))
    return np.concatenate(preds).astype(np.float32)


def predict(model, dataset, device, batch_size=16, num_workers=0,
            ordinal_mode='threshold'):
    """Run a single model on a Dataset and return one prediction per slide."""
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=_pin_memory_for(device),
    )
    return predict_loader(model, loader, device, ordinal_mode=ordinal_mode)


def predict_oof(model, df, image_dir, device, batch_size=16, num_workers=0,
                dataset_cls=PandaDataset, ordinal_mode='threshold'):
    """Predict one fold's held-out rows in dataframe order."""
    dataset = dataset_cls(df, image_dir, train=False)
    return predict(
        model,
        dataset,
        device,
        batch_size=batch_size,
        num_workers=num_workers,
        ordinal_mode=ordinal_mode,
    )


def ensemble_predict(models, dataset, device, batch_size=16, num_workers=0,
                     ordinal_mode='threshold'):
    """Average predictions across N trained models."""
    all_preds = [
        predict(
            m,
            dataset,
            device,
            batch_size=batch_size,
            num_workers=num_workers,
            ordinal_mode=ordinal_mode,
        )
        for m in models
    ]
    return np.mean(all_preds, axis=0)


def score(preds, targets, ordinal_mode='threshold'):
    """Wrap qwk for convenience — round, clip, compute kappa."""
    return qwk(preds, targets, ordinal_mode=ordinal_mode)
