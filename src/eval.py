"""
Shared evaluation utilities. Everyone uses these so that all reported scores
are computed the same way.

DO NOT define alternative QWK functions in your own modules — import from here.
"""
import numpy as np
from sklearn.metrics import cohen_kappa_score


def _sigmoid(x):
    """Numerically stable sigmoid on numpy arrays."""
    x = np.asarray(x, dtype=np.float32)
    out = np.empty_like(x)
    pos = x >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[neg])
    out[neg] = exp_x / (1.0 + exp_x)
    return out


def outputs_to_pred_values(preds, num_classes=6, ordinal_mode='threshold'):
    """Normalize model outputs into one prediction value per slide.

    Supported inputs:
      - regression outputs: shape [N] or [N, 1]
      - ordinal logits: shape [N, num_classes - 1]

    For ordinal models, ``ordinal_mode='threshold'`` matches the training loop:
    convert the K-1 logits to binary threshold decisions and count how many are
    positive. ``ordinal_mode='expected'`` instead sums the threshold
    probabilities into a softer continuous grade for ensembling experiments.
    """
    preds = np.asarray(preds)
    if preds.ndim == 0:
        return preds.reshape(1).astype(np.float32)
    if preds.ndim == 1:
        return preds.astype(np.float32)
    if preds.ndim == 2:
        if preds.shape[1] == 1:
            return preds[:, 0].astype(np.float32)
        if preds.shape[1] == num_classes - 1:
            probs = _sigmoid(preds)
            if ordinal_mode == 'threshold':
                return (probs > 0.5).sum(axis=1).astype(np.float32)
            if ordinal_mode == 'expected':
                return probs.sum(axis=1).astype(np.float32)
            raise ValueError(
                f"Unknown ordinal_mode={ordinal_mode!r}; expected 'threshold' or 'expected'"
            )
    raise ValueError(
        'Unsupported prediction shape '
        f'{tuple(preds.shape)}; expected [N], [N, 1], or [N, {num_classes - 1}]'
    )


def round_preds(preds, num_classes=6, ordinal_mode='threshold'):
    """Round model outputs to integer class labels.

    PANDA ISUP grade is in {0, 1, 2, 3, 4, 5}. We train as a regression problem
    on these as floats, then round at evaluation. Ordinal models emit 5 logits,
    which are decoded into slide-level grade predictions before rounding.
    Clipping prevents predictions outside the valid range from being assigned an
    invalid class.
    """
    pred_values = outputs_to_pred_values(
        preds,
        num_classes=num_classes,
        ordinal_mode=ordinal_mode,
    )
    rounded = np.round(pred_values)
    clipped = np.clip(rounded, 0, num_classes - 1)
    return clipped.astype(int)


def qwk(preds, targets, num_classes=6, ordinal_mode='threshold'):
    """Quadratic-weighted Cohen's kappa, the metric used by the PANDA challenge.

    preds: regression outputs, ordinal logits, or already-decoded predictions.
    targets: integer ground-truth ISUP grades in {0, ..., num_classes-1}.
    """
    rounded = round_preds(preds, num_classes, ordinal_mode=ordinal_mode)
    targets = np.asarray(targets).astype(int)
    return cohen_kappa_score(rounded, targets, weights='quadratic')


def summarize_scores(scores):
    """Return mean and std for a list/array of fold scores."""
    scores = np.asarray(scores, dtype=np.float32)
    if scores.ndim != 1 or len(scores) == 0:
        raise ValueError('scores must be a non-empty 1-D array')
    return float(scores.mean()), float(scores.std())


def mean_std_str(scores, precision=4):
    """Pretty-print the mean ± std summary used in results.md."""
    mean, std = summarize_scores(scores)
    return f'{mean:.{precision}f} ± {std:.{precision}f}'


def confusion_matrix_str(preds, targets, num_classes=6, ordinal_mode='threshold'):
    """Pretty-print a confusion matrix. Useful for debugging which classes are
    being confused.
    """
    from sklearn.metrics import confusion_matrix
    rounded = round_preds(preds, num_classes, ordinal_mode=ordinal_mode)
    cm = confusion_matrix(targets, rounded, labels=list(range(num_classes)))
    header = '    ' + ' '.join(f'p{i:>4}' for i in range(num_classes))
    lines = [header]
    for i, row in enumerate(cm):
        lines.append(f't{i:>2}  ' + ' '.join(f'{v:>4}' for v in row))
    return '\n'.join(lines)
