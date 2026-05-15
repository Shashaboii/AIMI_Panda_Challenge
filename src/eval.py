"""
Shared evaluation utilities. Everyone uses these so that all reported scores
are computed the same way.

DO NOT define alternative QWK functions in your own modules — import from here.
"""
import numpy as np
from sklearn.metrics import cohen_kappa_score


def round_preds(preds, num_classes=6):
    """Round continuous regression outputs to integer class labels.

    PANDA ISUP grade is in {0, 1, 2, 3, 4, 5}. We train as a regression problem
    on these as floats, then round at evaluation. Clipping prevents predictions
    outside the valid range from being assigned an invalid class.
    """
    preds = np.asarray(preds)
    rounded = np.round(preds)
    clipped = np.clip(rounded, 0, num_classes - 1)
    return clipped.astype(int)


def qwk(preds, targets, num_classes=6):
    """Quadratic-weighted Cohen's kappa, the metric used by the PANDA challenge.

    preds: continuous regression outputs (will be rounded), or already-integer
           predictions (rounding is a no-op for integers).
    targets: integer ground-truth ISUP grades in {0, ..., num_classes-1}.
    """
    rounded = round_preds(preds, num_classes)
    targets = np.asarray(targets).astype(int)
    return cohen_kappa_score(rounded, targets, weights='quadratic')


def confusion_matrix_str(preds, targets, num_classes=6):
    """Pretty-print a confusion matrix. Useful for debugging which classes are
    being confused.
    """
    from sklearn.metrics import confusion_matrix
    rounded = round_preds(preds, num_classes)
    cm = confusion_matrix(targets, rounded, labels=list(range(num_classes)))
    header = '    ' + ' '.join(f'p{i:>4}' for i in range(num_classes))
    lines = [header]
    for i, row in enumerate(cm):
        lines.append(f't{i:>2}  ' + ' '.join(f'{v:>4}' for v in row))
    return '\n'.join(lines)
