"""The project's primary metric: recall on Crystals at a fixed minimum precision.

See docs/primary_metric.md for why P_MIN=0.30 was chosen (from the classical baseline's
validation PR curve, before any model's test-set evaluation).
"""

import numpy as np
from sklearn.metrics import precision_recall_curve

P_MIN = 0.30


def recall_at_precision(y_true_binary: np.ndarray, scores: np.ndarray, p_min: float = P_MIN) -> float:
    """Recall at the first point on the PR curve where precision >= p_min, or 0.0 if that
    precision is never reached."""
    precision, recall, _ = precision_recall_curve(y_true_binary, scores)
    achievable = recall[precision >= p_min]
    return float(achievable.max()) if len(achievable) else 0.0
