"""Selected extension: replace hard training targets with uniform-smoothed targets.

q_c = (1 - epsilon) * 1[c == y] + epsilon / C
loss = -sum_c q_c * log_softmax(logits)_c, averaged over slides.
No model layers or validation/test targets are changed.
"""
import math
from torch import nn

METHODS = ('baseline', 'label_smoothing')

def make_training_loss(method, epsilon=0.1):
    if method not in METHODS:
        raise ValueError(f'Unknown method: {method}; expected {METHODS}')
    epsilon = float(epsilon)
    if not math.isfinite(epsilon) or not 0 <= epsilon <= 1:
        raise ValueError('epsilon must be finite and between 0 and 1.')
    return nn.CrossEntropyLoss(label_smoothing=0.0 if method == 'baseline' else epsilon)

def make_evaluation_loss():
    """Always use ordinary CE/NLL for comparable validation and test losses."""
    return nn.CrossEntropyLoss()
