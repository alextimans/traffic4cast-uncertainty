"""
Prediction interval calculations following the conformal prediction framework
as presented in Angelopoulos & Bates 2022, section 2.3.2.
"""

import torch
import numpy as np


def get_quantile(pred, n: int = None, alpha: float = 0.1):

    """
    Receives: prediction tensor (samples, 3, 6, H, W, Ch), where 2nd dim
    '3' is y_true (0), point prediction (1), uncertainty measure (2).
    Returns: the conformal prediction score function quantile across
    the sample dimension as tensor (6, H, W, Ch).

    Prediction tensor should contain all predictions for the calibration set
    on a single city (may be large in the first dim depending on calibration set size).
    Prediction tensor is expected to be float32 as required by torch.quantile.

    n: int
        Size of the calibration set.
    alpha: float
        Desired coverage of prediction interval is 1 - alpha, thus governing quantile selection.
    """

    n = n if n is not None else pred.shape[0]
    quant = np.ceil((1 - alpha) * (n + 1)) / n

    # clamp value max to 99% quantile to avoid outlier distortions due to small uncertainties
    # works with np.quantile, not with torch.quantile because "tensor too large"
    max_clamp = torch.tensor(np.quantile((torch.abs(pred[:, 0, ...] - pred[:, 1, ...]) / pred[:, 2, ...]).cpu().numpy()
                            , 0.99), dtype=torch.float32)

    return torch.quantile((torch.abs(pred[:, 0, ...] - pred[:, 1, ...]) / pred[:, 2, ...]
                           ).clamp(max=max_clamp), quant, dim=0)


def get_pred_interval(pred, quantiles):

    """
    Receives: prediction tensor (samples, 2, 6, H, W, Ch), where 2nd dim
    '2' is point prediction (0), uncertainty measure (1);
    quantile tensor (6, H, W, Ch) with calibration set quantiles.
    Returns: prediction interval tensor (samples, 2, 6, H, W, Ch), where 2nd dim
    '2' is interval lower bound (0), interval upper bound (1).
    The prediction intervals returned are symmetric about the prediction.
    
    Prediction tensor should contain the predictions for the test set on 
    a single city that matches the city for which the quantiles were computed.

    Note: Interval values are not clamped to [0, 255] and thus may exceed uint8 limits.
          Clamping will influence mean PI width metric if performed prior to evaluation.
    """

    return torch.stack(((pred[:, 0, ...] - pred[:, 1, ...] * quantiles),
                        (pred[:, 0, ...] + pred[:, 1, ...] * quantiles)), dim=1)


def coverage(pred):

    """
    Receives: prediction interval tensor (samples, 3, 6, H, W, Ch), where 2nd dim
    '3' is y_true(0), interval lower bound (1), interval upper bound (2).
    Returns: empirical coverage as a fraction across the sample dimension
    as tensor (6, H, W, Ch) with values in [0, 1].
    """

    bool_mask = torch.stack(((pred[:, 0, ...] >= pred[:, 1, ...]),
                             (pred[:, 0, ...] <= pred[:, 2, ...])), dim=1)

    bool_mask = torch.ones_like(bool_mask[:, 0, ...]) * (torch.sum(bool_mask, dim=1) > 1)

    return torch.sum(bool_mask, dim=0) / bool_mask.shape[0] # torch.float32


def mean_pi_width(pred):

    """
    Receives: prediction interval tensor (samples, 2, 6, H, W, Ch), where 2nd dim
    '2' is interval lower bound (0), interval upper bound (1).
    Returns: prediction interval width mean across the sample dimension as
    tensor (6, H, W, Ch).
    """

    return torch.mean(torch.abs(pred[:, 1, ...] - pred[:, 0, ...]), dim=0)
