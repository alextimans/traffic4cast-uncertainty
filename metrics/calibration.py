"""
Uncertainty prediction calibration for regression calculations, 
partially following the approach and metrics presented in Levi et al. 2020, section 3.
"""

import torch
import numpy as np


def corr(pred):

    """
    Receives: prediction tensor (samples, 2, 6, H, W, Ch), where 2nd dim '2'
    is error metric (0), uncertainty measure (1).
    Returns: Pearson correlation coefficient across the sample dimension as
    (6, H, W, Ch).
    Formula: sum[(x - x_mean)(y - y_mean)] * 1/sqrt(sum((x - x_mean)^2)) * 1/sqrt(sum((y - y_mean)^2))
    """

    return (torch.sum(
                (pred[:, 0, ...] - torch.mean(pred[:, 0, ...], dim=0)) * (pred[:, 1, ...] - torch.mean(pred[:, 1, ...], dim=0)), dim=0
                ) * torch.rsqrt(
                    torch.sum((pred[:, 0, ...] - torch.mean(pred[:, 0, ...], dim=0))**2, dim=0).clamp(min=1e-10)
                ) * torch.rsqrt(
                    torch.sum((pred[:, 1, ...] - torch.mean(pred[:, 1, ...], dim=0))**2, dim=0).clamp(min=1e-10)
                )
            )


def spearman_corr(error, unc, device):

    """
    Receives: error tensor (samples, H, W, Ch) containing error values,
    unc tensor (samples, H, W, Ch) containing uncertainty values, device.
    Returns: Spearman correlation coefficient across the sample dimension
    as tensor (H, W, Ch).
    Formula: obtain rank values across sample dimension, then compute Pearson corr.

    Note: Very inefficient code for current lack of a better solution.
    torch.argsort only accepts sorting across single dimensions.
    """

    assert error.shape[1:] == unc.shape[1:]
    s = tuple(error.shape)

    error_rank = torch.empty(size=s, dtype=torch.int, device=device)
    unc_rank = torch.empty(size=s, dtype=torch.int, device=device)

    for i in range(s[1]):
        for j in range(s[2]):
            for c in range(s[3]):

                error_tmp = error[:, i, j, c].argsort()
                error_rank[error_tmp, i, j, c] = torch.arange(s[0]).int().to(device)

                unc_tmp = unc[:, i, j, c].argsort()
                unc_rank[unc_tmp, i, j, c] = torch.arange(s[0]).int().to(device)

    return corr(torch.stack((error_rank.float(), unc_rank.float()), dim=1))


def ence(pred):

   """
   Receives: prediction tensor (samples, 3, 6, H, W, Ch), where 2nd dim
   '3' is y_true (0), point prediction (1), uncertainty measure (2).
   Returns: Expected normalized calibration error (ENCE) across the 
   sample dimension as tensor (6, H, W, Ch).
   Prediction tensor is expected to be float32 as required by torch.pow.

   Uncertainty measure is assumed to be a standard deviation.
   Every sample is treated as its own individual "bin".
   Then ENCE = mean(|std - rse| / std), mean over samples.
   """

   # clamp value max to 99% quantile to avoid outlier distortions due to small uncertainties
   # works with np.quantile, not with torch.quantile because "tensor too large"
   max_clamp = torch.tensor(np.quantile((torch.abs(pred[:, 2, ...] - torch.sqrt((pred[:, 0, ...] - pred[:, 1, ...])**2)) / pred[:, 2, ...]).cpu().numpy()
                           , 0.99), dtype=torch.float32)

   return torch.mean(
       (torch.abs(pred[:, 2, ...] - torch.sqrt((pred[:, 0, ...] - pred[:, 1, ...])**2)) / pred[:, 2, ...]
        ).clamp(max=max_clamp), dim=0)


def coeff_variation(pred):

    """
    Receives: prediction tensor (samples, 6, H, W, Ch) where the values
    represent the predicted uncertainty in form of e.g. a standard deviation.
    Returns: Predicted uncertainty coefficient of variation across the sample 
    dimension as tensor (6, H, W, Ch) with values in [0, 1].
    """

    return torch.std(pred, dim=0, unbiased=True) / torch.mean(pred, dim=0)


def get_rmv_rmse(pred, bins: int = 10):

    """
    Receives: prediction tensor (samples, 3), where 2nd dim
    '3' is y_true (0), point prediction (1), uncertainty measure (2);
    and a number of desired bins that divides the number of samples cleanly.
    Returns: tensor (bins, 2) where 2nd dim '2' is RMSE per bin (0), RMV per bin (1).
    Prediction tensor is expected to be float32 as required by torch.pow.
    
    These can then be used for plotting 2D calibration plots (reliability diagrams)
    for values across the sample dimension, e.g. for a fixed pred horizon, pixel & channel.
    """

    assert (pred.shape[0] / bins) % 1 == 0, "Select bins s.t. it divides #samples cleanly."

    bin_val = torch.empty(size=(bins, 2), dtype=torch.float32)
    samp_per_bin = int(pred.shape[0] / bins)
    sort_idx = torch.sort(pred[:, 2], descending=False)[1]

    for cbin in range(bins):
        idx = sort_idx[(cbin * samp_per_bin):(cbin * samp_per_bin + samp_per_bin)]
        bin_val[cbin, 0] = torch.sqrt(torch.mean((pred[idx, 0] - pred[idx, 1])**2)) # RMSE
        bin_val[cbin, 1] = torch.sqrt(torch.mean(pred[idx, 2]**2)) # RMV

    return bin_val


# =============================================================================
# from scipy.stats import pearsonr
# def corr2(pred): # inputted (samples, 2, H, W, Ch)
# 
#     per_cell_calib = torch.empty(tuple(pred.shape[2:]))
#     for i in range(per_cell_calib.shape[0]):
#         for j in range(per_cell_calib.shape[1]):
#             for c in range(per_cell_calib.shape[2]):
#                 per_cell_calib[i, j, c], _ = pearsonr(pred[:, 0, i, j, c], pred[:, 1, i, j, c])
# 
#     return per_cell_calib
# 
# from scipy.stats import spearmanr
# def corr2(pred): # inputted (samples, 2, H, W, Ch)
# 
#     per_cell_calib = torch.empty(tuple(pred.shape[2:]))
#     for i in range(per_cell_calib.shape[0]):
#         for j in range(per_cell_calib.shape[1]):
#             for c in range(per_cell_calib.shape[2]):
#                 per_cell_calib[i, j, c], _ = spearmanr(pred[:, 0, i, j, c], pred[:, 1, i, j, c])#, nan_policy="omit")
# 
#     return per_cell_calib
# 
# =============================================================================
