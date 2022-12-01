import torch
from torch.nn.functional import mse_loss


def mse(pred):

   """
   Receives: prediction tensor (samples, 2, 6, H, W, Ch), where 2nd dim
   '2' is y_true (0) and some prediction (1) (e.g. only point or point + uncertainty).
   Returns: MSE over all dimensions.
   """

   return mse_loss(pred[:, 1, ...], target=pred[:, 0, ...], reduction="mean")


def mse_samples(pred):

    """
    Receives: prediction tensor (samples, 2, 6, H, W, Ch), where 2nd dim
    '2' is y_true (0) and some prediction (1) (e.g. only point or point + uncertainty).
    Returns: MSE over the sample dimension as tensor (6, H, W, Ch).
    Prediction tensor is expected to be float32 as required by torch.pow.
    """

    return torch.mean((pred[:, 0, ...] - pred[:, 1, ...])**2, dim=0)


def mse_each_samp(pred):

    """
    Receives: prediction tensor (samples, 2, 6, H, W, Ch), where 2nd dim
    '2' is y_true (0) and some prediction (1) (e.g. only point or point + uncertainty).
    Returns: MSE for each sample as tensor (samples, 6, H, W, Ch).
    Prediction tensor is expected to be float32 as required by torch.pow.
    """

    return (pred[:, 0, ...] - pred[:, 1, ...])**2


def rmse_each_samp(pred):

    """
    Receives: prediction tensor (samples, 2, 6, H, W, Ch), where 2nd dim
    '2' is y_true (0) and some prediction (1) (e.g. only point or point + uncertainty).
    Returns: RMSE for each sample as tensor (samples, 6, H, W, Ch).
    Prediction tensor is expected to be float32 as required by torch.pow.
    """

    return torch.sqrt((pred[:, 0, ...] - pred[:, 1, ...])**2)
