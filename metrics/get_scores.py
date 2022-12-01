from typing import Tuple
import torch

from metrics.pred_interval import coverage, mean_pi_width
from metrics.calibration import ence, coeff_variation, corr, spearman_corr
from metrics.mse import mse_samples, mse_each_samp, rmse_each_samp


def get_scores(pred: torch.Tensor, pred_interval: torch.Tensor, device: str) -> torch.Tensor:

    # tensor (#metrics, H, W, Ch) containing all the metrics across the sample dimension
    return (
        torch.stack((
            torch.mean(pred[:, 0, ...], dim=0), # mean ground truth
            torch.mean(pred[:, 1, ...], dim=0), # mean predictions
            torch.mean(pred[:, 2, ...], dim=0), # mean uncertainty
            mse_samples(pred[:, :2, ...]), # mean MSE error

            torch.std(pred[:, 0, ...], dim=0), # std ground truth
            torch.std(pred[:, 1, ...], dim=0), # std predictions
            torch.std(pred[:, 2, ...], dim=0), # std uncertainty
            torch.std(mse_each_samp(pred[:, :2, ...]), dim=0), # std MSE error

            mean_pi_width(pred_interval), # mean PI width
            coverage(torch.cat((pred[:, 0, ...].unsqueeze(dim=1), pred_interval), dim=1)), # empirical PI coverage
            ence(pred), # ENCE
            coeff_variation(pred[:, 2, ...]), # coeff. of variation uncertainty
            corr(torch.stack((rmse_each_samp(pred[:, :2, ...]), pred[:, 2, ...]), dim=1)), # corr RMSE-uncertainty
            spearman_corr(rmse_each_samp(pred[:, :2, ...]), pred[:, 2, ...], device) # spearman corr RMSE-uncertainty
            ), dim=0)
        )


def get_score_names() -> str:
    return "[mean_gt, mean_pred, mean_unc, mean_mse, std_gt, std_pred, std_unc, std_mse, PI_width, cover, ENCE, CoV, corr, sp_corr]"


def get_scalar_scores(scores: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

    # taking means over (H, W, Ch) to return (#metrics) scalars

    # return (
    #     torch.cat(( # Speed channels
    #         torch.mean(scores[:12, :, :, [1, 3, 5, 7]], dim=(1,2,3)),
    #         torch.Tensor([np.nanmean(scores[12, :, :, [1, 3, 5, 7]].cpu().numpy())]).to(device)
    #         ), dim=0),
    #     torch.cat(( # Volume channels
    #         torch.mean(scores[:12, :, :, [0, 2, 4, 6]], dim=(1,2,3)),
    #         torch.Tensor([np.nanmean(scores[12, :, :, [0, 2, 4, 6]].cpu().numpy())]).to(device)
    #         ), dim=0)
    #     )

    return (torch.mean(scores[..., [1, 3, 5, 7]], dim=(1,2,3)), # speed
            torch.mean(scores[..., [0, 2, 4, 6]], dim=(1,2,3))) # vol
