import torch
from typing import Tuple
from tqdm import tqdm


class PointPred:
    def __init__(self):
        pass

    def constant_uncertainty(self, pred):
        # Std over cell predictions for test samples
        return torch.repeat_interleave(
            torch.std(pred[:, 1, ...], dim=0, unbiased=False
                      ).unsqueeze(dim=0), pred.shape[0], dim=0)
        # 0.5 * abs (pred_max - pred_min) per cell
        # return torch.repeat_interleave(
        #     (0.5 * torch.abs(torch.max(pred[:, 1, ...], dim=0)[0] - torch.min(pred[:, 1, ...], dim=0)[0])
        #      ).unsqueeze(dim=0), pred.shape[0], dim=0)

    @torch.no_grad()
    def __call__(self, device, loss_fct, dataloader, model, samp_limit,
                 parallel_use, post_transform) -> Tuple[torch.Tensor, float]:

        model.eval()
        loss_sum = 0
        bsize = dataloader.batch_size
        batch_limit = samp_limit // bsize
        pred = torch.empty(  # Pred contains y_true + point pred + uncertainty: (samples, 3, H, W, Ch)
            size=(batch_limit * bsize, 3, 495, 436, 8), dtype=torch.float32, device=device)

        with tqdm(dataloader) as tloader:
            for batch, (X, y) in enumerate(tloader):
                if batch == batch_limit:
                    break

                # X, y = X.to(device, non_blocking=parallel_use) / 255, y.to(device, non_blocking=parallel_use)
                X, y = X.to(device, non_blocking=parallel_use), y.to(
                    device, non_blocking=parallel_use)  # For UNet++

                y_pred = model(X)  # (1, 6 * Ch, H+pad, W+pad)
                loss = loss_fct(y_pred[:, :, 1:, 6:-6], y[:, :, 1:, 6:-6])
                y_pred = post_transform(torch.cat((y, y_pred, torch.zeros_like(y_pred)), dim=0)
                                        )[:, 5, ...].clamp(0, 255).unsqueeze(dim=0)  # (1, 3, H, W, Ch), only consider pred horizon 1h

                loss_sum += float(loss.item())
                loss_test = float(loss_sum/(batch+1))
                tloader.set_description(
                    f"Batch {batch+1}/{batch_limit} > eval")
                tloader.set_postfix(loss=loss_test)

                assert pred[(batch * bsize):(batch * bsize + bsize)
                            ].shape == y_pred.shape
                pred[(batch * bsize):(batch * bsize + bsize)
                     ] = y_pred  # Fill slice
                del X, y, y_pred

        # Constant uncertainty baseline per cell (pixel + channel)
        pred[:, 2, ...] = self.constant_uncertainty(pred).clamp(min=1e-4)

        return pred, loss_test
