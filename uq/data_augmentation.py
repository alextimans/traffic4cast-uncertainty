from functools import partial

import torch
import torchvision.transforms as tf
import torchvision.transforms.functional as TF

from typing import Tuple
from tqdm import tqdm


class DataAugmentation:
    def __init__(self):
        self.transformations = [
            TF.vflip,
            TF.hflip,
            partial(TF.rotate, angle=90, expand=True),
            partial(TF.rotate, angle=180, expand=True),
            partial(TF.rotate, angle=270, expand=True),
            tf.Compose([TF.vflip, partial(TF.rotate, angle=90, expand=True)]),
            tf.Compose([TF.vflip, partial(TF.rotate, angle=-90, expand=True)])
            ]

        self.detransformations = [
            TF.vflip,
            TF.hflip,
            partial(TF.rotate, angle=-90, expand=True),
            partial(TF.rotate, angle=-180, expand=True),
            partial(TF.rotate, angle=-270, expand=True),
            tf.Compose([partial(TF.rotate, angle=-90, expand=True), TF.vflip]),
            tf.Compose([partial(TF.rotate, angle=90, expand=True), TF.vflip])
            ]

        self.nr_augments = len(self.transformations)

    def transform(self, data: torch.Tensor) -> torch.Tensor:

        """
        Receives X = (1, 12 * Ch, H, W) and does k augmentations 
        returning X' = (1+k, 12 * Ch, H, W).
        """

        X = data
        for transform in self.transformations:
            X_aug = transform(data)
            X = torch.cat((X, X_aug), dim=0)

        assert list(X.shape) == [1+self.nr_augments] + list(data.shape[1:])

        return X

    def detransform(self, data: torch.Tensor) -> torch.Tensor:

        """
        Receives y_pred = (1+k, 6 * Ch, H, W), detransforms the 
        k augmentations and returns y_pred = (1+k, 6 * Ch, H, W).
        """

        y = data[0, ...].unsqueeze(dim=0)
        for i, detransform in enumerate(self.detransformations):
            y_deaug = detransform(data[i+1, ...].unsqueeze(dim=0))
            y = torch.cat((y, y_deaug), dim=0)

        assert y.shape == data.shape

        return y

    def aggregate(self, pred: torch.Tensor) -> torch.Tensor:

        """
        Receives: prediction tensor (1+k, 6 * Ch, H, W) and
        computes the aleatoric uncertainty obtained via test-time augmentation.
        Returns: tensor (2, 6 * Ch, H, W) where 1st dimension is point prediction (0),
        uncertainty measure (1).
        """

        # Aleatoric uncertainty estimation: std over original & augmented imgs; ensure uncertainty >0 for numerical reasons
        return torch.stack((pred[0, ...], torch.std(pred[1:, ...], dim=0, unbiased=False).clamp(min=1e-4)), dim=0)

    @torch.no_grad()
    def __call__(self, device, loss_fct, dataloader, model, samp_limit,
                 parallel_use, post_transform) -> Tuple[torch.Tensor, float]:

        model.eval()
        loss_sum = 0
        bsize = dataloader.batch_size
        batch_limit = samp_limit // bsize
        pred = torch.empty( # Pred contains y_true + pred original img + uncertainty: (samples, 3, H, W, Ch)
            size=(batch_limit * bsize, 3, 495, 436, 8), dtype=torch.float32, device=device)
    
        # Use only batch_size = 1 for dataloader since augmentations are interpreted as a batch
        with tqdm(dataloader) as tloader:
            for batch, (X, y) in enumerate(tloader):
                if batch == batch_limit:
                    break
    
                # X, y = X.to(device, non_blocking=parallel_use) / 255, y.to(device, non_blocking=parallel_use)
                X, y = X.to(device, non_blocking=parallel_use), y.to(device, non_blocking=parallel_use) # for UNet++
                X = self.transform(X) # (1+k, 12 * Ch, H+pad, W+pad) in [0, 1]
    
                y_pred = model(X) # (1+k, 6 * Ch, H+pad, W+pad) NOT in [0, 255]?
                loss = loss_fct(y_pred[0, :, 1:, 30:-30], y[:, :, 1:, 30:-30].squeeze(dim=0)) # For original img & unpadded
    
                y_pred[...] = self.detransform(y_pred) # (1+k, 6 * Ch, H+pad, W+pad)
                y_pred = self.aggregate(y_pred) # (2, 6 * Ch, H+pad, W+pad)
                y_pred = post_transform(torch.cat((y, y_pred), dim=0))[:, 5, ...].clamp(0, 255).unsqueeze(dim=0) # (1, 3, H, W, Ch), only consider pred horizon 1h
                # logging.info(f"{y_pred.shape, torch.min(y_pred), torch.max(y_pred)}")
    
                loss_sum += float(loss.item())
                loss_test = float(loss_sum/(batch+1))
                tloader.set_description(f"Batch {batch+1}/{batch_limit} > eval")
                tloader.set_postfix(loss = loss_test)
    
                assert pred[(batch * bsize):(batch * bsize + bsize)].shape == y_pred.shape
                pred[(batch * bsize):(batch * bsize + bsize)] = y_pred # Fill slice
                del X, y, y_pred
    
        return pred, loss_test
