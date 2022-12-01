import os
import glob
import logging
from typing import Tuple

import torch
from tqdm import tqdm

from model.checkpointing import load_torch_model_from_checkpoint


class DeepEnsemble:
    def __init__(self, load_from_epoch: list):
        self.ensemble = []
        self.load_from_epoch = load_from_epoch

    def load_ensemble(self, device: str, save_checkpoint: str, model_class, model_config):
        for i, ep in enumerate(self.load_from_epoch):
            # checkpt = glob.glob(os.path.join(save_checkpoint, f"unet_{i+1}", f"unet_ep{ep}_*.pt"))[0]
            checkpt = glob.glob(os.path.join(save_checkpoint, f"unet_pp_{i+8}", f"unet_pp_ep{ep}_*.pt"))[0] # For UNet++
            model = model_class(**model_config)
            if device != "cpu":
                model = torch.nn.DataParallel(model)
            load_torch_model_from_checkpoint(checkpt_path=checkpt, model=model, map_location=device)
            self.ensemble.append(model)
        logging.info(f"Ensemble of size {len(self.ensemble)} loaded.")

    def aggregate(self, pred):

        """
        Receives: prediction tensor (ensemble_size, 6 * Ch, H, W) and
        computes the average prediction and epistemic uncertainty.
        Returns: tensor (2, 6 * Ch, H, W) where 1st dimension is mean point prediction (0),
        uncertainty measure (1).
        """

        # Epistemic uncertainty estimation: std over ensemble predictions; ensure uncertainty >0 for numerical reasons
        return torch.stack((torch.mean(pred, dim=0),
                            torch.std(pred, dim=0, unbiased=False).clamp(min=1e-4)), dim=0)

    @torch.no_grad()
    def __call__(self, device, loss_fct, dataloader, model, samp_limit,
                 parallel_use, post_transform) -> Tuple[torch.Tensor, float]:

        for member in self.ensemble:
            member.eval()
        loss_sum = 0
        bsize = dataloader.batch_size
        batch_limit = samp_limit // bsize
        pred = torch.empty( # Pred contains y_true + avg ensemble pred + uncertainty: (samples, 3, H, W, Ch)
            size=(batch_limit * bsize, 3, 495, 436, 8), dtype=torch.float32, device=device)

        with tqdm(dataloader) as tloader:
            for batch, (X, y) in enumerate(tloader):
                if batch == batch_limit:
                    break
          
                # X, y = X.to(device, non_blocking=parallel_use) / 255, y.to(device, non_blocking=parallel_use)
                X, y = X.to(device, non_blocking=parallel_use), y.to(device, non_blocking=parallel_use) # For UNet++
                y_pred = self.aggregate(torch.cat((self.ensemble[0](X), # (2, 6 * Ch, H+pad, W+pad)
                                                   self.ensemble[1](X),
                                                   self.ensemble[2](X),
                                                   self.ensemble[3](X),
                                                   self.ensemble[4](X)), dim=0))

                loss = loss_fct(y_pred[0, :, 1:, 6:-6], y[:, :, 1:, 6:-6].squeeze(dim=0))
                y_pred = post_transform(torch.cat((y, y_pred), dim=0))[:, 5, ...].clamp(0, 255).unsqueeze(dim=0) # (1, 3, H, W, Ch), only consider pred horizon 1h

                loss_sum += float(loss.item())
                loss_test = float(loss_sum/(batch+1))
                tloader.set_description(f"Batch {batch+1}/{batch_limit} > eval")
                tloader.set_postfix(loss = loss_test)
          
                assert pred[(batch * bsize):(batch * bsize + bsize)].shape == y_pred.shape
                pred[(batch * bsize):(batch * bsize + bsize)] = y_pred # Fill slice
                del X, y, y_pred
          
        return pred, loss_test
