"""
Monte Carlo batch normalization (MCBN) for epistemic uncertainty estimation.
"""

import random
import logging
from typing import Tuple

import torch
from torch.utils.data import Subset, DataLoader
from tqdm import tqdm

from data.dataset import T4CDataset


class StochasticBatchNorm:
    def __init__(self, passes: int, train_batch_size: int):
        self.passes = passes
        self.train_batch_size = train_batch_size
        logging.info(f"Init StochasticBatchNorm with {self.passes=} and {self.train_batch_size=}.")

    def load_train_data(self, data_raw_path: str, dataset_config: dict):
        self.data_train = T4CDataset(root_dir=data_raw_path,
                                     dataset_type="train",
                                     **dataset_config)
        self.len = len(self.data_train)

    def set_bn_mode(self, model, set_all: bool = False):
        if set_all:
            # Set stochastic behaviour for all BatchNorm layers
            for m in model.modules():
                if isinstance(m, torch.nn.modules.batchnorm.BatchNorm2d):
                    m.train()
                    m.track_running_stats = False
        else:
            # Set stochastic batch behaviour only for first 2 BatchNorm layers
            
            # MODIFY HERE FOR DIFFERENT MODELS
            # model.down_path[0].block[2].train()
            # model.down_path[0].block[2].track_running_stats = False
            # model.down_path[0].block[5].train()
            # model.down_path[0].block[5].track_running_stats = False
            
            # For UNet++
            model.conv0_0.bn1.train()
            model.conv0_0.bn1.track_running_stats = False
            model.conv0_0.bn2.train()
            model.conv0_0.bn2.track_running_stats = False
        
        logging.info(f"Set BatchNorm in train mode, {set_all=}.")

    def aggregate(self, pred):
        
        """
        Receives: 
            prediction tensor (#passes, 6 * Ch, H, W) and computes the average prediction and epistemic uncertainty.
        
        Returns: 
            tensor (2, 6 * Ch, H, W) where 1st dimension is mean point prediction (0), uncertainty measure (1).
        """

        # Epistemic uncertainty estimation: std over forward pass preds; ensure uncertainty >0 for numerical reasons
        return torch.stack((torch.mean(pred, dim=0),
                            torch.std(pred, dim=0, unbiased=False).clamp(min=1e-4)), dim=0)

    @torch.no_grad()
    def __call__(self, device, loss_fct, dataloader, model, samp_limit,
                 parallel_use, post_transform) -> Tuple[torch.Tensor, float]:
    
        model.eval()
        self.set_bn_mode(model) # Stochastic batch norm behaviour

        loss_sum = 0
        bsize = dataloader.batch_size
        batch_limit = samp_limit // bsize
        pred = torch.empty( # Pred contains y_true + pred + uncertainty: (samples, 3, H, W, Ch)
            size=(batch_limit * bsize, 3, 495, 436, 8), dtype=torch.float32, device=device)
    
        # Use only batch_size = 1 for dataloader to match train_batch_size behaviour
        with tqdm(dataloader) as tloader:
            for batch, (X, y) in enumerate(tloader):
                if batch == batch_limit:
                    break

                # MODIFY HERE FOR DIFFERENT MODELS
                # X, y = X.to(device, non_blocking=parallel_use) / 255, y.to(device, non_blocking=parallel_use)
                X, y = X.to(device, non_blocking=parallel_use), y.to(device, non_blocking=parallel_use) # For UNet++
                
                preds = torch.empty(size=(self.passes, 48, 496, 448), dtype=torch.float32, device=device)

                for p in range(0, self.passes): # Stochastic forward passes
                    tr_idx = random.sample(range(0, self.len), self.train_batch_size)
                    loader = DataLoader(Subset(self.data_train, tr_idx), self.train_batch_size, shuffle=False)
                    
                    # MODIFY HERE FOR DIFFERENT MODELS
                    # tr_batch = next(iter(loader))[0].to(device, non_blocking=parallel_use) / 255
                    tr_batch = next(iter(loader))[0].to(device, non_blocking=parallel_use) # For UNet++
                    
                    # Pass X + train mini-batch through model and collect 'stochastic' pred for X
                    preds[p, ...] = model(torch.cat((X, tr_batch), dim=0))[0, ...]

                y_pred = self.aggregate(preds) # (2, 6 * Ch, H+pad, W+pad)
                del preds, loader, tr_batch
                
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


""" Some references for stochastic behaviour
- https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html
- https://discuss.pytorch.org/t/normalizing-batchnorm2d-in-train-and-eval-mode/114067/2
- https://discuss.pytorch.org/t/what-does-model-eval-do-for-batchnorm-layer/7146/27 (last comment)
- https://developpaper.com/detailed-explanation-of-bn-core-parameters-of-pytorch/
- https://github.com/icml-mcbn/mcbn/blob/7c3338272eff0096c27dd139278037ea57c90cf7/code/segnet/test_bayesian_segnet_mcbn_paperResults.py#L87 (Teye et al code)
"""
