import os
import sys
import logging
import glob
from pathlib import Path
from typing import Optional
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from data.dataset import T4CDataset
from data.data_layout import CITY_NAMES, CITY_TRAIN_ONLY
from model.configs import configs
from model.checkpointing import save_file_to_folder
from util.h5_util import write_data_to_h5, load_h5_file

from metrics.pred_interval import get_quantile, get_pred_interval
from metrics.get_scores import get_scores, get_score_names, get_scalar_scores


def eval_test(model: torch.nn.Module,
                batch_size: int,
                num_workers: int,
                dataset_config: dict,
                dataloader_config: dict,
                model_str: str,
                model_id: int,
                parallel_use: bool,
                data_raw_path: str,
                test_pred_path: str,
                device: str,
                uq_method: str,
                save_checkpoint: str,
                quantiles_path: str,
                alpha: float,
                test_samp_name: str,
                test_samp_path: str = None,
                dataset_limit: Optional[list] = None,
                pred_to_file: bool = True,
                pi_to_file: bool = True,
                scores_to_file: bool = True,
                masked_scores: bool = True,
                **kwargs):

    logging.info("Running %s..." %(sys._getframe().f_code.co_name)) # Current fct name
    
    model = model.to(device)
    loss_fct = torch.nn.functional.mse_loss
    uq_method_obj = configs[model_str]["uq_method"][uq_method] # Uncertainty object
    post_transform = configs[model_str]["post_transform"][uq_method]
    cities = [city for city in CITY_NAMES if city not in CITY_TRAIN_ONLY]

    if uq_method == "ensemble":
        uq_method_obj.load_ensemble(device, save_checkpoint, configs[model_str]["model_class"], configs[model_str]["model_config"])
    elif uq_method == "bnorm":
        uq_method_obj.load_train_data(data_raw_path, configs[model_str]["dataset_config"]["point"])
    elif uq_method == "patches":
        uq_method_obj.set_pre_transform(configs[model_str]["pre_transform"][uq_method])

    if dataset_limit[0] is not None: # Limit #cities
        assert dataset_limit[0] <= len(cities)
        cities = cities[:dataset_limit[0]]
    
    logging.info(f"Evaluating '{model_str}' on '{device}' for {cities} with {uq_method_obj.__class__}.")
    
    for city in cities:
    
        test_file_paths = sorted(glob.glob(f"{data_raw_path}/{city}/test/*8ch.h5", recursive=True))
        logging.info(f"{len(test_file_paths)} test files extracted from {data_raw_path}/{city}/test/...")

        if test_pred_path is None:
            city_pred_path = Path(f"{data_raw_path}/{city}/test_{uq_method}_{model_str+str(model_id)}")
            city_pred_path.mkdir(exist_ok=True, parents=True)
        else:
            h5_pred_path = Path(os.path.join(test_pred_path, "h5_files", city))
            score_pred_path = Path(os.path.join(test_pred_path, "scores", city))
            h5_pred_path.mkdir(exist_ok=True, parents=True)
            score_pred_path.mkdir(exist_ok=True, parents=True)
        
        data = T4CDataset(root_dir=data_raw_path,
                          file_filter=test_file_paths,
                          **dataset_config)
        
        if test_samp_path is not None:
            sub_idx = torch.from_numpy(np.loadtxt(os.path.join(test_samp_path, city, test_samp_name))).to(torch.int)
        else:
            sub_idx = range(0, len(data)) # All data
        logging.info(f"Evaluating for {city} on {len(sub_idx)} test set indices in range {torch.min(sub_idx).item(), torch.max(sub_idx).item()}.")

        dataloader = DataLoader(dataset=Subset(data, sub_idx),
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=num_workers,
                                pin_memory=parallel_use,
                                **dataloader_config)

        pred, loss_city = uq_method_obj(device=device, # (samples, 3, H, W, Ch) torch.float32
                                       loss_fct=loss_fct,
                                       dataloader=dataloader,
                                       model=model,
                                       samp_limit=len(sub_idx),
                                       parallel_use=parallel_use,
                                       post_transform=post_transform)

        logging.info(f"Obtained predictions with uncertainty {uq_method} as {pred.shape, pred.dtype}.")
        if pred_to_file:
            write_data_to_h5(data=pred, dtype=np.float16, compression="lzf", verbose=True,
                             filename=os.path.join(h5_pred_path, f"pred_{uq_method}.h5"))
        
        #quant = os.path.join(data_raw_path, city, quantiles_path)
        quant = os.path.join(quantiles_path, city, f"quant_{int((1-alpha)*100)}_{uq_method}.h5")
        logging.info(f"Using quantiles from '{quant}'.")
        pred_interval = get_pred_interval(pred[:, 1:, ...], load_h5_file(quant, dtype=torch.float16).to(device)) # (samples, 2, H, W, Ch) torch.float32
        logging.info(f"Obtained prediction intervals as {pred_interval.shape, pred_interval.dtype}.")
        if pi_to_file:
            write_data_to_h5(data=pred_interval, dtype=np.float16, compression="lzf", verbose=True,
                             filename=os.path.join(h5_pred_path, f"pi_{uq_method}.h5"))

        scores = get_scores(pred, pred_interval, device=device)
        if masked_scores:
            logging.info("Also calculating scores for mask where volume > 0.")
            # which pixels have sum of vol > 0 across sample dimension
            mask_vol = pred[:, 0, :, :, [0, 2, 4, 6]].sum(dim=(0, -1)) > 0
            mask_scores = get_scores(
                pred[..., mask_vol, :].unsqueeze(dim=-2),
                pred_interval[..., mask_vol, :].unsqueeze(dim=-2),
                device=device)
            del mask_vol
        del data, pred, pred_interval

        logging.info(f"Obtained metric scores across sample dimension as {scores.shape, scores.dtype}.")
        if scores_to_file:
            write_data_to_h5(data=scores, dtype=np.float16, compression="lzf", verbose=True,
                             filename=os.path.join(h5_pred_path, f"scores_{uq_method}.h5"))
            if masked_scores:
                write_data_to_h5(data=mask_scores, dtype=np.float16, compression="lzf", verbose=True,
                                 filename=os.path.join(h5_pred_path, f"scores_{uq_method}_mask.h5"))

        score_names = get_score_names()
        scalar_speed, scalar_vol = get_scalar_scores(scores)
        del scores
        if masked_scores:
            scalar_speed_mask, scalar_vol_mask = get_scalar_scores(mask_scores)
            del mask_scores

        logging.info(f"Scores ==> {score_names}")
        logging.info(f"Scores for pred horizon 1h and speed channels: {scalar_speed}")
        logging.info(f"Scores for pred horizon 1h and volume channels: {scalar_vol}")
        save_file_to_folder(file=scalar_speed.cpu().numpy(), filename=f"scores_{uq_method}_speed",
                            folder_dir=score_pred_path, fmt="%.4f", header=score_names)
        save_file_to_folder(file=scalar_vol.cpu().numpy(), filename=f"scores_{uq_method}_vol",
                            folder_dir=score_pred_path, fmt="%.4f", header=score_names)
        if masked_scores:
            save_file_to_folder(file=scalar_speed_mask.cpu().numpy(), filename=f"scores_{uq_method}_speed_mask",
                                folder_dir=score_pred_path, fmt="%.4f", header=score_names)
            save_file_to_folder(file=scalar_vol_mask.cpu().numpy(), filename=f"scores_{uq_method}_vol_mask",
                                folder_dir=score_pred_path, fmt="%.4f", header=score_names)

        logging.info(f"Evaluation via {uq_method} finished for {city}.")
    logging.info(f"Evaluation via {uq_method} finished for all cities in {cities}.")


def eval_calib(model: torch.nn.Module,
                batch_size: int,
                num_workers: int,
                dataset_config: dict,
                dataloader_config: dict,
                model_str: str,
                model_id: int,
                parallel_use: bool,
                data_raw_path: str,
                quantiles_path: str,
                device: str,
                uq_method: str,
                save_checkpoint: str,
                calibration_size: int,
                alpha: float,
                city_limit: Optional[int] = None,
                to_file: bool = True,
                **kwargs):

    logging.info("Running %s..." %(sys._getframe().f_code.co_name))

    model = model.to(device)
    loss_fct = torch.nn.functional.mse_loss
    uq_method_obj = configs[model_str]["uq_method"][uq_method] # Uncertainty object
    post_transform = configs[model_str]["post_transform"][uq_method]
    cities = [city for city in CITY_NAMES if city not in CITY_TRAIN_ONLY]

    if uq_method == "ensemble":
        uq_method_obj.load_ensemble(device, save_checkpoint, configs[model_str]["model_class"], configs[model_str]["model_config"])
    elif uq_method == "bnorm":
        uq_method_obj.load_train_data(data_raw_path, configs[model_str]["dataset_config"]["point"])
    elif uq_method == "patches":
        uq_method_obj.set_pre_transform(configs[model_str]["pre_transform"][uq_method])

    if city_limit is not None: # Limit #cities
        assert city_limit <= len(cities)
        cities = cities[:city_limit]

    logging.info(f"Evaluating for {cities} with calibration sets of size {calibration_size}.")

    for city in cities:
    
        val_file_paths = sorted(glob.glob(f"{data_raw_path}/{city}/val/*8ch.h5", recursive=True))
        logging.info(f"Calibration files extracted from {data_raw_path}/{city}/val/...")

        data = T4CDataset(root_dir=data_raw_path,
                          file_filter=val_file_paths,
                          **dataset_config)

        sub_idx = random.sample(range(0, len(data)), calibration_size)
        dataloader = DataLoader(dataset=Subset(data, sub_idx),
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=num_workers,
                                pin_memory=parallel_use,
                                **dataloader_config)
        
        pred, loss_city = uq_method_obj(device=device,
                                       loss_fct=loss_fct,
                                       dataloader=dataloader,
                                       model=model,
                                       samp_limit=calibration_size,
                                       parallel_use=parallel_use,
                                       post_transform=post_transform)
        logging.info(f"Obtained predictions with uncertainty {uq_method} as {pred.shape, pred.dtype}.")
        logging.info(f"MSE loss for {city}: {loss_city}.")

        #pred[0, 0, ...] = get_quantile(pred, n=calibration_size, alpha=alpha) # (H, W, Ch)
        quant = get_quantile(pred, n=calibration_size, alpha=alpha) # (H, W, Ch)
        logging.info(f"Obtained quantiles as {quant.shape, quant.dtype}.")

        if to_file:
            if quantiles_path is None:
                quant_path = os.path.join(data_raw_path, city)
            else:
                quant_path = Path(os.path.join(quantiles_path, city))
                quant_path.mkdir(exist_ok=True, parents=True)
            quant_name = f"quant_{int((1-alpha)*100)}_{uq_method}.h5"
            #write_data_to_h5(data=pred[0, 0, ...], dtype=np.float16, filename=os.path.join(quant_path, quant_name), compression="lzf", verbose=True)
            write_data_to_h5(data=quant, dtype=np.float16, filename=os.path.join(quant_path, quant_name), compression="lzf", verbose=True)

        del data, dataloader, pred, quant
    logging.info(f"Written all calibration set {(1-alpha)*100}% quantiles for {cities=} to file.")
