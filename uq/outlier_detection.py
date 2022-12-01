import os
import sys
import logging
import glob
from pathlib import Path
import argparse

import numpy as np
from scipy.stats import gaussian_kde
from scipy.stats import combine_pvalues

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from data.dataset import T4CDataset
from model.configs import configs
from model.checkpointing import load_torch_model_from_checkpoint
from util.h5_util import write_data_to_h5, load_h5_file
from util.logging import t4c_apply_basic_logging_config
from util.get_device import get_device
from util.set_seed import set_seed


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Parser for CLI arguments to run model.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model_str", type=str, default="unet", required=False, choices=["unet"],
                        help="Model string name.")
    parser.add_argument("--resume_checkpoint", type=str, default=None, required=False,
                        help="Path to torch model .pt checkpoint to be re-loaded.")
    parser.add_argument("--save_checkpoint", type=str, default="./checkpoints/", required=False,
                        help="Directory to store model checkpoints in.")
    parser.add_argument("--device", type=str, default=None, required=False, choices=["cpu", "cuda"],
                        help="Specify usage of specific device.")
    parser.add_argument("--random_seed", type=int, default=1234567, required=False,
                        help="Set manual random seed.")
    parser.add_argument("--data_parallel", type=str, default="False", required=False, choices=["True", "False"],
                        help="'Boolean' specifying use of DataParallel.")
    parser.add_argument("--num_workers", type=int, default=8, required=False,
                        help="Number of workers for data loader.")
    parser.add_argument("--batch_size", type=int, default=1, required=False,
                        help="Batch size for train, val and test data loaders. Preferably batch_size mod 2 = 0.")

    parser.add_argument("--data_raw_path", type=str, default="./data/raw", required=False,
                        help="Base directory of raw data.")
    parser.add_argument("--test_pred_path", type=str, default=None, required=False,
                        help="Specific directory to store test set model predictions in.")

    parser.add_argument("--uq_method", type=str, default=None, required=False, choices=["ensemble", "bnorm"],
                        help="Specify UQ method for epistemic uncertainty.")
    parser.add_argument("--fix_samp_idx", nargs=3, type=int, default=[None, None, None], required=False,
                        help="Fixed sample indices for time frame across training data per city [BANGKOK, BARCELONA, MOSCOW] in order.")
    parser.add_argument("--cities", nargs=3, type=str, default=[None, None, None], required=False,
                        help="City names to run outlier detection on, city in [BANGKOK, BARCELONA, MOSCOW].")

    parser.add_argument("--out_bound", type=float, default=0.01, required=False,
                        help="Outlier decision boundary: if p-value <= out_bound we consider value an outlier.")
    parser.add_argument("--out_name", type=str, default="out", required=False,
                        help="Outlier file name.")
    parser.add_argument("--test_pred_bool", type=str, default="True", required=False, choices=["True", "False"],
                        help="'Boolean' specifying if test_pred function should be called.")
    parser.add_argument("--detect_outliers_bool", type=str, default="True", required=False, choices=["True", "False"],
                        help="'Boolean' specifying if detect_outliers function should be called.")
    parser.add_argument("--train_pred_bool", type=str, default="True", required=False, choices=["True", "False"],
                        help="'Boolean' specifying if train set prediction in detect_outliers() should be called.")
    parser.add_argument("--get_pval_bool", type=str, default="True", required=False, choices=["True", "False"],
                        help="'Boolean' specifying if KDE fit + p-values in detect_outliers() should be called.")
    parser.add_argument("--agg_pval_bool", type=str, default="True", required=False, choices=["True", "False"],
                        help="'Boolean' specifying if aggregate p-values in detect_outliers() should be called.")

    return parser


def test_pred(model: torch.nn.Module,
                cities: list,
                fix_samp_idx: list,
                batch_size: int,
                num_workers: int,
                dataset_config: dict,
                dataloader_config: dict,
                model_str: str,
                parallel_use: bool,
                data_raw_path: str,
                test_pred_path: str,
                device: str,
                uq_method: str,
                save_checkpoint: str,
                pred_to_file: bool = True,
                **kwargs):

    logging.info("Running %s..." %(sys._getframe().f_code.co_name)) # Current fct name
    
    model = model.to(device)
    loss_fct = torch.nn.functional.mse_loss
    uq_method_obj = configs[model_str]["uq_method"][uq_method] # Uncertainty object
    post_transform = configs[model_str]["post_transform"][uq_method]

    if uq_method == "ensemble":
        uq_method_obj.load_ensemble(device, save_checkpoint, configs[model_str]["model_class"], configs[model_str]["model_config"])
    elif uq_method == "bnorm":
        uq_method_obj.load_train_data(data_raw_path, configs[model_str]["dataset_config"]["point"])

    logging.info(f"Evaluating '{model_str}' on '{device}' for {cities} with {uq_method_obj.__class__}.")
    
    for i, city in enumerate(cities):
    
        test_file_paths = sorted(glob.glob(f"{data_raw_path}/{city}/test/*8ch.h5", recursive=True))
        logging.info(f"{len(test_file_paths)} test files extracted from {data_raw_path}/{city}/test/...")

        if test_pred_path is None:
            raise AttributeError
        else:
            res_path = Path(os.path.join(test_pred_path, city))
            res_path.mkdir(exist_ok=True, parents=True)
        
        data = T4CDataset(root_dir=data_raw_path,
                          file_filter=test_file_paths,
                          **dataset_config)

        # idx of fixed sample index for each file
        logging.info(f"Using fixed sample index {fix_samp_idx[i]} for {city}.")
        sub_idx = [fix_samp_idx[i]+t*288 for t in range(len(test_file_paths))]

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

        logging.info(f"Obtained test set preds with uncertainty {uq_method} as {pred.shape, pred.dtype}.")
        if pred_to_file:
            write_data_to_h5(data=pred, dtype=np.float16, compression="lzf", verbose=True,
                             filename=os.path.join(res_path, f"pred_{uq_method}.h5"))
        del data, dataloader, pred

        logging.info(f"Evaluation via {uq_method} finished for {city}.")
    logging.info(f"Evaluation via {uq_method} finished for all cities in {cities}.")


def detect_outliers(model: torch.nn.Module,
                cities: list,
                fix_samp_idx: list,
                batch_size: int,
                num_workers: int,
                dataset_config: dict,
                dataloader_config: dict,
                model_str: str,
                parallel_use: bool,
                data_raw_path: str,
                test_pred_path: str,
                device: str,
                uq_method: str,
                save_checkpoint: str,
                out_bound: float,
                out_name: str,
                train_pred_bool: str,
                get_pval_bool: str,
                agg_pval_bool: str,
                pred_to_file: bool = True,
                pval_to_file: bool = True,
                out_to_file: bool = True,
                **kwargs):

    logging.info("Running %s..." %(sys._getframe().f_code.co_name)) # Current fct name
    
    model = model.to(device)
    loss_fct = torch.nn.functional.mse_loss
    uq_method_obj = configs[model_str]["uq_method"][uq_method] # Uncertainty object
    post_transform = configs[model_str]["post_transform"][uq_method]

    if uq_method == "ensemble":
        uq_method_obj.load_ensemble(device, save_checkpoint, configs[model_str]["model_class"], configs[model_str]["model_config"])
    elif uq_method == "bnorm":
        uq_method_obj.load_train_data(data_raw_path, configs[model_str]["dataset_config"]["point"])

    logging.info(f"Evaluating '{model_str}' on '{device}' for {cities} with {uq_method_obj.__class__}.")
    
    for i, city in enumerate(cities):
    
        train_file_paths = sorted(glob.glob(f"{data_raw_path}/{city}/train/*8ch.h5", recursive=True))
        logging.info(f"{len(train_file_paths)} train files extracted from {data_raw_path}/{city}/train/...")

        if test_pred_path is None:
            raise AttributeError
        else:
            res_path = Path(os.path.join(test_pred_path, city))
            res_path.mkdir(exist_ok=True, parents=True)
        
        data = T4CDataset(root_dir=data_raw_path,
                          file_filter=train_file_paths,
                          **dataset_config)

        # idx of fixed sample index for each file
        logging.info(f"Using fixed sample index {fix_samp_idx[i]} for {city}.")
        sub_idx = [fix_samp_idx[i]+t*288 for t in range(len(train_file_paths))]

        dataloader = DataLoader(dataset=Subset(data, sub_idx),
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=num_workers,
                                pin_memory=parallel_use,
                                **dataloader_config)

        # Train set predictions for fixed sample idx
        if eval(train_pred_bool) is not False:
            pred_tr, loss_city = uq_method_obj(device=device, # (samples, 3, H, W, Ch) torch.float32
                                           loss_fct=loss_fct,
                                           dataloader=dataloader,
                                           model=model,
                                           samp_limit=len(sub_idx),
                                           parallel_use=parallel_use,
                                           post_transform=post_transform)
    
            logging.info(f"Obtained train set preds with uncertainty {uq_method} as {pred_tr.shape, pred_tr.dtype}.")
            if pred_to_file:
                write_data_to_h5(data=pred_tr, dtype=np.float16, compression="lzf", verbose=True,
                                 filename=os.path.join(res_path, f"pred_tr_{uq_method}.h5"))
        else:
            logging.info(f"Train set preds assumed to be available as 'pred_tr_{uq_method}.h5'.")
            pred_tr = load_h5_file(os.path.join(res_path, f"pred_tr_{uq_method}.h5"), dtype=torch.float32)
        del data, dataloader

        # Train set uncertainties
        unc_tr = pred_tr[:, 2, ...].to("cpu")
        del pred_tr

        # Test set uncertainties
        pred = load_h5_file(os.path.join(test_pred_path, city, f"pred_{uq_method}.h5"), dtype=torch.float32)
        pix = get_nonzero_gt(pred)
        unc = pred[:, 2, ...]
        del pred

        # Cell-level uncertainty KDE Gaussian fit + p-values
        if eval(get_pval_bool) is not False:
            pval = get_pvalues(unc_tr, unc, pix, device)
            logging.info(f"Obtained tensor of p-values as {pval.shape, pval.dtype}.")
            if pval_to_file:
                write_data_to_h5(data=pval, dtype=np.float16, compression="lzf", verbose=True,
                                 filename=os.path.join(res_path, f"pval_{uq_method}.h5"))
        else:
            logging.info(f"P-values assumed to be available as 'pval_{uq_method}.h5'.")
            pval = load_h5_file(os.path.join(res_path, f"pval_{uq_method}.h5"), dtype=torch.float32)
        del unc_tr, unc

        # p-value aggregation and outlier labelling (channel-level, pixel-level)
        if eval(agg_pval_bool) is not False:
            out = aggregate_pvalues(pval, pix, out_bound, device)
            logging.info(f"Obtained tensor of outlier labels as {out.shape, out.dtype}.")
            if out_to_file:
                write_data_to_h5(data=out, dtype=bool, compression="lzf", verbose=True,
                                 filename=os.path.join(res_path, f"{out_name}_{uq_method}.h5"))
        else:
            logging.info(f"Aggregated & labelled outliers assumed to be available as 'out_{uq_method}.h5'.")
            out = load_h5_file(os.path.join(res_path, f"{out_name}_{uq_method}.h5"), dtype=torch.bool)

        # Outlier detection stats
        outlier_stats(out)
        del pval, out

        logging.info(f"Outlier detection via {uq_method} finished for {city}.")
    logging.info(f"Outlier detection via {uq_method} finished for all cities in {cities}.")


def get_nonzero_gt(pred):
    # get pixel indices that have ground truth sum of vol > 0 across sample dim
    return (pred[:, 0, :, :, [0, 2, 4, 6]].sum(dim=(0, -1)) > 0).nonzero(as_tuple=False)


def get_pvalues(unc_tr, unc, pix, device: str):
    samp, p_i, p_j, channels = tuple(unc.shape)

    # Tensor containing cell-level p-values for test set uncertainty vs. train set uncertainty KDE fit
    pval = torch.ones(size=(samp, p_i, p_j, channels), dtype=torch.float32, device="cpu")

    for i, j in tqdm(pix, desc="Pixel pairs"):
        for ch in range(channels):

            # i, j = i.item(), j.item()
            kde = gaussian_kde(unc_tr[:, i, j, ch], bw_method="scott")
            cell = unc[:, i, j, ch]

            for s in range(samp):
                pval[s, i, j, ch] = torch.tensor(
                    kde.integrate_box_1d(cell[s], np.inf),
                    dtype=torch.float32)

    assert pval.max() <= 1 and pval.min() >= 0, "p-values not in [0, 1]"
    return pval


def aggregate_pvalues(pval, pix, out_bound: float, device: str):
    logging.info(f"Making outlier decision based on prob. mass boundary {out_bound=}.")
    samp, p_i, p_j, _ = tuple(pval.shape)

    # Boolean tensor containing outlier labelling
    out = torch.zeros(size=(samp, p_i, p_j, 3), dtype=torch.bool, device="cpu")

    for i, j in tqdm(pix, desc="Pixel pairs"):
        for s in range(samp):

            # i, j = i.item(), j.item()
            out_ch = aggregate_channels(pval[s, i, j, :], out_bound)
            out_pix = aggregate_pixel(out_ch)
            # 1: Outlier vol, 2: Outlier speed, 3: Outlier pixel
            out[s, i, j, :] = torch.cat((out_ch, out_pix))

    return out


def aggregate_channels(pval, out_bound: float):
    # Channel group is outlier if combined p-value outside outlier bound
    p_vol_agg = combine_pvalues(pval[[0, 2, 4, 6]].clamp(min=1e-10), method="fisher")[1]
    p_sp_agg = combine_pvalues(pval[[1, 3, 5, 7]].clamp(min=1e-10), method="fisher")[1]

    out_vol = True if p_vol_agg <= out_bound else False
    out_sp = True if p_sp_agg <= out_bound else False

    return torch.tensor([out_vol, out_sp])


def aggregate_pixel(out_ch):
    # Pixel is outlier if at least one channel group is outlier
    out_pix = True if out_ch.sum() > 0 else False

    return torch.tensor([out_pix])


def outlier_stats(out):
    samp, p_i, p_j, _ = tuple(out.shape)
    tot, pix_tot = samp * p_i * p_j, p_i * p_j

    logging.info("### Outlier stats ###")

    ov, ov_pct = out[..., 0].sum(), out[..., 0].sum()/tot
    logging.info(f"Total outliers by vol ch: {ov}/({samp}*{p_i}*{p_j}) or {(ov_pct*100):.2f}%.")

    os, os_pct = out[..., 1].sum(), out[..., 1].sum()/tot
    logging.info(f"Total outliers by speed ch: {os}/({samp}*{p_i}*{p_j}) or {(os_pct*100):.2f}%.")

    op, op_pct = out[..., 2].sum(), out[..., 2].sum()/tot
    logging.info(f"Total outliers by pixel: {op}/({samp}*{p_i}*{p_j}) or {(op_pct*100):.2f}%.")

    om = out[..., 2].sum(dim=0).max() # max outlier counts across sample dim for pixels
    omc = (out[..., 2].sum(dim=0) == om).sum()
    omc_pct = omc / tot
    logging.info(f"""Pixels with max. outlier count by sample: {omc} pixels or
                 {(omc_pct*100):.2f}% with {om}/{samp} outliers.""")

    osamp = out[..., 2].sum(dim=(1,2)).to(torch.float32)
    osamp_m, osamp_std = int(osamp.mean().ceil()), int(osamp.std().ceil())
    osamp_pct_m, osamp_pct_std = osamp_m / pix_tot, osamp_std / pix_tot
    logging.info(f"""Avg. pixel outlier count by sample: {osamp_m} +/- {osamp_std} of ({p_i}*{p_j})
                 or {(osamp_pct_m*100):.2f} +/- {(osamp_pct_std*100):.2f}%.""")

    osmax, osmin = osamp.argmax(), osamp.argmin()
    osma, osma_pct = int(osamp[osmax].item()), osamp[osmax] / pix_tot
    osmi, osmi_pct = int(osamp[osmin].item()), osamp[osmin] / pix_tot
    logging.info(f"""Sample with most pixel outliers: test sample {osmax.item()}
                 with {osma}/({p_i}*{p_j}) outliers or {(osma_pct*100):.2f}%.""")
    logging.info(f"""Sample with least pixel outliers: test sample {osmin.item()}
                 with {osmi}/({p_i}*{p_j}) outliers or {(osmi_pct*100):.2f}%.""")


def main():
    t4c_apply_basic_logging_config()
    logging.info("Running %s..." %(sys._getframe().f_code.co_name)) # Current fct name
    parser = create_parser()
    args = parser.parse_args()

    # Named args (from parser + config file)
    model_str = args.model_str
    resume_checkpoint = args.resume_checkpoint
    device = args.device
    data_parallel = args.data_parallel
    random_seed = args.random_seed
    uq_method = args.uq_method
    test_pred_bool = args.test_pred_bool
    detect_outliers_bool = args.detect_outliers_bool

    cities = [city for city in args.cities if city != "None"]
    vars(args).pop("cities")

    model_class = configs[model_str]["model_class"]
    model_config = configs[model_str]["model_config"]
    dataset_config = configs[model_str]["dataset_config"][uq_method]
    dataloader_config = configs[model_str]["dataloader_config"]

    # Set (all) seeds
    random_seed = set_seed(random_seed)
    logging.info(f"Used {random_seed=} for seeds.")

    # Model setup
    model = model_class(**model_config)
    assert model_class == model.__class__, f"{model.__class__=} invalid."
    logging.info(f"Created model of class {model_class}.")

    # Device setting
    device, parallel_use = get_device(device, data_parallel)
    if parallel_use: # Multiple GPU usage
        model = torch.nn.DataParallel(model)
        logging.info(f"Using {len(model.device_ids)} GPUs: {model.device_ids}.")
        device = f"cuda:{model.device_ids[0]}" # cuda:0 is main process device
    logging.info(f"Using {device=}, {parallel_use=}.")
    vars(args).pop("device")
     
    # Checkpoint loading
    if resume_checkpoint is not None:
        load_torch_model_from_checkpoint(checkpt_path=resume_checkpoint,
                                         model=model, map_location=device)
    else:
        logging.info("No model checkpoint given.")

    if eval(test_pred_bool) is not False:
        logging.info("Collecting test set preds, in particular uncertainties.")
        test_pred(model=model,
                       cities=cities,
                       dataset_config=dataset_config,
                       dataloader_config=dataloader_config,
                       parallel_use=parallel_use,
                       device=device,
                       **(vars(args)))
        logging.info("Test set preds collected.")
    else:
        logging.info("Test set preds assumed to be available.")

    if eval(detect_outliers_bool) is not False:
        logging.info("Detecting outliers on test set via train set distr. fits.")
        detect_outliers(model=model,
                       cities=cities,
                       dataset_config=dataset_config,
                       dataloader_config=dataloader_config,
                       parallel_use=parallel_use,
                       device=device,
                       **(vars(args)))
        logging.info("Outliers detected.")
    else:
        logging.info("No outlier detection occuring.")
    logging.info("Main finished.")


if __name__ == "__main__":
    main()
