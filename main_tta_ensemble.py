# base code from https://github.com/iarai/NeurIPS2021-traffic4cast

import argparse
import logging
import sys
import torch

from data.dataset import T4CDataset

from model.configs import configs
from model.train import run_model
from model.train_val_split import train_val_split
from model.checkpointing import load_torch_model_from_checkpoint

from util.logging import t4c_apply_basic_logging_config
from util.get_device import get_device
from util.set_seed import set_seed

# from uq.eval_model import eval_test, eval_calib
# replace and uncomment below for evaluating TTA + Ens combination and call eval_tta_ensemble
from uq.eval_tta_ensemble import eval_test, eval_calib
# replace and uncomment below for evaluating Patches + Ens combination and call eval_patches_ensemble
# from uq.eval_patches_ensemble import eval_test, eval_calib


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Parser for CLI arguments to run model.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--model_str", type=str, default="unet", required=False, choices=["unet", "unet_patches", "unet_pp"],
                        help="Model string name.")
    parser.add_argument("--model_id", type=int, default=1, required=False,
                        help="Model ID to differentiate models easier than via timestamp.")
    parser.add_argument("--model_training", type=str, default="True", required=False, choices=["True", "False"],
                        help="'Boolean' specifying if model training should be called.")
    parser.add_argument("--model_evaluation", type=str, default="True", required=False, choices=["True", "False"],
                        help="'Boolean' specifying if model evaluation should be called.")
    parser.add_argument("--resume_checkpoint", type=str, default=None, required=False,
                        help="Path to torch model .pt checkpoint to be re-loaded.")
    parser.add_argument("--save_checkpoint", type=str, default="./checkpoints/", required=False,
                        help="Directory to store model checkpoints in.")

    parser.add_argument("--batch_size", type=int, default=4, required=False,
                        help="Batch size for train, val and test data loaders. Preferably batch_size mod 2 = 0.")
    parser.add_argument("--epochs", type=int, default=2, required=False,
                        help="Number of epochs to train.")

    parser.add_argument("--data_limit", type=int, default=None, required=False,
                        help="Cap train + val dataset size at this limit. Refers to 2h samples, not data files.")
    parser.add_argument("--train_data_limit", type=int, default=None, required=False,
                        help="Cap train dataset size at this limit. Refers to 2h samples, not data files.")
    parser.add_argument("--val_data_limit", type=int, default=None, required=False,
                        help="Cap validation dataset size at this limit. Refers to 2h samples, not data files.")
    parser.add_argument("--test_data_limit", nargs=3, type=int, default=[None, None, None], required=False,
                        help="Limit values of (#cities, #files per city, #samples per file) to test on.")
    parser.add_argument("--train_file_filter", type=str, default=None, required=False,
                        help="Filter files for train dataset. Defaults to '**/*8ch.h5' i.e. no filtering.")
    parser.add_argument("--val_file_filter", type=str, default=None, required=False,
                        help="Filter files for val dataset. Defaults to '**/*8ch.h5' i.e. no filtering.")

    parser.add_argument("--num_workers", type=int, default=8, required=False,
                        help="Number of workers for data loader.")
    parser.add_argument("--device", type=str, default=None, required=False, choices=["cpu", "cuda"],
                        help="Specify usage of specific device.")
    parser.add_argument("--device_ids", type=str, nargs="*", default=None, required=False,
                        help="Whitelist of device ids. If not given, all device ids are taken. For GPUs training.")
    parser.add_argument("--data_parallel", type=str, default="False", required=False, choices=["True", "False"],
                        help="'Boolean' specifying use of DataParallel.")

    parser.add_argument("--loglevel", type=str, default="info", required=False,
                        help="Provide logging level. Ex.: --loglevel debug, default=warning.")
    parser.add_argument("--random_seed", type=int, default=1234567, required=False,
                        help="Set manual random seed.")
    parser.add_argument("--display_model", type=str, default="False", required=False, choices=["True", "False"],
                        help="'Boolean' to display model architecture in CLI.")
    parser.add_argument("--display_system_status", type=str, default="False", required=False, choices=["True", "False"],
                        help="'Boolean' to display system status during training in CLI.")

    #parser.add_argument("--config_path", default="config.yaml", required=False, help="Configuration file location.")
    parser.add_argument("--data_raw_path", type=str, default="./data/raw", required=False,
                        help="Base directory of raw data.")
    parser.add_argument("--data_compressed_path", type=str, default="./data/compressed", required=False,
                        help="Data is extracted from this location if no data at data_raw_path.")
    parser.add_argument("--test_pred_path", type=str, default=None, required=False,
                        help="Specific directory to store test set model predictions in.")

    parser.add_argument("--calibration", type=str, default="False", required=False, choices=["True", "False"],
                        help="Specify if should run calibration script to obtain quantiles for prediction intervals.")
    parser.add_argument("--calibration_size", type=int, default=100, required=False,
                        help="Specify calibration set size to obtain quantiles for prediction intervals.")
    parser.add_argument("--uq_method", type=str, default="point", required=False, choices=["point", "tta", "ensemble", "bnorm", "patches"],
                        help="Specify UQ method for test set and/or calibration set evaluation.")
    parser.add_argument("--quantiles_path", type=str, default=None, required=False,
                        help="Quantiles file path.")
    parser.add_argument("--test_samp_path", type=str, default=None, required=False,
                        help="Test sample indices file path.")
    parser.add_argument("--test_samp_name", type=str, default="test_indices.txt", required=False,
                        help="Test sample indices file name.")
    return parser


def main():

    """
    - CLI arguments handling
    - Seed setting
    - Dataset creation
    - Model creation (optional: checkpoint reloading)
    - Device setting
    - Call on model training
    - Call on calibration set script
    - Call on model test set evaluation
    """

    # Parsing
    t4c_apply_basic_logging_config()
    logging.info("Running %s..." %(sys._getframe().f_code.co_name)) # Current fct name
    parser = create_parser()
    args = parser.parse_args()
    logging.info("CLI arguments parsed.")

    # Named args (from parser + config file)
    model_str = args.model_str
    model_training = args.model_training
    model_evaluation = args.model_evaluation
    resume_checkpoint = args.resume_checkpoint
    display_model = args.display_model
    device = args.device
    device_ids = args.device_ids
    data_parallel = args.data_parallel
    random_seed = args.random_seed

    data_limit = args.data_limit
    train_data_limit = args.train_data_limit
    val_data_limit = args.val_data_limit
    test_data_limit = args.test_data_limit # list[x, y, z]
    data_raw_path = args.data_raw_path
    train_file_filter = args.train_file_filter
    val_file_filter = args.val_file_filter

    calibration = args.calibration
    uq_method = args.uq_method

    model_class = configs[model_str]["model_class"]
    model_config = configs[model_str]["model_config"]
    dataset_config = configs[model_str]["dataset_config"][uq_method]
    dataloader_config = configs[model_str]["dataloader_config"]
    optimizer_config = configs[model_str]["optimizer_config"]
    lr_scheduler_config = configs[model_str]["lr_scheduler_config"]
    earlystop_config = configs[model_str]["earlystop_config"]

    # Set (all) seeds
    random_seed = set_seed(random_seed)
    logging.info(f"Used {random_seed=} for seeds.")

    # Datasets
    logging.info("Building datasets...")
    if data_limit is not None:
        train_data_limit, val_data_limit = train_val_split(data_limit)

    data_train = T4CDataset(root_dir=data_raw_path,
                            file_filter=train_file_filter,
                            dataset_type="train",
                            dataset_limit=train_data_limit,
                            **configs["unet"]["dataset_config"]["point"])
    data_val = T4CDataset(root_dir=data_raw_path,
                          file_filter=val_file_filter,
                          dataset_type="val",
                          dataset_limit=val_data_limit,
                          **configs["unet"]["dataset_config"]["point"])

    assert (len(data_train) > 0) and (len(data_val) > 0)
    logging.info("Dataset sizes: Train = %s, Val = %s." %(len(data_train), len(data_val)))

    # Model setup
    model = model_class(**model_config)
    assert model_class == model.__class__, f"{model.__class__=} invalid."
    logging.info(f"Created model of class {model_class}.")

    if eval(display_model) is not False: # str to bool
        logging.info(model)

    # Device setting
    device, parallel_use = get_device(device, data_parallel)
    if parallel_use: # Multiple GPU usage
        model = torch.nn.DataParallel(model, device_ids=device_ids)
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

    # Model training
    if eval(model_training) is not False:
        logging.info("Training model...")
        model = run_model(model=model,
                          data_train=data_train,
                          data_val=data_val,
                          dataloader_config=dataloader_config,
                          optimizer_config=optimizer_config,
                          lr_scheduler_config=lr_scheduler_config,
                          earlystop_config=earlystop_config,
                          device=device,
                          parallel_use=parallel_use,
                          **(vars(args)))
        logging.info("Model trained.")
    else:
        del data_train, data_val
        logging.info("Model training not called, loaded datasets removed.")

    # Get calibration set quantiles for prediction intervals
    if eval(calibration) is not False:
        logging.info("Running calibration script...")
        eval_calib(model=model,
                    dataset_config=dataset_config,
                    dataloader_config=dataloader_config,
                    device=device,
                    parallel_use=parallel_use,
                    alpha=0.1, # 90% PIs
                    city_limit=test_data_limit[0],
                    to_file = True,
                    **(vars(args)))
        logging.info("Calibration script finished.")

    # Test set evaluation
    if eval(model_evaluation) is not False:
        logging.info("Evaluating model on test set...")
        logging.info(f"Evaluating uncertainty using '{uq_method}'...")
        eval_test(model=model,
                    dataset_limit=test_data_limit,
                    dataset_config=dataset_config,
                    dataloader_config=dataloader_config,
                    device=device,
                    parallel_use=parallel_use,
                    alpha=0.1,
                    **(vars(args)))
        logging.info("Model evaluated.")
    else:
        logging.info("Model test set evaluation not called.")

    logging.info("Main finished.")


if __name__ == "__main__":
    main()
