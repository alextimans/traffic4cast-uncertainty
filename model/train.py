# adapted from base code from https://github.com/iarai/NeurIPS2021-traffic4cast

"""
Model training logic.
"""

import os
import logging
import sys
from typing import Tuple

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.early_stopping import EarlyStopping
from model.checkpointing import load_torch_opt_from_checkpoint
from model.checkpointing import save_torch_model_to_checkpoint
from model.checkpointing import save_file_to_folder
from data.dataset import T4CDataset
from util.monitoring import system_status


def run_model(model: torch.nn.Module,
              data_train: T4CDataset,
              data_val: T4CDataset,
              batch_size: int,
              num_workers: int,
              epochs: int,
              dataloader_config: dict,
              optimizer_config: dict,
              lr_scheduler_config: dict,
              earlystop_config: dict,
              model_str: str,
              model_id: int,
              resume_checkpoint: str,
              save_checkpoint: str,
              parallel_use: bool,
              display_system_status: bool,
              device: str,
              **kwargs):

    logging.info("Running %s..." %(sys._getframe().f_code.co_name)) # Current fct name

    # Load data
    train_loader = DataLoader(dataset=data_train,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=parallel_use,
                              **dataloader_config)
    val_loader = DataLoader(dataset=data_val,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers,
                            pin_memory=parallel_use,
                            **dataloader_config)
    logging.info(f"Created data loaders with {batch_size=}.")

    # Model logic
    model = model.to(device, non_blocking=parallel_use)
    loss_fct = torch.nn.functional.mse_loss #torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), **optimizer_config)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, **lr_scheduler_config)
    early_stopping = EarlyStopping(**earlystop_config)

    # Load last training state
    if resume_checkpoint is not None:
        last_epoch, last_loss = load_torch_opt_from_checkpoint(resume_checkpoint, optimizer, device)
    else:
        last_epoch, last_loss = -1, None

    # Training
    loss_train, loss_val = train_model(device, epochs, optimizer, loss_fct,
                                       train_loader, val_loader, model, model_str,
                                       model_id, save_checkpoint, early_stopping,
                                       lr_scheduler, display_system_status, parallel_use,
                                       last_epoch, last_loss)
    logging.info("Finished training of model %s on %s for %s epochs.",
                 model_str, device, epochs)
    logging.info("Final loss '{}' -> Train: {:.4f}, Val: {:.4f}"
                 .format(loss_fct.__name__, loss_train[-1], loss_val[-1]))

    return model


def train_model(device, epochs, optimizer, loss_fct, train_loader, val_loader,
                model, model_str, model_id, save_checkpoint, early_stopping,
                lr_scheduler, display_system_status, parallel_use,
                last_epoch, last_loss) -> Tuple[list, list]:

    path_checkpt = os.path.join(save_checkpoint, f"{model_str}_{model_id}")
    next_epoch = last_epoch + 1
    l_train, l_val = [], [] # Loss per epoch

    for epoch in range(next_epoch, next_epoch + epochs):

        loss_train, l_t = _train_epoch(device, epoch, optimizer, loss_fct, train_loader, model, parallel_use)
        loss_val, l_v = _val_epoch(device, epoch, loss_fct, val_loader, model, parallel_use)

        l_train.append(loss_train)
        l_val.append(loss_val)

        save_file_to_folder(file=l_t, filename=f"loss_t_bybatch_{epoch}", folder_dir=path_checkpt,
                            fmt="%.4f", header=f"train loss by batch for {model_str}_{model_id} for {epoch=}")
        save_file_to_folder(file=l_v, filename=f"loss_v_bybatch_{epoch}", folder_dir=path_checkpt,
                            fmt="%.4f", header=f"val loss by batch for {model_str}_{model_id} for {epoch=}")

        logging.info("Epoch: {}, Train loss: {:.4f}, Val loss: {:.4f}"
                     .format(epoch, loss_train, loss_val))
        if eval(display_system_status) is not False:
            logging.info(system_status()) # Visualize GPU, memory, disk usage

        lr_scheduler.step(loss_val)

        save_to_checkpt = early_stopping(model, loss_val, last_loss)
        if save_to_checkpt:
            save_torch_model_to_checkpoint(model, optimizer, model_str=model_str, epoch=epoch,
                                           loss=loss_val, save_checkpoint=path_checkpt)
        if early_stopping.early_stop:
            logging.info(f"Early stopping at {epoch=}.")
            break

    comment = f"loss by epoch for {model_str}_{model_id} for epochs {next_epoch} to {epoch}"
    save_file_to_folder(file=l_train, filename=f"loss_train_ep{next_epoch}_{epoch}", folder_dir=path_checkpt,
                        fmt="%.4f", header="train " + comment)
    save_file_to_folder(file=l_val, filename=f"loss_val_ep{next_epoch}_{epoch}", folder_dir=path_checkpt,
                        fmt="%.4f", header="val " + comment)

    return l_train, l_val


def _train_epoch(device, epoch, optimizer, loss_fct, dataloader, model, parallel_use) -> Tuple[float, list]:
    model.train()
    loss_sum = 0
    l_t = [] # Loss per batch

    with tqdm(dataloader) as tepoch:
        for batch, (X, y) in enumerate(tepoch):
            X, y = X.to(device, non_blocking=parallel_use), y.to(device, non_blocking=parallel_use)
            # X = X / 255 # in [0, 1] # Comment for UNet++
            y_pred = model(X) # (batch, 6 * Ch, 496, 448) in [0, 255]
            loss = loss_fct(y_pred[:, :, 1:, 6:-6], y[:, :, 1:, 6:-6]) # Mean over batch samples + channels + pixels

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            l_t.append(float(loss.item()))
            loss_sum += float(loss.item())
            loss_train = float(loss_sum/(batch+1)) # Add mean over batches -> mean over all samples
            tepoch.set_description(f"Epoch {epoch} > train")
            tepoch.set_postfix(loss = loss_train)

    return loss_train, l_t


@torch.no_grad()
def _val_epoch(device, epoch, loss_fct, dataloader, model, parallel_use) -> Tuple[float, list]:
    model.eval()
    loss_sum = 0
    l_v = [] # Loss per batch

    with tqdm(dataloader) as tepoch:
        for batch, (X, y) in enumerate(tepoch):
            X, y = X.to(device, non_blocking=parallel_use), y.to(device, non_blocking=parallel_use)
            # X = X / 255 # Comment for UNet++
            y_pred = model(X)
            loss = loss_fct(y_pred[:, :, 1:, 6:-6], y[:, :, 1:, 6:-6]) # Loss on unpadded data

            l_v.append(float(loss.item()))
            loss_sum += float(loss.item())
            loss_val = float(loss_sum/(batch+1))
            tepoch.set_description(f"Epoch {epoch} > val")
            tepoch.set_postfix(loss = loss_val)

    return loss_val, l_v
