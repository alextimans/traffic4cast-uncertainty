# base code from https://github.com/iarai/NeurIPS2021-traffic4cast

import os
import datetime
import logging
from pathlib import Path
from typing import Union, Tuple

import numpy as np
import torch


def load_torch_opt_from_checkpoint(checkpt_path: Union[str, Path],
                                   optimizer: torch.optim.Optimizer,
                                   map_location: str = None) -> Tuple[int, float]:

    state_dict = torch.load(checkpt_path, map_location)

    msg = f"Calling on model training but '{checkpt_path}' has no optimizer state_dict."
    assert isinstance(state_dict, dict) and ("optimizer" in state_dict), msg
    optimizer.load_state_dict(state_dict["optimizer"])
    logging.info(f"Loaded optimizer state_dict from checkpoint '{checkpt_path}'.")

    if (state_dict["epoch"] >= 0) and ("loss" in state_dict):
        last_epoch, last_loss = state_dict["epoch"], state_dict["loss"]
        logging.info(f"Picking up training from {last_epoch=} with last val loss {last_loss}.")
    else:
        last_epoch, last_loss = -1, None
        logging.info("No previous training info, returning {last_epoch=}, {last_loss=}.")

    return last_epoch, last_loss


def load_torch_model_from_checkpoint(checkpt_path: Union[str, Path],
                                     model: torch.nn.Module,
                                     map_location: str = None):

    state_dict = torch.load(checkpt_path, map_location)
    assert isinstance(state_dict, dict) and ("model" in state_dict)

    state_model = state_dict["model"]
    parallel_checkpt = all("module." in key for key in list(state_model.keys()))

    if not isinstance(model, torch.nn.DataParallel) and parallel_checkpt:
        new_state_model = state_model.copy()
        for key, val in state_model.items(): # remove "module." for successful match
            new_state_model[key[7:]] = new_state_model.pop(key)
        state_model = new_state_model
        logging.info("Mismatch model <-> state_dict, removed 'module.' from keys.")

    elif isinstance(model, torch.nn.DataParallel) and not parallel_checkpt:
        new_state_model = state_model.copy()
        for key, val in state_model.items(): # add "module." for successful match
            new_state_model["module." + key] = new_state_model.pop(key)
        state_model = new_state_model
        logging.info("Mismatch model <-> state_dict, added 'module.' to keys.")

    model.load_state_dict(state_model)
    logging.info(f"Loaded model from checkpoint '{checkpt_path}'.")


def save_torch_model_to_checkpoint(model: torch.nn.Module,
                                   optimizer: torch.optim.Optimizer = None,
                                   model_str: str = None,
                                   model_id: int = None,
                                   epoch: int = -1, # No trained epochs
                                   loss: float = None,
                                   save_checkpoint: str = ""):

    """ 
    Saves a torch model as a checkpoint in specified location.

    Parameters
    ----------
    model: torch.nn.Module
        Model to create checkpoint of.
    optimizer: torch.optim.Optimizer
        Optimizer to add to checkpoint.
    model_str: str
        Model string name.
    model_id: int
        Model ID to create unique checkpoints folder.
    epoch: int
        Nr. of epochs model was trained.
    loss: float
        Loss we want to save e.g. validation loss.
    save_checkpoint: str
        Path to checkpoints folder. Default is local directory.
    """

    if model_str is not None and model_id is not None:
        save_checkpoint = Path(os.path.join(save_checkpoint, f"{model_str}_{model_id}"))
    else:
        save_checkpoint = Path(save_checkpoint)

    save_checkpoint.mkdir(exist_ok=True, parents=True)

    timestamp = datetime.datetime.strftime(datetime.datetime.now(), "%m%d%H%M")
    path_checkpt = os.path.join(save_checkpoint, f"{model_str}_ep{epoch}_{timestamp}.pt")

    save_dict = {"epoch": epoch,
                 "model": model.state_dict()}

    if optimizer is not None:
        save_dict.update({"optimizer": optimizer.state_dict()})

    if loss is not None:
        save_dict.update({"loss": loss})

    torch.save(save_dict, path_checkpt)
    logging.info(f"Model {model_str} trained to {epoch=} saved as '{path_checkpt}'.")


def save_file_to_folder(file = None, filename: str = None,
                        folder_dir: Union[Path, str] = None, **kwargs):

    """ 
    Stores file in specified folder as .txt file.
    """

    folder_path = Path(folder_dir) if isinstance(folder_dir, str) else folder_dir
    folder_path.mkdir(exist_ok=True, parents=True)

    np.savetxt(os.path.join(folder_path, f"{filename}.txt"), file, **kwargs)
    logging.info(f"Written {filename}.txt to {folder_path}.")
