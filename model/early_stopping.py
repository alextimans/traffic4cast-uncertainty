# base code from https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py

import logging
import numpy as np
import torch


class EarlyStopping:
    def __init__(self,
                 patience: int = 3,
                 delta: float = 0,
                 verbose: bool = False,
                 save_each_epoch: bool = False,
                 loss_improve: str = "min"):

        """
        Early stops model training if val loss doesn't improve after a given patience.

        Parameters
        ----------
        patience: int
            How long to wait after the last time val loss improved before stopping.
        delta: float
            Minimum change in the monitored loss to qualify as an improvement.
        verbose: bool
            If True prints a message for each validation loss improvement.
        save_each_epoch: bool
            Should the model be saved each epoch regardless of improvement or not.
        loss_improve: str
            Loss function-specific improvement direction. One in ["min", "max"].       
        """

        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.save_each_epoch = save_each_epoch
        self.loss_improve = loss_improve

        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.loss_val_min = -np.Inf if self.loss_improve == "max" else np.Inf

    def __call__(self,
                 model: torch.nn.Module,
                 loss_val: float = None,
                 last_loss: float = None) -> bool:

        save_to_checkpt = False

        if last_loss is not None:
            self.best_loss = last_loss
            self.loss_val_min = last_loss

        loss = loss_val
        if (self.loss_improve == "max"):
            loss = -loss_val
            self.delta = -self.delta

        if self.best_loss is None: # First call
            self.best_loss = loss
            if self.verbose:
                logging.info(f"Val loss change: {self.loss_val_min:.4f} -> {loss:.4f}")
            self.loss_val_min = loss
            save_to_checkpt = True

        elif (loss >= self.best_loss - self.delta): # No improvement in val loss
            self.counter += 1
            logging.info(f"EarlyStopping being patient: {self.counter}/{self.patience+1}.")

            if (self.counter > self.patience): # Init early stopping
                self.early_stop = True

            if self.save_each_epoch:
                if self.verbose:
                    logging.info(f"Val loss change: {self.loss_val_min:.4f} -> {loss:.4f}")
                self.loss_val_min = loss
                save_to_checkpt = True

        else: # Improvement in val loss
            self.counter = 0
            self.best_loss = loss
            if self.verbose:
                logging.info(f"Val loss change: {self.loss_val_min:.4f} -> {loss:.4f}")
            self.loss_val_min = loss
            save_to_checkpt = True

        return save_to_checkpt
