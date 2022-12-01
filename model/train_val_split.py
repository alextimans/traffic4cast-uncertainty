import logging
from typing import Tuple
import numpy as np

from data.data_layout import TRAIN_FILES, VAL_FILES


def train_val_split(data_limit: int = None) -> Tuple[int, int]:

    if data_limit <= 1:
        raise ValueError(f"Cannot train and validate on {data_limit} samples only.")

    train_fraction = np.round(TRAIN_FILES / (TRAIN_FILES + VAL_FILES), 3)
    val_fraction = np.round(1 - train_fraction, 3)

    assert np.isclose(train_fraction + val_fraction, 1.0)

    train_data_limit = int(np.floor(data_limit * train_fraction))
    val_data_limit = data_limit - train_data_limit
    
    assert (train_data_limit + val_data_limit) == data_limit

    logging.info(f"Given {data_limit=}, split train + val data into {train_data_limit} " +
                 f"train samples and {val_data_limit} val samples " +
                 f"based on train/val fractions {100*train_fraction}%/{100*val_fraction}%.")

    return train_data_limit, val_data_limit
