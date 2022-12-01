# base code from https://github.com/iarai/NeurIPS2021-traffic4cast

import logging
from pathlib import Path
from typing import Optional, Union

import h5py
import numpy as np
import torch


def load_h5_file(file_path: Union[str, Path], sl: Optional[slice] = None,
                 to_torch: bool = True, dtype=None):
    """
    Given a file path to an h5 file assumed to house a tensor, load that
    tensor into memory and return a pointer.

    Parameters
    ----------
    file_path: str
        h5 file to load.
    sl: Optional[slice]
        Slice to load (data is written in chunks for faster access to rows).
    to_torch: bool
        Transform loaded numpy array into torch tensor
    dtype
        Set the specific dtype of the transformed torch tensor
    """

    file_path = str(file_path) if isinstance(file_path, Path) else file_path
    with h5py.File(file_path, "r") as file:
        data = file.get("array")

        if sl is not None:
            data = data[sl]  # Auto. np.ndarray
        else:
            data = data[:]  # Auto. np.ndarray

        if to_torch:
            if dtype is not None:
                data = torch.from_numpy(data).to(
                    dtype=dtype)  # e.g. torch.uint8
            else:
                data = torch.from_numpy(data).to(dtype=torch.float)

        return data


def write_data_to_h5(data: Union[np.ndarray, torch.Tensor],
                     dtype: str = None,
                     filename: Union[str, Path] = None,
                     compression: str = "gzip",  # "lzf"
                     compression_level: int = 9,  # 6
                     verbose: bool = False):
    """
    Write data in compressed h5 format.

    Parameters
    ----------
    data: np.ndarray or torch.Tensor
        Data to be written to h5 file.
    dtype: str
        Data type to be stored as.
    filename: str or Path object
        Name of h5 file which is written.
    compression: str
        Compression strategy, one of ['gzip', 'szip', 'lzf'].
    compression_level: int
        Compression setting. Int 0-9 for 'gzip', 2-tuple for 'szip'.
    verbose: bool
        Print writing logs.
    """

    if isinstance(data, torch.Tensor):
        if data.requires_grad:
            data = data.detach().cpu().numpy()
        else:
            data = data.cpu().numpy()

    if compression == "lzf":
        compression_level = None

    filename = str(filename) if isinstance(filename, Path) else filename
    with h5py.File(filename, "w", libver="latest") as file:

        # https://docs.h5py.org/en/stable/faq.html#faq
        # https://www.oreilly.com/library/view/python-and-hdf5/9781491944981/ch04.html
        file.create_dataset("array",
                            data=data,  # Infers shape and dtype
                            dtype=dtype,  # If explicitly given overrides inferred dtype
                            # Optimize for row access
                            chunks=(1, *data.shape[1:]),
                            compression=compression,
                            compression_opts=compression_level)

        if verbose:
            logging.info(
                f"Written {filename} to .h5 file with {compression=}.")
