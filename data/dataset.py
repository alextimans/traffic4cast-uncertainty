# base code from https://github.com/iarai/NeurIPS2021-traffic4cast

from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from util.h5_util import load_h5_file
from data.data_layout import MAX_ONE_DAY_SMP_IDX, MAX_FILE_DAY_IDX, TWO_HOURS


class T4CDataset(Dataset):
    def __init__(self,
                 root_dir: str,
                 file_filter: Union[str, list] = None,
                 dataset_type: Optional[str] = None,
                 dataset_limit: Optional[int] = None,
                 transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
                 zeropad2d: Optional[Tuple[int, int, int, int]] = None):

        """ 
        Create custom torch dataset from data.

        Parameters
        ----------
        root_dir: str
            Data root folder, by convention should be './data/raw'. 
            All '**/*8ch.h5' files will be added unless filtered.
        file_filter: str or list
            Filter files under 'root_dir' for specific files e.g. by city.
            Caution: doesn't discriminate dataset_type, relies on correct path.
            Defaults to '**/*ch8.h5' i.e. no filtering.
            Is str when specifying correct path (regex-style) e.g. 'BARCELONA/train/*8ch.h5'.
            Is list when passing list of file paths directly. 
        dataset_type: str
            One of ['train', 'val', 'test'].
        dataset_limit: int
            Truncate dataset size (by 2h samples, not files).
        transform: Callable
            Transform applied to both the input and label.
        zeropad2d: Tuple
            Optional specific padding argument used for transform applied to data.
        """

        self.root_dir = root_dir
        self.dataset_limit = dataset_limit
        self.transform = transform
        self.zeropad2d = zeropad2d
        self.dataset_type = dataset_type

        if file_filter is None:
            assert self.dataset_type in ["train", "val", "test"]
            self.file_filter = "**/" + self.dataset_type + "/*8ch.h5"
        else:
            self.file_filter = file_filter

        self._load_dataset()
        self._count_files_per_city_and_year()

    def _load_dataset(self):
        if isinstance(self.file_filter, list): # Used in eval scripts
            self.files = self.file_filter
        else:
            self.files = sorted(list(Path(self.root_dir).rglob(self.file_filter)))
        self.files = [str(file) for file in self.files] # Path to str

    def _count_files_per_city_and_year(self):
        city_and_year = [file.split("/")[-1].split("_")[0].split("-")[0] + file.split("/")[-1].split("_")[-2] for file in self.files]
        assert len(self.files) > 0 and len(self.files) == len(city_and_year)
        self.count = int(len(self.files) / len(np.unique(city_and_year)))

    def _load_h5_file(self, file_path, sl: Optional[slice], to_torch: bool = True):
        return load_h5_file(file_path, sl, to_torch)

    def __len__(self):
        dataset_size = len(self.files) * MAX_FILE_DAY_IDX
        if self.dataset_limit is not None:
            dataset_size = min(dataset_size, self.dataset_limit)

        return dataset_size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if idx > self.__len__():
            raise IndexError(f"Sample {idx=} out of bounds for len {self.__len__()}.")

        file_idx = idx // MAX_FILE_DAY_IDX
        start_hour = idx % MAX_FILE_DAY_IDX

        if file_idx == len(self.files): # Very last file idx 288
            file_idx -= 1
            start_hour = MAX_ONE_DAY_SMP_IDX
        elif (file_idx + 1) % self.count == 0 and start_hour > MAX_ONE_DAY_SMP_IDX: # Last file per city idx 264-287
            start_hour = MAX_ONE_DAY_SMP_IDX # Replicate last full sample 22-24h

        if start_hour > MAX_ONE_DAY_SMP_IDX: # Two hours stretch across two h5 files
            slots_1st_day = MAX_FILE_DAY_IDX - start_hour
            slots_2nd_day = TWO_HOURS - slots_1st_day

            sl_1st_day = slice(start_hour, start_hour + slots_1st_day) # X x 5m slots
            sl_2nd_day = slice(0, slots_2nd_day) # Y x 5m slots s.t. X + Y = 24

            time_1st_day = self._load_h5_file(self.files[file_idx], sl_1st_day)
            time_2nd_day = self._load_h5_file(self.files[file_idx + 1], sl_2nd_day)

            two_hours = torch.cat((time_1st_day, time_2nd_day), dim=0)

        else: # Two hours stretch across one h5 file
            sl = slice(start_hour, start_hour + TWO_HOURS) # 24 x 5m slots
            two_hours = self._load_h5_file(self.files[file_idx], sl)

        assert two_hours.size(0) == TWO_HOURS, f"two_hours of size {two_hours.size(0)}"
        X, y = data_split_xy(two_hours)

        if self.transform is not None:
            if self.zeropad2d is not None:
                X = self.transform(X, zeropad2d=self.zeropad2d)
                y = self.transform(y, zeropad2d=self.zeropad2d)
            else:
                X = self.transform(X)
                y = self.transform(y)

        assert isinstance(X, torch.Tensor) and isinstance(y, torch.Tensor)

        return X, y


def data_split_xy(data, offset: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:

    """ 
    Split data into sample input and label.

    Parameters
    ----------
    data: Data, tensor of shape (24, H, W, Ch).
    offset: int
        Extra offsetting of sample.

    Returns
    -------
        X: Input, tensor of shape (12, H, W, Ch).
        y: Label, tensor of shape (6, H, W, Ch).
    """

    X = data[offset:(offset + 12)]
    pred_steps = np.add(11 + offset, [1, 2, 3, 6, 9, 12]) # 6 pred horizons
    y = data[pred_steps]

    return X, y


"""
def test(idx):
    files = ["file1_city1", "file2_city1", "file3_city1", "file1_city2", "file2_city2", "file3_city2"]
    MAX_FILE_DAY_IDX = 288
    MAX_ONE_DAY_SMP_IDX = 264
    TWO_HOURS = 24

    if idx > (len(files) * MAX_FILE_DAY_IDX):
        raise IndexError(f"Sample index {idx} out of bounds.")

    file_idx = idx // MAX_FILE_DAY_IDX # Div and floor to int
    start_hour = idx % MAX_FILE_DAY_IDX # Modulo

    nr_files_per_city = 3 # i.e. self.count

    if (file_idx == len(files)): # Last file case idx 288
        print("Last index")
        file_idx -= 1
        start_hour = MAX_ONE_DAY_SMP_IDX
    elif ((file_idx + 1) % nr_files_per_city == 0 and start_hour > MAX_ONE_DAY_SMP_IDX): # Idx 264-287
        start_hour = MAX_ONE_DAY_SMP_IDX
    
    if (start_hour > MAX_ONE_DAY_SMP_IDX): # Two hours stretch across two h5 files
        slots_1st_day = MAX_FILE_DAY_IDX - start_hour
        slots_2nd_day = TWO_HOURS - slots_1st_day
    
        sl_1st_day = slice(start_hour, start_hour + slots_1st_day) # X x 5m slots
        sl_2nd_day = slice(0, slots_2nd_day) # Y x 5m slots s.t. X + Y = 24
    
        print(f"1st day // File: {files[file_idx]}, slice: {sl_1st_day}")
        print(f"2nd day // File: {files[file_idx+1]}, slice: {sl_2nd_day}")
    
    else: # Two hours stretch across one h5 file
        sl = slice(start_hour, start_hour + TWO_HOURS) # 24 x 5m slots
        print(f"One day // File: {files[file_idx]}, slice: {sl}")
"""
