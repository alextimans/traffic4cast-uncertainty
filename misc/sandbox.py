#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: atimans
"""

import glob
import matplotlib.pyplot as plt
import numpy as np
import torch

from util.h5_util import load_h5_file
from data.data_layout import volume_channel_indices
from data.data_layout import speed_channel_indices
from data.data_layout import heading_list

from baselines.naive_average import NaiveAverage
from competition.prepare_test_data.prepare_test_data import prepare_test
from metrics.mse import mse


BASE_FOLDER = 'data/raw'
files = sorted(glob.glob(f'{BASE_FOLDER}/**/test/*8ch.h5', recursive=True))
file_idx = 156
file = files[file_idx]

file = "./data/raw_samp/ANTWERP/test/2020-06-07_ANTWERP_8ch.h5"
file = "./data/raw_samp/BANGKOK/test/2020-04-29_BANGKOK_8ch.h5"
file = "./data/raw_samp/BARCELONA/test/2019-05-13_BARCELONA_8ch.h5"
file = "./data/raw_samp/MOSCOW/test/2019-02-22_MOSCOW_8ch.h5"

city, date = file.split('/')[3], file.split('/')[5].split('_')[0]
print(f'Selected file: {file}')

data = load_h5_file(file)
data = data[1:11, ...]

print(f'Data: shape {data.shape} and type {data.dtype}')

pred = load_h5_file("./data/raw_samp/ANTWERP/test_pred/2020-06-07_ANTWERP_8ch.h5")
pred = load_h5_file("./data/raw_samp/BANGKOK/test_pred/2020-04-29_BANGKOK_8ch.h5")
pred = load_h5_file("./data/raw_samp/BARCELONA/test_pred/2019-05-13_BARCELONA_8ch.h5")
pred = load_h5_file("./data/raw_samp/MOSCOW/test_pred/2019-02-22_MOSCOW_8ch.h5")

pred = pred[:, 0, :, :] # next time step pred only

print(f'Data: shape {pred.shape} and type {pred.dtype}')

torch.nonzero(data).shape
torch.nonzero(pred).shape

### Plots

plt.title(f'{city} {date} sum all channels')
plt.imshow(data.sum(axis=(0, -1)))

plt.title(f'{city} {date} sum all channels pred')
plt.imshow(pred.sum(axis=(0, -1)))

fig_idx = [[0,0], [0,1], [1,0], [1,1]]
_, ax = plt.subplots(2, 2, figsize=(30, 150))
for h, heading in enumerate(heading_list):
    i, j = fig_idx[h][0], fig_idx[h][1]
    ax[i, j].set_title(f'{heading} vol')
    ax[i, j].imshow(data[:, :, :, volume_channel_indices[h]].sum(axis=0))

_, ax = plt.subplots(2, 2, figsize=(30, 150))
for h, heading in enumerate(heading_list):
    i, j = fig_idx[h][0], fig_idx[h][1]
    ax[i, j].set_title(f'{heading} speed')
    ax[i, j].imshow(data[:, :, :, speed_channel_indices[h]].sum(axis=0))

base_map = load_h5_file(f'{BASE_FOLDER}/{city}/{city}_static.h5')[0]
plt.title(f'{city} {base_map.shape} map')
plt.imshow(base_map, cmap='gray_r', vmin=0, vmax=255)

### Naive avg

city, day, offset = 'MOSCOW', 50, 200
train_files = sorted(glob.glob(f'{BASE_FOLDER}/{city}/training/*8ch.h5', recursive=True))
date_first, date_last = train_files[0].split('/')[4].split('_')[0], train_files[-1].split('/')[4].split('_')[0]
print(f'{len(train_files)} files for {city} from {date_first} to {date_last}')
print(f'Selected file {day}/{len(train_files)} for test data starting at time {offset}/288')

data = load_h5_file(train_files[day])
test_data, y_true = prepare_test(data, offset=offset)
print(f"Test data as {type(test_data)} with {test_data.dtype} and shape {test_data.shape}")
print(f"Test data min={np.min(test_data)}, max={np.max(test_data)} and mean={np.mean(test_data)}")
print(f"Ground truth min={np.min(y_true)}, max={np.max(y_true)} and mean={np.mean(y_true)}")

def naive_average(x: np.ndarray):
    return torch.squeeze(NaiveAverage().forward(torch.unsqueeze(torch.from_numpy(x).float(), dim=0)), dim=0).numpy()

def plot_pred(test_data: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray):
    """plots sum over all channels"""
    for pred_idx in range(0, 6):
        _, axs = plt.subplots(1, 3, figsize=(30, 150))
        axs[0].set_title(f"y_true {pred_idx}")
        axs[0].imshow(y_true[pred_idx].sum(-1))
        axs[1].set_title(f"y_pred {pred_idx}")
        axs[1].imshow(y_pred[pred_idx].sum(-1))
        axs[2].set_title(f"sq pred error {pred_idx}")
        axs[2].imshow(((y_true[pred_idx] - y_pred[pred_idx]) ** 2).sum(-1))

y_pred = naive_average(test_data)
print(f"MSE for predicted 6 frames: {round(mse(y_pred, y_true).item(), 4)}")
plot_pred(test_data, y_true, y_pred)
print("Testing this from the console with vi command")
print("Testing this from the console with nano command")
