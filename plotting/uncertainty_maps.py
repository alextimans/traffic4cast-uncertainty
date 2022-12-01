import os
from pathlib import Path

import torch
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap #, LinearSegmentedColormap

from util.h5_util import load_h5_file
from metrics.mse import mse_samples


def normalize(v, new_min = 0, new_max = 1):
    v_min, v_max = torch.min(v), torch.max(v)
    return (v - v_min)/(v_max - v_min) * (new_max - new_min) + new_min


def normalize_by(v, norm_max, new_min=0, new_max = 1):
    v_min, v_max = torch.min(v), norm_max
    return (v - v_min)/(v_max - v_min) * (new_max - new_min) + new_min


def save_fig(fig_path: str, city: str, uq_method: str, filename: str):
    file_path = Path(os.path.join(fig_path, city))
    file_path.mkdir(exist_ok=True, parents=True)
    plt.savefig(os.path.join(file_path, f"{uq_method}_{filename}.png"),
                dpi=300, format="png", bbox_inches='tight')
    print(f"Saved figure to {file_path}.")


def make_cmap(cmap_str: str = "OrRd"):
    cmap = plt.get_cmap(cmap_str)
    my_cmap = cmap(np.arange(cmap.N))
    # add transparency gradient only for first half of color bar
    my_cmap[:, -1] = np.concatenate([np.linspace(0, 1, int(cmap.N/2)),
                                     np.ones(int(cmap.N/2))])
    my_cmap = ListedColormap(my_cmap)
    return my_cmap


def make_cmap_light(cmap_str: str = "OrRd"):
    cmap = plt.get_cmap(cmap_str)
    my_cmap = cmap(np.arange(cmap.N))
    # add transparency gradient for full color bar
    my_cmap[:, -1] = np.linspace(0, 1, cmap.N)
    my_cmap = ListedColormap(my_cmap)
    return my_cmap


# =============================================================================
# STATIC VALUES
# =============================================================================
map_path = "./data/raw"
fig_path = "./figures/test_100/test_100_1_report"
base_path = "./results/test_100/test_100_1/h5_files"

city = "MOSCOW" # ANTWERP, BANGKOK, BARCELONA, MOSCOW
uq_method = "combo" # bnorm, combo, point, ensemble, patches, tta
m = "False" # Masked values: True, False
ch = "speed" # Channel group: speed, vol

sc_idx = {"mean_gt": 0, "mean_pred": 1, "mean_unc": 2, "mean_mse": 3,
             "std_gt": 4, "std_pred": 5, "std_unc": 6, "std_mse": 7,
             "pi_width": 8, "cover": 9, "ence": 10, "cov": 11,
             "corr": 12, "sp_corr": 13}
m_idx = {"True": "_mask", "False": ""}
ch_idx = {"vol":[0, 2, 4, 6], "speed":[1, 3, 5, 7]}

""" Data shapes
pred: (samp, 3, H, W, Ch); 3: GT, pred, unc
pred combo: (samp, 4, H, W, Ch); 4: GT, pred, epi, alea
pi: (samp, 2, H, W, Ch); 2: lower bound, upper bound
scores: (14, H, W, Ch); 14: metrics in order of sc_idx
scores_mask: (14, x, 1, Ch); x: subset masking values e.g. 130966
"""
""" City crops
r: center pixel height (y-axis T to B)
c: center pixel width (x-axis L to R)
num: display size r-num:r+num, c-num:c+num

ANTWERP
r=235; c=242; num=20 #highway loop
r=212; c=178; num=20 #highway loop
r=197; c=220; num=20 #highway loop
r=270; c=230; num=20 #bridge
r=240; c=220; num=20 #city center

BANGKOK
r=285; c=240; num=20 #city section
r=265; c=100; num=20 #outside highway junction
r=300; c=293; num=20 #outside highway junction
r=295; c=190; num=30 #city center
r=328; c=200; num=20 #bridge

BARCELONA
r=405; c=172; num=20 #city junction
r=368; c=105; num=20 #junctions parallel
r=350; c=255; num=20 #city center grid
r=195; c=307; num=30 #outside city hub 

MOSCOW
r=174; c=247; num=30 #bridge 
r=152; c=60; num=30 #Losinoostrovksy district
r=358; c=226; num=30 #highway w merge
r=230; c=190; num=30 #city center
"""


# =============================================================================
# All key metrics, fixed ch, spatial map whole city
# =============================================================================
sc_str = ["mean_gt", "mean_mse", "mean_unc", "pi_width", "ence", "sp_corr"]
titles = ["Ground truth", "MSE", "Uncertainty", "MPIW", "ENCE", r"Correlation $\rho_{sp}$"]

for uq_method in ["point", "combo"]:
    scores = load_h5_file(os.path.join(base_path, city, f"scores_{uq_method}{m_idx[m]}.h5"))
    
    for ch in list(ch_idx.keys()):
        dat = torch.empty((len(sc_str), 495, 436))
        for i, s in enumerate(sc_str):
            d = torch.mean(scores[sc_idx[s], :, :, ch_idx[ch]], dim=-1)
            if s in ["sp_corr", "corr"]:
                dat[i, ...] = d
            else:
                dat[i, ...] = normalize(torch.log(d.clamp(min=1e-5)))
        
        my_cmap = "coolwarm"
        fig, axes = plt.subplots(1, len(sc_str), figsize=(10, 2.4))
        fig.subplots_adjust(wspace=0.1)
        for i, ax in enumerate(axes.flat):
            ax.set_title(titles[i], fontsize="small")
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            im = ax.imshow(dat[i, ...], cmap=my_cmap, vmin=-1, vmax=1)
        fig.colorbar(im, ax=axes.ravel().tolist(), location="right", aspect=20, pad=0.01, shrink=0.62)
        fig.suptitle(f"{city.capitalize()}, UQ: {uq_method}, Ch: {ch}, " +
                     "Mean log-norm metrics", fontsize="small")
        
        save_fig(fig_path, city, uq_method, f"metrics_{ch}")


# =============================================================================
# Mean MSE vs. uncertainty, fixed ch, spatial map whole city
# =============================================================================
for uq_method in ["point", "tta", "ensemble", "combo"]:
    scores = load_h5_file(os.path.join(base_path, city, f"scores_{uq_method}{m_idx[m]}.h5"))

    for ch in list(ch_idx.keys()):
        unc = torch.mean(scores[sc_idx["mean_unc"], :, :, ch_idx[ch]], dim=-1)#[150:300, 150:300] #[150:300, 150:300], [220:250, 230:260]
        mse = torch.mean(scores[sc_idx["mean_mse"], :, :, ch_idx[ch]], dim=-1)#[150:300, 150:300]

        my_cmap = make_cmap()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(4, 2.4))
        fig.subplots_adjust(wspace=0.1)
        ax1.set_title("MSE", fontsize="small")
        ax1.get_yaxis().set_visible(False)
        ax1.get_xaxis().set_visible(False)
        ax2.set_title("Uncertainty", fontsize="small")
        ax2.get_yaxis().set_visible(False)
        ax2.get_xaxis().set_visible(False)
        
        im1 = ax1.imshow(normalize(torch.log(mse.clamp(min=1e-5))), cmap=my_cmap, vmin=0, vmax=1)
        im2 = ax2.imshow(normalize(torch.log(unc)), cmap=my_cmap, vmin=0, vmax=1)
        fig.colorbar(im1, ax=[ax1, ax2], location="right", aspect=20, pad=0.03, shrink=0.75)
        fig.suptitle(f"{city.capitalize()}, UQ: {uq_method}, Ch: {ch}, " +
                     "Mean log-norm MSE vs. unc", fontsize="small")
        
        save_fig(fig_path, city, uq_method, f"mse_unc_{ch}")
uq_method="combo"

# =============================================================================
# Mean MSE vs. unc decomp. for "combo", fixed ch, spatial map whole city
# =============================================================================
pred = load_h5_file(os.path.join(base_path, city, "pred_combo.h5"))
data = torch.stack((mse_samples(pred[:, :2, ...]),
                    torch.mean(pred[:, 2, ...] + pred[:, 3, ...], dim=0),
                    torch.mean(pred[:, 2, ...], dim=0),
                    torch.mean(pred[:, 3, ...], dim=0)))

for ch in list(ch_idx.keys()):
    mse = torch.mean(data[0, :, :, ch_idx[ch]], dim=-1)
    pred = torch.mean(data[1, :, :, ch_idx[ch]], dim=-1)
    epi = torch.mean(data[2, :, :, ch_idx[ch]], dim=-1)
    alea = torch.mean(data[3, :, :, ch_idx[ch]], dim=-1)

    my_cmap = make_cmap()
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(8, 2.4))
    fig.subplots_adjust(wspace=0.1)
    ax1.set_title("MSE", fontsize="small")
    ax1.get_yaxis().set_visible(False)
    ax1.get_xaxis().set_visible(False)
    ax2.set_title("Predictive unc.", fontsize="small")
    ax2.get_yaxis().set_visible(False)
    ax2.get_xaxis().set_visible(False)
    ax3.set_title("Epistemic unc.", fontsize="small")
    ax3.get_yaxis().set_visible(False)
    ax3.get_xaxis().set_visible(False)
    ax4.set_title("Aleatoric unc.", fontsize="small")
    ax4.get_yaxis().set_visible(False)
    ax4.get_xaxis().set_visible(False)

    pred_max = torch.log(pred).max()

    im1 = ax1.imshow(normalize(torch.log(mse.clamp(min=1e-5))), cmap=my_cmap, vmin=0, vmax=1)
    im2 = ax2.imshow(normalize(torch.log(pred)), cmap=my_cmap, vmin=0, vmax=1)
    im3 = ax3.imshow(normalize_by(torch.log(epi), pred_max), cmap=my_cmap, vmin=0, vmax=1)
    im4 = ax4.imshow(normalize_by(torch.log(alea), pred_max), cmap=my_cmap, vmin=0, vmax=1)
    fig.colorbar(im1, ax=[ax1, ax2, ax3, ax4], location="right", aspect=20, pad=0.015, shrink=0.75)
    fig.suptitle(f"{city.capitalize()}, UQ: combo, Ch: {ch}, " +
                 "Mean log-norm MSE vs. unc decomposition", fontsize="small")

    save_fig(fig_path, city, "combo", f"mse_unc_decomp_{ch}")


# =============================================================================
# Mean MSE vs. unc decomp for "combo", fixed ch, spatial map city crop
# =============================================================================
res_map = load_h5_file(os.path.join(map_path, city, f"{city}_map_high_res.h5"))
pred = load_h5_file(os.path.join(base_path, city, "pred_combo.h5"))

for ch in list(ch_idx.keys()):

    dat = torch.stack((torch.mean(mse_samples(pred[:, :2, :, :, ch_idx[ch]]), dim=-1).clamp(min=1e-5),
                        torch.mean(pred[:, 2, :, :, ch_idx[ch]] + pred[:, 3, :, :, ch_idx[ch]], dim=(0,-1)),
                        torch.mean(pred[:, 2, :, :, ch_idx[ch]], dim=(0, -1)),
                        torch.mean(pred[:, 3, :, :, ch_idx[ch]], dim=(0, -1))))
    
    r=240; c=220; num=20 #city center

    my_cmap = make_cmap()
    fig, axes = plt.subplots(1, 4, figsize=(8, 2.6))
    fig.subplots_adjust(wspace=0.1)
    labs = ["MSE", "Predictive unc.", "Epistemic unc.", "Aleatoric unc."]
    pred_max = torch.log(dat[1, r-num:r+num, c-num:c+num]).max()

    for o, ax in enumerate(axes.flat):

        if o == 0:
            data = normalize(torch.log(dat[o, ...]))
        else:
            data = normalize_by(torch.log(dat[o, ...]), pred_max)

        blup = torch.empty((2*10*num, 2*10*num))
        for i in range(-10*num, 10*num):
            for j in range(-10*num, 10*num):
                if r+i//10 >= 495 or c+j//10 >= 436:
                    continue
                blup[i+10*num, j+10*num] = data[r+i//10, c+j//10]

        # im = ax.imshow(blup, cmap=my_cmap, vmin=0, vmax=1)#, alpha=0.5)
        # ax.imshow(res_map[10*(r-num):10*(r+num), 10*(c-num):10*(c+num)], cmap='gray_r', alpha=0.55, vmin=0, vmax=255)
        ax.imshow(res_map[10*(r-num):10*(r+num), 10*(c-num):10*(c+num)], cmap='gray_r', vmin=0, vmax=255)
        im = ax.imshow(blup, cmap=my_cmap, vmin=0, vmax=1, alpha=0.55)
        ax.set_title(f"{labs[o]}", fontsize="small")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    fig.colorbar(mappable=im, ax=axes.ravel().tolist(), location="right", aspect=20, pad=0.015, shrink=0.61)
    fig.suptitle(f"{city.capitalize()}, UQ: combo, Ch: {ch}, " +
                 f"Crop ({r-num}:{r+num}, {c-num}:{c+num}),\n" +
                 "Mean log-norm MSE vs. unc decomposition", fontsize="small")
    
    save_fig(fig_path, city, "combo", f"mse_unc_decomp_{ch}_crop_{r}_{c}")


# =============================================================================
# Mean X, vol and speed, spatial map whole city
# =============================================================================
dat_str = "std_gt" #"sp_corr"
lab = "GT std." #r"Corr. $\rho_{sp}$"

scores = load_h5_file(os.path.join(base_path, city, f"scores_{uq_method}{m_idx[m]}.h5"))
dat_v = torch.mean(scores[sc_idx[dat_str], :, :, ch_idx["vol"]], dim=-1)
dat_s = torch.mean(scores[sc_idx[dat_str], :, :, ch_idx["speed"]], dim=-1)

my_cmap = make_cmap() # "rainbow" "coolwarm"
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(4, 2.4))
fig.subplots_adjust(wspace=0.1)
ax1.set_title(f"{lab} volume", fontsize="small")
ax1.get_yaxis().set_visible(False)
ax1.get_xaxis().set_visible(False)
ax2.set_title(f"{lab} speed", fontsize="small")
ax2.get_yaxis().set_visible(False)
ax2.get_xaxis().set_visible(False)

# im1 = ax1.imshow(dat_v.clamp(min=1e-5), cmap=my_cmap, vmin=-1, vmax=1)
# im2 = ax2.imshow(dat_s.clamp(min=1e-5), cmap=my_cmap, vmin=-1, vmax=1)
im1 = ax1.imshow(normalize(torch.log(dat_v.clamp(min=1e-5))), cmap=my_cmap, vmin=0, vmax=1)
im2 = ax2.imshow(normalize(torch.log(dat_s.clamp(min=1e-5))), cmap=my_cmap, vmin=0, vmax=1)
fig.colorbar(im1, ax=[ax1, ax2], location="right", aspect=20, pad=0.03, shrink=0.75)
fig.suptitle(f"{city.capitalize()}, UQ: {uq_method}, " +
             f"Mean log-norm {lab} by channels", fontsize="small")

save_fig(fig_path, city, uq_method, f"{dat_str}_bych")


# =============================================================================
# Mean X, vol and speed, spatial map city crop
# =============================================================================
dat_str = "mean_unc"
lab = "Unc"

res_map = load_h5_file(os.path.join(map_path, city, f"{city}_map_high_res.h5"))
scores = load_h5_file(os.path.join(base_path, city, f"scores_{uq_method}{m_idx[m]}.h5"))

r=358; c=226; num=30 #highway w merge

my_cmap = make_cmap()
fig, axes = plt.subplots(1, 2, figsize=(4, 2.6))
fig.subplots_adjust(wspace=0.1)
labs = ["vol", "speed"]
for o, ax in enumerate(axes.flat):

    data = normalize(torch.log(torch.mean(dat[sc_idx[dat_str], :, :, ch_idx[labs[o]]], dim=-1).clamp(min=1e-5)))

    blup = torch.empty((2*10*num, 2*10*num))
    for i in range(-10*num, 10*num):
        for j in range(-10*num, 10*num):
            if r+i//10 >= 495 or c+j//10 >= 436:
                continue
            blup[i+10*num, j+10*num] = data[r+i//10, c+j//10]

    # im = ax.imshow(blup, cmap=my_cmap, vmin=0, vmax=1)#, alpha=0.5)
    # ax.imshow(res_map[10*(r-num):10*(r+num), 10*(c-num):10*(c+num)], cmap='gray_r', alpha=0.55, vmin=0, vmax=255)
    ax.imshow(res_map[10*(r-num):10*(r+num), 10*(c-num):10*(c+num)], cmap='gray_r', vmin=0, vmax=255)
    im = ax.imshow(blup, cmap=my_cmap, vmin=0, vmax=1, alpha=0.55)
    ax.set_title(f"{lab} {labs[o]}", fontsize="small")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
fig.colorbar(mappable=im, ax=axes.ravel().tolist(), location="right", aspect=20, pad=0.03, shrink=0.63)
fig.suptitle(f"{city.capitalize()}, UQ: {uq_method}, " +
             f"Crop ({r-num}:{r+num}, {c-num}:{c+num}),\n" +
             f"Mean log-norm {lab} by channels", fontsize="small")

save_fig(fig_path, city, uq_method, f"{dat_str}_bych_crop_{r}_{c}")


# =============================================================================
# Mean X, fixed ch, spatial map city crop multiple at once
# =============================================================================
dat_str = "mean_unc"
lab = "Unc."

res_map = load_h5_file(os.path.join(map_path, city, f"{city}_map_high_res.h5"))
scores = load_h5_file(os.path.join(base_path, city, f"scores_{uq_method}{m_idx[m]}.h5"))
dat = torch.log(torch.mean(scores[sc_idx[dat_str], :, :, ch_idx[ch]], dim=-1).clamp(min=1e-5))
# dat = normalize(torch.log(dat.clamp(min=1e-5)))

r_list, c_list, num = [405, 368, 350], [172, 105, 255], 20

norm_max = torch.max(torch.tensor([
    dat[r_list[0]-num:r_list[0]+num, c_list[0]-num:c_list[0]+num].max(),
    dat[r_list[1]-num:r_list[1]+num, c_list[1]-num:c_list[1]+num].max(),
    dat[r_list[2]-num:r_list[2]+num, c_list[2]-num:c_list[2]+num].max()]))
dat = normalize_by(dat, norm_max)

my_cmap = make_cmap()
fig, axes = plt.subplots(1, 3, figsize=(8, 2.8))
fig.subplots_adjust(wspace=0.1)
for o, ax in enumerate(axes.flat):
    r, c = r_list[o], c_list[o]

    blup = torch.empty((2*10*num, 2*10*num))
    for i in range(-10*num, 10*num):
        for j in range(-10*num, 10*num):
            if r+i//10 >= 495 or c+j//10 >= 436:
                continue
            blup[i+10*num, j+10*num] = dat[r+i//10, c+j//10]

    # im = ax.imshow(blup, cmap=my_cmap, vmin=0, vmax=1)#, alpha=0.5)
    # ax.imshow(res_map[10*(r-num):10*(r+num), 10*(c-num):10*(c+num)], cmap='gray_r', alpha=0.55, vmin=0, vmax=255)
    ax.imshow(res_map[10*(r-num):10*(r+num), 10*(c-num):10*(c+num)], cmap='gray_r', vmin=0, vmax=255)
    im = ax.imshow(blup, cmap=my_cmap, vmin=0, vmax=1, alpha=0.55)
    ax.set_title(f"{lab}, ({r-num}:{r+num}, {c-num}:{c+num})", fontsize="small")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
fig.colorbar(mappable=im, ax=axes.ravel().tolist(), location="right", aspect=20, pad=0.02, shrink=0.76)
fig.suptitle(f"{city.capitalize()}, UQ: {uq_method},\n" +
             f"Mean log-norm {lab} for {ch} channels", fontsize="small")

save_fig(fig_path, city, uq_method, f"{dat_str}_{ch}_crops")


# =============================================================================
# All unc methods, fixed ch, spatial map whole city
# =============================================================================
uq_str = ["point", "combo", "ensemble", "bnorm", "tta", "patches"]
titles = ["CUB (P)", "TTA + Ens (P)", "Ens (E)", "MCBN (E)", "TTA (A)", "Patches (A)"]

dat = torch.empty((len(uq_str), 495, 436))

for ch in list(ch_idx.keys()):
    for i, s in enumerate(uq_str):
        scores = load_h5_file(os.path.join(base_path, city, f"scores_{uq_str[i]}{m_idx[m]}.h5"))
        d = torch.mean(scores[sc_idx["mean_unc"], :, :, ch_idx[ch]], dim=-1)
        dat[i, ...] = torch.log(d)
    norm_max = torch.max(torch.tensor([dat[0,...].quantile(0.99), dat[1,...].quantile(0.99)]))

    my_cmap = make_cmap()
    fig, axes = plt.subplots(1, len(uq_str), figsize=(10, 2.4))
    fig.subplots_adjust(wspace=0.1)
    for i, ax in enumerate(axes.flat):
        ax.set_title(titles[i], fontsize="small")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        im = ax.imshow(normalize_by(dat[i, ...], norm_max), cmap=my_cmap, vmin=0, vmax=1)
    fig.colorbar(im, ax=axes.ravel().tolist(), location="right", aspect=20, pad=0.01, shrink=0.63)
    fig.suptitle(f"{city.capitalize()}, Ch: {ch}, " +
                 "Mean log-norm uncertainties", fontsize="small")
    
    save_fig(fig_path, city, "all", f"unc_{ch}")


# =============================================================================
# All unc methods, fixed ch, spatial map city crop
# =============================================================================
uq_str = ["point", "combo", "ensemble", "bnorm", "tta", "patches"]
titles = ["CUB (P)", "TTA + Ens (P)", "Ens (E)", "MCBN (E)", "TTA (A)", "Patches (A)"]

r=235; c=242; num=20 #highway loop

res_map = load_h5_file(os.path.join(map_path, city, f"{city}_map_high_res.h5"))
dat = torch.empty((len(uq_str), 495, 436))

for ch in list(ch_idx.keys()):
    for i, s in enumerate(uq_str):
        scores = load_h5_file(os.path.join(base_path, city, f"scores_{uq_str[i]}{m_idx[m]}.h5"))
        d = torch.mean(scores[sc_idx["mean_unc"], :, :, ch_idx[ch]], dim=-1)
        dat[i, ...] = torch.log(d)
    norm_max = torch.max(torch.tensor([dat[0, r-num:r+num, c-num:c+num].quantile(0.99),
                                       dat[1, r-num:r+num, c-num:c+num].quantile(0.99)]))

    my_cmap = make_cmap()
    fig, axes = plt.subplots(1, len(uq_str), figsize=(10, 2.4))
    fig.subplots_adjust(wspace=0.1)
    for o, ax in enumerate(axes.flat):

        blup = torch.empty((2*10*num, 2*10*num))
        for i in range(-10*num, 10*num):
            for j in range(-10*num, 10*num):
                if r+i//10 >= 495 or c+j//10 >= 436:
                    continue
                blup[i+10*num, j+10*num] = dat[o, r+i//10, c+j//10]
    
        # im = ax.imshow(blup, cmap=my_cmap, vmin=0, vmax=1)#, alpha=0.5)
        # ax.imshow(res_map[10*(r-num):10*(r+num), 10*(c-num):10*(c+num)], cmap='gray_r', alpha=0.55, vmin=0, vmax=255)
        ax.imshow(res_map[10*(r-num):10*(r+num), 10*(c-num):10*(c+num)], cmap='gray_r', vmin=0, vmax=255)
        im = ax.imshow(normalize_by(blup, norm_max), cmap=my_cmap, vmin=0, vmax=1, alpha=0.55)
        ax.set_title(titles[o], fontsize="small")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    fig.colorbar(im, ax=axes.ravel().tolist(), location="right", aspect=20, pad=0.01, shrink=0.55)
    fig.suptitle(f"{city.capitalize()}, Ch: {ch}, " +
                 f"Crop ({r-num}:{r+num}, {c-num}:{c+num}), " +
                 "Mean log-norm uncertainties", fontsize="small")
    
    save_fig(fig_path, city, "all", f"unc_{ch}_crop_{r}_{c}")


# =============================================================================
# Mean Correlation, filtered by non-zero GT, fixed ch, spatial map whole city
# =============================================================================
scores = load_h5_file(os.path.join(base_path, city, f"scores_{uq_method}{m_idx[m]}.h5"))
gt = scores[sc_idx["mean_gt"], :, :, ch_idx["vol"]]
gt_zero = gt.sum(dim=-1) <= 0
# gt_nonzero = gt.sum(dim=-1) > 0

cbar_kws={"location": "right", "aspect": 40, "pad": 0.02}

for ch in list(ch_idx.keys()):
    corr = torch.mean(scores[sc_idx["sp_corr"], :, :, ch_idx[ch]], dim=-1)
    ax = sns.heatmap(corr.numpy(), mask=gt_zero.numpy(), cmap="coolwarm",
                     vmin=-1, vmax=1, cbar_kws=cbar_kws, edgecolors="black")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_title(f"{city.capitalize()}, UQ: {uq_method}, Ch: {ch}\n " +
                 r"Correlation $\rho_{sp}$ for non-zero ground truths", fontsize="small")
    for _, spine in ax.spines.items():
        spine.set_visible(True)

    save_fig(fig_path, city, uq_method, f"corr_masked_{ch}")
    plt.clf()


# =============================================================================
# Corr histograms zero vs. non-zero GT, fixed ch
# =============================================================================
for uq_method in ["point", "combo"]:
    scores = load_h5_file(os.path.join(base_path, city, f"scores_{uq_method}{m_idx[m]}.h5"))
    gt = scores[sc_idx["mean_gt"], :, :, ch_idx["vol"]]
    gt_zero = gt.sum(dim=-1) <= 0
    gt_nonzero = gt.sum(dim=-1) > 0
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 2.4))
    # fig.subplots_adjust(wspace=0.1)
    labs = ["Volume", "Speed"]
    
    for i, ax in enumerate(axes.flat):
        corr = torch.mean(scores[sc_idx["sp_corr"], :, :, list(ch_idx.values())[i]], dim=-1)
        m_zero, m_nonzero = corr[gt_zero].mean().item(), corr[gt_nonzero].mean().item()
        ax.hist(corr[gt_zero].numpy(), bins=20, alpha=0.6, range=(-1,1),
                # color="blue", label=r"Zero GT, $\bar{\rho}_{sp}$ = " + f"{m_zero:.2f}", density=True)
                color="blue", label=r"Zero ground truth", density=True)
        ax.hist(corr[gt_nonzero].numpy(), bins=20, alpha=0.6, range=(-1,1),
                # color="red", label=r"Non-zero GT, $\bar{\rho}_{sp}$ = " + f"{m_nonzero:.2f}", density=True)
                color="red", label=r"Non-zero ground truth", density=True)
        ax.legend(loc="upper left", fontsize="small") # large
        ax.set_ylabel("Density")
        ax.set_xlabel(r"Correlation $\rho_{sp}$")
        ax.set_title(labs[i], fontsize="small")
        # ax.grid()
    fig.suptitle(f"{city.capitalize()}, UQ: {uq_method}, Corr histograms by GT",
                 y=1.1, x=0.5, fontsize="small")
    
    save_fig(fig_path, city, uq_method, "corr_hist")


# =============================================================================
# Mean PI widths vs. coverage, fixed ch
# =============================================================================
scores = load_h5_file(os.path.join(base_path, city, f"scores_{uq_method}{m_idx[m]}.h5"))

fig, axes = plt.subplots(1, 2, figsize=(12, 2.4))
labs = ["Volume", "Speed"]

for i, ax in enumerate(axes.flat):
    cov = torch.mean(scores[sc_idx["cover"], :, :, list(ch_idx.values())[i]], dim=-1).flatten()
    pi = torch.mean(scores[sc_idx["pi_width"], :, :, list(ch_idx.values())[i]], dim=-1).flatten()

    bins = 30
    bin_val = torch.empty(size=(bins, 2))
    samp_per_bin = int(pi.shape[0] / bins)
    sort_idx = torch.sort(pi, descending=False)[1]

    for cbin in range(bins):
        idx = sort_idx[(cbin * samp_per_bin):(cbin * samp_per_bin + samp_per_bin)]
        bin_val[cbin, 0] = torch.mean(pi[idx])
        bin_val[cbin, 1] = torch.mean(cov[idx])

    ax.plot(bin_val[:, 0], bin_val[:, 1], color="red", label="Emp. coverage")
    ax.set_xscale("log")
    ax.set_ylim(0.85, 0.95)
    ax.set_xlabel("Binned MPIW (log scale)")
    ax.set_ylabel("Coverage level")
    ax.set_title(labs[i], fontsize="small")
    ax.axhline(y=0.9, color="grey", ls=":", label="Nominal coverage")
    ax.legend(loc="upper left", fontsize="small")
fig.suptitle(f"{city.capitalize()}, UQ: {uq_method}, " +
             "Emp. and nom. cover for binned PI widths",
             y=1.1, x=0.5, fontsize="small")

save_fig(fig_path, city, uq_method, "pi_width_cover")


# =============================================================================
# Hist empirical coverage vs. beta distr, fixed ch
# =============================================================================
from scipy.stats import beta
n = 100; alpha=0.1 # calibration samples, coverage
a, b = n + 1 - np.floor((n+1)*alpha), np.floor((n+1)*alpha) # beta shape params
x = np.linspace(beta.ppf(0, a, b), beta.ppf(1, a, b), 1000)

scores = load_h5_file(os.path.join(base_path, city, f"scores_{uq_method}{m_idx[m]}.h5"))

fig, axes = plt.subplots(1, 2, figsize=(12, 2.4))
labs = ["Volume", "Speed"]

for i, ax in enumerate(axes.flat):
    cov = torch.mean(scores[sc_idx["cover"], :, :, list(ch_idx.values())[i]], dim=-1).flatten()

    ax.hist(cov.numpy(), bins=20, alpha=0.6, range=(0.7, 1),
            color="blue", density=True,
            label=r"Emp. coverage, $\bar{x}$ = " + f"{cov.mean():.2f}")
    ax.plot(x, beta.pdf(x, a, b), color="red", alpha=0.8,
            label = "Nominal Beta fit")
            # label=f"Beta({int(a)},{int(b)})")
    ax.set_xlim(0.75, 1)
    ax.legend(loc="upper left", fontsize="small") # large
    ax.set_ylabel("Density")
    ax.set_xlabel("Coverage level")
    ax.set_title(labs[i], fontsize="small")
    # ax.grid()
fig.suptitle(f"{city.capitalize()}, UQ: {uq_method}, Coverage histogram",
             y=1.1, x=0.5, fontsize="small")

save_fig(fig_path, city, uq_method, "cov_hist")


# =============================================================================
# Zero ground truth percentages per city
# =============================================================================

for city in ["ANTWERP","BANGKOK","BARCELONA","MOSCOW"]:
    scores = load_h5_file(os.path.join(base_path, city, f"scores_{uq_method}{m_idx[m]}.h5"))
    pred = load_h5_file(os.path.join(base_path, city, f"pred_{uq_method}.h5"))

    print("="*4, city, "="*4)
    gt = scores[sc_idx["mean_gt"], :, :, ch_idx["vol"]]
    gt_zero = gt.sum(dim=-1) <= 0
    print(f"GT Zero: {(gt_zero.sum()/(495*436))*100:.3f} %")
    # Sanity check
    gt_zero2 = (pred[:, 0, :, :, [0, 2, 4, 6]].sum(dim=(0, -1)) <= 0)
    print(f"GT Zero 2: {(gt_zero2.sum()/(495*436))*100:.3f} %")
