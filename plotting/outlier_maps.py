import os
from pathlib import Path

import torch
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap #, LinearSegmentedColormap
from matplotlib.animation import FuncAnimation #, PillowWriter

from util.h5_util import load_h5_file


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


def outlier_stats(out):
    samp, p_i, p_j, _ = tuple(out.shape)
    tot, pix_tot = samp * p_i * p_j, p_i * p_j

    print("### Outlier stats ###")

    ov, ov_pct = out[..., 0].sum(), out[..., 0].sum()/tot
    print(f"Total outliers by vol ch: {ov}/({samp}*{p_i}*{p_j}) or {(ov_pct*100):.2f}%.")

    os, os_pct = out[..., 1].sum(), out[..., 1].sum()/tot
    print(f"Total outliers by speed ch: {os}/({samp}*{p_i}*{p_j}) or {(os_pct*100):.2f}%.")

    op, op_pct = out[..., 2].sum(), out[..., 2].sum()/tot
    print(f"Total outliers by pixel: {op}/({samp}*{p_i}*{p_j}) or {(op_pct*100):.2f}%.")

    om = out[..., 2].sum(dim=0).max() # max outlier counts across sample dim for pixels
    omc = (out[..., 2].sum(dim=0) == om).sum()
    omc_pct = omc / tot
    print(f"""Pixels with max. outlier count by sample: {omc} pixels or
                 {(omc_pct*100):.2f}% with {om}/{samp} outliers.""")

    osamp = out[..., 2].sum(dim=(1,2)).to(torch.float32)
    osamp_m, osamp_std = int(osamp.mean().ceil()), int(osamp.std().ceil())
    osamp_pct_m, osamp_pct_std = osamp_m / pix_tot, osamp_std / pix_tot
    print(f"""Avg. pixel outlier count by sample: {osamp_m} +/- {osamp_std} of ({p_i}*{p_j})
                 or {(osamp_pct_m*100):.2f} +/- {(osamp_pct_std*100):.2f}%.""")

    osmax, osmin = osamp.argmax(), osamp.argmin()
    osma, osma_pct = int(osamp[osmax].item()), osamp[osmax] / pix_tot
    osmi, osmi_pct = int(osamp[osmin].item()), osamp[osmin] / pix_tot
    print(f"""Sample with most pixel outliers: test sample {osmax.item()}
                 with {osma}/({p_i}*{p_j}) outliers or {(osma_pct*100):.2f}%.""")
    print(f"""Sample with least pixel outliers: test sample {osmin.item()}
                 with {osmi}/({p_i}*{p_j}) outliers or {(osmi_pct*100):.2f}%.""")


# =============================================================================
# STATIC VALUES
# =============================================================================
map_path = "./data/raw"
fig_path = "./figures/out_morning/report" # out_morning, out_evening
base_path = "./results/out_morning"

city = "BANGKOK" # BANGKOK, BARCELONA, MOSCOW
uq_method = "ensemble"

out_b = {"ob01": 0.1, "ob025": 0.25, "ob05": 0.5, "ob075": 0.75, "ob1": 1, "ob2": 2}
ob = "ob01"


# =============================================================================
# Outliers whole city (vol, speed, pixel); mean over samp, fixed samp
# =============================================================================
out = load_h5_file(os.path.join(base_path, city, f"out_{ob}_{uq_method}.h5"))
base_map = load_h5_file(os.path.join(map_path, city, f"{city}_static.h5"))[0]

# Mean outlier percentage per pixel across samp dim
my_cmap = make_cmap()
fig, axes = plt.subplots(1, 3, figsize=(6, 2.6))
fig.subplots_adjust(wspace=0.1)
lab = ["Volume", "Speed", "Pixel"]
for i, ax in enumerate(axes.flat):
    data = out[..., i].mean(dim=0)
    ax.imshow(base_map, cmap="gray_r", vmin=0, vmax=255)
    im = ax.imshow(data, cmap=my_cmap)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_title(f"{lab[i]}", fontsize="small")
fig.suptitle(f"{city.capitalize()}, UQ: {uq_method}, OB: {out_b[ob]}%, " +
             "Mean outlier perc over test samp", fontsize="small")
fig.colorbar(mappable=im, ax=axes.ravel().tolist(), location="right", aspect=20, pad=0.02, shrink=0.7)

save_fig(fig_path, city, uq_method, f"mean_samp_{ob}")

# Outlier per pixel for fixed samp idx
samp = 44 # fixed
cmap = ListedColormap(["None", "red"])
# cmap = mpl.cm.get_cmap("OrRd", 2)
fig, axes = plt.subplots(1, 3, figsize=(6, 2.6))
fig.subplots_adjust(wspace=0.1)
lab = ["Volume", "Speed", "Pixel"]
for i, ax in enumerate(axes.flat):
    data = out[samp, :, :, i]
    ax.imshow(base_map, cmap="gray_r", vmin=0, vmax=255)
    ax.imshow(data, cmap=cmap, alpha=0.5)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_title(f"{lab[i]}", fontsize="small")
fig.suptitle(f"{city.capitalize()}, UQ: {uq_method}, OB: {out_b[ob]}%, " +
             f"Outliers for fixed test samp {samp}", fontsize="small")

save_fig(fig_path, city, uq_method, f"samp_{samp}_{ob}")


# =============================================================================
# Outliers crop city (vol, speed, pixel); mean over samp
# =============================================================================
out = load_h5_file(os.path.join(base_path, city, f"out_{ob}_{uq_method}.h5"))
res_map = load_h5_file(os.path.join(map_path, city, f"{city}_map_high_res.h5"))

# Mean outlier perc across samp dim and fixed city crop
r=210; c=305; num=20 #outside city hub

my_cmap = make_cmap()
fig, axes = plt.subplots(1, 3, figsize=(6, 2.4))
fig.subplots_adjust(wspace=0.1)
lab = ["Volume", "Speed", "Pixel"]
for o, ax in enumerate(axes.flat):

    blup = torch.empty((2*10*num, 2*10*num))
    for i in range(-10*num, 10*num):
        for j in range(-10*num, 10*num):
            if r+i//10 >= 495 or c+j//10 >= 436:
                continue
            # Mean outlier percentage per pixel across samp dim
            blup[i+10*num, j+10*num] = out[:, r+i//10, c+j//10, o].mean(dim=0)

    ax.imshow(res_map[10*(r-num):10*(r+num), 10*(c-num):10*(c+num)], cmap='gray_r', vmin=0, vmax=255)
    im = ax.imshow(blup, cmap=my_cmap, alpha=0.65, vmin=0, vmax=1)
    ax.set_title(f"{lab[o]}", fontsize="small")
    ax.set_xticks(np.arange(-0.5, 2 * (num+1) * 10 - 0.5, 20))
    ax.set_xticklabels(np.arange(c-num, c+num+1, 2))
    ax.set_yticks(np.arange(-0.5, 2 * (num+1) * 10 - 0.5, 20))
    ax.set_yticklabels(np.arange(r-num, r+num+1, 2))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
fig.suptitle(f"{city.capitalize()}, UQ: {uq_method}, OB: {out_b[ob]}%, " +
             f"Crop ({r-num}:{r+num}, {c-num}:{c+num}), " +
             "Mean outlier perc over test samp", fontsize="small")
fig.colorbar(mappable=im, ax=axes.ravel().tolist(), location="right", aspect=20, pad=0.02, shrink=0.65)

save_fig(fig_path, city, uq_method, f"mean_samp_{ob}_crop_{r}_{c}")


# =============================================================================
# Outliers crop city (vol, speed, pixel); fixed samp
# =============================================================================
out = load_h5_file(os.path.join(base_path, city, f"out_{ob}_{uq_method}.h5"))
res_map = load_h5_file(os.path.join(map_path, city, f"{city}_map_high_res.h5"))

# Outlier per pixel for fixed samp idx and fixed city crop
samp = 72
r=152; c=60; num=30 #Losinoostrovksy district

cmap = ListedColormap(["None", "red"])
fig, axes = plt.subplots(1, 3, figsize=(6, 2.4))
fig.subplots_adjust(wspace=0.1)
lab = ["Volume", "Speed", "Pixel"]
for o, ax in enumerate(axes.flat):

    blup = torch.empty((2*10*num, 2*10*num))
    for i in range(-10*num, 10*num):
        for j in range(-10*num, 10*num):
            if r+i//10 >= 495 or c+j//10 >= 436:
                continue
            # Outlier per pixel for fixed samp idx
            blup[i+10*num, j+10*num] = out[samp, r+i//10, c+j//10, o]

    im = ax.imshow(blup, cmap=cmap)
    ax.imshow(res_map[10*(r-num):10*(r+num), 10*(c-num):10*(c+num)], cmap='gray_r', alpha=0.8, vmin=0, vmax=255)
    ax.set_title(f"{lab[o]}", fontsize="small")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
fig.suptitle(f"{city.capitalize()}, UQ: {uq_method}, OB: {out_b[ob]}%, " +
             f"Crop ({r-num}:{r+num}, {c-num}:{c+num}), " +
             f"Outliers for fixed test samp {samp}", fontsize="small")

save_fig(fig_path, city, uq_method, f"samp_{samp}_{ob}_crop_{r}_{c}")


# =============================================================================
# ANIMATION: Outliers whole city (vol, speed, pixel); over samp dim
# =============================================================================
# takes ~4m to run
out = load_h5_file(os.path.join(base_path, city, f"out_{ob}_{uq_method}.h5"))
base_map = load_h5_file(os.path.join(map_path, city, f"{city}_static.h5"))[0]

cmap = ListedColormap(["None", "red"])
fig, axes = plt.subplots(1, 3, figsize=(6, 2.4))
fig.subplots_adjust(wspace=0.1)
lab = ["Volume", "Speed", "Pixel"]

def update(samp):
    for i, ax in enumerate(axes.flat):
        data = out[samp, :, :, i]
        ax.imshow(base_map, cmap="gray_r", vmin=0, vmax=255)
        ax.imshow(data, cmap=cmap, alpha=0.5)
        ax.set_title(f"out {lab[i]}", fontsize="small")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    fig.suptitle(f"{city.capitalize()}, UQ: {uq_method}, OB: {out_b[ob]}%, " +
                 f"Outliers for test samp {samp}", fontsize="small")

ani = FuncAnimation(fig, update, frames=out.shape[0], interval=200)
path = os.path.join(fig_path, city, f"{uq_method}_by_samp_{ob}.mp4")

# ani.save(path, writer=PillowWriter(fps=3)) # .gif
ani.save(path, fps=3) # .mp4


# =============================================================================
# ANIMATION: Outliers crop city (vol, speed, pixel); over samp dim
# =============================================================================
# takes ~15m to run
out = load_h5_file(os.path.join(base_path, city, f"out_{ob}_{uq_method}.h5"))
res_map = load_h5_file(os.path.join(map_path, city, f"{city}_map_high_res.h5"))

r=152; c=60; num=30 #Losinoostrovksy district

cmap = ListedColormap(["None", "red"])
fig, axes = plt.subplots(1, 3, figsize=(6, 2.4))
fig.subplots_adjust(wspace=0.1)
lab = ["Volume", "Speed", "Pixel"]

def update(samp):
    for o, ax in enumerate(axes.flat):

        blup = torch.empty((2*10*num, 2*10*num))
        for i in range(-10*num, 10*num):
            for j in range(-10*num, 10*num):
                if r+i//10 >= 495 or c+j//10 >= 436:
                    continue
                blup[i+10*num, j+10*num] = out[samp, r+i//10, c+j//10, o]

        ax.imshow(blup, cmap=cmap)
        ax.imshow(res_map[10*(r-num):10*(r+num), 10*(c-num):10*(c+num)],
                  cmap='gray_r', alpha=0.8, vmin=0, vmax=255)
        ax.set_title(f"out {lab[o]}", fontsize="small")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    fig.suptitle(f"{city.capitalize()}, UQ: {uq_method}, OB: {out_b[ob]}%, " +
                 f"Crop ({r-num}:{r+num}, {c-num}:{c+num}), " +
                 f"Outliers for test samp {samp}", fontsize="small")

ani = FuncAnimation(fig, update, frames=out.shape[0], interval=200)
path = os.path.join(fig_path, city, f"{uq_method}_by_samp_{ob}_crop_{r}_{c}.mp4")

# ani.save(path, writer=PillowWriter(fps=3)) # .gif
ani.save(path, fps=3) # .mp4


# =============================================================================
# ANIMATION: Outliers whole city per channel (8); over samp dim
# =============================================================================
# takes ~8m to run
from data.data_layout import channel_labels as ch_lab

pval = load_h5_file(os.path.join(base_path, city, f"pval_{uq_method}.h5"))
base_map = load_h5_file(os.path.join(map_path, city, f"{city}_static.h5"))[0]

out_bound = out_b[ob]/100
lab = ["volume_NW", "volume_NE", "speed_NW", "speed_NE",
       "volume_SW", "volume_SE", "speed_SW", "speed_SE"]
ch_id = [ch_lab.index(ch) for ch in lab] # change order of ch for better readability

cmap = ListedColormap(["None", "red"])
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
fig.subplots_adjust(wspace=0.1)

def update(samp):
    for i, ax in enumerate(axes.flat):
        data = pval[samp, :, :, ch_id[i]] <= out_bound # outliers per channel
        ax.imshow(base_map, cmap="gray_r", vmin=0, vmax=255)
        ax.imshow(data, cmap=cmap, alpha=0.5)
        ax.set_title(f"{lab[i]}", fontsize="small")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    fig.suptitle(f"{city.capitalize()}, UQ: {uq_method}, OB: {out_b[ob]}%, " +
                 f"Outliers for test samp {samp}", fontsize="small")

ani = FuncAnimation(fig, update, frames=pval.shape[0], interval=200)
path = os.path.join(fig_path, city, f"{uq_method}_by_samp_{ob}_per_ch.mp4")

# ani.save(path, writer=PillowWriter(fps=3)) # .gif
ani.save(path, fps=3) # .mp4


# =============================================================================
# Total pixel outlier counts whole city (vol, speed, pixel) vs. samp dim
# =============================================================================
out = load_h5_file(os.path.join(base_path, city, f"out_{ob}_{uq_method}.h5"))

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.subplots_adjust(wspace=0.25)
lab = ["Volume", "Speed", "Pixel"]
for i, ax in enumerate(axes.flat):
    data = (out[..., i].sum(dim=(1,2)) / (495*436))*100
    ax.plot(data, ls="-", lw=0.7, marker=".", color="black", mec="black", mfc="red")
    ax.set_title(f"{lab[i]}", fontsize="small")
    ax.set_ylabel("Outlier share [%]")
    ax.set_xlabel("Test sample")
fig.suptitle(f"{city.capitalize()}, UQ: {uq_method}, OB: {out_b[ob]}%, " +
             "Total outlier count (sum over pixels) vs. test samples", fontsize="small")

save_fig(fig_path, city, uq_method, f"out_counts_{ob}")


# =============================================================================
# P-values & outlier bounds vs. samp dim; fixed pixel
# =============================================================================
from scipy.stats import combine_pvalues

# pred_tr = load_h5_file(os.path.join(base_path, city, f"pred_tr_{uq_method}.h5"))
pval = load_h5_file(os.path.join(base_path, city, f"pval_{uq_method}.h5"))
out = load_h5_file(os.path.join(base_path, city, f"out_{ob}_{uq_method}.h5"))

# find pixel
(out[..., 2].sum(dim=0) == 71).nonzero(as_tuple=False)

pix_h, pix_w = 245, 405 # fixed pixel
samp_c = 90 # sample count, not fixed sample!
pval_pix = pval[:, pix_h, pix_w, :]

agg_pval = torch.empty(size=(samp_c, 2), dtype=torch.float32)
for samp in range(samp_c):
    agg_pval[samp, 0] = 1 - combine_pvalues(pval_pix[samp, [0, 2, 4, 6]].clamp(min=1e-10), method="fisher")[1]
    agg_pval[samp, 1] = 1 - combine_pvalues(pval_pix[samp, [1, 3, 5, 7]].clamp(min=1e-10), method="fisher")[1]

fig, axes = plt.subplots(1, 2, figsize=(10, 3.1))
fig.subplots_adjust(wspace=0.3)
lab = ["Volume", "Speed"]
for i, ax in enumerate(axes.flat):
    data = agg_pval[:, i]
    ax.scatter(range(len(data)), data, label="1 - p-value (Fisher)", marker=".", color="black")
    # ax.plot(data, label="1 - p-value (Fisher)", color="black")
    ax.set_title(f"{lab[i]}", fontsize="small")

    out_bound = 0.01
    ax.axhline(1-out_bound, color="sienna", linestyle=":", label="Outlier Bound 1%")
    ax.text(x=0, y=1-out_bound-1e-3, s=rf"{(agg_pval[:, i] >= 1-out_bound).sum().item()}$\uparrow$",
            color="sienna", fontweight="bold", fontsize="small")

    out_bound = 0.005
    ax.axhline(1-out_bound, color="orange", linestyle=":", label="Outlier Bound 0.5%")
    ax.text(x=0, y=1-out_bound-1e-3, s=rf"{(agg_pval[:, i] >= 1-out_bound).sum().item()}$\uparrow$",
            color="orange", fontweight="bold", fontsize="small")

    out_bound = 0.001
    ax.axhline(1-out_bound, color="red", linestyle=":", label="Outlier Bound 0.1%")
    ax.text(x=0, y=1-out_bound-1e-3, s=rf"{(agg_pval[:, i] >= 1-out_bound).sum().item()}$\uparrow$",
            color="red", fontweight="bold", fontsize="small")

    ax.set_ylabel("1 - p-value (Fisher)")
    ax.set_ylim(0.98, 1.0003)
    ax.set_xlabel("Test sample")
    ax.legend(fontsize="small", loc="lower right")
fig.suptitle(f"{city.capitalize()}, UQ: {uq_method}, Pixel ({pix_h}, {pix_w}), " +
             "P-values (Fisher) & outlier bounds vs. test samples", fontsize="small")

save_fig(fig_path, city, uq_method, f"pval_ob_pix_{pix_h}_{pix_w}")


# =============================================================================
# Total (pixel+samp) outlier counts whole city (vol, speed, pixel) vs. out bounds
# =============================================================================
ob_str = list(out_b.keys())
out_ob = torch.empty(size=(len(ob_str), 3), dtype=torch.float32)

for i, ob in enumerate(ob_str):
    out = load_h5_file(os.path.join(base_path, city, f"out_{ob}_{uq_method}.h5"))
    out_ob[i, 0] = (out[..., 0].sum(dim=(0,1,2))/(90*495*436))*100
    out_ob[i, 1] = (out[..., 1].sum(dim=(0,1,2))/(90*495*436))*100
    out_ob[i, 2] = (out[..., 2].sum(dim=(0,1,2))/(90*495*436))*100
    del out

fig, axes = plt.subplots(1, 3, figsize=(16, 6))
lab = ["Volume", "Speed", "Pixel"]
for i, ax in enumerate(axes.flat):
    data = out_ob[:, i]
    ax.plot(data, ls="-", lw=1, marker="o", color="black", mec="black", mfc="red")
    ax.set_title(f"out {lab[i]}", fontsize="small")
    ax.set_ylabel("Outlier portion (in %)")
    ax.set_xlabel("Outlier bound (in %)")
    ax.set_xticks(ticks=[0, 1, 2, 3, 4, 5], labels=['0.1', '0.25', '0.5', '0.75', '1', '2'])
fig.suptitle(f"{city.capitalize()}, UQ: {uq_method}, " +
             "Perc of total outlier count (sum over pixels + samples) vs. outlier bounds", fontsize="small")

save_fig(fig_path, city, uq_method, "out_perc_vs_ob")


# =============================================================================
# Hist comparison test/train uncertainties; fixed pixel
# =============================================================================
out = load_h5_file(os.path.join(base_path, city, f"out_{ob}_{uq_method}.h5"))
pred = load_h5_file(os.path.join(base_path, city, f"pred_{uq_method}.h5"))
pred_tr = load_h5_file(os.path.join(base_path, city, f"pred_tr_{uq_method}.h5"))

# find high outlier pixel
(out[..., 2].sum(dim=0) == 86).nonzero(as_tuple=False)
# fixed pixel
pix_h, pix_w = 213, 302
# where do the outlier labels come from
out[:, pix_h, pix_w, :].sum(dim=0)

# Mean over channel uncertainties
pred_pix = torch.stack((pred[:, 2, pix_h, pix_w, [0, 2, 4, 6]].mean(dim=-1),
                        pred[:, 2, pix_h, pix_w, [1, 3, 5, 7]].mean(dim=-1)), dim=1)
pred_tr_pix = torch.stack((pred_tr[:, 2, pix_h, pix_w, [0, 2, 4, 6]].mean(dim=-1),
                           pred_tr[:, 2, pix_h, pix_w, [1, 3, 5, 7]].mean(dim=-1)), dim=1)

fig, axes = plt.subplots(1, 2, figsize=(8, 2.4))
lab = ["Volume", "Speed"]
# ranges = [0.3, 1.5]
for i, ax in enumerate(axes.flat):
    ax.hist(pred_pix[:, i].numpy(), bins=15, alpha=0.6, #range=(0,ranges[i]),
            color="red", label="Test set", density=True)
    ax.hist(pred_tr_pix[:, i].numpy(), bins=15, alpha=0.6, #range=(0,ranges[i]),
            color="blue", label="Train set", density=True)

    # out_bound = out_b[ob]
    # vl = torch.quantile(pred_tr_pix[:, i], 1-(out_bound/100)).item()
    oc = int(out[:, pix_h, pix_w, i].sum(dim=0).item())
    # ax.axvline(vl, color="black", linestyle=":", label=f"Outlier bound {out_bound}%")
    # ax.text(x=vl+0.1, y=ax.get_ylim()[1]-0.5, s=rf"{oc}$\rightarrow$", color="black", fontweight="bold")
    # ax.plot([], [], ' ', label=f"{oc}/90 outliers")
    ax.set_title(f"{lab[i]} ({oc}/90 outliers)", fontsize="small")
    ax.set_ylabel("Density")
    ax.set_xlabel("Mean uncertainty")
    ax.legend()
    # ax.grid()
fig.suptitle(f"{city.capitalize()}, UQ: {uq_method}, Pixel ({pix_h}, {pix_w}), " +
             "Histograms of test vs. train set uncertainties",
             y=1.1, x=0.5, fontsize="small")

save_fig(fig_path, city, uq_method, f"hist_unc_pix_{pix_h}_{pix_w}")


# =============================================================================
# GT/pred/unc activity vs. test samples: weekends & weekdays; fixed crop
# =============================================================================
pred = load_h5_file(os.path.join(base_path, city, f"pred_{uq_method}.h5"))

# Test data: Thu Apr 4 - Tue Jun 30, 2020
weekend_idx = [(2+7*i) for i in range(13)] + [(3+7*i) for i in range(13)]
weekend_idx.sort()
week_idx = [x for x in [i for i in range(90)] if x not in weekend_idx]
val = {"GT": 0, "pred": 1, "unc": 2}
lab = {"Volume": [0, 2, 4, 6], "Speed": [1, 3, 5, 7]}

d = "GT" # GT, pred, unc
label = "Ground truth"
r = 174 # center pixel height (y-axis T to B)
c = 247 # center pixel width (x-axis L to R)
num = 11 # display: center +- num

fig, axes = plt.subplots(1, 2, figsize=(10, 3))
# fig.subplots_adjust(wspace=0.3)
for i, ax in enumerate(axes.flat):
    l = list(lab.keys())[i]
    data = pred[:, val[d], r-num:r+num, c-num:c+num, lab[l]].sum(dim=(-1)).mean(dim=(1,2))
    # data2 = pred[:, val["GT"], r-num:r+num, c-num:c+num, lab[l]].sum(dim=(-1)).mean(dim=(1,2))

    # mse = (pred[:, val["GT"], r-num:r+num, c-num:c+num, lab[l]]
    #        - pred[:, val["pred"], r-num:r+num, c-num:c+num, lab[l]])**2
    # data = mse.mean(dim=(1,2,3))

    ax.plot(data, ls="-", lw=0.7, marker=".", color="black", mec="black",
            mfc="black", zorder=1, label=label)
    # ax.plot(data2, ls="-", lw=0.7, marker=".", color="blue", mec="blue",
    #         mfc="blue", zorder=1, label="GT")
    # ax.scatter(range(90), data, c="black", marker=".", zorder=2, label="Week day")
    ax.scatter(x=weekend_idx, y=data[weekend_idx], c="red", edgecolors="red",
               marker=".", zorder=2, label="Weekend")
    # ax.scatter(x=weekend_idx, y=data2[weekend_idx], c="red", edgecolors="red",
    #            marker=".", zorder=2)
    ax.set_title(f"{l} (Central area)", fontsize="small")
    ax.set_ylabel(label)
    ax.set_xlabel("Test sample")
    ax.legend(fontsize="small")

fig.suptitle(f"{city.capitalize()}, UQ: {uq_method}, OB: {out_b[ob]}%, " +
             f"Crop ({r-num}:{r+num}, {c-num}:{c+num}), " +
             f"{d} (Mean over pixels, sum channels) vs. test samples",
             y=1.1, x=0.5, fontsize="small")

save_fig(fig_path, city, uq_method, f"val_{d}_{ob}_crop_{r}_{c}")


# =============================================================================
# GT/pred/unc activity vs. train samples: weekends & weekdays; fixed crop
# =============================================================================
pred_tr = load_h5_file(os.path.join(base_path, city, f"pred_tr_{uq_method}.h5"))

# Train data: Wed Jan 2 - Sun Jun 30, 2019
weekend_idx = [(3+7*i) for i in range(26)] + [(4+7*i) for i in range(26)]
weekend_idx.sort()
week_idx = [x for x in [i for i in range(180)] if x not in weekend_idx]
val = {"GT": 0, "pred": 1, "unc": 2}
lab = {"Volume": [0, 2, 4, 6], "Speed": [1, 3, 5, 7]}

d = "GT" # GT, pred, unc
label = "Ground truth"
r = 358 # center pixel height (y-axis T to B)
c = 226 # center pixel width (x-axis L to R)
num = 10 # display: center +- num

fig, axes = plt.subplots(1, 2, figsize=(12, 3))
# fig.subplots_adjust(wspace=0.3)
for i, ax in enumerate(axes.flat):
    l = list(lab.keys())[i]
    data = pred_tr[:, val[d], r-num:r+num, c-num:c+num, lab[l]].sum(dim=(-1)).mean(dim=(1,2))
    # data2 = pred_tr[:, val["GT"], r-num:r+num, c-num:c+num, lab[l]].sum(dim=(-1)).mean(dim=(1,2))
    ax.plot(data, ls="-", lw=0.7, marker=".", color="black", mec="black",
            mfc="black", zorder=1, label=label)
    # ax.plot(data2, ls="-", lw=0.7, marker=".", color="blue", mec="blue",
    #         mfc="blue", zorder=1, label="GT")
    ax.scatter(x=weekend_idx, y=data[weekend_idx], c="red", edgecolors="red",
               marker=".", zorder=2, label="Weekend")
    # ax.scatter(x=weekend_idx, y=data2[weekend_idx], c="red", edgecolors="red",
    #             marker=".", zorder=2)
    ax.set_title(f"{l} (Central area)", fontsize="small")
    ax.set_ylabel(label)
    ax.set_xlabel("Train sample")
    ax.legend(fontsize="small")

fig.suptitle(f"{city.capitalize()}, UQ: {uq_method}, OB: {out_b[ob]}%, " +
             f"Crop ({r-num}:{r+num}, {c-num}:{c+num}), " +
             f"{d} (Mean over pixels, sum channels) vs. train samples",
             y=1.1, x=0.5, fontsize="small")

save_fig(fig_path, city, uq_method, f"val_{d}_train_{ob}_crop_{r}_{c}")


# =============================================================================
# GT/pred/unc activity vs. test samples: weekends & weekdays; OUT fixed crop
# =============================================================================
pred = load_h5_file(os.path.join(base_path, city, f"pred_{uq_method}.h5"))

# Test data: Thu Apr 4 - Tue Jun 30, 2020
weekend_idx = [(2+7*i) for i in range(13)] + [(3+7*i) for i in range(13)]
weekend_idx.sort()
week_idx = [x for x in [i for i in range(90)] if x not in weekend_idx]
val = {"GT": 0, "pred": 1, "unc": 2}
lab = {"Volume": [0, 2, 4, 6], "Speed": [1, 3, 5, 7]}

d = "GT" # GT, pred, unc
label = "Ground truth"
# NOTE: this is the macro crop we want to capture (square)
r_m = 150 # center pixel height (y-axis T to B)
c_m = 60 # center pixel width (x-axis L to R)
num_m = 30 # display: center +- num

# NOTE: this is the crop we want to remove from the macro crop (square)
r = 152 # center pixel height (y-axis T to B)
c = 60 # center pixel width (x-axis L to R)
num = 8 # display: center +- num

fig, axes = plt.subplots(1, 2, figsize=(10, 3))
# fig.subplots_adjust(wspace=0.3)
for i, ax in enumerate(axes.flat):
    l = list(lab.keys())[i]

    # collect data from sliced rectangles around out-crop but within-macro crop
    data = torch.stack((pred[:, val[d], r_m-num_m:r_m+num_m, c_m-num_m:c-num, lab[l]].sum(dim=(-1)).mean(dim=(1,2)),
                        pred[:, val[d], r_m-num_m:r_m+num_m, c+num:c_m+num_m, lab[l]].sum(dim=(-1)).mean(dim=(1,2)),
                        pred[:, val[d], r_m-num_m:r-num, c-num:c+num, lab[l]].sum(dim=(-1)).mean(dim=(1,2)),
                        pred[:, val[d], r+num:r_m+num_m, c-num:c+num, lab[l]].sum(dim=(-1)).mean(dim=(1,2))),
                        dim=1)

    # data = torch.stack((
    #     ((pred[:, val["GT"], r_m-num_m:r_m+num_m, c_m-num_m:c-num, lab[l]] - pred[:, val["pred"], r_m-num_m:r_m+num_m, c_m-num_m:c-num, lab[l]])**2).mean(dim=(1,2,3)),
    #     ((pred[:, val["GT"], r_m-num_m:r_m+num_m, c+num:c_m+num_m, lab[l]] - pred[:, val["pred"], r_m-num_m:r_m+num_m, c+num:c_m+num_m, lab[l]])**2).mean(dim=(1,2,3)),
    #     ((pred[:, val["GT"], r_m-num_m:r-num, c-num:c+num, lab[l]] - pred[:, val["pred"], r_m-num_m:r-num, c-num:c+num, lab[l]])**2).mean(dim=(1,2,3)),
    #     ((pred[:, val["GT"], r+num:r_m+num_m, c-num:c+num, lab[l]] - pred[:, val["pred"], r+num:r_m+num_m, c-num:c+num, lab[l]])**2).mean(dim=(1,2,3))
    #     ), dim=1)

    # weighed mean of collected values according to pixel counts in each rectangle
    w0 = (num_m * 2) * ((c-num)-(c_m-num_m))
    w1 = (num_m * 2) * ((c_m+num_m)-(c+num))
    w2 = ((r-num)-(r_m-num_m)) * (num * 2)
    w3 = ((r_m+num_m)-(r+num)) * (num * 2)
    wt = w0 + w1 + w2 + w3
    data = data[:, 0]*(w0/wt) + data[:, 1]*(w1/wt) + data[:, 2]*(w2/wt) + data[:, 3]*(w3/wt)

    # data2 = pred[:, val["GT"], r-num:r+num, c-num:c+num, lab[l]].sum(dim=(-1)).mean(dim=(1,2))
    ax.plot(data, ls="-", lw=0.7, marker=".", color="black", mec="black",
            mfc="black", zorder=1, label=label)
    # ax.plot(data2, ls="-", lw=0.7, marker=".", color="blue", mec="blue",
    #         mfc="blue", zorder=1, label="GT")
    # ax.scatter(range(90), data, c="black", marker=".", zorder=2, label="Week day")
    ax.scatter(x=weekend_idx, y=data[weekend_idx], c="red", edgecolors="red",
               marker=".", zorder=2, label="Weekend")
    # ax.scatter(x=weekend_idx, y=data2[weekend_idx], c="red", edgecolors="red",
    #            marker=".", zorder=2)
    ax.set_title(f"{l} (Context area)", fontsize="small")
    ax.set_ylabel(label)
    ax.set_xlabel("Test sample")
    ax.legend(fontsize="small")

fig.suptitle(f"{city.capitalize()}, UQ: {uq_method}, OB: {out_b[ob]}%, " +
             f"Crop ({r_m-num_m}:{r_m+num_m}, {c_m-num_m}:{c_m+num_m}) - ({r-num}:{r+num}, {c-num}:{c+num}), " +
             f"{d} (Mean over pixels, sum channels) vs. test samples",
             y=1.1, x=0.5, fontsize="small")

save_fig(fig_path, city, uq_method, f"val_{d}_{ob}_crop_{r_m}_{c_m}_OUT_{r}_{c}")


# =============================================================================
# GT/pred/unc activity vs. train samples: weekends & weekdays; OUT fixed crop
# =============================================================================
pred_tr = load_h5_file(os.path.join(base_path, city, f"pred_tr_{uq_method}.h5"))

# Train data: Wed Jan 2 - Sun Jun 30, 2019
weekend_idx = [(3+7*i) for i in range(26)] + [(4+7*i) for i in range(26)]
weekend_idx.sort()
week_idx = [x for x in [i for i in range(180)] if x not in weekend_idx]
val = {"GT": 0, "pred": 1, "unc": 2}
lab = {"Volume": [0, 2, 4, 6], "Speed": [1, 3, 5, 7]}

d = "GT" # GT, pred, unc
label = "Ground truth"
# NOTE: this is the macro crop we want to capture (square)
r_m = 150 # center pixel height (y-axis T to B)
c_m = 60 # center pixel width (x-axis L to R)
num_m = 30 # display: center +- num

# NOTE: this is the crop we want to remove from the macro crop (square)
r = 152 # center pixel height (y-axis T to B)
c = 60 # center pixel width (x-axis L to R)
num = 8 # display: center +- num

fig, axes = plt.subplots(1, 2, figsize=(12, 3))
for i, ax in enumerate(axes.flat):
    l = list(lab.keys())[i]

    # collect data from sliced rectangles around out-crop but within-macro crop
    data = torch.stack((pred_tr[:, val[d], r_m-num_m:r_m+num_m, c_m-num_m:c-num, lab[l]].sum(dim=(-1)).mean(dim=(1,2)),
                        pred_tr[:, val[d], r_m-num_m:r_m+num_m, c+num:c_m+num_m, lab[l]].sum(dim=(-1)).mean(dim=(1,2)),
                        pred_tr[:, val[d], r_m-num_m:r-num, c-num:c+num, lab[l]].sum(dim=(-1)).mean(dim=(1,2)),
                        pred_tr[:, val[d], r+num:r_m+num_m, c-num:c+num, lab[l]].sum(dim=(-1)).mean(dim=(1,2))),
                       dim=1)
    # weighed mean of collected values according to pixel counts in each rectangle
    w0 = (num_m * 2) * ((c-num)-(c_m-num_m))
    w1 = (num_m * 2) * ((c_m+num_m)-(c+num))
    w2 = ((r-num)-(r_m-num_m)) * (num * 2)
    w3 = ((r_m+num_m)-(r+num)) * (num * 2)
    wt = w0 + w1 + w2 + w3
    data = data[:, 0]*(w0/wt) + data[:, 1]*(w1/wt) + data[:, 2]*(w2/wt) + data[:, 3]*(w3/wt)

    # data2 = pred[:, val["GT"], r-num:r+num, c-num:c+num, lab[l]].sum(dim=(-1)).mean(dim=(1,2))
    ax.plot(data, ls="-", lw=0.7, marker=".", color="black", mec="black",
            mfc="black", zorder=1, label=label)
    # ax.plot(data2, ls="-", lw=0.7, marker=".", color="blue", mec="blue",
    #         mfc="blue", zorder=1, label="GT")
    # ax.scatter(range(90), data, c="black", marker=".", zorder=2, label="Week day")
    ax.scatter(x=weekend_idx, y=data[weekend_idx], c="red", edgecolors="red",
               marker=".", zorder=2, label="Weekend")
    # ax.scatter(x=weekend_idx, y=data2[weekend_idx], c="red", edgecolors="red",
    #            marker=".", zorder=2)
    ax.set_title(f"{l} (Context area)", fontsize="small")
    ax.set_ylabel(label)
    ax.set_xlabel("Train sample")
    ax.legend(fontsize="small")

fig.suptitle(f"{city.capitalize()}, UQ: {uq_method}, OB: {out_b[ob]}%, " +
             f"Crop ({r_m-num_m}:{r_m+num_m}, {c_m-num_m}:{c_m+num_m}) - ({r-num}:{r+num}, {c-num}:{c+num}), " +
             f"{d} (Mean over pixels, sum channels) vs. train samples",
             y=1.1, x=0.5, fontsize="small")

save_fig(fig_path, city, uq_method, f"val_{d}_train_{ob}_crop_{r_m}_{c_m}_OUT_{r}_{c}")
