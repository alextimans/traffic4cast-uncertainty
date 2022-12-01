import os
import torch
from scipy.stats import spearmanr
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from model.configs import configs
from uq.eval_tta_ensemble import load_ensemble

model_class = configs["unet"]["model_class"]
model_config = configs["unet"]["model_config"]
device = "cpu"
save_checkpoint="./checkpoints"

ensemble = load_ensemble(device, save_checkpoint, model_class, model_config, [1,1,1,1,1])
n = len(ensemble)

bias_corr = torch.empty((n, n))
weights_corr = torch.empty((n, n))

for i in range(n):
    bias = ensemble[i].final.bias.detach()
    weight = ensemble[i].final.weight.detach().reshape(48,64).ravel()

    for j in range(n):
        bias2 = ensemble[j].final.bias.detach()
        weight2 = ensemble[j].final.weight.detach().reshape(48,64).ravel()

        bias_corr[i, j] = spearmanr(bias, bias2)[0]
        weights_corr[i, j] = spearmanr(weight, weight2)[0]

name = ["bias_corr", "weights_corr"]
for i, corr in enumerate([bias_corr, weights_corr]):
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    y_lab=[f"U-Net #{i+1}" for i in range(n)]
    x_lab=[f"#{i+1}" for i in range(n)]
    sns.heatmap(corr, annot=True, cmap="coolwarm",
                vmin=-1, vmax=1, square=True, xticklabels=x_lab,
                yticklabels=y_lab, linewidths=0.5, mask=mask)

    plt.savefig(os.path.join(save_checkpoint, f"model_last_layer_{name[i]}.png"),
                dpi=300, format="png", bbox_inches='tight')
    plt.clf()
