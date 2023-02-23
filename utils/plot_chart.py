import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

def draw_chart(train_loss, train_r2_score, val_loss, val_r2_score, test_loss_list, test_r2_list, region, type_model="dis", output_prefix = 'images_lowpass'):
    if not os.path.exists(output_prefix):
        os.mkdir(output_prefix)
    fig, axs = plt.subplots(2, 3)
    fig.suptitle(f'Region {region}')
    axs[0,0].set_title("Training Loss")
    axs[0,0].plot(train_loss)
    axs[1,0].set_title("Training R2 score")
    axs[1,0].plot(train_r2_score)
    axs[0,1].set_title("Validation Loss")
    axs[0,1].plot(val_loss)
    axs[1,1].set_title("Validation R2 score")
    axs[1,1].plot(val_r2_score)
    axs[0,2].set_title("Testing loss")
    axs[0,2].plot(test_loss_list)
    axs[1,2].set_title("Testing R2 score")
    axs[1,2].plot(test_r2_list)
    fig.tight_layout()

    axs[1,0].grid()
    axs[1,1].grid()
    axs[1,2].grid()

    axs[1,0].set_yticks(np.arange(0, 1, 0.2))
    axs[1,0].set_ylim(0,1)
    axs[1,1].set_yticks(np.arange(0, 1, 0.2))
    axs[1,1].set_ylim(0,1)
    # axs[1,2].set_yticks(np.arange(0, 1, 0.2))
    # axs[1,2].set_ylim(0,1)
    plt.savefig(os.path.join(output_prefix,f"grud_{type_model}_region_{region}.png"))
    plt.close(fig)


def draw_MAF_R2(pred, label, a1_freq_list, region, bins=30, output_prefix = 'images'):
    if not os.path.exists(output_prefix):
        os.mkdir(output_prefix)
    a1_freq_list = a1_freq_list.tolist()
    bins_list = pd.cut(a1_freq_list, bins, labels=range(bins))
    pred_bins = [[] for _ in range(bins)]
    label_bins = [[] for _ in range(bins)]
    for index, bin in enumerate(bins_list):
        pred_bins[bin].append(pred[:, index])
        label_bins[bin].append(label[:, index])
    r2_score_list = []
    for index in range(bins):
        y = torch.stack(label_bins[index]).detach().numpy().T
        y_pred = torch.stack(pred_bins[index]).detach().numpy().T
        n_samples = len(y)
        _r2_score = sum([r2_score(y[i], y_pred[i]) for i in range(n_samples)])/n_samples
        r2_score_list.append(_r2_score)
    x_axis = np.unique(pd.cut(a1_freq_list, bins, labels=np.linspace(start=0, stop=0.5, num=bins)))
    print(np.unique(bins_list))
    plt.plot(x_axis, r2_score_list)
    plt.grid(linestyle='--')
    plt.xlabel("Minor Allele Frequency")
    plt.ylabel("R2")
    # plt.ylim(0.65, 1)
    plt.savefig(os.path.join(output_prefix, f'grud_region_{region}_MAF_R2.png'))
