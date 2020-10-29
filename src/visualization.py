import numpy as np
import os
import matplotlib.pyplot as plt


def visualization(file_name):
    pixels = np.load(file_name + ".npz")

    val_loss = pixels['val_total']
    val_bce = pixels['val_bce']
    val_kl = pixels['val_kld']
    train_loss = pixels['train_total']
    train_bce = pixels['train_bce']
    train_kl = pixels['train_kld']
    n_samples = np.arange(len(val_loss))

    figure, axis = plt.subplots()
    axis.plot(n_samples, val_loss, color='black', label='Val Loss')
    axis.plot(n_samples, val_bce, color='blue', label='Val Recon Loss')
    axis.plot(n_samples, train_loss, linestyle='--',
              color='black', label='Train Loss')
    axis.plot(n_samples, train_bce, linestyle='--',
              color='blue', label='Train Recon Loss')
    axis.set_ylabel("Total/Recon Loss")
    ax2 = axis.twinx()
    ax2.plot(n_samples, val_kl, color='red', label='Val KL Loss')
    ax2.plot(n_samples, train_kl, linestyle='--',
             color='red', label='Train KL Loss')
    axis.set_xlabel('Num epochs')
    ax2.set_ylabel("KL Loss")
    figure.legend(loc="upper right")
    plt.subplots_adjust(top=0.88)
    axis.set_title(file_name + 'Loss')
    figure.savefig(file_name + '.png')
