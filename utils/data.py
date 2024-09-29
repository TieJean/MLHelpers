#@title Generate 2D GMM Dataset
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import namedtuple


DataInfo = namedtuple('data_info', ['centers', 'cluster_idx'])
def sample_circular_gmm(batch_size, num_modes=4, radius=1.0, std=0.05, viz=False):
    angles = torch.linspace(0, 2 * torch.pi, num_modes+1)[:-1] + torch.pi/2
    centers = torch.stack([radius * torch.cos(angles), radius * torch.sin(angles)], dim=1)
    cluster_idx = torch.randint(0, num_modes, (batch_size,))
    selected_centers = centers[cluster_idx]
    batch_data = selected_centers + std * torch.randn(batch_size, 2)

    data_info = DataInfo(centers=centers, cluster_idx=cluster_idx)

    if viz:
        plt.figure(figsize=(2, 2))
        for i in range(len(data_info.centers)):
            idx = (data_info.cluster_idx == i)
            plt.scatter(batch_data[idx, 0], batch_data[idx, 1])

    return batch_data, data_info
