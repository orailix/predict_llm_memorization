import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm


class SimpleNetwork(nn.Module):
    def __init__(self, n_layer, d) -> None:
        super().__init__()
        self.n_layer = n_layer
        self.d = d
        self.dummy_param = nn.Parameter(torch.empty(0))
        self.weights = [nn.Linear(d, d) for _ in range(n_layer)]
        self.final_classifier = nn.Linear(d, 2)

    def to(self, device):
        self = super().to(device)
        for idx in range(len(self.weights)):
            self.weights[idx] = self.weights[idx].to(device)

        return self

    @property
    def device(self):
        return self.dummy_param.device

    def forward(self, input: torch.Tensor):

        # values_memo = []

        value = input
        # values_memo.append(value.clone().detach())
        for w in self.weights:
            value = w(value)
            value = F.relu(value)
            # values_memo.append(value.clone().detach())

        # Final classification
        output = self.final_classifier(value)
        return output  # , values_memo

    def get_decision_boundaries_fig(self, X, labels, bs=100, resolution=100):

        x_min, x_max = min(X[:, 0]) - 0.3, max(X[:, 0]) + 0.3
        y_min, y_max = min(X[:, 1]) - 0.3, max(X[:, 1]) + 0.3

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_xlim(xmin=x_min, xmax=x_max)
        ax.set_ylim(ymin=y_min, ymax=y_max)

        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, resolution), np.linspace(y_min, y_max, resolution)
        )
        background_points = np.c_[xx.ravel(), yy.ravel()]
        background_points = torch.Tensor(background_points).to(self.device)

        # Evaluating
        self.eval()
        zz = []
        with torch.no_grad():
            num_batch = background_points.shape[0] // bs
            for batch_idx in tqdm(range(num_batch)):
                inputs = background_points[
                    batch_idx
                    * bs : min((batch_idx + 1) * bs, background_points.shape[0]),
                    :,
                ]
                outputs = self.forward(inputs)
                zz.append(outputs[:, 0] - outputs[:, 1])
        zz = torch.cat(zz, dim=0)
        zz = zz.view(resolution, resolution).cpu().numpy()

        ax.imshow(zz, extent=(x_min, x_max, y_min, y_max), origin="lower", cmap="bwr")

        # Scatter
        ax.scatter(
            X[labels == 0, 0],
            X[labels == 0, 1],
            c="red",
            marker="o",
            edgecolors="black",
        )
        ax.scatter(
            X[labels == 1, 0],
            X[labels == 1, 1],
            c="blue",
            marker="o",
            edgecolors="black",
        )

        return fig
