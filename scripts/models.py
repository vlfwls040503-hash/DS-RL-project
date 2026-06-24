# -*- coding: utf-8 -*-
"""BC model architectures."""
import torch
import torch.nn as nn


class MLP(nn.Module):
    """Single-frame state -> action regressor."""
    def __init__(self, in_dim, out_dim=3, hidden=(256, 256, 128), p=0.1):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(p)]
            d = h
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class GRUNet(nn.Module):
    """Sequence of frames -> action at last step."""
    def __init__(self, in_dim, out_dim=3, hidden=128, layers=2, p=0.1):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden, num_layers=layers, batch_first=True,
                          dropout=p if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(p),
                                  nn.Linear(hidden, out_dim))

    def forward(self, x):                 # x: (B, L, F)
        out, _ = self.gru(x)
        return self.head(out[:, -1, :])   # last timestep


class GRUGaussian(nn.Module):
    """Sequence -> per-action Gaussian (mean, log_std). Captures behavioral spread."""
    def __init__(self, in_dim, out_dim=3, hidden=128, layers=2, p=0.1,
                 logstd_min=-3.0, logstd_max=2.0):
        super().__init__()
        self.out_dim = out_dim
        self.logstd_min, self.logstd_max = logstd_min, logstd_max
        self.gru = nn.GRU(in_dim, hidden, num_layers=layers, batch_first=True,
                          dropout=p if layers > 1 else 0.0)
        self.trunk = nn.Sequential(nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(p))
        self.mu = nn.Linear(hidden, out_dim)
        self.log_std = nn.Linear(hidden, out_dim)

    def forward(self, x):
        out, _ = self.gru(x)
        h = self.trunk(out[:, -1, :])
        mu = self.mu(h)
        log_std = torch.clamp(self.log_std(h), self.logstd_min, self.logstd_max)
        return mu, log_std


def gaussian_nll(mu, log_std, y):
    """Mean Gaussian negative log-likelihood over batch and action dims."""
    inv_var = torch.exp(-2.0 * log_std)
    return (0.5 * inv_var * (y - mu) ** 2 + log_std).mean()
