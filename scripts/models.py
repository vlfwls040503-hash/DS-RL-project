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


class CVAE(nn.Module):
    """Conditional VAE over distance-indexed trajectories.
      q(z | behavior, geometry) : GRU encoder -> (mu_z, logvar_z)
      p(behavior | z, geometry) : GRU decoder (geometry fed every step + z broadcast)
    behavior=(B,W,2), geometry=(B,W,G). Seq2seq decode (no teacher forcing)."""
    def __init__(self, beh_dim=2, geo_dim=5, z_dim=16, hidden=128, layers=2, p=0.1):
        super().__init__()
        self.beh_dim, self.geo_dim, self.z_dim = beh_dim, geo_dim, z_dim
        self.enc_gru = nn.GRU(beh_dim + geo_dim, hidden, num_layers=layers,
                              batch_first=True, dropout=p if layers > 1 else 0.0)
        self.mu_z = nn.Linear(hidden, z_dim)
        self.logvar_z = nn.Linear(hidden, z_dim)
        self.dec_gru = nn.GRU(geo_dim + z_dim, hidden, num_layers=layers,
                              batch_first=True, dropout=p if layers > 1 else 0.0)
        self.dec_out = nn.Sequential(nn.LayerNorm(hidden), nn.GELU(), nn.Linear(hidden, beh_dim))

    def encode(self, behavior, geometry):
        out, _ = self.enc_gru(torch.cat([behavior, geometry], dim=-1))
        last = out[:, -1, :]
        return self.mu_z(last), self.logvar_z(last)

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z, geometry):
        W = geometry.shape[1]
        zb = z.unsqueeze(1).expand(-1, W, -1)              # (B,W,z)
        out, _ = self.dec_gru(torch.cat([geometry, zb], dim=-1))
        return self.dec_out(out)                            # (B,W,beh)

    def forward(self, behavior, geometry):
        mu, logvar = self.encode(behavior, geometry)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, geometry), mu, logvar


def cvae_loss(recon, behavior, mu, logvar, beta, recon_fn):
    """recon_fn: e.g. nn.SmoothL1Loss(). Returns (total, recon, kl) — kl in nats/sample."""
    rec = recon_fn(recon, behavior)
    kl = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    return rec + beta * kl, rec, kl
