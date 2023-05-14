import torch
import torch.nn as nn

from core.network.model import model_register
from core.network.utils import cosine_similarity


@model_register
class VariationalAutoEncoder(torch.nn.Module):
    def __init__(self, cfg):
        super(VariationalAutoEncoder, self).__init__()
        self.cfg = cfg
        self.emb_dim = cfg.emb_dim
        self.loss = cfg.loss
        loss = self.loss
        self.detach_target = cfg.detach_target
        para_key = ['beta']
        default_key_val = [1]
        for key, v in zip(para_key, default_key_val):
            custom_v = self.cfg.get(key, v)
            self.__setattr__(key, custom_v)

        self.criterion = None
        if loss == 'l1':
            self.criterion = nn.L1Loss()
        elif loss == 'l2':
            self.criterion = nn.MSELoss()
        elif loss == 'cosine':
            self.criterion = cosine_similarity

        self.fc_mu = nn.Linear(self.emb_dim, self.emb_dim)
        self.fc_var = nn.Linear(self.emb_dim, self.emb_dim)

        self.decoder = nn.Sequential(
            nn.Linear(self.emb_dim, self.emb_dim),
            nn.BatchNorm1d(self.emb_dim),
            nn.ReLU(),
            nn.Linear(self.emb_dim, self.emb_dim),
        )
        return

    def encode(self, x):
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, y):
        if self.detach_target:
            y = y.detach()

        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        y_hat = self.decoder(z)

        reconstruction_loss = self.criterion(y_hat, y)
        kl_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        loss = reconstruction_loss + self.beta * kl_loss

        return loss

