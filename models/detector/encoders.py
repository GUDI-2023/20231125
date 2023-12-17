# -*- coding: utf-8 -*-
"""Deep Anomaly Detection on Attributed Networks (DOMINANT)"""


import torch
import numpy as np
from torch_geometric.nn import GCN

from .base import DeepDetector
from ..nn.encoders import encodersBase


class encoders(DeepDetector):

    def __init__(self,
                 hid_dim=64,
                 num_layers=4,
                 dropout=0.,
                 weight_decay=0.,
                 act=torch.nn.functional.relu,
                 sigmoid_s=False,
                 backbone=GCN,
                 contamination=0.1,
                 lr=4e-3,
                 epoch=100,
                 gpu=-1,
                 batch_size=0,
                 num_neigh=-1,
                 weight=0.5, # ori 0.5 assert 0 <= weight <= 1, "weight must be a float between 0 and 1.",1 是纯attribute
                 verbose=0,
                 save_emb=False,
                 compile_model=False,
                 **kwargs):

        super(encoders, self).__init__(hid_dim=hid_dim,
                                       num_layers=num_layers,
                                       dropout=dropout,
                                       weight_decay=weight_decay,
                                       act=act,
                                       backbone=backbone,
                                       contamination=contamination,
                                       lr=lr,
                                       epoch=epoch,
                                       gpu=gpu,
                                       batch_size=batch_size,
                                       num_neigh=num_neigh,
                                       verbose=verbose,
                                       save_emb=save_emb,
                                       compile_model=compile_model,
                                       **kwargs)

        self.weight = weight
        self.sigmoid_s = sigmoid_s
        self.T = 3
        self.in_node_nf = 0
        self.gamma = PredefinedNoiseSchedule('polynomial_2', timesteps=self.T, precision=1e-05)

    def process_graph(self, data):
        encodersBase.process_graph(data)

    def init_model(self, **kwargs):
        if self.save_emb:
            self.emb = torch.zeros(self.num_nodes, self.hid_dim)
        return encodersBase(in_dim=self.in_dim,
                            hid_dim=self.hid_dim,
                            num_layers=self.num_layers,
                            dropout=self.dropout,
                            act=self.act,
                            sigmoid_s=self.sigmoid_s,
                            backbone=self.backbone,
                            **kwargs).to(self.device)

    def forward_model(self, data):
        batch_size = data.batch_size
        node_idx = data.n_id

        x = data.x.to(self.device)
        s = data.s.to(self.device)

        edge_index = data.edge_index.to(self.device)

        x_, s_ = self.model(x, edge_index)

        score = self.model.loss_func( x[:batch_size],
                                     x_[:batch_size],
                                     s[:batch_size, node_idx],
                                     s_[:batch_size],
                                     self.weight)

        loss = torch.mean(score)

        return loss, score.detach().cpu()

    def conditional_di(self, data):
        return 0
    def forward_di(self, data, prop):
        return 0
    def toden(self, s, f):
        return 0
        # The article pertains to enterprise collaboration, and this part of the code is currently undergoing the patent application process. Once the patent application is finalized, the code will be made publicly available in the first quarter of 2024.



class PredefinedNoiseSchedule(torch.nn.Module):
    """
    Predefined noise schedule. Essentially creates a lookup array for predefined (non-learned) noise schedules.
    """

    def __init__(self, noise_schedule, timesteps, precision):
        super(PredefinedNoiseSchedule, self).__init__()
        self.timesteps = timesteps

        splits = noise_schedule.split('_')
        assert len(splits) == 2
        power = float(splits[1])
        alphas2 = polynomial_schedule(timesteps, s=precision, power=power)

        sigmas2 = 1 - alphas2

        log_alphas2 = np.log(alphas2)
        log_sigmas2 = np.log(sigmas2)

        log_alphas2_to_sigmas2 = log_alphas2 - log_sigmas2

        self.gamma = torch.nn.Parameter(
            torch.from_numpy(-log_alphas2_to_sigmas2).float(),
            requires_grad=False)

    def forward(self, t):
        t_int = torch.round(t * self.timesteps).long()
        return self.gamma[t_int]


def polynomial_schedule(timesteps: int, s=1e-4, power=1.):
    """
    A noise schedule based on a simple polynomial equation: 1 - x^power.
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas2 = (1 - np.power(x / steps, power)) ** 2

    alphas2 = clip_noise_schedule(alphas2, clip_value=0.001)

    precision = 1 - 2 * s

    alphas2 = precision * alphas2 + s

    return alphas2


def clip_noise_schedule(alphas2, clip_value=0.001):
    """
    For a noise schedule given by alpha^2, this clips alpha_t / alpha_t-1. This may help improve stability during
    sampling.
    """
    alphas2 = np.concatenate([np.ones(1), alphas2], axis=0)

    alphas_step = (alphas2[1:] / alphas2[:-1])

    alphas_step = np.clip(alphas_step, a_min=clip_value, a_max=1.)
    alphas2 = np.cumprod(alphas_step, axis=0)

    return alphas2