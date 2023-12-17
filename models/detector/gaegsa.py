# -*- coding: utf-8 -*-
""" Graph Autoencoder
"""

import torch
import numpy as np
import warnings
import torch.nn.functional as F
from torch_geometric.nn import MLP, GCN, GIN

from . import DeepDetector
from ..nn.gaegsa import encoderfBase


class encoderf(DeepDetector):

    def __init__(self,
                 hid_dim=64,
                 num_layers=4,
                 dropout=0.,
                 weight_decay=0.,
                 act=F.relu,
                 backbone=GCN,
                 recon_s=False,
                 sigmoid_s=False,
                 contamination=0.1,
                 lr=4e-3,
                 epoch=100,
                 gpu=-1,
                 batch_size=0,
                 num_neigh=-1,
                 verbose=False,
                 save_emb=False,
                 compile_model=False,
                 **kwargs):

        if num_neigh != 0 and backbone == MLP:
            warnings.warn('MLP does not use neighbor information.')
            num_neigh = 0

        self.recon_s = recon_s
        self.sigmoid_s = sigmoid_s


        super(encoderf, self).__init__(hid_dim=hid_dim,
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

    def process_graph(self, data):
        encoderfBase.process_graph(data, recon_s=self.recon_s)

    def init_model(self, **kwargs):
        if self.save_emb:
            self.emb = torch.zeros(self.num_nodes,
                                   self.hid_dim)
        return encoderfBase(in_dim=self.in_dim,
                       hid_dim=self.hid_dim,
                       num_layers=self.num_layers,
                       dropout=self.dropout,
                       act=self.act,
                       recon_s=self.recon_s,
                       sigmoid_s=self.sigmoid_s,
                       backbone=self.backbone,

                       **kwargs)

    def forward_model(self, data):

        batch_size = data.batch_size
        node_idx = data.n_id

        x = data.x.to(self.device)
        s = data.s.to(self.device)
        edge_index = data.edge_index.to(self.device)


        self.model.to(self.device)
        h, hs, out, prop = self.model(data, x, edge_index)

        target = s if self.recon_s else x
        # Training of denoising network is hidden here, will be soon released after patents application

        score_c = torch.mean(self.model.loss_func(target[:batch_size],
                                                h[:batch_size],
                                                reduction='none'), dim=1)


        data.y = data.y.bool()
        data.y = data.y.long()

        loss_c = torch.mean(score_c)


        return loss_c, score_c.detach().cpu()

    def conditional_di(self, data):
        return 0
    def forward_di(self, data, prop):
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