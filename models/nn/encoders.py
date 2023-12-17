import math
import torch
import torch.nn as nn
from torch_geometric.nn import GCN
from torch_geometric.utils import to_dense_adj

from .decoder import DotProductDecoder
from .functional import double_recon_loss


class encodersBase(nn.Module):

    def __init__(self,
                 in_dim,
                 hid_dim=64,
                 num_layers=4,
                 dropout=0.,
                 act=torch.nn.functional.relu,
                 sigmoid_s=False,
                 backbone=GCN,
                 **kwargs):
        super(encodersBase, self).__init__()

        # split the number of layers for the encoder and decoders
        assert num_layers >= 2, \
            "Number of layers must be greater than or equal to 2."
        encoder_layers = math.floor(num_layers / 2)
        decoder_layers = math.ceil(num_layers / 2)

        self.shared_encoder = backbone(in_channels=in_dim,
                                       hidden_channels=hid_dim,
                                       num_layers=encoder_layers,
                                       out_channels=hid_dim,
                                       dropout=dropout,
                                       act=act,
                                       **kwargs)

        self.attr_decoder = backbone(in_channels=hid_dim,
                                     hidden_channels=hid_dim,
                                     num_layers=decoder_layers,
                                     out_channels=in_dim,
                                     dropout=dropout,
                                     act=act,
                                     **kwargs)

        self.struct_decoder = DotProductDecoder(in_dim=hid_dim,
                                                hid_dim=hid_dim,
                                                num_layers=decoder_layers - 1,
                                                dropout=dropout,
                                                act=act,
                                                sigmoid_s=sigmoid_s,
                                                backbone=backbone,
                                                **kwargs)

        self.loss_func = double_recon_loss
        self.emb = None

    def forward(self, x, edge_index):
        """
        Forward computation.

        Parameters
        ----------
        x : torch.Tensor
            Input attribute embeddings.
        edge_index : torch.Tensor
            Edge index.

        Returns
        -------
        x_ : torch.Tensor
            Reconstructed attribute embeddings.
        s_ : torch.Tensor
            Reconstructed adjacency matrix.
        """

        self.emb = self.shared_encoder(x, edge_index)

        # denoising network will be soon released after patents application

        # reconstruct feature matrix
        x_ = self.attr_decoder(self.emb, edge_index)

        # decode adjacency matrix
        s_ = self.struct_decoder(self.emb, edge_index)

        return x_, s_

    @staticmethod
    def process_graph(data):
        """
        Obtain the dense adjacency matrix of the graph.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Input graph.
        """
        data.s = to_dense_adj(data.edge_index)[0]
