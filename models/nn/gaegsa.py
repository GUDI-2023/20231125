import math
from torch import nn
import torch
import torch.nn.functional as F
from torch_geometric.nn import MLP, GCN, GIN
from torch_geometric.utils import to_dense_adj

from ..nn.decoder import DotProductDecoder
from .functional import double_recon_loss

class encoderfBase(nn.Module):

    """
    Graph Autoencoder

    See :cite:`kipf2016variational` for details.
    """

    def __init__(self,
                 in_dim,
                 hid_dim=64,
                 num_layers=4,
                 dropout=0.,
                 act=F.relu,
                 backbone=GCN,
                 recon_s=False,
                 sigmoid_s=False,


                 **kwargs):
        super(encoderfBase, self).__init__()

        self.backbone = GCN
        # split the number of layers for the encoder and decoders
        assert num_layers >= 2, \
            "Number of layers must be greater than or equal to 2."
        encoder_layers = math.floor(num_layers / 2)
        decoder_layers = math.ceil(num_layers / 2)

        # mlpae component for attribute
        self.encoder = self.backbone(in_channels=in_dim,
                                     hidden_channels=hid_dim,
                                     out_channels=hid_dim,
                                     num_layers=encoder_layers,
                                     dropout=dropout,
                                     act=act,
                                     **kwargs)
        self.recon_s = recon_s
        if self.recon_s:
            self.decoder = DotProductDecoder(in_dim=hid_dim,
                                             hid_dim=hid_dim,
                                             num_layers=decoder_layers,
                                             dropout=dropout,
                                             act=act,
                                             sigmoid_s=sigmoid_s,
                                             backbone=self.backbone,
                                             **kwargs)
        else:
            self.decoder = self.backbone(in_channels=hid_dim,
                                         hidden_channels=hid_dim,
                                         out_channels=in_dim,
                                         num_layers=decoder_layers,
                                         dropout=dropout,
                                         act=act,
                                         **kwargs)

        self.loss_func_s = double_recon_loss
        self.emb_s = None
        self.emb_c = None


        self.loss_func = F.mse_loss
        self.emb = None

    def forward(self, data, x, edge_index):
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
            Reconstructed embeddings.
        """

        if self.backbone == MLP or True:
            self.emb_c = self.encoder(x, edge_index).to(x.device)
            self.emb_s = self.encoder(x, edge_index)
            self.emb = torch.cat([self.emb_c, self.emb_s], dim=1)
            # print('a', x.shape,self.emb.shape)
            # out, prop = self.gsa(data, self.emb)
            out, prop = 1,1

            # decode contexural matrix
            x_ = self.decoder(self.emb_c, edge_index).to(x.device)



        else: # GCN + AE
            self.emb = self.encoder(x, edge_index).to(x.device)
            out, prop = self.gsa(data, self.emb)
            x_ = self.decoder(self.emb, edge_index).to(x.device)



        return x_, out, prop

    @staticmethod
    def process_graph(data, recon_s=False):
        """
        Obtain the dense adjacency matrix of the graph.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Input graph.
        recon_s : bool, optional
            Reconstruct the structure instead of node feature .
        """
        # if recon_s or True:
        #
        data.s = to_dense_adj(data.edge_index)[0]
