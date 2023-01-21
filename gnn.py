import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GCNConv
import scipy


def LSIGF(weights, S, x):
    '''
    weights is a list of length k, with each element of shape d_in x d_out
    S is N x N, sparse matrix
    x is N x d, d-dimensional feature (i.e., unique node ID in the featureless setting)
    '''    
    # Number of filter taps
    K = len(weights)
    d = x.shape[1]

    # Create list to store graph diffused signals
    zs = [x]
    
    # Loop over the number of filter taps / different degree of S
    for k in range(1, K):        
        # diffusion step, S^k*x
        x = torch.spmm(S, x) #torch.matmul(x, S) -- slow
        # append the S^k*x in the list z
        zs.append(x)
    
    # sum up
    out = [z @ weight/torch.sqrt(torch.tensor(d)) for z, weight in zip(zs, weights)]
    out = torch.stack(out)
    y = torch.sum(out, axis=0)
    return y

class GraphFilter(torch.nn.Module):

    def __init__(self, Fin, Fout, K, normalize=False):

        super(GraphFilter, self).__init__()
        self.Fin = Fin 
        self.Fout = Fout
        self.K = K
        self.normalize = normalize
        self.weight = nn.ParameterList([nn.Parameter(torch.randn(self.Fin,self.Fout)) for k in range(self.K)])
        #self.reset_parameters()


    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.Fin * self.K)
        for elem in self.weight:
          elem.data.uniform_(-stdv, stdv)

    def forward(self, x, edge_index, edge_weight):
        N = x.shape[0]
        E = edge_index.shape[1]
        if edge_weight is None:
            edge_weight = torch.ones(E)
            edge_weight = edge_weight.to(x.device)
        S = torch.sparse_coo_tensor(edge_index, edge_weight, (N,N))

        if self.normalize:
            edge_weight_np = edge_weight.numpy()
            edge_index_np = edge_index.numpy()
            aux = scipy.sparse.coo_matrix((edge_weight_np, (edge_index_np[0],edge_index_np[1])), shape=(N,N))
            u, s, vh = scipy.sparse.linalg.svds(aux, k=1)
            S = S/s[0]

        return LSIGF(self.weight,S,x)


class GNN(torch.nn.Module):

    def __init__(self, name, GNNtype, Flist, MLPlist, softmax=False, Klist = None):

        super(GNN, self).__init__()
        self.name = name
        self.type = GNNtype
        self.Flist = Flist
        self.L = len(Flist)
        self.MLPlist = MLPlist 
        self.Lmlp = len(MLPlist)
        self.softmax = softmax
        self.Klist = Klist

        self.layers = nn.ModuleList()
        self.MLPlayers = nn.ModuleList()

        for i in range(self.L-1):
            if self.type == 'sage':
                self.layers.append(SAGEConv(Flist[i],Flist[i+1]))
            elif self.type == 'gcn':
                self.layers.append(GCNConv(Flist[i],Flist[i+1]))
            elif self.type =='gnn':
                self.layers.append(GraphFilter(Flist[i],Flist[i+1],Klist[i]))

        for i in range(self.Lmlp-1):
            self.MLPlayers.append(nn.Linear(MLPlist[i],MLPlist[i+1],bias=False))
        #for i in range(self.Lmlp-1):
        #    self.MLPlayers[i].reset_parameters()
            
    def forward(self, data):

        y, edge_index, edge_weight, batch = data.x, data.edge_index, data.edge_weight, data.batch

        for i, layer in enumerate(self.layers):
            if self.type == 'gnn':
                y = layer(y, edge_index=edge_index, edge_weight=edge_weight)
            else:
                y = layer(y, edge_index=edge_index)
            y = F.relu(y)

        for i, layer in enumerate(self.MLPlayers):
            y = layer(y)/torch.sqrt(torch.tensor(self.MLPlist[i]))
        
        if self.softmax == True:
            y = F.log_softmax(y, dim=1)

        return y

    def get_intermediate_features(self, data):

        y, edge_index, edge_weight, batch = data.x, data.edge_index, data.edge_weight, data.batch

        outputs = [y]

        for i, layer in enumerate(self.layers):
            if self.type == 'gnn':
                y = layer(y, edge_index=edge_index, edge_weight=edge_weight)
            else:
                y = layer(y, edge_index=edge_index)
            y = F.relu(y)
            outputs.append(y)

        for i, layer in enumerate(self.MLPlayers):
            y = layer(y)
            outputs.append(y)
        
        if self.softmax == True:
            y = F.log_softmax(y, dim=1)

        return outputs

    def get_weights(self):

        weight_list = []
        for i, layer in enumerate(self.layers):
            if self.type == 'gnn':
                weight_list.append(layer.weight)
        for i, layer in enumerate(self.MLPlayers):
            weight_list.append([layer.weight])

        return weight_list
