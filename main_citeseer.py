# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 18:50:43 2023

@author: Luana Ruiz
"""

import os
import datetime
import numpy as np
import torch
import copy
import pickle as pkl

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import Planetoid

import matplotlib.pyplot as plt

import gnn
import train_test
import kernel as ker

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

thisFilename = 'citeseer' # This is the general name of all related files

saveDirRoot = 'experiments' # In this case, relative location
saveDir = os.path.join(saveDirRoot, thisFilename) 

#\\\ Create .txt to store the values of the setting parameters for easier
# reference when running multiple experiments
today = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
# Append date and time of the run to the directory, to avoid several runs of
# overwritting each other.
saveDir = saveDir + '-' + today
# Create directory 
if not os.path.exists(saveDir):
    os.makedirs(saveDir)

""
""
# Aux classes and functions

class objectview(object):
    def __init__(self, d):
        self.__dict__ = d

# Global variables

plt.rcParams['text.usetex'] = True

n_realizations = 10

nTrain = 1
nValid = 1
nTest = 1

# Graph

n_vector = [400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]

gnn_results = np.zeros((n_realizations, len(n_vector), 3))
kernel_results = np.zeros((n_realizations, len(n_vector), 3))
gnn_results_transf = np.zeros((n_realizations, len(n_vector), 3))
kernel_results_transf = np.zeros((n_realizations, len(n_vector), 3))
x_label = np.zeros((n_realizations, nTest))
y_plot_1 = np.zeros((n_realizations, len(n_vector), 3, nTest))
y_plot_2 = np.zeros((n_realizations, len(n_vector), 3, nTest))
y_plot_3 = np.zeros((n_realizations, len(n_vector), 3, nTest))
# Copy
eig_plot = np.zeros((n_realizations, len(n_vector), 3, 6))

S_list = []
S_large_list = []

# Transferability data

# Data

dataset = Planetoid(root='/tmp/citeseer', name='CiteSeer', split='public')
F0 = dataset.num_node_features
print(F0)
F0 = 1000
C = dataset.num_classes
print(C)
data = dataset.data # Save it to have the same test samples in the transferability test
N = data.num_nodes
print(N)
save_test_mask = data.test_mask
edge_list_large = data.edge_index
E = edge_list_large.shape[1]
edge_weights_large = torch.ones(E,dtype=torch.float32)/N
S_large = torch.sparse_coo_tensor(edge_list_large, edge_weights_large, (N,N)).to_dense()
xTest2 = data.x[:,0:F0]*torch.tile(torch.reshape(save_test_mask,(N,1)),(1,F0))
yTest2 = data.y*save_test_mask
aux = Data(x=xTest2, edge_index=edge_list_large, edge_attr=edge_weights_large, y=yTest2)
aux = aux.to(device)
dataTest2 = [aux]

# Arch. parameters

F = [F0,10]
MLP = [10,C]
K = [2]

F2 = [F0,50]
MLP2 = [50,C]
K2 = [2]

F3 = [F0,100]
MLP3 = [100,C]
K3 = [2]

for rlz in range(n_realizations):

    S_aux_list = []

    for idx,n in enumerate(n_vector):

        # Create training graph
        flag = False
        while flag == False:
            sampledData = data.subgraph(torch.randint(0, N, (n,)))
            print(sampledData)
            flag = True
            for i in range(C):
                if torch.sum(sampledData.y*sampledData.train_mask==i) == 0:
                    flag = False
        sampledData.x = sampledData.x[:,0:F0]

        edge_list = sampledData.edge_index
        E = edge_list.shape[1]
        edge_weights = torch.ones(E,dtype=torch.float32)/n
        S = torch.sparse_coo_tensor(edge_list, edge_weights, (n,n)).to_dense()
        S_aux_list.append(S)

        # Training data

        xTrain = sampledData.x*torch.tile(torch.reshape(sampledData.train_mask,(n,1)),(1,F0))
        yTrain = sampledData.y*sampledData.train_mask
        xValid = sampledData.x*torch.tile(torch.reshape(sampledData.val_mask,(n,1)),(1,F0))
        yValid = sampledData.y*sampledData.val_mask
        xTest = sampledData.x*torch.tile(torch.reshape(sampledData.test_mask,(n,1)),(1,F0))
        yTest = sampledData.y*sampledData.test_mask

        # Train GNN

        dataTrain = []
        dataValid = []
        dataTest = []

        aux = Data(x=xTrain, edge_index=edge_list, edge_attr=edge_weights, y=yTrain)
        aux = aux.to(device)
        dataTrain.append(aux)
        aux = Data(x=xValid, edge_index=edge_list, edge_attr=edge_weights, y=yValid)
        aux = aux.to(device)
        dataValid.append(aux)
        aux = Data(x=xTest, edge_index=edge_list, edge_attr=edge_weights, y=yTest)
        aux = aux.to(device)
        dataTest.append(aux)

        # GNN models

        modelList = []

        GNN = gnn.GNN('gnn1', 'gnn', F, MLP, False, K)
        GNN.to(device)
        modelList.append(GNN)

        GNN2 = gnn.GNN('gnn2', 'gnn', F2, MLP2, False, K)
        GNN2.to(device)
        modelList.append(GNN2)

        GNN3 = gnn.GNN('gnn3', 'gnn', F3, MLP3, False, K)
        GNN3.to(device)
        modelList.append(GNN3)

        SAGE = gnn.GNN('sage', 'sage', F, MLP, False)
        SAGE.to(device)
        #modelList.append(SAGE)

        GCN = gnn.GNN('gcn', 'gcn', F, MLP, False)
        GCN.to(device)
        #modelList.append(GCN)

        # Loss

        loss = torch.nn.CrossEntropyLoss()
        for args in [
                {'batch_size': 32, 'epochs': 500, 'opt': 'adam', 'opt_scheduler': 'none', 'opt_restart': 0, 'weight_decay': 5e-3, 'lr': 0.001},
            ]:
                args = objectview(args)

        # Training and testing

        for m_idx, model in enumerate(modelList):

            # Kernel

            consF = model.Flist[:-1] + model.MLPlist
            consK = model.Klist + list(np.ones(len(model.MLPlist),dtype=int)) 

            first_model = copy.deepcopy(model)
            weight_list_layers = first_model.get_weights()
            tensor_weight_list = []
            for weight_list in weight_list_layers:
                for i, weight in enumerate(weight_list):
                    if i == 0:
                        tensor_weight = torch.transpose(weight.detach(),0,1)
                        tensor_weight = tensor_weight.unsqueeze(-1)
                    else:
                        tensor_weight = torch.cat((tensor_weight,torch.transpose(weight.unsqueeze(-1).detach(),0,1)),axis=-1)
                tensor_weight_list.append(tensor_weight/torch.sqrt(torch.tensor(consF[i])))

            loader = DataLoader(dataTrain, batch_size=args.batch_size, shuffle=False)
            val_loader = DataLoader(dataValid, batch_size=nValid, shuffle=False)

            val_losses, losses, best_model, best_loss = train_test.train(loader, val_loader, model, loss, args, logistic=True) 

            print()

            print("Minimum validation set loss: {0}".format(min(val_losses)))
            print("Minimum loss: {0}".format(min(losses)))

            # Run test for our best model to save the predictions!
            test_loader = DataLoader(dataTest, batch_size=nTest, shuffle=False)
            test_loss = train_test.test(test_loader, best_model, logistic=True)
            print("Test loss: {0}".format(test_loss))
            gnn_results[rlz,idx,m_idx] = test_loss
            # Run test for our best model to save the predictions!
            test_loader2 = DataLoader(dataTest2, batch_size=nTest, shuffle=False)
            test_loss2 = train_test.test(test_loader2, best_model, logistic=True)
            print("Test loss transf.: {0}".format(test_loss2))
            gnn_results_transf[rlz,idx,m_idx] = test_loss2

            print()

            fig = plt.figure()
            plt.title('Training loss (MSE)')
            plt.plot(losses, label="training loss" + "-" + model.name)
            plt.plot(val_losses, label="test loss" + "-" + model.name)
            plt.legend()
            #plt.show()
            fig.savefig(os.path.join(saveDir,'loss-' + model.name +  "-" + str(rlz) + '-' + str(idx) + '.pdf'), bbox_inches = 'tight')
            plt.close()

            aux_loader = DataLoader(dataTrain, batch_size=nTrain, shuffle=False)
            for batch in aux_loader:
                yTrain = batch.y.detach()
                yTrain = torch.reshape(yTrain,(nTrain,n,-1))
                featsTrain = first_model.get_intermediate_features(batch)
            for i in range(len(featsTrain)):
                featsTrain[i] = torch.reshape(featsTrain[i],(nTrain,n,-1)).detach()

            kernel = ker.KernelRegression(len(consF)-1,consK,consF,torch.tensor(S,dtype=torch.float32,device=device),logistic=True)

            for batch in test_loader:
                yTest = batch.y.detach()
                yTest = torch.reshape(yTest,(nTest,n,-1))
                featsTest = first_model.get_intermediate_features(batch)
                gnn_preds = best_model(batch)
                gnn_preds = torch.reshape(gnn_preds,(nTest,n,-1)).detach()
            for i in range(len(featsTest)):
                featsTest[i] = torch.reshape(featsTest[i],(nTest,n,-1)).detach()

            kernel_preds = kernel.predict(featsTrain,tensor_weight_list,yTrain,featsTest)
            
            # Copy
            eig_plot[rlz,idx,m_idx,:] = kernel.get_eigenvalues()/n

            # Run test for our best model to save the predictions!
            test_loss = torch.nn.functional.cross_entropy(kernel_preds[0],yTest[0,:,0])
            print("Test loss for kernel regression: {0}".format(test_loss))
            kernel_results[rlz,idx,m_idx] = test_loss

            for batch in test_loader2:
                yTest2 = batch.y.detach()
                yTest2 = torch.reshape(yTest2,(nTest,N,-1))
                featsTest2 = first_model.get_intermediate_features(batch)
                gnn_preds2 = best_model(batch)
                gnn_preds2 = torch.reshape(gnn_preds2,(nTest,N,-1)).detach()
            for i in range(len(featsTest2)):
                featsTest2[i] = torch.reshape(featsTest2[i],(nTest,N,-1)).detach()

            kernel_preds2 = kernel.predict(featsTrain,tensor_weight_list,yTrain,featsTest2,torch.tensor(S_large,dtype=torch.float32).to(device))

             # Run test for our best model to save the predictions!
            test_loss2 = torch.nn.functional.cross_entropy(kernel_preds2[0],yTest2[0,:,0])
            print("Test loss for kernel regression transf.: {0}".format(test_loss2))
            kernel_results_transf[rlz,idx,m_idx] = test_loss2

        S_list.append(S_aux_list)

# Plots

# Kernel transferability plot

kernel_transf = np.abs(kernel_results-kernel_results_transf)/kernel_results_transf
kernel_transf_avg = np.mean(kernel_transf,axis=0)
kernel_transf_std = np.std(kernel_transf,axis=0)

for i in range(3):
    fig = plt.figure()
    #plt.title('Kernel transferability')
    plt.xlabel('Training graph size')
    plt.ylabel('Transferability error')
    plt.errorbar(n_vector,kernel_transf_avg[:,i],kernel_transf_std[:,i])
    #plt.show()
    fig.savefig(os.path.join(saveDir,'transf_kernel' + str(i) + '.pdf'), bbox_inches = 'tight')
    plt.close()

# GNN and kernel transferability plot

gnn_transf = np.abs(gnn_results-gnn_results_transf)/gnn_results_transf
gnn_transf_avg = np.mean(gnn_transf,axis=0)
gnn_transf_std = np.std(gnn_transf,axis=0)

for i in range(3):
    fig = plt.figure()
    #plt.title('Kernel transferability')
    plt.xlabel('Training graph size')
    plt.ylabel('Transferability error')
    plt.errorbar(n_vector,kernel_transf_avg[:,i],kernel_transf_std[:,i],label='GNTK')
    plt.errorbar(n_vector,gnn_transf_avg[:,i],gnn_transf_std[:,i],label='GNN')
    plt.legend()
    #plt.show()
    fig.savefig(os.path.join(saveDir,'transf_kernel_gnn' + str(i) + '.pdf'), bbox_inches = 'tight')
    plt.close()

# Copy

k=0
for rlz in range(n_realizations):
    fig = plt.figure()
    for i in range(len(n_vector)):
        plt.scatter(np.arange(0,3),eig_plot[rlz,i,:,k]/eig_plot[rlz,-1,:,k], label='n='+str(n_vector[i]))
    plt.xlabel('Width')
    plt.xticks(np.arange(0,3),['10','50','100'])
    plt.legend()
    fig.savefig(os.path.join(saveDir,'eigenvalues-' + str(rlz) + '.pdf'), bbox_inches = 'tight')
    plt.close()
    
fig = plt.figure()
for i in range(len(n_vector)):
    i = len(n_vector)-1-i
    plt.scatter(np.mean(eig_plot[:,i,:,k]/eig_plot[:,-1,:,k],axis=0),np.arange(0,3),label='n='+str(n_vector[i]))
    plt.errorbar(np.mean(eig_plot[:,i,:,k]/eig_plot[:,-1,:,k],axis=0),np.arange(0,3),xerr=np.std(eig_plot[:,i,:,k]/eig_plot[:,-1,:,k],axis=0),fmt="o")
plt.ylabel('Width')
plt.yticks(np.arange(0,3),['10','50','100'])
plt.legend(ncols=3)
plt.ylim([-1, 3.5])
fig.savefig(os.path.join(saveDir,'eigenvalues.pdf'), bbox_inches = 'tight')
plt.close()

with open(os.path.join(saveDir,'results.txt'), 'w') as f:
    f.write('gnn results: ')
    f.write(str(gnn_results))
    f.write('\n')
    f.write('gnn results transf: ')
    f.write(str(gnn_results_transf))
    f.write('\n')
    f.write('kernel results: ')
    f.write(str(kernel_results))
    f.write('\n')
    f.write('kernel results transf: ')
    f.write(str(kernel_results_transf))

pkl.dump(eig_plot, open(os.path.join(saveDir,'eigs.p'), 'wb'))
