# Expand dims of features!!!!!!! (look consensus)
# Check if pyG is doing the right thing when doing sparse matrix multiplication. I think so

import os
import datetime
import numpy as np
import torch
import copy

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

import matplotlib.pyplot as plt

import gnn
import train_test
import kernel as ker
import movie2 as movie

zeroTolerance = 1e-9

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

thisFilename = 'movie' # This is the general name of all related files

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

n_realizations = 1
n_vector = [50, 100, 150, 200, 250]

gnn_results = np.zeros((n_realizations, len(n_vector), 3))
kernel_results = np.zeros((n_realizations, len(n_vector), 3))
gnn_results_transf = np.zeros((n_realizations, len(n_vector), 3))
kernel_results_transf = np.zeros((n_realizations, len(n_vector), 3))
x_label = np.zeros((n_realizations, 61))
y_plot_1 = np.zeros((n_realizations, len(n_vector), 3, 61))
y_plot_2 = np.zeros((n_realizations, len(n_vector), 3, 61))
y_plot_3 = np.zeros((n_realizations, len(n_vector), 3, 61))

S_list = []
S_large_list = []

# Arch. parameters

F0 = 1
C = 1

F = [F0,5]
MLP = [5,C]
K = [2]

F2 = [F0,10]
MLP2 = [10,C]
K2 = [2]

F3 = [F0,50]
MLP3 = [50,C]
K3 = [2]

for rlz in range(n_realizations):
    
    S_aux_list = []
    
    for idx, min_r in enumerate(n_vector):
        
        # Create training graph
        
        X, idxContact = movie.load_data(movie=257, n=min_r, min_ratings=10)
        nTotal = X.shape[0] # total number of users (samples)
        permutation = np.random.permutation(nTotal)
        nTrain = int(np.ceil(0.9*nTotal)) # number of training samples
        idxTrain = permutation[0:nTrain] # indices of training samples
        nTest = nTotal-nTrain # number of test samples
        idxTest=permutation[nTrain:nTotal] # indices of test samples

        # Creating and sparsifying the graph

        S = movie.create_graph(X=X, idxTrain=idxTrain, knn=5)
        S = (S>zeroTolerance).astype(int)
        print(S.shape)
        n = S.shape[0]

        # Creating the training and test sets

        xTrain, yTrain, xTest, yTest = movie.split_data(X, idxTrain, idxTest, idxContact)
        nTrain = xTrain.shape[0]
        nTest = xTest.shape[0]
        nValid = nTest
        xValid = xTest
        yValid = yTest
        
        S_aux_list.append(S)
        edges = np.nonzero(S)
        nb_edges = edges[0].shape[0]
        edge_list = np.zeros((2,nb_edges))
        for i in range(2):
            edge_list[i,:] = edges[i]
        edge_weights = S[edges]
        
        # Transferability graph and data
        
        X2, idxContact2 = movie.load_data(movie=257, n=500, nb_ratings=10)

        S_large = movie.create_graph(X=X2, idxTrain=idxTrain, knn=5)
        S_large = S_large>zeroTolerance
        N = S_large.shape[0]
        
        _, _, xTest2, yTest2 = movie.split_data(X2, idxTrain, idxTest, idxContact2)
        nTest2 = xTest2.shape[0]
        
        # Transferability data
        
        S_large_list.append(S_large)
        edges_large = np.nonzero(S_large)
        nb_edges_large = edges_large[0].shape[0]
        edge_list_large = np.zeros((2,nb_edges_large))
        for i in range(2):
            edge_list_large[i,:] = edges_large[i]
        edge_weights_large = S_large[edges_large]
        
        dataTest2 = []
        
        for t in range(xTest2.shape[0]):
            data2 = Data(x=xTest2[t].float(), edge_index=torch.tensor(edge_list_large,dtype=torch.long), 
                edge_attr=torch.tensor(edge_weights_large,dtype=torch.float32), y=yTest2[t].float())
            data2 = data2.to(device)
            dataTest2.append(data2)
        
        # Train GNN
        
        dataTrain = []
        dataValid = []
        dataTest = []
        
        for t in range(xTrain.shape[0]):
            data = Data(x=xTrain[t].float(), edge_index=torch.tensor(edge_list), 
                edge_attr=torch.tensor(edge_weights,dtype=torch.float32), y=yTrain[t].float())
            data = data.to(device)
            dataTrain.append(data)
        for t in range(xValid.shape[0]):
            data = Data(x=xValid[t].float(), edge_index=torch.tensor(edge_list), 
                edge_attr=torch.tensor(edge_weights,dtype=torch.float32), y=yValid[t].float())
            data = data.to(device)
            dataValid.append(data)
        for t in range(xTest.shape[0]):
            data = Data(x=xTest[t].float(), edge_index=torch.tensor(edge_list), 
                edge_attr=torch.tensor(edge_weights,dtype=torch.float32), y=yTest[t].float())
            data = data.to(device)
            dataTest.append(data)
        
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
        
        loss = movie.movieMSELoss
        for args in [
                {'batch_size': 32, 'epochs': 20, 'opt': 'adam', 'opt_scheduler': 'none', 'opt_restart': 0, 'weight_decay': 5e-3, 'lr': 0.001},
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
            
            val_losses, losses, best_model, best_loss = train_test.train(loader, val_loader, model, loss, args, idxContact, n) 
        
            print()
        
            print("Minimum validation set loss: {0}".format(min(val_losses)))
            print("Minimum loss: {0}".format(min(losses)))
        
            # Run test for our best model to save the predictions!
            test_loader = DataLoader(dataTest, batch_size=nTest, shuffle=False)
            test_loss = train_test.test(test_loader, best_model, idxContact, n)
            print("Test loss: {0}".format(test_loss))
            gnn_results[rlz,idx,m_idx] = test_loss
            # Run test for our best model to save the predictions!
            test_loader2 = DataLoader(dataTest2, batch_size=nTest, shuffle=False)
            test_loss2 = train_test.test(test_loader2, best_model, idxContact2, N)
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
  
            kernel = ker.KernelRegression(len(consF)-1,consK,consF,torch.tensor(S,dtype=torch.float32).to(device))
                
            for batch in test_loader:
                yTest = batch.y.detach()
                yTest = torch.reshape(yTest,(nTest,n,-1))
                featsTest = first_model.get_intermediate_features(batch)
                gnn_preds = best_model(batch)
                gnn_preds = torch.reshape(gnn_preds,(nTest,n,-1)).detach()
            for i in range(len(featsTest)):
                featsTest[i] = torch.reshape(featsTest[i],(nTest,n,-1)).detach()
    
            kernel_preds = kernel.predict(featsTrain,tensor_weight_list,yTrain,featsTest)
        
            # Run test for our best model to save the predictions!
            test_loss = torch.nn.functional.mse_loss(kernel_preds[:,idxContact],yTest[:,idxContact])
            print("Test loss for kernel regression: {0}".format(test_loss))
            kernel_results[rlz,idx,m_idx] = test_loss
            
            for batch in test_loader2:
                yTest2 = batch.y.detach()
                print(yTest2.shape)
                yTest2 = torch.reshape(yTest2,(nTest2,N,-1))
                featsTest2 = first_model.get_intermediate_features(batch)
                gnn_preds2 = best_model(batch)
                gnn_preds2 = torch.reshape(gnn_preds2,(nTest2,N,-1)).detach()
            for i in range(len(featsTest2)):
                featsTest2[i] = torch.reshape(featsTest2[i],(nTest2,N,-1)).detach()
    
            kernel_preds2 = kernel.predict(featsTrain,tensor_weight_list,yTrain,featsTest2,torch.tensor(S_large,dtype=torch.float32).to(device))
            
             # Run test for our best model to save the predictions!
            test_loss2 = torch.nn.functional.mse_loss(kernel_preds2[:,idxContact2],yTest2[:,idxContact2])
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
    plt.xlabel('Graph size ($n$)')
    plt.ylabel('Relative MSE difference')
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