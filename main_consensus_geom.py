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
from graph import createGraph, createSBM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 14

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# Arch. parameters

F = [1,10]
MLP = [10,1]
K = [2]

F2 = [1,50]
MLP2 = [50,1]
K2 = [2]

F3 = [1,250]
MLP3 = [250,1]
K3 = [2]

thisFilename = 'op_dyn_geom' # This is the general name of all related files

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

n_realizations = 5

nTrain = 300
nValid = int(0.1*nTrain)
nTest = int(0.1*nTrain)

mu = 0
sigma = 2

p = 0.1
q = 0.05

# Graph

n_vector = [20, 30, 40, 50, 60, 70, 80, 90, 100]
N = 300

gnn_results = np.zeros((n_realizations, len(n_vector), 3))
kernel_results = np.zeros((n_realizations, len(n_vector), 3))
gnn_results_transf = np.zeros((n_realizations, len(n_vector), 3))
kernel_results_transf = np.zeros((n_realizations, len(n_vector), 3))
x_label = np.zeros((n_realizations, nTest))
y_plot_1 = np.zeros((n_realizations, len(n_vector), 3, nTest))
y_plot_2 = np.zeros((n_realizations, len(n_vector), 3, nTest))
y_plot_3 = np.zeros((n_realizations, len(n_vector), 3, nTest))

S_list = []
S_large_list = []

# Transferability data
    
S_large = createGraph(N)
#S_large = createSBM(N,p,q,N/2)
S_large_list.append(S_large)
edges = np.nonzero(S_large)
nb_edges_large = edges[0].shape[0]
edge_list_large = np.zeros((2,nb_edges_large))
for i in range(2):
    edge_list_large[i,:] = edges[i]
edge_weights_large = S_large[edges]

dataTest2 = []

xTest2 = np.random.multivariate_normal(mu*np.ones(N),sigma*np.eye(N),size=nTest)
xTest2 = np.expand_dims(xTest2,axis=2)
yTest2 = np.mean(xTest2, axis=-1, keepdims=True)
for t in range(xTest2.shape[0]):
    data = Data(x=torch.tensor(xTest2[t],dtype=torch.float32), edge_index=torch.tensor(edge_list_large,dtype=torch.long), 
        edge_attr=torch.tensor(edge_weights_large,dtype=torch.float32), y=torch.tensor(yTest2[t],dtype=torch.float32))
    data = data.to(device)
    dataTest2.append(data)

for rlz in range(n_realizations):
    
    S_aux_list = []
    
    for idx,n in enumerate(n_vector):
        
        # Create training graph
        
        S = createGraph(n)
        #S = createSBM(n,p,q,n/2)
        S_aux_list.append(S)
        edges = np.nonzero(S)
        nb_edges = edges[0].shape[0]
        edge_list = np.zeros((2,nb_edges))
        for i in range(2):
            edge_list[i,:] = edges[i]
        edge_weights = S[edges]
        
        # Training data
        
        xTrain = np.random.multivariate_normal(mu*np.ones(n),sigma*np.eye(n),size=nTrain)
        xTrain = np.expand_dims(xTrain,axis=2)
        yTrain = np.mean(xTrain, axis=-1, keepdims=True)
        xValid = np.random.multivariate_normal(mu*np.ones(n),sigma*np.eye(n),size=nValid)
        xValid = np.expand_dims(xValid,axis=2)
        yValid = np.mean(xValid, axis=-1, keepdims=True)
        xTest = np.random.multivariate_normal(mu*np.ones(n),sigma*np.eye(n),size=nTest)
        xTest = np.expand_dims(xTest,axis=2)
        yTest = np.mean(xTest, axis=-1, keepdims=True)
        
        # Train GNN
        
        dataTrain = []
        dataValid = []
        dataTest = []
        
        for t in range(xTrain.shape[0]):
            data = Data(x=torch.tensor(xTrain[t],dtype=torch.float32), edge_index=torch.tensor(edge_list,dtype=torch.long), 
                edge_attr=torch.tensor(edge_weights,dtype=torch.float32), y=torch.tensor(yTrain[t],dtype=torch.float32))
            data = data.to(device)
            dataTrain.append(data)
        for t in range(xValid.shape[0]):
            data = Data(x=torch.tensor(xValid[t],dtype=torch.float32), edge_index=torch.tensor(edge_list,dtype=torch.long), 
                edge_attr=torch.tensor(edge_weights,dtype=torch.float32), y=torch.tensor(yValid[t],dtype=torch.float32))
            data = data.to(device)
            dataValid.append(data)
        for t in range(xTest.shape[0]):
            data = Data(x=torch.tensor(xTest[t],dtype=torch.float32), edge_index=torch.tensor(edge_list,dtype=torch.long), 
                edge_attr=torch.tensor(edge_weights,dtype=torch.float32), y=torch.tensor(yTest[t],dtype=torch.float32))
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
        
        loss = torch.nn.MSELoss()
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
            
            loader = DataLoader(dataTrain, batch_size=args.batch_size, shuffle=False)
            val_loader = DataLoader(dataValid, batch_size=nValid, shuffle=False)
            
            val_losses, losses, best_model, best_loss = train_test.train(loader, val_loader, model, loss, args) 
        
            print()
        
            print("Minimum validation set loss: {0}".format(min(val_losses)))
            print("Minimum loss: {0}".format(min(losses)))
        
            # Run test for our best model to save the predictions!
            test_loader = DataLoader(dataTest, batch_size=nTest, shuffle=False)
            test_loss = train_test.test(test_loader, best_model)
            print("Test loss: {0}".format(test_loss))
            gnn_results[rlz,idx,m_idx] = test_loss
            # Run test for our best model to save the predictions!
            test_loader2 = DataLoader(dataTest2, batch_size=nTest, shuffle=False)
            test_loss2 = train_test.test(test_loader2, best_model)
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
                
            #first_model = best_model # Comment!
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
            test_loss = torch.nn.functional.mse_loss(kernel_preds,yTest)
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
            test_loss2 = torch.nn.functional.mse_loss(kernel_preds2,yTest2)
            print("Test loss for kernel regression transf.: {0}".format(test_loss2))
            kernel_results_transf[rlz,idx,m_idx] = test_loss2
            
            # Plot against projection of xTest onto graph eigenvectors
            L, V = np.linalg.eig(S_large)
            ind = np.argsort(-np.abs(L))
            V = V[:,ind]
            V = torch.tensor(V,dtype=torch.float32)
            xPlot = torch.tensordot(featsTest2[0].squeeze().cpu(),V[:,0],dims=([1],[0]))
            ind = np.argsort(xPlot)
            x_label[rlz,:] = xPlot[ind]
            yPlot = torch.tensordot(kernel_preds2.squeeze().cpu(),V[:,0],dims=([1],[0]))
            yPlot2 = torch.tensordot(gnn_preds2.squeeze().cpu(),V[:,0],dims=([1],[0]))
            yPlot3 = torch.tensordot(yTest2.squeeze().cpu(),V[:,0],dims=([1],[0]))
            y_plot_1[rlz,idx,m_idx,:] = yPlot[ind]
            y_plot_2[rlz,idx,m_idx,:] = yPlot2[ind]
            y_plot_3[rlz,idx,m_idx,:] = yPlot3[ind]
            
            fig = plt.figure()
            #plt.title('Projection along 1st eigenvector of graph')
            plt.xlabel('$[\hat{\mathbf{x}}]_1$, sorted')
            plt.ylabel('$[\hat{\mathbf{y}}]_1$')
            plt.plot(xPlot[ind],yPlot[ind],label="GNTK")  
            plt.plot(xPlot[ind],yPlot2[ind],label="GNN") 
            plt.plot(xPlot[ind],yPlot3[ind],label="True") 
            plt.legend()
            #plt.show()
            fig.savefig(os.path.join(saveDir,'projection-' + model.name + '-' + str(rlz) + '-' + str(idx) + '.pdf'), bbox_inches = 'tight')
            plt.close()
            print()
            
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


for i in range(len(n_vector)):
    for m_idx in range(3):
        fig = plt.figure()
        #plt.title('Projection along 1st eigenvector of graph')
        plt.xlabel('$[\hat{\mathbf{x}}]_1$, sorted')
        plt.ylabel('$[\hat{\mathbf{y}}]_1$')
        plt.plot(np.mean(x_label,axis=0),np.mean(y_plot_3[:,i,m_idx],axis=0),label="True",color='black') 
        plt.fill_between(np.mean(x_label,axis=0),np.mean(y_plot_1[:,i,m_idx],axis=0)-np.std(y_plot_1[:,i,m_idx],axis=0),
                np.mean(y_plot_1[:,i,m_idx],axis=0)+np.std(y_plot_1[:,i,m_idx],axis=0),alpha=0.2,facecolor='#089FFF',
                linewidth=1, antialiased=True) 
        plt.plot(np.mean(x_label,axis=0),np.mean(y_plot_1[:,i,m_idx],axis=0),label="GNTK",color='#089FFF')  
        plt.fill_between(np.mean(x_label,axis=0),np.mean(y_plot_2[:,i,m_idx],axis=0)-np.std(y_plot_2[:,i,m_idx],axis=0),
                np.mean(y_plot_2[:,i,m_idx],axis=0)+np.std(y_plot_2[:,i,m_idx],axis=0),alpha=0.2,facecolor='#FF9848',
                linewidth=1, antialiased=True) 
        plt.plot(np.mean(x_label,axis=0),np.mean(y_plot_2[:,i,m_idx],axis=0),label="GNN",color='#FF9848') 
        plt.legend()
        #plt.show()
        fig.savefig(os.path.join(saveDir,'final_projection-gnn' + str(m_idx) + '-' + str(i) + '.pdf'), bbox_inches = 'tight')
        plt.close()
        
        fig2 = plt.figure()
        #plt.title('Projection along 1st eigenvector of graph')
        plt.xlabel('$[\hat{\mathbf{x}}]_1$, sorted')
        plt.ylabel('$[\hat{\mathbf{y}}]_1$')
        plt.plot(np.mean(x_label,axis=0),np.mean(y_plot_3[:,i,m_idx],axis=0),label="True",color='black') 
        plt.fill_between(np.mean(x_label,axis=0),np.mean(y_plot_1[:,i,m_idx],axis=0)-np.std(y_plot_1[:,i,m_idx],axis=0),
                np.mean(y_plot_1[:,i,m_idx],axis=0)+np.std(y_plot_1[:,i,m_idx],axis=0),alpha=0.2,facecolor='#089FFF',
                linewidth=1, antialiased=True) 
        plt.plot(np.mean(x_label,axis=0),np.mean(y_plot_1[:,i,m_idx],axis=0),label="GNTK",color='#089FFF')  
        #plt.fill_between(np.mean(x_label,axis=0),np.mean(y_plot_2[:,i,m_idx],axis=0)-np.std(y_plot_2[:,i,m_idx],axis=0),
        #        np.mean(y_plot_2[:,i,m_idx],axis=0)+np.std(y_plot_2[:,i,m_idx],axis=0),alpha=0.2,facecolor='#FF9848',
        #        linewidth=1, antialiased=True) 
        #plt.plot(np.mean(x_label,axis=0),np.mean(y_plot_2[:,i,m_idx],axis=0),label="GNN",color='#FF9848') 
        plt.legend()
        #plt.show()
        fig2.savefig(os.path.join(saveDir,'final_projection-gntk' + str(m_idx) + '-' + str(i) + '.pdf'), bbox_inches = 'tight')
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