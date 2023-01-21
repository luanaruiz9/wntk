# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 16:12:10 2023

@author: Luana Ruiz
"""

import numpy as np
import scipy.spatial.distance as spy
import networkx as nx
import matplotlib.pyplot as plt

def createGraph(n, l=50, distMax=15, ratio=0.1):
    k = int(ratio*n)                                         
    coordinates = l*np.random.rand(n,2)
    distanceMatrix = spy.cdist(coordinates,coordinates)
    S = distanceMatrix
    threshold = np.sort(S,axis=-1)[:,k]
    S[S > np.tile(threshold,(n,1))] = 0
    S = S > 0
    S[S + np.transpose(S) > 0] = 1
    #D = np.power(np.sum(S,axis=1),-0.5)
    #S = D*S
    #S = np.transpose(D)*S
    commG = nx.from_numpy_matrix(S)
    plt.figure()
    nx.draw(commG, coordinates)
    plt.draw()
    plt.close()
    return S/n

def createSBM(n,p,q,n1):
    n1 = int(n1)
    n2 = n-n1
    Ptop = np.concatenate((p*np.ones((n1,n1)),q*np.ones((n1,n2))),axis=1)
    Pbottom = np.concatenate((q*np.ones((n2,n1)),p*np.ones((n2,n2))),axis=1)
    P = np.concatenate((Ptop,Pbottom),axis=0)
    S = np.random.binomial(1,P).squeeze()
    S[S + np.transpose(S) > 0] = 1
    commG = nx.from_numpy_matrix((S-np.eye(n))>0)
    coordinates = np.zeros((n,2))
    coordinates[0:n1,0] = np.arange(0,n1*0.2,0.2)
    coordinates[n1:n,0] = np.arange((n1-n2)*0.2/2,(n1-n2)*0.2/2+n2*0.2,0.2)
    coordinates[0:n1,1] = 0.5*np.random.rand(n1)
    coordinates[n1:n,1] = 0.05*n + 0.5*np.random.rand(n2)
    plt.figure()
    nx.draw(commG,coordinates)
    plt.draw()
    plt.close()
    return S/n