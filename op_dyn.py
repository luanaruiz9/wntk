# -*- coding: utf-8 -*-
"""
Created on Sun Jan  8 16:17:58 2023

@author: Luana Ruiz
"""

import numpy as np

class OpinionDynamics:
    
    def __init__(self,S,mu,sigma):
        
        self.S = S
        self.nodes = S.shape[1]
        self.mu = mu
        self.sigma = sigma

        self.iterations = 1000 
        self.epsilon = 0.3 #this is the confidence bound parameter. If two opinions are within this bound then they influence each other in the update, otherwise they don't. determines consensus vs polarization vs fragmentation (more than 2 final clusters). 
        self.c = 0.1 #influence parameter - determines speed of convergence.


    def opn_update(self, opn):
         n = len(opn)
         for i in range(n):
              set_nbhd=np.where(self.S[i,:]!=0)[0]
              set_withinconf = np.where(np.abs(opn-opn[i])<self.epsilon)[0]
              setj = list(set(set_nbhd).intersection(set(set_withinconf)))
              if len(setj)==0:
                   continue
              mean_opn_neighbors = np.mean(np.array([self.S[i,j] for j in setj]))
              opn[i]+= self.c*mean_opn_neighbors
         return opn
     
    def getData(self,m):
        opn0 = np.random.multivariate_normal(self.mu*np.ones(self.nodes),self.sigma*np.eye(self.nodes),size=m)
        opnT = np.zeros((m,self.nodes))
        
        for sample in range(m):
            opn = opn0[sample]
            count = 0
            for t in range(self.iterations): 
                 opn_new = self.opn_update(opn)
                 opn_old = opn
                 opn = opn_new
                 if np.linalg.norm(opn_old - opn_new) < 0.001: 
                     count+=1
                 else:
                    count = 0
                 if count==50:
                    convergence_time=t-50
                    break
            opnT[sample,:] = opn
            
        return opn0, opnT