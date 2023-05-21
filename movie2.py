# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 12:30:15 2020

@author: Luana Ruiz
"""

import numpy as np
import os
import zipfile # To handle zip files
import torch as torch

def load_data(movie, n, min_ratings):
    
    # Extract all from zip file
    dataDir=os.getcwd()
    zipObject = zipfile.ZipFile(os.path.join(dataDir,'ml-100k.zip'))
    zipObject.extractall(dataDir)
    zipObject.close()
    
    rawDataFilename = os.path.join(dataDir,'ml-100k','ratings.dat')
    
    # Initialize rating matrix
    rawMatrix = np.empty([0, 0]) 
    
    # From each row of u.data, extract userID, movieID and rating
    with open(rawDataFilename, 'r') as rawData:
        for dataLine in rawData:
            dataLineSplit = dataLine.rstrip('\n').split('::')
            userID = int(dataLineSplit[0])
            movieID = int(dataLineSplit[1])
            rating = int(dataLineSplit[2])
            if userID > rawMatrix.shape[0]:
                rowDiff = userID - rawMatrix.shape[0]
                zeroPadRows = np.zeros([rowDiff, rawMatrix.shape[1]])
                rawMatrix = np.concatenate((rawMatrix, zeroPadRows),
                                           axis = 0)
            if movieID > rawMatrix.shape[1]:
                colDiff = movieID - rawMatrix.shape[1]
                zeroPadCols = np.zeros([rawMatrix.shape[0], colDiff])
                rawMatrix = np.concatenate((rawMatrix, zeroPadCols),
                                           axis = 1)
                
            # Assign rating to rating matrix
            rawMatrix[userID - 1, movieID - 1] = rating
          
    # Define X
    X = rawMatrix
    
    # Count number of ratings per column, i.e., per movie
    nbRatingsCols = np.sum(X>0,axis=0)
    
    # Mask to identify movies with at least min_ratings
    print(nbRatingsCols[movie])
    mask = nbRatingsCols >= min_ratings

    # Save new index of the input argument "movie"
    idxMovie = np.sum(mask[0:movie])

    # Remove matrix columns
    idx = np.argwhere(mask>0).squeeze()
    X = X[:,idx.squeeze()]
    
    if n != 'all':
    # Select n movies
        mask2 = np.zeros(X.shape[1])
        mask2[:n-1] = 1
        mask2 = np.random.permutation(mask2)
        if mask2[idxMovie] == 0:
            mask2[idxMovie] = 1
        else:
            i = np.random.randint(0,X.shape[1])
            while mask2[i] == 1:
                i = np.random.randint(0,X.shape[1])
            mask2[i] = 1

        # Save new index of the input argument "movie"
        idxMovie = np.sum(mask2[0:idxMovie])

        # Remove matrix columns
        idx = np.argwhere(mask2>0).squeeze()
        X = X[:,idx.squeeze()]

    # Make sure there are no rows of all zeros
    nbRatingsRows = np.sum(X>0,axis=1)
    idx = np.argwhere(nbRatingsRows>0).squeeze()
    X=X[idx,:]
    
    # Return cleaned-up X and new index of input argument "movie"
    return X, int(idxMovie)


def create_graph(X, idxTrain, knn):
    
    # Everything below 1e-9 is considered zero
    zeroTolerance = 1e-9
    
    # Number of nodes is equal to the number of columns (movies)
    N = X.shape[1]
    
    # Isolating users used for training
    XTrain = np.transpose(X[idxTrain,:])
    
    # Calculating correlation matrix
    binaryTemplate = (XTrain > 0).astype(XTrain.dtype)
    sumMatrix = XTrain.dot(binaryTemplate.T)
    countMatrix = binaryTemplate.dot(binaryTemplate.T)
    countMatrix[countMatrix == 0] = 1
    avgMatrix = sumMatrix / countMatrix
    sqSumMatrix = (XTrain ** 2).dot(binaryTemplate.T)
    correlationMatrix = sqSumMatrix / countMatrix - avgMatrix ** 2
    
    # Normalizing by diagonal weights
    sqrtDiagonal = np.sqrt(np.diag(correlationMatrix))
    nonzeroSqrtDiagonalIndex = (sqrtDiagonal > zeroTolerance)\
                                                 .astype(sqrtDiagonal.dtype)
    sqrtDiagonal[sqrtDiagonal < zeroTolerance] = 1.
    invSqrtDiagonal = 1/sqrtDiagonal
    invSqrtDiagonal = invSqrtDiagonal * nonzeroSqrtDiagonalIndex
    normalizationMatrix = np.diag(invSqrtDiagonal)
    
    # Zero-ing the diagonal
    normalizedMatrix = normalizationMatrix.dot(
                            correlationMatrix.dot(normalizationMatrix)) \
                            - np.eye(correlationMatrix.shape[0])

    # Keeping only edges with weights above the zero tolerance
    normalizedMatrix[np.abs(normalizedMatrix) < zeroTolerance] = 0.
    W = normalizedMatrix
    
    # Sparsifying the graph
    WSorted = np.sort(W,axis=1)
    threshold = WSorted[:,-knn].squeeze()
    thresholdMatrix = (np.tile(threshold,(N,1))).transpose()
    W[W<thresholdMatrix] = 0
    
    # Normalizing by eigenvalue with largest magnitude
    E, V = np.linalg.eig(W)
    W = W/np.max(np.abs(E))
    
    return W
    
    
def split_data(X, idxTrain, idxTest, idxMovie):  
    
    N = X.shape[1]
    
    xTrain = X[idxTrain,:]
    idx = np.argwhere(xTrain[:,idxMovie]>0).squeeze()
    xTrain = xTrain[idx,:]
    yTrain = np.zeros(xTrain.shape)
    yTrain[:,idxMovie] = xTrain[:,idxMovie]
    xTrain[:,idxMovie] = 0
    
    xTrain = torch.tensor(xTrain)
    xTrain = xTrain.reshape([-1,1,N])
    yTrain = torch.tensor(yTrain)
    yTrain = yTrain.reshape([-1,1,N])
    
    xTest = X[idxTest,:]
    idx = np.argwhere(xTest[:,idxMovie]>0).squeeze()
    xTest = xTest[idx,:]
    yTest = np.zeros(xTest.shape)
    yTest[:,idxMovie] = xTest[:,idxMovie]
    xTest[:,idxMovie] = 0
    
    xTest = torch.tensor(xTest)
    xTest = xTest.reshape([-1,1,N])
    yTest = torch.tensor(yTest)
    yTest = yTest.reshape([-1,1,N])
    
    return xTrain, yTrain, xTest, yTest

def movieMSELoss(yHat,y,idxMovie,n):
    mse = nn.MSELoss()
    yHat = torch.reshape(yHat,(-1,n))
    y = torch.reshape(y,(-1,n))
    return mse(yHat[:,idxMovie].reshape([-1,1]),y[:,idxMovie].reshape([-1,1]))