import numpy as np
import torch
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge

# NTK

zeroTolerance = 1e-9

class KernelRegression():

	def __init__(self, L, K, F, S, logistic=False):
		self.L = L 
		self.K = K 
		self.F = F 
		self.S = S
		self.logistic = logistic
		self.kernel = None
		self.reg = None

	def computeGradient(self, x, weights, S=None): # x is L-long list with N x n x F[l] elts
										   # weights is L-long list with F[l+1] x F[l] x K[l] elts
		L = self.L # scalar
		K = self.K # len L
		F = self.F # len L + 1
		if S is None:
			S = self.S # n x n
		N = x[0].shape[0]
		n = x[0].shape[1]

		grads = []

		for l in range(L):
			gradL = torch.zeros(N, n, F[l+1], F[l], K[l],device=S.device)
			Sk = torch.unsqueeze(x[l],2) # N x n x 1 x F[l]
			Sk = torch.tile(Sk,(1,1,F[l+1],1)) # N x n x F[l+1] x F[l]
			gradL[:,:,:,:,0] = Sk # N, n, F[l+1], F[l], K[l]
			S = torch.reshape(S,(1,n,n)) # 1 x n x n
			for k in range(K[l]-1):
				Sk = torch.matmul(S,torch.permute(Sk,(0,2,1,3))) # N, F[l+1], n, F[l]
				Sk = torch.permute(Sk,(0,2,1,3)) # N, n, F[l+1], F[l]
				gradL[:,:,:,:,k+1] = Sk # N, n, F[l+1], F[l], K[l]
			if l < L-1:
				gradL[gradL < 0] = 0
			for l2 in range(l+1,L):
				Sk = gradL # N, n, F[l+1], F[l], K[l]
				gradL = torch.unsqueeze(gradL,5) 
				gradL = torch.tile(gradL,(1,1,1,1,1,K[l2])) # N, n, F[l+1], F[l], K[l], K[l+1]
				gradL[:,:,:,:,:,0] = Sk
				for k in range(K[l2]-1):
					Sk = torch.matmul(S, torch.permute(Sk,(0,2,1,3))) # N, F[l+1], n, F[l]
					Sk = torch.permute(Sk,(0,2,1,3)) # N, n, F[l+1], F[l]
					gradL[:,:,:,:,:,k+1] = Sk # N, n, F[l+1], F[l], K[l], K[l+1]
				gradL = torch.permute(gradL,(0,1,3,4,2,5)) # N, n, F[l], K[l], F[l+1], K[l+1]
				gradL = torch.reshape(gradL,(N,n,F[l],K[l],-1)) # N, n, F[l], K[l], F[l+1]K[l+1]
				gradL = torch.permute(gradL,(0,1,2,4,3)) # N, n, F[l], F[l+1]K[l+1], K[l]
				gradL = torch.matmul(torch.reshape(weights[l2],(1,F[l2+1],-1)),gradL) # N, n, F[l], F[l2+1], K[l]
				gradL = torch.permute(gradL,(0,1,3,2,4)) # N, n, F[l2+1], F[l], K[l]
				if l2 < L-1:
					gradL[gradL < 0] = 0
			gradL = torch.reshape(gradL,(N,n,F[-1],-1))
			if self.logistic:
				y = torch.nn.functional.softmax(x[-1]) # N x n x F[-1]
				soft = torch.reshape(torch.kron(y,y),(N,N,n,n,F[-1],F[-1]))
				soft = -torch.permute(soft,(1,3,5,0,2,4)) # N x n x F[-1] x N x n x F[-1]
				diag = torch.diag(torch.reshape(y*(1-y),(N*n*F[-1],)))
				soft = soft + torch.reshape(diag, (N,n,F[-1],N,n,F[-1]))
				soft = torch.reshape(soft,(N,n,F[-1],-1))
				gradL = torch.reshape(gradL,(N*n*F[-1],-1))
				gradL = torch.max(torch.matmul(soft,gradL),dim=2,keepdim=True)[0]
			if l == 0:
				grads = gradL
			else:
				grads = torch.cat((grads,gradL),axis=-1)

		return grads

	def fit(self, xTrain, weights, yTrain):
		gradsTrain = self.computeGradient(xTrain, weights)
		nTrain = gradsTrain.shape[0]
		n = gradsTrain.shape[1]
		Fout = self.F[-1]
		kernel = torch.tensordot(gradsTrain, gradsTrain, dims=([3],[3])) # nTrain x n x Fout x nTrain x n x Fout
		if not self.logistic:
			self.kernel = torch.reshape(kernel,(nTrain*n*Fout,nTrain*n*Fout))
			X = self.kernel.cpu().numpy()
			y = torch.reshape(yTrain,(nTrain*n*Fout,)).cpu().numpy()
			reg = Ridge(alpha=0.0,fit_intercept=False,solver='lsqr').fit(X,y)
		else:
			class_weights = dict()
			for i in range(Fout):
				class_weights[i] = 1
			self.kernel = torch.reshape(kernel,(nTrain*n,nTrain*n))
			X = self.kernel.cpu().numpy()
			y = torch.reshape(yTrain,(nTrain*n,)).cpu().numpy()
			reg = LogisticRegression(penalty='none',class_weight=class_weights,fit_intercept=False).fit(X,y)
		return reg

	def predict(self, xTrain, weights, yTrain, xTest, S=None, reg=None):
		gradsTrain = self.computeGradient(xTrain, weights)
		nTrain = gradsTrain.shape[0]
		n = gradsTrain.shape[1]
		Fout = self.F[-1]        
		gradsTest = self.computeGradient(xTest, weights, S)
		n2 = gradsTest.shape[1]
		nTest = gradsTest.shape[0]
		kernel = torch.tensordot(gradsTrain, gradsTest, dims=([3],[3])) # nTrain x n x Fout x nTest x n x Fout
		if not self.logistic:
			kernel = torch.reshape(kernel,(nTrain*n*Fout,nTest,n2,Fout))
		else:
			kernel = torch.reshape(kernel,(nTrain*n,nTest,n2,1))
		if self.reg is None:
			reg = self.fit(xTrain, weights, yTrain)
			self.reg = reg 
		else:
			reg = self.reg
		if not self.logistic:
			X = torch.reshape(kernel,(nTrain*n*Fout,nTest*n2*Fout)).cpu().numpy()
			predictions = reg.predict(np.transpose(X))
		else:
			X = torch.reshape(kernel,(nTrain*n,nTest*n2)).cpu().numpy()
			predictions = reg.predict_proba(np.transpose(X))
		predictions = torch.tensor(predictions,device=gradsTrain.device)
		return torch.reshape(predictions,(nTest,n2,Fout))
    
	def get_eigenvalues(self):
		L, _ = torch.lobpcg(self.kernel, k=6)
		return L.cpu().numpy()


 
		


