from sklearn.neighbors import kneighbors_graph,radius_neighbors_graph
import torch
import torch.nn.functional as F
import numpy as np
from scipy.linalg import eigh
from scipy import ndimage,sparse

# torch cluster
# from torch_cluster import knn_graph
def to_graph(X,sigma,e,n_neighbors,similarity_matrix,knn_aprox,eps=1e-7):
    '''
    Compute similarity matrix.
    return: similarity matrix
    '''
    if type(X) == torch.Tensor:
      X = X.detach().to("cpu").numpy()
      
    if similarity_matrix == 'e-NG':
      A = radius_neighbors_graph(X, e, mode='connectivity',include_self=False, n_jobs=-1)
      return A
    
    elif similarity_matrix == 'full':
        pass
    
    elif similarity_matrix == 'precomputed':
      return A

    else:
        
        if knn_aprox:
            A = PyNNDescentTransformer(n_neighbors=n_neighbors,metric="euclidean",n_jobs=-1).fit_transform(X)
        else:
            A = kneighbors_graph(X, n_neighbors, mode='distance',include_self=False, n_jobs=-1)
            
        if sigma == 'max':
            sigma_2 = 2*np.power(A.max(axis=1).toarray(),2) + eps
            A = ( ( A.power(2,dtype=np.float32) ).multiply(1/sigma_2) ).tocsr()
            np.exp(-A.data,out=A.data)
        
        elif sigma == 'mean':
            sigma_2 = 2*np.power(A.sum(axis=1) / A.getnnz(axis=1).reshape(-1,1),2) + eps
            A = ( ( A.power(2,dtype=np.float32) ).multiply(1/sigma_2) ).tocsr()
            np.exp(-A.data,out=A.data)
        
        else:
            sigma_2 = 2*np.power(sigma,2) + eps
            A = ( ( A.power(2,dtype=np.float32) ).multiply(1/sigma_2) ).tocsr()
            np.exp(-A.data,out=A.data)
        
        if knn_aprox:
            A = A - sparse.identity(A.shape[0])

        if similarity_matrix == 'k-hNNG':
            return (A + A.T)/2
            
        if similarity_matrix == 'k-NNG':
            return A.maximum(A.T)

        if similarity_matrix == 'k-mNNG':
            return A.minimum(A.T)
