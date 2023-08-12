
import numpy as np

def compute_top_k_eigenvectors(matrix, k):
    """
    计算矩阵的前k个特征向量，并按特征值从大到小排序。
    
    参数：
    matrix (numpy.ndarray): 输入的矩阵。
    k (int): 要选择的特征向量数量。
    
    返回：
    top_k_eigenvectors (numpy.ndarray): 排序后的前k个特征向量。
    """
    # 计算特征向量和特征值
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    
    # 对特征值进行排序
    sorted_indices = np.argsort(eigenvalues.real)[::-1]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    
    # 选择前k个特征向量
    top_k_eigenvectors = sorted_eigenvectors[:, :k]
    
    return top_k_eigenvectors

# 创建输入矩阵
#A = np.array([[1.0, 2.0,3.0], [3.0, 4.0,5.0],[5.0, 6.0,7.0]])
# A=np.diag((1, 2, 3,4,5,6))
A=np.diag((6, 5, 4,3,2,1))

# 指定要选择的特征向量数量
k = 4

# 调用函数计算并输出结果
top_k_eigenvectors = compute_top_k_eigenvectors(A, k)
print("Top", k, "Eigenvectors:")
print(top_k_eigenvectors)

# import torch
# import torch.linalg as la
# # 创建输入矩阵
# A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
# # 计算特征向量和特征值
# eigenvalues, eigenvectors = la.eig(A)
# # 对特征值进行排序
# sorted_indices = torch.argsort(eigenvalues.real, descending=True)
# sorted_eigenvalues = eigenvalues[sorted_indices]
# sorted_eigenvectors = eigenvectors[:, sorted_indices]
# print("Sorted Eigenvalues:")
# print(sorted_eigenvalues)
# print("\nSorted Eigenvectors:")
# print(sorted_eigenvectors)



# import numpy as np

# # 创建输入矩阵
# # A = np.array([[1.0, 2.0], [3.0, 4.0]])
# A=np.diag((1, 2, 3))

# # 计算特征向量和特征值
# eigenvalues, eigenvectors = np.linalg.eig(A)

# # 对特征值进行排序
# sorted_indices = np.argsort(eigenvalues.real)[::-1]
# sorted_eigenvalues = eigenvalues[sorted_indices]
# sorted_eigenvectors = eigenvectors[:, sorted_indices]

# print("Sorted Eigenvalues:")
# print(sorted_eigenvalues)
# print("\nSorted Eigenvectors:")
# print(sorted_eigenvectors)
