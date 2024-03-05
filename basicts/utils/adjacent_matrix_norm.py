import numpy as np
import scipy.sparse as sp
from scipy.sparse import linalg
import torch

def calculate_symmetric_normalized_laplacian(adj: np.ndarray) -> np.matrix:
    """Calculate yymmetric normalized laplacian.
    Assuming unnormalized laplacian matrix is `L = D - A`,
    then symmetric normalized laplacian matrix is:
    `L^{Sym} =  D^-1/2 L D^-1/2 =  D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2`
    For node `i` and `j` where `i!=j`, L^{sym}_{ij} <=0.

    Args:
        adj (np.ndarray): Adjacent matrix A

    Returns:
        np.matrix: Symmetric normalized laplacian L^{Sym}
    """

    adj = sp.coo_matrix(adj)
    degree = np.array(adj.sum(1))
    # diagonals of D^{-1/2}
    degree_inv_sqrt = np.power(degree, -0.5).flatten()
    degree_inv_sqrt[np.isinf(degree_inv_sqrt)] = 0.
    matrix_degree_inv_sqrt = sp.diags(degree_inv_sqrt)   # D^{-1/2}
    symmetric_normalized_laplacian = sp.eye(
        adj.shape[0]) - matrix_degree_inv_sqrt.dot(adj).dot(matrix_degree_inv_sqrt).tocoo()
    return symmetric_normalized_laplacian


def calculate_scaled_laplacian(adj: np.ndarray, lambda_max: int = 2, undirected: bool = True) -> np.matrix:
    """Re-scaled the eigenvalue to [-1, 1] by scaled the normalized laplacian matrix for chebyshev pol.
    According to `2017 ICLR GCN`, the lambda max is set to 2, and the graph is set to undirected.
    Note that rescale the laplacian matrix is equal to rescale the eigenvalue matrix.
    `L_{scaled} = (2 / lambda_max * L) - I`

    Args:
        adj (np.ndarray): Adjacent matrix A
        lambda_max (int, optional): Defaults to 2.
        undirected (bool, optional): Defaults to True.

    Returns:
        np.matrix: The rescaled laplacian matrix.
    """

    if undirected:
        adj = np.maximum.reduce([adj, adj.T])
    laplacian_matrix = calculate_symmetric_normalized_laplacian(adj)
    if lambda_max is None:  # manually cal the max lambda
        lambda_max, _ = linalg.eigsh(laplacian_matrix, 1, which='LM')
        lambda_max = lambda_max[0]
    laplacian_matrix = sp.csr_matrix(laplacian_matrix)
    num_nodes, _ = laplacian_matrix.shape
    identity_matrix = sp.identity(
        num_nodes, format='csr', dtype=laplacian_matrix.dtype)
    laplacian_res = (2 / lambda_max * laplacian_matrix) - identity_matrix
    return laplacian_res


def calculate_symmetric_message_passing_adj(adj: np.ndarray) -> np.matrix:
    """Calculate the renormalized message passing adj in `GCN`.
    A = A + I
    return D^{-1/2} A D^{-1/2}

    Args:
        adj (np.ndarray): Adjacent matrix A

    Returns:
        np.matrix: Renormalized message passing adj in `GCN`.
    """

    # add self loop
    adj = adj + np.diag(np.ones(adj.shape[0], dtype=np.float32))
    # print("calculating the renormalized message passing adj, please ensure that self-loop has added to adj.")
    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    mp_adj = d_mat_inv_sqrt.dot(adj).transpose().dot(
        d_mat_inv_sqrt).astype(np.float32)
    return mp_adj


def calculate_transition_matrix(adj: np.ndarray) -> np.matrix:
    """Calculate the transition matrix `P` proposed in DCRNN and Graph WaveNet.
    P = D^{-1}A = A/rowsum(A)

    Args:
        adj (np.ndarray): Adjacent matrix A

    Returns:
        np.matrix: Transition matrix P
    """

    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(row_sum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv)
    prob_matrix = d_mat.dot(adj).astype(np.float32).todense()
    return prob_matrix

def calculate_symmetric_normalized_laplacian(adj: np.ndarray,data: str) -> np.matrix:
    # 调用calculate_normalized_laplacian函数，计算邻接矩阵的归一化拉普拉斯矩阵L和孤立点的个数
    L, isolated_point_num = calculate_normalized_laplacian(adj)
    # 特征值分解，得到L的特征值EigVal和特征向量EigVec
    EigVal, EigVec = np.linalg.eig(L.toarray())

    # 特征值排序，得到排序索引idx
    idx = EigVal.argsort()

    # 利用特征值的排序索引对特征值和特征向量排序
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])
    # 去掉孤立点的特征向量，取前x个特征向量组成图的嵌入X_spe
    if data=="PEMS04" or data=="PEMS-BAY":
        x = 64
    elif data=="PEMS08" or data=="TFA":
        x = 32
    elif data=="PEMS07":
        x = 96
    laplacian_pe = torch.from_numpy(EigVec[:, isolated_point_num + 1: x + isolated_point_num + 1]).float()
    # 设置嵌入的梯度为False，表示不需要更新
    laplacian_pe.require_grad = False
    # 将嵌入转换为numpy数组
    laplacian_pe = laplacian_pe.numpy()
    # 将嵌入转换为稀疏矩阵
    laplacian_pe = sp.csr_matrix(laplacian_pe)

    # 返回嵌入
    return laplacian_pe

def calculate_normalized_laplacian(adj: np.ndarray):
    # 将邻接矩阵转换为稀疏矩阵
    adj = sp.coo_matrix(adj)
    # 计算邻接矩阵的每一行的和，得到度矩阵D的对角线元素
    d = np.array(adj.sum(1))

    # 计算孤立点的个数，即度为0的点
    isolated_point_num = np.sum(np.where(d, 0, 1))
    # 计算度矩阵D的逆平方根，得到D^{-1/2}
    d_inv_sqrt = np.power(d, -0.5).flatten()
    # 将无穷大的值替换为0
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    # 将D^{-1/2}转换为对角矩阵
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    # 计算归一化拉普拉斯矩阵L = I - D^{-1/2}AD^{-1/2}
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    # 返回归一化拉普拉斯矩阵和孤立点的个数
    return normalized_laplacian, isolated_point_num


def calculate_normalized_laplacianzihuanb(adj: np.ndarray):
    i = sp.eye(adj.shape[0])
    adj = adj + i
    # 将邻接矩阵转换为稀疏矩阵
    adj = sp.coo_matrix(adj)
    # 计算邻接矩阵的每一行的和，得到度矩阵D的对角线元素
    d = np.array(adj.sum(1))
    # 计算孤立点的个数，即度为0的点
    isolated_point_num = np.sum(np.where(d, 0, 1))
    # 计算度矩阵D的逆平方根，得到D^{-1/2}
    d_inv_sqrt = np.power(d, -0.5).flatten()
    # 将无穷大的值替换为0
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    # 将D^{-1/2}转换为对角矩阵
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

    normalized_laplacian = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    # 返回归一化拉普拉斯矩阵和孤立点的个数
    return normalized_laplacian, isolated_point_num