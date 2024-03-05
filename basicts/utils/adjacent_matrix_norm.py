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
    # ����calculate_normalized_laplacian�����������ڽӾ���Ĺ�һ��������˹����L�͹�����ĸ���
    L, isolated_point_num = calculate_normalized_laplacian(adj)
    # ����ֵ�ֽ⣬�õ�L������ֵEigVal����������EigVec
    EigVal, EigVec = np.linalg.eig(L.toarray())

    # ����ֵ���򣬵õ���������idx
    idx = EigVal.argsort()

    # ��������ֵ����������������ֵ��������������
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])
    # ȥ�������������������ȡǰx�������������ͼ��Ƕ��X_spe
    if data=="PEMS04" or data=="PEMS-BAY":
        x = 64
    elif data=="PEMS08" or data=="TFA":
        x = 32
    elif data=="PEMS07":
        x = 96
    laplacian_pe = torch.from_numpy(EigVec[:, isolated_point_num + 1: x + isolated_point_num + 1]).float()
    # ����Ƕ����ݶ�ΪFalse����ʾ����Ҫ����
    laplacian_pe.require_grad = False
    # ��Ƕ��ת��Ϊnumpy����
    laplacian_pe = laplacian_pe.numpy()
    # ��Ƕ��ת��Ϊϡ�����
    laplacian_pe = sp.csr_matrix(laplacian_pe)

    # ����Ƕ��
    return laplacian_pe

def calculate_normalized_laplacian(adj: np.ndarray):
    # ���ڽӾ���ת��Ϊϡ�����
    adj = sp.coo_matrix(adj)
    # �����ڽӾ����ÿһ�еĺͣ��õ��Ⱦ���D�ĶԽ���Ԫ��
    d = np.array(adj.sum(1))

    # ���������ĸ���������Ϊ0�ĵ�
    isolated_point_num = np.sum(np.where(d, 0, 1))
    # ����Ⱦ���D����ƽ�������õ�D^{-1/2}
    d_inv_sqrt = np.power(d, -0.5).flatten()
    # ��������ֵ�滻Ϊ0
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    # ��D^{-1/2}ת��Ϊ�ԽǾ���
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    # �����һ��������˹����L = I - D^{-1/2}AD^{-1/2}
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    # ���ع�һ��������˹����͹�����ĸ���
    return normalized_laplacian, isolated_point_num


def calculate_normalized_laplacianzihuanb(adj: np.ndarray):
    i = sp.eye(adj.shape[0])
    adj = adj + i
    # ���ڽӾ���ת��Ϊϡ�����
    adj = sp.coo_matrix(adj)
    # �����ڽӾ����ÿһ�еĺͣ��õ��Ⱦ���D�ĶԽ���Ԫ��
    d = np.array(adj.sum(1))
    # ���������ĸ���������Ϊ0�ĵ�
    isolated_point_num = np.sum(np.where(d, 0, 1))
    # ����Ⱦ���D����ƽ�������õ�D^{-1/2}
    d_inv_sqrt = np.power(d, -0.5).flatten()
    # ��������ֵ�滻Ϊ0
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    # ��D^{-1/2}ת��Ϊ�ԽǾ���
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

    normalized_laplacian = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    # ���ع�һ��������˹����͹�����ĸ���
    return normalized_laplacian, isolated_point_num