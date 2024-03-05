import os
import csv
import pickle
import pandas as pd
import numpy as np



def get_adjacency_matrix(distance_df_filename: str, num_of_vertices: int, id_filename: str = None) -> tuple:
    """Generate adjacency matrix.

    Args:
        distance_df_filename (str): path of the csv file contains edges information
        num_of_vertices (int): number of vertices
        id_filename (str, optional): id filename. Defaults to None.

    Returns:
        tuple: two adjacency matrix.
            np.array: connectivity-based adjacency matrix A (A[i, j]=0 or A[i, j]=1)
            np.array: distance-based adjacency matrix A
    """
    data = pd.read_csv(distance_df_filename, header=None)
    # 将DataFrame转换为numpy数组
    adjacency_matrix_connectivity = data.values.astype(np.float32)
    adjacency_matrix_distance = adjacency_matrix_connectivity.astype(np.float32)
    # 将大于0的值设置为1
    adjacency_matrix_connectivity[adjacency_matrix_connectivity > 0] = 1
    return adjacency_matrix_connectivity, adjacency_matrix_distance


def generate_adj_tfa():
    distance_df_filename, num_of_vertices = "datasets/raw_data/TFA/TFA_adj.csv", 179
    if os.path.exists(distance_df_filename.split(".", maxsplit=1)[0] + ".txt"):
        id_filename = distance_df_filename.split(".", maxsplit=1)[0] + ".txt"
    else:
        id_filename = None
    adj_mx, distance_mx = get_adjacency_matrix(
        distance_df_filename, num_of_vertices, id_filename=id_filename)
    # the self loop is missing
    add_self_loop = False
    if add_self_loop:
        print("adding self loop to adjacency matrices.")
        adj_mx = adj_mx + np.identity(adj_mx.shape[0])
        distance_mx = distance_mx + np.identity(distance_mx.shape[0])
    else:
        adj_mx = adj_mx - np.identity(adj_mx.shape[0])
        distance_mx = distance_mx - np.identity(distance_mx.shape[0])
        print("kindly note that there is no self loop in adjacency matrices.")
    with open("datasets/raw_data/TFA/adj_TFA.pkl", "wb") as f:
        pickle.dump(adj_mx, f)
    with open("datasets/raw_data/TFA/adj_TFA_distance.pkl", "wb") as f:
        pickle.dump(distance_mx, f)
