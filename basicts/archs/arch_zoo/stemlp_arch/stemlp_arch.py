import torch
from torch import nn
import torch.nn.functional as F
import math
from .mlp import MultiLayerPerceptron
from .revin import RevIN


class STEMLP(nn.Module):
    """
    Paper: Spatial-Temporal Identity: A Simple yet Effective Baseline for Multivariate Time Series Forecasting
    Link: https://arxiv.org/abs/2208.05233
    Official Code: https://github.com/zezhishao/STID
    """

    def __init__(self, **model_args):
        super().__init__()
        # attributes
        self.num_nodes = model_args["num_nodes"]
        self.node_dim = model_args["node_dim"]
        self.input_len = model_args["input_len"]
        self.input_dim = model_args["input_dim"]
        self.embed_dim = model_args["embed_dim"]
        self.output_len = model_args["output_len"]
        self.num_layerA = model_args["num_layerA"]
        self.num_layerB = model_args["num_layerB"]
        self.num_layerC = model_args["num_layerC"]
        self.lape_dim = model_args["lape_dim"]
        self.adp_dim = model_args["adp_dim"]
        self.time_of_day_size = model_args["time_of_day_size"]
        self.day_of_week_size = model_args["day_of_week_size"]
        self.time_of_tp2_size = model_args["time_of_tp2_size"]
        self.time_of_tp3_size = model_args["time_of_tp3_size"]
        self.if_time_in_day = model_args["if_T_i_D"]
        self.if_day_in_week = model_args["if_D_i_W"]
        self.if_time_in_tp2 = model_args["if_T_i_TP2"]
        self.if_time_in_tp3 = model_args["if_T_i_TP3"]
        self.if_prespatial = model_args["if_prenode"]
        self.if_adaspatial = model_args["if_adanode"]
        self.lap_mx = model_args["lap_mx"]
        self.t = self.s = 0
        # spatial embeddings
        if self.if_adaspatial:
            self.node_embeddings1 = nn.Parameter(torch.randn(self.num_nodes, self.adp_dim), requires_grad=True)
            self.node_embeddings2 = nn.Parameter(torch.randn(self.adp_dim, self.num_nodes), requires_grad=True)
            self.Adp_spatial_embedding = LaplacianPE(self.lape_dim, self.lape_dim)
        if self.if_prespatial:
            self.pred_spatial_embedding = LaplacianPE(self.lape_dim, self.lape_dim)
        # temporal embeddings
        if self.if_time_in_day:
            self.time_in_day_emb = nn.Parameter(
                torch.empty(self.time_of_day_size, self.node_dim))
            nn.init.xavier_uniform_(self.time_in_day_emb)
        if self.if_day_in_week:
            self.day_in_week_emb = nn.Parameter(
                torch.empty(self.day_of_week_size, self.node_dim))
            nn.init.xavier_uniform_(self.day_in_week_emb)
        if self.if_time_in_tp2:
            self.time_in_tp2_emb = nn.Parameter(
                torch.empty(self.time_of_tp2_size, self.node_dim))
            nn.init.xavier_uniform_(self.time_in_tp2_emb)
        if self.if_time_in_tp3:
            self.time_in_tp3_emb = nn.Parameter(
                torch.empty(self.time_of_tp3_size, self.node_dim))
            nn.init.xavier_uniform_(self.time_in_tp3_emb)
        # embedding layer
        self.time_series_emb_layer = DataEmbedding(self.input_dim * self.input_len, self.embed_dim)

        # encoding
        if self.if_time_in_day or self.if_day_in_week or self.if_time_in_tp2 or self.if_time_in_tp3:
            self.t = 1
            
        if self.if_adaspatial or self.if_prespatial:
            self.s = 1
        

        self.hidden_dim = self.embed_dim + self.lape_dim * int(self.if_prespatial)+self.lape_dim * int(self.if_adaspatial) +\
                            self.node_dim * int(self.if_day_in_week) + \
                          self.node_dim * int(self.if_time_in_day) + self.node_dim * int(self.if_time_in_tp2) + self.node_dim * int(self.if_time_in_tp3)
        self.encoder = nn.Sequential(
            *[MultiLayerPerceptron(self.hidden_dim, int(self.hidden_dim * 1.5)) for _ in range(3)])

        # regression
        self.regression_layer = nn.Conv2d(
            in_channels=self.hidden_dim, out_channels=self.output_len, kernel_size=(1, 1), bias=True)
        self.rev = RevIN(self.num_nodes, affine=False, subtract_last=False)

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool,
                **kwargs) -> torch.Tensor:
        """Feed forward of STID.

        Args:
            history_data (torch.Tensor): history data with shape [B, L, N, C]

        Returns:
            torch.Tensor: prediction with shape [B, L, N, C]
        """

        # prepare data
        input_data = history_data[..., range(self.input_dim)]
        x = history_data[..., 0]
        history_data[..., 0] = self.rev(x, 'norm')
        if self.if_time_in_day:
            t_i_d_data = history_data[..., 1]
            # In the datasets used in STID, the time_of_day feature is normalized to [0, 1]. We multiply it by 288 to get the index.
            # If you use other datasets, you may need to change this line.
            time_in_day_emb = self.time_in_day_emb[
                (t_i_d_data[:, -1, :] * self.time_of_day_size).type(torch.LongTensor)]
        else:
            time_in_day_emb = None
        if self.if_day_in_week:
            d_i_w_data = history_data[..., 2]
            day_in_week_emb = self.day_in_week_emb[
                (d_i_w_data[:, -1, :] * self.day_of_week_size).type(torch.LongTensor)]
        else:
            day_in_week_emb = None
        if self.if_time_in_tp2:
            t_i_d_data = history_data[..., 2]
            # In the datasets used in STID, the time_of_day feature is normalized to [0, 1]. We multiply it by 288 to get the index.
            # If you use other datasets, you may need to change this line.
            time_in_tp2_emb = self.time_in_tp2_emb[
                (t_i_d_data[:, -1, :] * self.time_of_tp2_size).type(torch.LongTensor)]
        else:
            time_in_tp2_emb = None
        if self.if_time_in_tp3:
            t_i_d_data = history_data[..., 3]
            # In the datasets used in STID, the time_of_day feature is normalized to [0, 1]. We multiply it by 288 to get the index.
            # If you use other datasets, you may need to change this line.
            time_in_tp3_emb = self.time_in_tp3_emb[
                (t_i_d_data[:, -1, :] * self.time_of_tp3_size).type(torch.LongTensor)]
        else:
            time_in_tp3_emb = None

        # time series embedding
        batch_size, _, num_nodes, _ = input_data.shape
        input_data = input_data.transpose(1, 2).contiguous()
        input_data = input_data.view(
            batch_size, num_nodes, -1).transpose(1, 2).unsqueeze(-1)
        time_series_emb = self.time_series_emb_layer(input_data)
        if self.if_adaspatial :
            # Adp spatial embeddings
            Adpgraph = F.softmax(F.relu(torch.mm(self.node_embeddings1, self.node_embeddings2)), dim=1)
            sa_emb = STEMLP.calculate_symmetric_normalized_laplacian(Adpgraph, self.num_nodes,self.lape_dim).to(torch.float32)
            sa_emb = self.Adp_spatial_embedding(sa_emb).expand(batch_size, -1, -1, -1)
            s_emb= sa_emb
        if self.if_prespatial :
            # pred spatial embeddings
            self.lap_mx = self.lap_mx.to(input_data.device)
            sp_emb = self.pred_spatial_embedding(self.lap_mx).expand(batch_size, -1, -1, -1)
            s_emb = sp_emb
            
        if self.if_prespatial and self.if_adaspatial:
                s_emb = torch.cat([sa_emb, sp_emb], dim=1)
        
        # temporal embeddings
        tem_emb = []
        if time_in_day_emb is not None:
            tem_emb.append(time_in_day_emb.transpose(1, 2).unsqueeze(-1))
        if day_in_week_emb is not None:
            tem_emb.append(day_in_week_emb.transpose(1, 2).unsqueeze(-1))
        if time_in_tp2_emb is not None:
            tem_emb.append(time_in_tp2_emb.transpose(1, 2).unsqueeze(-1))
        if time_in_tp3_emb is not None:
            tem_emb.append(time_in_tp3_emb.transpose(1, 2).unsqueeze(-1))
        if self.t==1:
            tem_emb = torch.cat(tem_emb, dim=1)
        if self.s==1 and self.t==1:
            hidden = torch.cat([tem_emb, s_emb], dim=1)
            hidden = torch.cat([time_series_emb, hidden], dim=1)
            
        if self.t!=1:
            hidden = torch.cat([time_series_emb, s_emb], dim=1)
            
        if self.s!=1:
            hidden = torch.cat([time_series_emb, tem_emb], dim=1)
            
            
        # encoding
        hidden = self.encoder(hidden)

        # regression
        prediction = self.regression_layer(hidden)
        prediction = self.rev(prediction.squeeze(-1), 'denorm').unsqueeze(-1)
        return prediction

    def calculate_symmetric_normalized_laplacian(adj: torch.Tensor, num_nodes,lap_dim) -> torch.Tensor:
        # 调用calculate_normalized_laplacian函数，计算邻接矩阵的归一化拉普拉斯矩阵L和孤立点的个数
        adj = adj.masked_fill(torch.isinf(adj), 0)
        # 特征值分解，得到L的特征值EigVal和特征向量EigVec
        adj = adj.masked_fill(torch.isnan(adj), 0)
        degree_matrix = torch.sum(adj, dim=1, keepdim=False)
        # print(degree_matrix)
        # 计算孤立点的个数，即度为0的点

        isolated_point_num = torch.sum(degree_matrix == 0).item()

        if isolated_point_num == num_nodes:
            isolated_point_num = isolated_point_num - lap_dim-1
        EigVal, EigVec = torch.linalg.eigh(adj)
        # 特征值排序，得到排序索引idx
        idx = torch.argsort(torch.abs(EigVal), dim=0, descending=True)

        # 利用特征值的排序索引对特征值和特征向量排序
        EigVal, EigVec = EigVal[idx], EigVec[:, idx]
        # 去掉孤立点的特征向量，取前8个特征向量组成图的嵌入X_spe

        laplacian_pe = EigVec[:, isolated_point_num + 1: lap_dim+ isolated_point_num + 1]

        # 返回嵌入
        return laplacian_pe


class DataEmbedding(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.data_embed = nn.Conv2d(
            in_channels=input_dim, out_channels=embed_dim, kernel_size=(1, 1), bias=True)

    def forward(self, x):
        x = self.data_embed(x)

        return x


class LaplacianPE(nn.Module):
    def __init__(self, lape_dim, embed_dim):
        super().__init__()
        #self.embedding_lap_pos_enc = nn.Linear(lape_dim, embed_dim)

    def forward(self, lap_mx):
        #lap_pos_enc = self.embedding_lap_pos_enc(lap_mx).unsqueeze(0).unsqueeze(-1).transpose(1, 2)
        lap_pos_enc = lap_mx.unsqueeze(0).unsqueeze(-1).transpose(1, 2)
        return lap_pos_enc