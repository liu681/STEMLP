import torch
from torch import nn
import torch.nn.functional as F
import math
from .mlp import MultiLayerPerceptron

class STEMLP(nn.Module):
    """
    Paper:
    Link:
    Official Code:
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
        if self.if_adaspatial:  # adp spatial embeddings
            self.node_embeddings1 = nn.Parameter(torch.randn(self.num_nodes, self.adp_dim), requires_grad=True)
            self.node_embeddings2 = nn.Parameter(torch.randn(self.adp_dim, self.num_nodes), requires_grad=True)
            self.Adp_spatial_embedding = LaplacianPE(self.lape_dim, self.lape_dim)
        if self.if_prespatial:  # pre spatial emdeding
            self.pred_spatial_embedding = LaplacianPE(self.lape_dim, self.lape_dim)

        # temporal embeddings
        if self.if_time_in_day:  # period 1
            self.time_in_day_emb = nn.Parameter(
                torch.empty(self.time_of_day_size, self.node_dim))
            nn.init.xavier_uniform_(self.time_in_day_emb)
        if self.if_day_in_week:  # period week
            self.day_in_week_emb = nn.Parameter(
                torch.empty(self.day_of_week_size, self.node_dim))
            nn.init.xavier_uniform_(self.day_in_week_emb)
        if self.if_time_in_tp2:  # period 2
            self.time_in_tp2_emb = nn.Parameter(
                torch.empty(self.time_of_tp2_size, self.node_dim))
            nn.init.xavier_uniform_(self.time_in_tp2_emb)
        if self.if_time_in_tp3:  # period 3
            self.time_in_tp3_emb = nn.Parameter(
                torch.empty(self.time_of_tp3_size, self.node_dim))
            nn.init.xavier_uniform_(self.time_in_tp3_emb)

        # data embedding
        self.time_series_emb_layer = DataEmbedding(self.input_dim * self.input_len, self.embed_dim)

        # MLP encoding
        # data_temporal MLP Module
        if self.if_time_in_day or self.if_day_in_week or self.if_time_in_tp2 or self.if_time_in_tp3:
            self.t = 1
            self.data_time_hidden_dim = self.embed_dim + self.node_dim * int(self.if_day_in_week) + \
                                        self.node_dim * int(self.if_time_in_day) + self.node_dim * int(self.if_time_in_tp2) + self.node_dim * int(self.if_time_in_tp3)
            self.data_time_encoder = nn.Sequential(
                *[MultiLayerPerceptron(self.data_time_hidden_dim, int(self.data_time_hidden_dim * 1.5)) for _ in
                  range(self.num_layerA)])
        # data_spatial MLP Module
        if self.if_adaspatial or self.if_prespatial:
            self.s = 1
            self.data_spatial_hidden_dim = self.embed_dim + self.lape_dim * int(self.if_prespatial)+self.lape_dim * int(self.if_adaspatial)
            self.data_spatial_encoder = nn.Sequential(
                *[MultiLayerPerceptron(self.data_spatial_hidden_dim, int(self.data_spatial_hidden_dim * 1.5)) for _ in
                  range(self.num_layerB)])
        # total MLP Module
        self.hidden_dim = self.embed_dim * self.s + self.embed_dim * self.t + self.lape_dim * int(self.if_prespatial)+self.lape_dim * int(self.if_adaspatial) +\
                            self.node_dim * int(self.if_day_in_week) + \
                          self.node_dim * int(self.if_time_in_day) + self.node_dim * int(self.if_time_in_tp2) + self.node_dim * int(self.if_time_in_tp3)
        self.encoder = nn.Sequential(
            *[MultiLayerPerceptron(self.hidden_dim, int(self.hidden_dim * 1.5)) for _ in range(self.num_layerC)])

        # regression
        self.regression_layer = nn.Conv2d(
            in_channels=self.hidden_dim, out_channels=self.output_len, kernel_size=(1, 1), bias=True)


    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool,
                **kwargs) -> torch.Tensor:
        """Feed forward of STEMLP.

        Args:
            history_data (torch.Tensor): history data with shape [B, L, N, C]

        Returns:
            torch.Tensor: prediction with shape [B, L, N, C]
        """

        # prepare data
        input_data = history_data[..., range(self.input_dim)]

        # temporal embeddings
        if self.if_time_in_day:  # period 1
            t_i_d_data = history_data[..., 1]
            time_in_day_emb = self.time_in_day_emb[
                (t_i_d_data[:, -1, :] * self.time_of_day_size).type(torch.LongTensor)]
        else:
            time_in_day_emb = None
        if self.if_day_in_week:  # period week
            d_i_w_data = history_data[..., 2]
            day_in_week_emb = self.day_in_week_emb[
                (d_i_w_data[:, -1, :] * self.day_of_week_size).type(torch.LongTensor)]
        else:
            day_in_week_emb = None
        if self.if_time_in_tp2:  # period 2
            t_i_d_data = history_data[..., 3]
            time_in_tp2_emb = self.time_in_tp2_emb[
                (t_i_d_data[:, -1, :] * self.time_of_tp2_size).type(torch.LongTensor)]
        else:
            time_in_tp2_emb = None
        if self.if_time_in_tp3:  # period 3
            t_i_d_data = history_data[..., 4]
            time_in_tp3_emb = self.time_in_tp3_emb[
                (t_i_d_data[:, -1, :] * self.time_of_tp3_size).type(torch.LongTensor)]
        else:
            time_in_tp3_emb = None

        # data embedding
        batch_size, _, num_nodes, _ = input_data.shape
        input_data = input_data.transpose(1, 2).contiguous()
        input_data = input_data.view(
            batch_size, num_nodes, -1).transpose(1, 2).unsqueeze(-1)
        time_series_emb = self.time_series_emb_layer(input_data)

        # spatial embeddings
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
        if self.s==1:
            if self.if_prespatial and self.if_adaspatial:
                s_emb = torch.cat([sa_emb, sp_emb], dim=1)
            ds_emb = torch.cat([time_series_emb, s_emb], dim=1)
            ds_emb = self.data_spatial_encoder(ds_emb)
            hidden = ds_emb

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
            dt_emb = torch.cat([time_series_emb, tem_emb], dim=1)
            dt_emb = self.data_time_encoder(dt_emb)
            hidden =dt_emb

        # concate all embeddings
        if self.t==1 and self.s==1:
            hidden = torch.cat([dt_emb, ds_emb], dim=1)
        hidden = self.encoder(hidden)

        # regression
        prediction = self.regression_layer(hidden)
        return prediction

    def calculate_symmetric_normalized_laplacian(adj: torch.Tensor, num_nodes, lap_dim) -> torch.Tensor:

        adj = adj.masked_fill(torch.isinf(adj), 0)

        adj = adj.masked_fill(torch.isnan(adj), 0)
        degree_matrix = torch.sum(adj, dim=1, keepdim=False)
        isolated_point_num = torch.sum(degree_matrix == 0).item()
        if isolated_point_num == num_nodes:
            isolated_point_num = isolated_point_num - lap_dim-1
        EigVal, EigVec = torch.linalg.eigh(adj)
        idx = torch.argsort(torch.abs(EigVal), dim=0, descending=True)
        EigVal, EigVec = EigVal[idx], EigVec[:, idx]
        laplacian_pe = EigVec[:, isolated_point_num + 1: lap_dim+ isolated_point_num + 1]
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