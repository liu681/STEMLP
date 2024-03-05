import torch
from torch import nn
import torch.nn.functional as F
import math
from .gcn import DGCN, GCN_1, MultiLayerPerceptron


class DGSTAN(nn.Module):
    """
    Paper: Spatial-Temporal Identity: A Simple yet Effective Baseline for Multivariate Time Series Forecasting
    Link: https://arxiv.org/abs/2208.05233
    Official Code: https://github.com/zezhishao/STID
    """

    def __init__(self, **model_args):
        super().__init__()
        # attributes
        self.num_nodes = model_args["num_nodes"]
        self.hidden_dim = model_args["hidden_dim"]
        self.input_len = model_args["input_len"]
        self.input_dim = model_args["input_dim"]
        self.hidden_dim_1 = model_args["hidden_dim_1"]
        self.output_len = model_args["output_len"]
        self.layer = model_args["layer"]
        self.dropout_rate = model_args["dropout_rate"]
        self.adj_mx = model_args["adj_mx"]

        self.GCN_1 = GCN_1(self.input_len, self.hidden_dim, self.output_len, self.dropout_rate)
        self.Multi_Scale_Residual_Graph_Convolution = Multi_Scale_Residual_Graph_Convolution(self.input_len, self.hidden_dim, self.output_len, self.dropout_rate, self.num_nodes, self.adj_mx, self.layer)
        self.mlp = MultiLayerPerceptron(int(self.output_len * (self.layer+1)), self.hidden_dim_1, self.output_len)
        self.conv = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=(3, 1), padding=(1, 0))
        self.m = torch.nn.GLU()
        self.sa = SpatialAttention(self.dropout_rate)
        self.se = SEAttention(channel=self.output_len, reduction=1)
        self.g1 = torch.nn.Parameter(torch.rand(self.num_nodes, self.output_len))  # GCN7的权重
        self.g2 = torch.nn.Parameter(torch.rand(self.num_nodes, self.output_len))  # GCN7的权重
        # regression
        self.regression_layer = nn.Conv2d(
            in_channels=self.output_len, out_channels=self.output_len, kernel_size=(1, 1), bias=True)



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
        # print(self.adj_mx.shape)
        self.adj_mx = torch.Tensor(self.adj_mx)
        self.adj_mx = self.adj_mx.to(input_data.device).squeeze(0)
        #print(self.adj_mx.shape)
        self.adj_mx = DGSTAN.process_graph(self.adj_mx)
        #print(self.adj_mx.shape)
        input_data = input_data.squeeze(3).transpose(1, 2).contiguous()
        output, input = self.GCN_1(input_data, self.adj_mx)
        out = output
        output, input, out = self.Multi_Scale_Residual_Graph_Convolution(output, input, out)
        # print(out.shape)
        out = out.unsqueeze(3).permute(0, 2, 1, 3)
        # print(out.shape)
        out = self.mlp(out)
        # 时间维度卷积
        out = out.permute(0, 3, 1, 2)
        out = self.conv(out)
        out = out.permute(0, 3, 2, 1)
        # GLU实现
        out = self.m(out)
        # print(out.shape)
        out1 = self.sa(out, self.adj_mx)  # batch*n*t
        # print(out1.shape,111)
        # 时间通道注意力
        out = out.transpose(1, 2)
        out = self.se(out)
        out = out.transpose(1, 2)
        out = out.squeeze(3)
        # print(out.shape)
        out = self.g1 * out + self.g2 * out1  #
        out = out.unsqueeze(3).permute(0, 2, 1, 3)
        #print(out.shape)
        # regression
        prediction = self.regression_layer(out)
        return prediction

    def process_graph(graph_data):  # 这个就是在原始的邻接矩阵之上，再次变换，也就是\hat A = D_{-1/2}*A*D_{-1/2}
        N = graph_data.size(0)  # 获得节点的个数
        matrix_i = torch.eye(N, dtype=torch.float, device=graph_data.device)  # 定义[N, N]的单位矩阵
        graph_data = graph_data + matrix_i  # [N, N]  ,就是 A+I
        degree_matrix = torch.sum(graph_data, dim=1, keepdim=False)  # [N],计算度矩阵，塌陷成向量，其实就是将上面的A+I每行相加
        degree_matrix = torch.pow(degree_matrix, -0.5).flatten()  # 计算度矩阵的逆，若为0，-1次方可能计算结果为无穷大的数
        degree_matrix[torch.isinf(degree_matrix)] = 0.  # 让无穷大的数为0
        degree_matrix = torch.diag(degree_matrix)  # 转换成对角矩阵
        out = graph_data.matmul(degree_matrix).transpose(0, 1).matmul(degree_matrix)
        return out  # 返回 A = D_{-1/2}*A*D_{-1/2}

class Multi_Scale_Residual_Graph_Convolution(nn.Module):
    def __init__(self, input_len, hidden_dim, output_len, dropout_rate, num_nodes, adj_mx, layer):
        super(Multi_Scale_Residual_Graph_Convolution, self).__init__()
        self.dgcn_layers = nn.ModuleList([DGCN(input_len, hidden_dim, output_len, dropout_rate, num_nodes, adj_mx) for _ in range(layer)])

    def forward(self, output, input, out):
        for dgcn in self.dgcn_layers:
            output, input, out = dgcn(output, input, out)
        return output, input, out


class SpatialAttention(nn.Module):
    def __init__(self, Gdropout_rate):
        super(SpatialAttention, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.linear_1 = nn.Linear(2, 1)  # 定义一个线性层
        self.dropout = nn.Dropout(Gdropout_rate)
        self.act = nn.ReLU()  # 定义激活函数

    def forward(self, x,graph_data):
        x = x.squeeze(3)
        max_result, _ = torch.max(x, dim=2, keepdim=True)
        avg_result = torch.mean(x, dim=2, keepdim=True)
        flow_x = torch.cat([max_result, avg_result], 2)
        # 第一个图卷积层
        output_1 = torch.matmul(graph_data, flow_x)  # [B, N, N] ,[B, N, hid_c]，就是 \hat AWX
        output_1 = self.linear_1(output_1)  # [B, N, hid_C],这个就是 WX，其中W是可学习的参数，X是输入的流量数据（就是flow_x）

        # output_1 = self.linear_1(flow_x)  # [B, N, hid_C],这个就是 WX，其中W是可学习的参数，X是输入的流量数据（就是flow_x）
        # output_1 = torch.matmul(graph_data, output_1)  # [B, N, N] ,[B, N, hid_c]，就是 \hat AWX
        output_1 = self.dropout(output_1)
        output_1 = self.act(output_1)
        # 第二个图卷积层
        output_2 = self.sigmoid(output_1)
        output_2=output_2.expand_as(x)
        output_2=output_2*x
        return output_2

class SEAttention(nn.Module):
    def __init__(self, channel, reduction):
        super(SEAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        #print(x.shape)
        y = self.avg_pool(x).view(b, c)
        #print(y.shape)
        y = self.fc(y).view(b, c, 1, 1)
        #print(y.expand_as(x).shape)
        return x * y.expand_as(x)

