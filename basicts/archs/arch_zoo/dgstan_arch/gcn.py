import torch
from torch import nn


class DGCN(nn.Module):
    """Multi-Layer Perceptron with residual links."""

    def __init__(self, input_dim, hidden_dim, out_dim, dropout_rate, nodes,adj_mx) -> None:
        super(DGCN, self).__init__()
        self.linear_1 = nn.Linear(input_dim, hidden_dim)  # 定义一个线性层
        self.linear_2 = nn.Linear(hidden_dim, out_dim)  # 定义一个线性层
        self.dropout = nn.Dropout(dropout_rate)
        self.act = nn.ReLU()  # 定义激活函数
        self.weight = torch.nn.Parameter(torch.rand(nodes, input_dim))
        self.adj_mx =adj_mx

    def forward(self, flow: torch.Tensor, input: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        """Feed forward of MLP.

        Args:
            input_data (torch.Tensor): input data with shape [B, D, N]

        Returns:
            torch.Tensor: latent repr
        """
        # 第一个图卷积层
        self.adj_mx = torch.Tensor(self.adj_mx)
        self.adj_mx = self.adj_mx.to(flow.device).squeeze(0)

        graph_data = DGCN.process_graph(self.adj_mx)
        flow_x = input * self.weight
        flow_x = flow + flow_x
        output_1 = torch.matmul(graph_data, flow_x)  # [B, N, N] ,[B, N, hid_c]，就是 \hat AWX
        output_1 = self.linear_1(output_1)  # [B, N, hid_C],这个就是 WX，其中W是可学习的参数，X是输入的流量数据（就是flow_x）

        # output_1 = self.linear_1(flow_x)  # [B, N, hid_C],这个就是 WX，其中W是可学习的参数，X是输入的流量数据（就是flow_x）
        # output_1 = torch.matmul(graph_data, output_1)  # [B, N, N] ,[B, N, hid_c]，就是 \hat AWX
        output_1 = self.dropout(output_1)
        output_1 = self.act(output_1)
        # 第二个图卷积层
        output_2 = self.linear_2(output_1)  # WX
        output_2 = self.act(torch.matmul(graph_data, output_2))
        out = torch.cat((out, output_2), dim=2)
        return output_2, flow, out

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


class GCN_1(nn.Module):
    """Multi-Layer Perceptron with residual links."""

    def __init__(self, input_dim, hidden_dim, out_dim, dropout_rate) -> None:
        super(GCN_1, self).__init__()
        self.linear_1 = nn.Linear(input_dim, hidden_dim)  # 定义一个线性层
        self.linear_2 = nn.Linear(hidden_dim, out_dim)  # 定义一个线性层
        self.dropout = nn.Dropout(dropout_rate)
        self.act = nn.ReLU()  # 定义激活函数

    def forward(self, flow_x: torch.Tensor, graph_data: torch.Tensor) -> torch.Tensor:
        """Feed forward of MLP.

        Args:
            input_data (torch.Tensor): input data with shape [B, D, N]

        Returns:
            torch.Tensor: latent repr
        """
        # 第一个图卷积层

        output_1 = torch.matmul(graph_data, flow_x)  # [B, N, N] ,[B, N, hid_c]，就是 \hat AWX
        output_1 = self.linear_1(output_1)  # [B, N, hid_C],这个就是 WX，其中W是可学习的参数，X是输入的流量数据（就是flow_x）

        # output_1 = self.linear_1(flow_x)  # [B, N, hid_C],这个就是 WX，其中W是可学习的参数，X是输入的流量数据（就是flow_x）
        # output_1 = torch.matmul(graph_data, output_1)  # [B, N, N] ,[B, N, hid_c]，就是 \hat AWX
        output_1 = self.dropout(output_1)
        output_1 = self.act(output_1)
        # 第二个图卷积层
        output_2 = self.linear_2(output_1)  # WX
        output_2 = self.act(torch.matmul(graph_data, output_2))

        return output_2, flow_x

class MultiLayerPerceptron(nn.Module):
    """Multi-Layer Perceptron with residual links."""

    def __init__(self, input_dim, hidden_dim,output_dim) -> None:
        super().__init__()
        self.fc1 = nn.Conv2d(
            in_channels=input_dim,  out_channels=hidden_dim, kernel_size=(1, 1), bias=True)
        self.fc2 = nn.Conv2d(
            in_channels=hidden_dim, out_channels=output_dim, kernel_size=(1, 1), bias=True)
        self.act = nn.ReLU()
        #self.drop = nn.Dropout(p=0.15)

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """Feed forward of MLP.

        Args:
            input_data (torch.Tensor): input data with shape [B, D, N]

        Returns:
            torch.Tensor: latent repr
        """

        hidden = self.fc2(self.act(self.fc1(input_data)))     # MLP
        #hidden = hidden + input_data                           # residual
        return hidden