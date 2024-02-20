import torch 
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import BatchNorm, LayerNorm

class GCN(torch.nn.Module):
    def __init__(self, 
        hidden_channels:int = 10, 
        num_features:int = 7, 
        num_classes:int = 8):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.bn = BatchNorm(num_features)
        self.ln = LayerNorm(hidden_channels)
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        
        x = self.bn(x)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.ln(x)
        # x = F.dropout(x, training=self.training)
        # x = self.ln(x)
        # x = self.conv2(x, edge_index)
        # x = F.relu(x)
        # x = self.ln(x)
        # x = self.conv3(x, edge_index)
        # x = self.ln(x)
        # x = self.lin(x)
        x = global_mean_pool(x,batch)
        x = self.lin(x)
        
        return x

