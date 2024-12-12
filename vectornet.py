import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, max_pool
from torch_geometric.data import DataLoader
from dataset import GraphDataset
import os

class GraphLayer(MessagePassing):
    def __init__(self, in_channels, hidden_state, verbose = False):
        super().__init__(aggr='max')
        self.verbose = verbose
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_state),
            nn.LayerNorm(hidden_state),
            nn.ReLU(),
            nn.Linear(hidden_state, in_channels))
   
    def forward(self, x, edge_index):
        x = self.mlp(x)
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j):
        return x_j
    
    def update(self, agrr_out, x):
        return torch.cat([x, agrr_out], dim = 1)


class SubGraph(nn.Module):
    def __init__(self, in_channels, sub_graph_layers, hidden_state):  
        super().__init__()   
        self.layer_seq = nn.Sequential()
        for i in range(sub_graph_layers):
            self.layer_seq.add_module(f'glp_{i}',GraphLayer(in_channels, hidden_state))
            in_channels*=2

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for layer in self.layer_seq:
            x = layer(x, edge_index)
        data.x = x
        result = max_pool(data.cluster, data)
        result.x = result.x / result.x.norm(dim=0)
        return result


def masked_softmax(x, valid_len):
    shape = x.shape
    if valid_len.dim() == 1:
        valid_len = torch.repeat_interleave(valid_len, repeats=shape[1], dim = 0)
    else:
        valid_len = valid_len.reshape(-1)
    
    x = x.reshape(-1, shape[-1])
    max_len = x.size(1)
    mask = torch.arange((max_len), dtype=torch.float32)[None,:] <= valid_len[:, None]
    x[~mask] = -1e-6
    return nn.functional.softmax(x.reshape(shape), dim = -1)

class SelfAttention(nn.Module):
    def __init__(self, in_channels, global_graph_width, need_scale = False):
        super().__init__()
        self.in_channels = in_channels
        self.q_lin = nn.Linear(in_channels, global_graph_width)
        self.k_lin = nn.Linear(in_channels, global_graph_width)
        self.v_lin = nn.Linear(in_channels, global_graph_width)
    
    def forward(self, x, valid_len):
        # shape [query_size, global_graph_width]
        query = self.q_lin(x)
        # shape [query_size, global_graph_width]
        key = self.k_lin(x)
        # shape [query_size, global_graph_width]
        val = self.v_lin(x)

        #shape [query_size, query_size]
        scores = torch.bmm(query, key.transpose(1,2))
        attention_weights = masked_softmax(scores, valid_len)
        return torch.bmm(attention_weights, val)



class PredTrajMLP(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_state):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(in_channels, hidden_state),
                                 nn.LayerNorm(hidden_state),
                                 nn.ReLU(),
                                 nn.Linear(hidden_state, out_channels))

    def forward(self, x):
        return self.mlp(x)


class VectorNet(nn.Module):
    def __init__(self, in_channels, out_channels, sub_graph_layers, sub_graph_state = 64, global_graph_width = 64, traj_pred_mlp_width = 64):
        super().__init__()
        self.poly_feature_shape = in_channels * (2 ** sub_graph_layers)
        self.subgraph = SubGraph(in_channels, sub_graph_layers, sub_graph_state)
        self.self_attention_layer = SelfAttention(self.poly_feature_shape, global_graph_width)
        self.pred_traj_mlp = PredTrajMLP(global_graph_width, out_channels, traj_pred_mlp_width)
    
    def forward(self, data):
        time_step_len = int(data[0].time_step_len[0])
        
        valid_len = []
        for i in range(data.num_graphs):
            valid_len.append(data[i].valid_len[0])
        valid_len = torch.tensor(valid_len)
        sub_graph_data = self.subgraph(data)

        x = sub_graph_data.x.view(-1, time_step_len, self.poly_feature_shape)
        self_attention_out = self.self_attention_layer(x, valid_len)
        pred = self.pred_traj_mlp(self_attention_out[:,0])
        return pred


if __name__ == "__main__":
    device = torch.device('cpu')
    in_channels = 8
    out_channels = 60
    sub_graph_layers = 3

    model = VectorNet(in_channels, out_channels, sub_graph_layers).to(device)

    train_dir = os.path.join('./interm_data','train_intermediate')
    dataset = GraphDataset(train_dir)
    batch_size = 2

    data_iter = DataLoader(dataset, batch_size=batch_size)

    for data in data_iter:
        out = model(data)

    