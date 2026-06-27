import torch
from torch.nn import Sequential
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric_temporal.nn.recurrent import DCRNN, A3TGCN, AGCRN, A3TGCN2
from torch_geometric_temporal.nn.attention import ASTGCN, GMAN, MSTGCN
from traffic_prediction.astgcn_customized import Customized_ASTGCN


# Fully connected nueral network for test
class FNNTest(nn.Module):
    def __init__(self,
                 num_for_predict,
                ):
        super(FNNTest, self).__init__()
        self.linear1 = torch.nn.Linear(num_for_predict, num_for_predict)

    def forward(self, x):
        x_squeezed = torch.squeeze(x, dim=2)
        o1 = F.relu(self.linear1(x_squeezed))
        return o1
        
    
# Fully connected nueral network for test
class FNNTestLarge(nn.Module):
    def __init__(self,
                 num_for_predict,
                ):
        super(FNNTestLarge, self).__init__()
        self.linear1 = torch.nn.Linear(num_for_predict, num_for_predict*10)
        self.linear2 = torch.nn.Linear(num_for_predict*10, num_for_predict)

    def forward(self, x):
        x_squeezed = torch.squeeze(x, dim=2)
        o1 = F.relu(self.linear1(x_squeezed))
        o2 = F.relu(self.linear2(o1))
        return o2
        
    
# Define the AverageModel
class AverageModel(nn.Module):
    def __init__(self):
        super(AverageModel, self).__init__()
    
    def forward(self, x):
        # Compute the average along dimension 2
        avg = x[0].mean(dim=3, keepdim=True)  # Input shape (A, B, C, D), Output shape (A, B, C, 1)
        # Repeat the averaged values along dimension 3 to match the original shape
        avg_repeated = avg.repeat(1, 1, 1, x[0].size(3))  # Shape (A, B, C, D)
        avg_repeated_squeezed = torch.squeeze(avg_repeated, dim=2)
        return avg_repeated_squeezed
    
# Define the AverageSurgeModel
class AverageSurgeModel(nn.Module):
    def __init__(self, 
                 surge_mode,
                ):
        super(AverageSurgeModel, self).__init__()
        self.surge_mode = surge_mode
        
    def forward(self, x):
        # Inflow
        # Compute the average along dimension 2
        avg_inflow = x[0].mean(dim=3, keepdim=True)  # Input shape (A, B, C, D), Output shape (A, B, C, 1)
        # Repeat the averaged values along dimension 3 to match the original shape
        avg_inflow_repeated = avg_inflow.repeat(1, 1, 1, x[0].size(3))  # Shape (A, B, C, D)
        avg_inflow_repeated_squeezed = torch.squeeze(avg_inflow_repeated, dim=2)
        
        # Outflow
        # Compute the average along dimension 2
        avg_outflow = x[1].mean(dim=3, keepdim=True)  # Input shape (A, B, C, D), Output shape (A, B, C, 1)
        # Repeat the averaged values along dimension 3 to match the original shape
        avg_outflow_repeated = avg_outflow.repeat(1, 1, 1, x[1].size(3))  # Shape (A, B, C, D)
        avg_outflow_repeated_squeezed = torch.squeeze(avg_outflow_repeated, dim=2)
        
        # Compute surge
        if self.surge_mode == 1: # Shortage
            shortage_pred = avg_outflow_repeated_squeezed - avg_inflow_repeated_squeezed
            surge_pred = torch.max(shortage_pred, torch.zeros_like(shortage_pred))
        elif self.surge_mode == 2: # Surplus
            surplus_pred = avg_inflow_repeated_squeezed - avg_outflow_repeated_squeezed
            surge_pred = torch.max(surplus_pred, torch.zeros_like(surplus_pred))
        elif self.surge_mode == 3: # Diff as o-i
            surge_pred = avg_outflow_repeated_squeezed - avg_inflow_repeated_squeezed
        
        return surge_pred
    
    
# Why using ASTGCN_AttentionGCN_customized instead of Customized_ASTGCN? To include ReLu activation
class ASTGCN_AttentionGCN_customized(torch.nn.Module):
    def __init__(self, 
                num_of_vertices,
                num_for_predict,
                len_input,
                nb_block,
                in_channels,
                K,
                nb_chev_filter,
                nb_time_filter,
                time_strides,
                hidden,
                edge_index):
        super(ASTGCN_AttentionGCN_customized, self).__init__()
        self.astgcn = Customized_ASTGCN(nb_block, in_channels, K, nb_chev_filter, nb_time_filter, time_strides, num_for_predict, len_input, num_of_vertices, edge_index = edge_index)
        self.linear = torch.nn.Linear(num_for_predict, num_for_predict)
        self.hidden = hidden
        self.edge_index = edge_index

    def forward(self, x):
        y = self.astgcn(X=x)
        if self.hidden==0:
            y = self.linear(F.relu(y))
        return y


class MSTGCN_temp(torch.nn.Module):
    def __init__(self,
                   nb_block, 
                   in_channels, 
                   K, 
                   nb_chev_filter, 
                   nb_time_filter, 
                   time_strides, 
                   num_for_predict, 
                   len_input,
                   edge_index,
                ):
        super(MSTGCN_temp, self).__init__()
        self.mstgcn = MSTGCN(nb_block = nb_block, 
                               in_channels = in_channels, 
                               K = K, 
                               nb_chev_filter = nb_chev_filter, 
                               nb_time_filter = nb_time_filter, 
                               time_strides = time_strides, 
                               num_for_predict = num_for_predict, 
                               len_input = len_input,
                            )
        self.edge_index = edge_index

    def forward(self, x):
        return self.mstgcn(X=x, edge_index=self.edge_index)
        


# https://github.com/benedekrozemberczki/pytorch_geometric_temporal/blob/6c98fb346ae5f0727a5b628fb4965225750482c3/notebooks/astgcn_for_traffic_flow_forecasting.ipynb
# https://github.com/benedekrozemberczki/pytorch_geometric_temporal/blob/6c98fb346ae5f0727a5b628fb4965225750482c3/notebooks/processing_traffic_data_for_deep_learning_projects.ipynb
# https://pytorch-geometric-temporal.readthedocs.io/en/latest/_modules/torch_geometric_temporal/nn/attention/astgcn.html
# https://pytorch-geometric-temporal.readthedocs.io/en/latest/modules/root.html#torch_geometric_temporal.nn.attention.astgcn.ASTGCN
class ASTGCN_AttentionGCN(torch.nn.Module):
    def __init__(self, 
                 num_of_vertices,
                 num_for_predict,
                len_input,
                nb_block,
                in_channels,
                K,
                nb_chev_filter,
                nb_time_filter,
#                         time_strides = num_of_hours,
                time_strides,
                hidden,
                edge_index):
        super(ASTGCN_AttentionGCN, self).__init__()
        self.astgcn = ASTGCN(nb_block, in_channels, K, nb_chev_filter, nb_time_filter, time_strides, num_for_predict, 
                             len_input, num_of_vertices)
        self.linear_1 = torch.nn.Linear(num_for_predict, hidden)
        self.linear_2 = torch.nn.Linear(hidden, num_for_predict)
        self.linear_single = torch.nn.Linear(num_for_predict, num_for_predict)
        self.hidden = hidden
        self.edge_index = edge_index

#     def forward(self, x, edge_index):
    def forward(self, x):
        y = self.astgcn(X=x, edge_index=self.edge_index)
#         y = F.relu(y)
#         y = self.linear(y)
        if self.hidden==0:
            y = self.linear_single(F.relu(y))
#         else:
#             y = self.linear_1(F.relu(y))
#             y = self.linear_2(F.relu(y))
        return y
    
    

# ENSEMBLE 1: Predict inflow, outflow, and speed
# https://datascience.stackexchange.com/questions/117110/is-it-possible-to-combine-models-in-pytorch-and-pytorch-geometric
# https://discuss.pytorch.org/t/how-can-i-connect-a-new-neural-network-after-a-trained-neural-network-and-optimize-them-together/48293
# https://discuss.pytorch.org/t/combining-trained-models-in-pytorch/28383/2
class EnsembleNet(torch.nn.Module): 
    def __init__(self, modelA, modelB, modelC, periods, n_nodes):
        super(EnsembleNet, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.modelC = modelC
        self.linear_single = torch.nn.Linear(periods*sum(n_nodes), periods*sum(n_nodes))
        
    def forward(self, x, ):
        x1 = self.modelA(x[0])
        x2 = self.modelB(x[1])
        x3 = self.modelC(x[2])
        x1 = x1.view(x1.shape[0], 1, 1, -1)
        x2 = x2.view(x2.shape[0], 1, 1, -1)
        x3 = x3.view(x3.shape[0], 1, 1, -1)
        y = torch.concat((x1, x2, x3), dim=-1)
        y = self.linear_single(F.relu(y))
        return y
    
    
# Input: Speed, Prediction: inflow, outflow, and speed
class MultiPredictionNet1(torch.nn.Module): 
    def __init__(self, 
                 n_input_periods,
                 n_output_periods,
                 node_features,
                 n_nodes, 
                 edge_index_list,
                 device,
                ):
        super(MultiPredictionNet1, self).__init__()
        self.node_features = node_features
        self.n_nodes = n_nodes
        self.n_input_periods = n_input_periods
        self.linear_embedding = torch.nn.Linear(n_output_periods*n_nodes[0], 
                                                n_input_periods*n_nodes[1]) # n_nodes[1]=n_nodes[2]
        
        self.model_list = []
        for model_idx in range(len(edge_index_list)):
            self.model_list.append(ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                            num_for_predict = n_output_periods,
                                            in_channels = node_features[model_idx],
                                            num_of_vertices = n_nodes[model_idx],
                                            nb_block = 2,
                                            K = 3,
                                            nb_chev_filter = 64,
                                            nb_time_filter = 64,
                                            time_strides = 1, # time_strides = num_of_hours 
                                            hidden = 0,
                                            edge_index=edge_index_list[model_idx]).to(device))

    def forward(self, x):
        o_speed = self.model_list[0](x)
        o_speed_flattened = o_speed.view(o_speed.shape[0], 1, 1, -1)
        o_speed_linear = self.linear_embedding(o_speed_flattened)
        o_speed_linear_reshaped = o_speed_linear.view(o_speed.shape[0], # Batch size 
                                                      self.n_nodes[1], # Number of nodes, n_nodes[1]=n_nodes[2]
                                                      self.node_features[1], # Number of features,  node_features[1]=node_features[2]
                                                      self.n_input_periods) # Number of periods
        o_inflow = self.model_list[1](o_speed_linear_reshaped)
        o_outflow = self.model_list[2](o_speed_linear_reshaped)
        return [o_speed, o_inflow, o_outflow]
    
    

# Input: Speed, Prediction: inflow, outflow, and speed
class MultiPredictionNet2(torch.nn.Module): 
    def __init__(self, 
                 n_input_periods,
                 n_output_periods,
                 node_features,
                 n_nodes, 
                 edge_index_list,
                 device,
                ):
        super(MultiPredictionNet2, self).__init__()
        self.node_features = node_features
        self.n_nodes = n_nodes
        self.n_input_periods = n_input_periods
        self.linear_embedding = torch.nn.Linear(n_output_periods*n_nodes[0], 
                                                n_input_periods*n_nodes[1]) # n_nodes[1]=n_nodes[2]
        
        self.model_list = []
        for model_idx in range(len(edge_index_list)):
            self.model_list.append(ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                            num_for_predict = n_output_periods,
                                            in_channels = node_features[model_idx],
                                            num_of_vertices = n_nodes[model_idx],
                                            nb_block = 2,
                                            K = 3,
                                            nb_chev_filter = 64,
                                            nb_time_filter = 64,
                                            time_strides = 1, # time_strides = num_of_hours 
                                            hidden = 0,
                                            edge_index=edge_index_list[model_idx]).to(device))
        self.linear_fusion = torch.nn.Linear(n_output_periods*sum(n_nodes), 
                                                n_output_periods*sum(n_nodes)) # n_nodes[1]=n_nodes[2]

    def forward(self, x):
        o_speed = self.model_list[0](x)
#         print(f"o_speed.shape: {o_speed.shape}")
        o_speed_flattened = o_speed.view(o_speed.shape[0], 1, 1, -1)
        o_speed_linear = self.linear_embedding(o_speed_flattened)
        o_speed_linear_reshaped = o_speed_linear.view(o_speed.shape[0], # Batch size 
                                                      self.n_nodes[1], # Number of nodes, n_nodes[1]=n_nodes[2]
                                                      self.node_features[1], # Number of features,  node_features[1]=node_features[2]
                                                      self.n_input_periods) # Number of periods
        o_inflow = self.model_list[1](o_speed_linear_reshaped)
        o_outflow = self.model_list[2](o_speed_linear_reshaped)
#         print(f"o_inflow.shape: {o_inflow.shape}")
#         print(f"o_outflow.shape: {o_outflow.shape}")
        o_inflow_flattened = o_inflow.reshape(o_inflow.shape[0], 1, 1, -1)
        o_outflow_flattened = o_outflow.reshape(o_outflow.shape[0], 1, 1, -1)
        input_fusion = torch.concat((o_speed_flattened, o_inflow_flattened, o_outflow_flattened), dim=-1)
        output_fusion = self.linear_fusion(F.relu(input_fusion))
#         print(f"output_fusion.shape: {output_fusion.shape}")
        return output_fusion
    
    
# Input: Speed, Prediction: inflow, outflow, and speed
class MultiPredictionNet2_2(torch.nn.Module): 
    def __init__(self, 
                 n_input_periods,
                 n_output_periods,
                 node_features,
                 n_nodes, 
                 edge_index_list,
                 device,
                ):
        super(MultiPredictionNet2_2, self).__init__()
        self.node_features = node_features
        self.n_nodes = n_nodes
        self.n_input_periods = n_input_periods
        self.linear_embedding_1 = torch.nn.Linear(n_output_periods*n_nodes[0], 
                                                n_input_periods*n_nodes[1]) 
        self.linear_embedding_2 = torch.nn.Linear(n_output_periods*n_nodes[0], 
                                                n_input_periods*n_nodes[2]) 
        self.model_list = []
        for model_idx in range(len(edge_index_list)):
            self.model_list.append(ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                            num_for_predict = n_output_periods,
                                            in_channels = node_features[model_idx],
                                            num_of_vertices = n_nodes[model_idx],
                                            nb_block = 2,
                                            K = 3,
                                            nb_chev_filter = 64,
                                            nb_time_filter = 64,
                                            time_strides = 1, # time_strides = num_of_hours 
                                            hidden = 0,
                                            edge_index=edge_index_list[model_idx]).to(device))
        self.linear_fusion = torch.nn.Linear(n_output_periods*sum(n_nodes), 
                                                n_output_periods*sum(n_nodes)) # n_nodes[1]=n_nodes[2]

    def forward(self, x):
        o_speed = F.relu(self.model_list[0](x))
#         print(f"o_speed.shape: {o_speed.shape}")
        o_speed_flattened = o_speed.view(o_speed.shape[0], 1, 1, -1)
        o_speed_linear_1 = self.linear_embedding_1(o_speed_flattened)
        o_speed_linear_2 = self.linear_embedding_2(o_speed_flattened)
        o_speed_linear_reshaped_1 = o_speed_linear_1.view(o_speed.shape[0], # Batch size 
                                                      self.n_nodes[1], # Number of nodes
                                                      self.node_features[1], # Number of features
                                                      self.n_input_periods) # Number of periods
        o_speed_linear_reshaped_2 = o_speed_linear_2.view(o_speed.shape[0], # Batch size 
                                                      self.n_nodes[2], # Number of nodes
                                                      self.node_features[2], # Number of features
                                                      self.n_input_periods) # Number of periods
        o_inflow = F.relu(self.model_list[1](o_speed_linear_reshaped_1))
        o_outflow = F.relu(self.model_list[2](o_speed_linear_reshaped_2))
#         print(f"o_inflow.shape: {o_inflow.shape}")
#         print(f"o_outflow.shape: {o_outflow.shape}")
        o_inflow_flattened = o_inflow.reshape(o_inflow.shape[0], 1, 1, -1)
        o_outflow_flattened = o_outflow.reshape(o_outflow.shape[0], 1, 1, -1)
        input_fusion = torch.concat((o_speed_flattened, o_inflow_flattened, o_outflow_flattened), dim=-1)
        output_fusion = F.relu(self.linear_fusion(input_fusion))
#         print(f"output_fusion.shape: {output_fusion.shape}")
        return output_fusion
    
    

# Input: Speed, Prediction: inflow, outflow, and speed
class MultiPredictionNet3(torch.nn.Module): 
    def __init__(self, 
                 n_input_periods,
                 n_output_periods,
                 node_features,
                 n_nodes, 
                 edge_index_list,
                 device,
                ):
        super(MultiPredictionNet3, self).__init__()
        self.node_features = node_features
        self.n_nodes = n_nodes
        self.n_input_periods = n_input_periods
        self.linear_embedding = torch.nn.Linear(n_output_periods*n_nodes[0], 
                                                n_input_periods*n_nodes[1]) # n_nodes[1]=n_nodes[2]
        
        self.model_list = []
        for model_idx in range(len(edge_index_list)):
            self.model_list.append(ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                            num_for_predict = n_output_periods,
                                            in_channels = node_features[model_idx],
                                            num_of_vertices = n_nodes[model_idx],
                                            nb_block = 2,
                                            K = 3,
                                            nb_chev_filter = 64,
                                            nb_time_filter = 64,
                                            time_strides = 1, # time_strides = num_of_hours 
                                            hidden = 0,
                                            edge_index=edge_index_list[model_idx]).to(device))
        self.linear_fusion = torch.nn.Linear(n_output_periods*sum(n_nodes), 
                                                n_output_periods*sum(n_nodes)) # n_nodes[1]=n_nodes[2]

    def forward(self, x):
        o_speed = self.model_list[0](x)
#         print(f"o_speed.shape: {o_speed.shape}")
        i_speed_flattened = x.view(x.shape[0], 1, 1, -1)
        o_speed_flattened = o_speed.view(o_speed.shape[0], 1, 1, -1)
        avg_i_o_speed_flattened = (i_speed_flattened + o_speed_flattened)/2
        avg_i_o_speed_linear = self.linear_embedding(avg_i_o_speed_flattened)
        avg_i_o_speed_linear_reshaped = avg_i_o_speed_linear.view(o_speed.shape[0], # Batch size 
                                                      self.n_nodes[1], # Number of nodes, n_nodes[1]=n_nodes[2]
                                                      self.node_features[1], # Number of features,  node_features[1]=node_features[2]
                                                      self.n_input_periods) # Number of periods
        o_inflow = self.model_list[1](avg_i_o_speed_linear_reshaped)
        o_outflow = self.model_list[2](avg_i_o_speed_linear_reshaped)
#         print(f"o_inflow.shape: {o_inflow.shape}")
#         print(f"o_outflow.shape: {o_outflow.shape}")
        o_inflow_flattened = o_inflow.reshape(o_inflow.shape[0], 1, 1, -1)
        o_outflow_flattened = o_outflow.reshape(o_outflow.shape[0], 1, 1, -1)
        input_fusion = torch.concat((o_speed_flattened, o_inflow_flattened, o_outflow_flattened), dim=-1)
        output_fusion = self.linear_fusion(F.relu(input_fusion))
#         print(f"output_fusion.shape: {output_fusion.shape}")
        return output_fusion




#  Input: Speed, Prediction: inflow, outflow, and speed. Trains a GNN model for speed. Then, a linear layer predicts output.
class MultiPredictionNet4(torch.nn.Module): 
    def __init__(self, 
                 n_input_periods,
                 n_output_periods,
                 node_features,
                 n_nodes, 
                 edge_index_list,
                 device,
                ):
        super(MultiPredictionNet4, self).__init__()
        self.node_features = node_features
        self.n_nodes = n_nodes
        self.n_input_periods = n_input_periods
        
        self.model_gnn = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                        num_for_predict = n_output_periods,
                                        in_channels = node_features[0],
                                        num_of_vertices = n_nodes[0],
                                        nb_block = 2,
                                        K = 3,
                                        nb_chev_filter = 64,
                                        nb_time_filter = 64,
                                        time_strides = 1, # time_strides = num_of_hours 
                                        hidden = 0,
                                        edge_index=edge_index_list[0]).to(device)
        
        self.linear_fusion = torch.nn.Linear(n_output_periods*n_nodes[0], 
                                                n_output_periods*sum(n_nodes)) # n_nodes[1]=n_nodes[2]

    def forward(self, x):
        o_speed = self.model_gnn(x)
        o_speed_flattened = o_speed.view(o_speed.shape[0], 1, 1, -1)
        output_fusion = self.linear_fusion(F.relu(o_speed_flattened))
        return output_fusion
    
    
#  Input: Speed, Prediction: inflow
class MultiPredictionNet5(torch.nn.Module): 
    def __init__(self, 
                 n_input_periods,
                 n_output_periods,
                 node_features,
                 n_nodes, 
                 hidden_warmup,
                 edge_index_list,
                 model_path,
                ):
        super(MultiPredictionNet5, self).__init__()
        self.model_warmup = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = node_features[0],
                                                        num_of_vertices = n_nodes[0],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[0],
                                                        edge_index=edge_index_list[0],
                                                    )  

        # Load state dicts
        self.model_warmup.load_state_dict(torch.load("../models/" + model_path[0] + ".pth"))
        self.linear = torch.nn.Linear(n_output_periods*n_nodes[0], 
                                                n_output_periods*n_nodes[1]) 

    def forward(self, x):
        o_speed = self.model_warmup(x)
        o_speed_flattened = o_speed.view(o_speed.shape[0], 1, 1, -1)
        output = self.linear(F.relu(o_speed_flattened))
        return output
    
    
#  Input: Speed, Prediction: outflow
class MultiPredictionNet6(torch.nn.Module): 
    def __init__(self, 
                 n_input_periods,
                 n_output_periods,
                 node_features,
                 n_nodes, 
                 hidden_warmup,
                 edge_index_list,
                 model_path,
                ):
        super(MultiPredictionNet6, self).__init__()
        self.model_warmup = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = node_features[0],
                                                        num_of_vertices = n_nodes[0],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[0],
                                                        edge_index=edge_index_list[0],
                                                    )  

        # Load state dicts
        self.model_warmup.load_state_dict(torch.load("../models/" + model_path[0] + ".pth"))
        self.linear = torch.nn.Linear(n_output_periods*n_nodes[0], 
                                                n_output_periods*n_nodes[2]) 

    def forward(self, x):
        o_speed = self.model_warmup(x)
        o_speed_flattened = o_speed.view(o_speed.shape[0], 1, 1, -1)
        output = self.linear(F.relu(o_speed_flattened))
        return output
    
    
    
#  Input: Speed AND Inflow, Prediction: inflow. Speed prediction GNN has warmup. Inflow GNN has no warmup
class MultiPredictionNet7(torch.nn.Module): 
    def __init__(self, 
                 n_input_periods,
                 n_output_periods,
                 node_features,
                 n_nodes, 
                 hidden_warmup,
                 edge_index_list,
                 model_path,
                ):
        super(MultiPredictionNet7, self).__init__()
        # Speed prediction GNN
        self.model_warmup_speed = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = node_features[0],
                                                        num_of_vertices = n_nodes[0],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[0],
                                                        edge_index=edge_index_list[0],
                                                    )  
        ## Load state dicts
        self.model_warmup_speed.load_state_dict(torch.load("../models/" + model_path[0] + ".pth"))
        
        # Inflow prediction GNN
        self.model_inflow = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = node_features[1],
                                                        num_of_vertices = n_nodes[1],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[1],
                                                        edge_index=edge_index_list[1],
                                                    )  

        self.linear = torch.nn.Linear(n_output_periods* (n_nodes[0]+n_nodes[1]), 
                                                n_output_periods*n_nodes[1]) 

    def forward(self, x):
        o_speed = self.model_warmup_speed(x[0])
        o_speed_flattened = o_speed.view(o_speed.shape[0], 1, 1, -1)
        
        o_inflow = self.model_inflow(x[1])
        o_inflow_flattened = o_inflow.view(o_inflow.shape[0], 1, 1, -1)
        
        input_fusion = torch.concat((o_speed_flattened, o_inflow_flattened), dim=-1)
        output = self.linear(F.relu(input_fusion))
        return output
   
    
#  Input: Speed AND Outflow, Prediction: outflow. Speed prediction GNN has warmup. Outflow GNN has no warmup
class MultiPredictionNet8(torch.nn.Module): 
    def __init__(self, 
                 n_input_periods,
                 n_output_periods,
                 node_features,
                 n_nodes, 
                 hidden_warmup,
                 edge_index_list,
                 model_path,
                ):
        super(MultiPredictionNet8, self).__init__()
        # Speed prediction GNN
        self.model_warmup_speed = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = node_features[0],
                                                        num_of_vertices = n_nodes[0],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[0],
                                                        edge_index=edge_index_list[0],
                                                    )  
        ## Load state dicts
        self.model_warmup_speed.load_state_dict(torch.load("../models/" + model_path[0] + ".pth"))
        
        # Outflow prediction GNN
        self.model_outflow = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = node_features[2],
                                                        num_of_vertices = n_nodes[2],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[2],
                                                        edge_index=edge_index_list[2],
                                                    )  

        self.linear = torch.nn.Linear(n_output_periods* (n_nodes[0]+n_nodes[2]), 
                                                n_output_periods*n_nodes[2]) 

    def forward(self, x):
        o_speed = self.model_warmup_speed(x[0])
        o_speed_flattened = o_speed.view(o_speed.shape[0], 1, 1, -1)
        
        o_outflow = self.model_outflow(x[1])
        o_outflow_flattened = o_outflow.view(o_outflow.shape[0], 1, 1, -1)
        
        input_fusion = torch.concat((o_speed_flattened, o_outflow_flattened), dim=-1)
        output = self.linear(F.relu(input_fusion))
        return output 
    
#  Input: Speed AND Inflow AND DP Outflow prediction, Prediction: speed AND inflow. Speed prediction GNN has warmup. Inflow GNN has warmup. DP outflow GNN is already trained and will NOT have backprop. 
class MultiPredictionNet9(torch.nn.Module): 
    def __init__(self, 
                 n_input_periods,
                 n_output_periods,
                 node_features,
                 n_nodes, 
                 hidden_warmup,
                 edge_index_list,
                 model_path,
                ):
        super(MultiPredictionNet9, self).__init__()
        # Speed prediction GNN
        self.model_warmup_speed = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = node_features[0],
                                                        num_of_vertices = n_nodes[0],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[0],
                                                        edge_index=edge_index_list[0],
                                                    )  
        ## Load state dicts
        self.model_warmup_speed.load_state_dict(torch.load("../models/" + model_path[0] + ".pth"))
        
        # Inflow prediction GNN
        self.model_warmup_inflow = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = node_features[1],
                                                        num_of_vertices = n_nodes[1],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[1],
                                                        edge_index=edge_index_list[1],
                                                    )  
        ## Load state dicts
        self.model_warmup_inflow.load_state_dict(torch.load("../models/" + model_path[1] + ".pth"))
        
        # DP outflow prediction GNN
        self.model_dp_trained_outflow = MultiPredictionNet6(n_input_periods=n_input_periods,
                                             n_output_periods=n_output_periods,
                                             node_features=node_features,
                                             n_nodes=n_nodes, 
                                             hidden_warmup=hidden_warmup,
                                             edge_index_list=edge_index_list,
                                             model_path=model_path,
                                           )
        ## Load state dicts
        self.model_dp_trained_outflow.load_state_dict(torch.load("../models/" + model_path[3] + ".pth"))
        
        ## Freeze parameters of model_dp_trained_outflow
        for param in self.model_dp_trained_outflow.parameters():
            param.requires_grad = False

        self.linear = torch.nn.Linear(n_output_periods * sum(n_nodes), 
                                                n_output_periods * (n_nodes[0]+n_nodes[1])) 

    def forward(self, x):
        o_speed = self.model_warmup_speed(x[0])
        o_speed_flattened = o_speed.view(o_speed.shape[0], 1, 1, -1)
        
        o_inflow = self.model_warmup_inflow(x[1])
        o_inflow_flattened = o_inflow.view(o_inflow.shape[0], 1, 1, -1)
        
        o_outflow = self.model_dp_trained_outflow(x[0])
        o_outflow_flattened = o_outflow.view(o_outflow.shape[0], 1, 1, -1)
        
        input_fusion = torch.concat((o_speed_flattened, o_inflow_flattened, o_outflow_flattened), dim=-1)
        output = self.linear(F.relu(input_fusion))
        return output 
    
#  Input: Speed AND Outflow AND DP Inflow prediction, Prediction: speed AND outflow. Speed prediction GNN has warmup. Outflow GNN has warmup.  DP inflow GNN is already trained and will NOT have backprop. 
class MultiPredictionNet10(torch.nn.Module): 
    def __init__(self, 
                 n_input_periods,
                 n_output_periods,
                 node_features,
                 n_nodes, 
                 hidden_warmup,
                 edge_index_list,
                 model_path,
                ):
        super(MultiPredictionNet10, self).__init__()
        # Speed prediction GNN
        self.model_warmup_speed = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = node_features[0],
                                                        num_of_vertices = n_nodes[0],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[0],
                                                        edge_index=edge_index_list[0],
                                                    )  
        ## Load state dicts
        self.model_warmup_speed.load_state_dict(torch.load("../models/" + model_path[0] + ".pth"))
        
        # Inflow prediction GNN
        self.model_warmup_outflow = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = node_features[2],
                                                        num_of_vertices = n_nodes[2],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[2],
                                                        edge_index=edge_index_list[2],
                                                    )  
        ## Load state dicts
        self.model_warmup_outflow.load_state_dict(torch.load("../models/" + model_path[2] + ".pth"))
        
        # DP outflow prediction GNN
        self.model_dp_trained_inflow = MultiPredictionNet5(n_input_periods=n_input_periods,
                                             n_output_periods=n_output_periods,
                                             node_features=node_features,
                                             n_nodes=n_nodes, 
                                             hidden_warmup=hidden_warmup,
                                             edge_index_list=edge_index_list,
                                             model_path=model_path,
                                           )
        ## Load state dicts
        self.model_dp_trained_inflow.load_state_dict(torch.load("../models/" + model_path[3] + ".pth"))
        
        ## Freeze parameters of model_dp_trained_outflow
        for param in self.model_dp_trained_inflow.parameters():
            param.requires_grad = False

        self.linear = torch.nn.Linear(n_output_periods * sum(n_nodes), 
                                                n_output_periods * (n_nodes[0]+n_nodes[2])) 

    def forward(self, x):
        o_speed = self.model_warmup_speed(x[0])
        o_speed_flattened = o_speed.view(o_speed.shape[0], 1, 1, -1)
        
        o_inflow = self.model_dp_trained_inflow(x[0])
        o_inflow_flattened = o_inflow.view(o_inflow.shape[0], 1, 1, -1)
        
        o_outflow = self.model_warmup_outflow(x[1])
        o_outflow_flattened = o_outflow.view(o_outflow.shape[0], 1, 1, -1)
        
        input_fusion = torch.concat((o_speed_flattened, o_inflow_flattened, o_outflow_flattened), dim=-1)
        output = self.linear(F.relu(input_fusion))
        return output 
    

#  Input: Speed AND Inflow, Prediction: Speed AND Inflow. Both speed prediction GNN and inflow prediction GNN have warmup.
class MultiPredictionNet11(torch.nn.Module): 
    def __init__(self, 
                 n_input_periods,
                 n_output_periods,
                 node_features,
                 n_nodes, 
                 hidden_warmup,
                 edge_index_list,
                 model_path,
                ):
        super(MultiPredictionNet11, self).__init__()
        # Speed prediction GNN
        self.model_warmup_speed = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = node_features[0],
                                                        num_of_vertices = n_nodes[0],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[0],
                                                        edge_index=edge_index_list[0],
                                                    )  
        ## Load state dicts
        self.model_warmup_speed.load_state_dict(torch.load("../models/" + model_path[0] + ".pth"))
        
        # Inflow prediction GNN
        self.model_warmup_inflow = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = node_features[1],
                                                        num_of_vertices = n_nodes[1],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[1],
                                                        edge_index=edge_index_list[1],
                                                    )  
        ## Load state dicts
        self.model_warmup_inflow.load_state_dict(torch.load("../models/" + model_path[1] + ".pth"))
        
        self.linear = torch.nn.Linear(n_output_periods * (n_nodes[0]+n_nodes[1]), 
                                                n_output_periods * (n_nodes[0]+n_nodes[1])) 

    def forward(self, x):
        o_speed = self.model_warmup_speed(x[0])
        o_speed_flattened = o_speed.view(o_speed.shape[0], 1, 1, -1)
        
        o_inflow = self.model_warmup_inflow(x[1])
        o_inflow_flattened = o_inflow.view(o_inflow.shape[0], 1, 1, -1)
        
        input_fusion = torch.concat((o_speed_flattened, o_inflow_flattened), dim=-1)
        output = self.linear(F.relu(input_fusion))
        return output
   
    

#  Input: Speed AND Outflow, Prediction: outflow. Speed prediction GNN has warmup. Outflow GNN has no warmup
class MultiPredictionNet12(torch.nn.Module): 
    def __init__(self, 
                 n_input_periods,
                 n_output_periods,
                 node_features,
                 n_nodes, 
                 hidden_warmup,
                 edge_index_list,
                 model_path,
                ):
        super(MultiPredictionNet12, self).__init__()
        # Speed prediction GNN
        self.model_warmup_speed = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = node_features[0],
                                                        num_of_vertices = n_nodes[0],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[0],
                                                        edge_index=edge_index_list[0],
                                                    )  
        ## Load state dicts
        self.model_warmup_speed.load_state_dict(torch.load("../models/" + model_path[0] + ".pth"))
        
        # Outflow prediction GNN
        self.model_warmup_outflow = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = node_features[2],
                                                        num_of_vertices = n_nodes[2],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[2],
                                                        edge_index=edge_index_list[2],
                                                    )  
        ## Load state dicts
        self.model_warmup_outflow.load_state_dict(torch.load("../models/" + model_path[2] + ".pth"))

        self.linear = torch.nn.Linear(n_output_periods * (n_nodes[0]+n_nodes[2]), 
                                                n_output_periods * (n_nodes[0]+n_nodes[2])) 

    def forward(self, x):
        o_speed = self.model_warmup_speed(x[0])
        o_speed_flattened = o_speed.view(o_speed.shape[0], 1, 1, -1)
        
        o_outflow = self.model_warmup_outflow(x[1])
        o_outflow_flattened = o_outflow.view(o_outflow.shape[0], 1, 1, -1)
        
        input_fusion = torch.concat((o_speed_flattened, o_outflow_flattened), dim=-1)
        output = self.linear(F.relu(input_fusion))
        return output 
    

    
#  Input: Speed AND Inflow AND DP Outflow. prediction, Prediction: speed AND inflow. Speed prediction GNN has warmup. Inflow GNN has warmup. DP outflow GNN is already trained and will have backprop. 
class MultiPredictionNet13(torch.nn.Module): 
    def __init__(self, 
                 n_input_periods,
                 n_output_periods,
                 node_features,
                 n_nodes, 
                 hidden_warmup,
                 edge_index_list,
                 model_path,
                ):
        super(MultiPredictionNet13, self).__init__()
        # Speed prediction GNN
        self.model_warmup_speed = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = node_features[0],
                                                        num_of_vertices = n_nodes[0],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[0],
                                                        edge_index=edge_index_list[0],
                                                    )  
        ## Load state dicts
        self.model_warmup_speed.load_state_dict(torch.load("../models/" + model_path[0] + ".pth"))
        
        # Inflow prediction GNN
        self.model_warmup_inflow = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = node_features[1],
                                                        num_of_vertices = n_nodes[1],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[1],
                                                        edge_index=edge_index_list[1],
                                                    )  
        ## Load state dicts
        self.model_warmup_inflow.load_state_dict(torch.load("../models/" + model_path[1] + ".pth"))
        
        # DP outflow prediction GNN
        self.model_dp_trained_outflow = MultiPredictionNet6(n_input_periods=n_input_periods,
                                             n_output_periods=n_output_periods,
                                             node_features=node_features,
                                             n_nodes=n_nodes, 
                                             hidden_warmup=hidden_warmup,
                                             edge_index_list=edge_index_list,
                                             model_path=model_path,
                                           )
        ## Load state dicts
        self.model_dp_trained_outflow.load_state_dict(torch.load("../models/" + model_path[3] + ".pth"))

        self.linear = torch.nn.Linear(n_output_periods * sum(n_nodes), 
                                                n_output_periods * (n_nodes[0]+n_nodes[1])) 

    def forward(self, x):
        o_speed = self.model_warmup_speed(x[0])
        o_speed_flattened = o_speed.view(o_speed.shape[0], 1, 1, -1)
        
        o_inflow = self.model_warmup_inflow(x[1])
        o_inflow_flattened = o_inflow.view(o_inflow.shape[0], 1, 1, -1)
        
        o_outflow = self.model_dp_trained_outflow(x[0])
        o_outflow_flattened = o_outflow.view(o_outflow.shape[0], 1, 1, -1)
        
        input_fusion = torch.concat((o_speed_flattened, o_inflow_flattened, o_outflow_flattened), dim=-1)
        output = self.linear(F.relu(input_fusion))
        return output 
    
#  Input: Speed AND Outflow AND DP Inflow prediction. Prediction: speed AND outflow. Speed prediction GNN has warmup. Outflow GNN has warmup. DP inflow GNN is already trained and will have backprop. 
class MultiPredictionNet14(torch.nn.Module): 
    def __init__(self, 
                 n_input_periods,
                 n_output_periods,
                 node_features,
                 n_nodes, 
                 hidden_warmup,
                 edge_index_list,
                 model_path,
                ):
        super(MultiPredictionNet14, self).__init__()
        # Speed prediction GNN
        self.model_warmup_speed = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = node_features[0],
                                                        num_of_vertices = n_nodes[0],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[0],
                                                        edge_index=edge_index_list[0],
                                                    )  
        ## Load state dicts
        self.model_warmup_speed.load_state_dict(torch.load("../models/" + model_path[0] + ".pth"))
        
        # Inflow prediction GNNDo we 
        self.model_warmup_outflow = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = node_features[2],
                                                        num_of_vertices = n_nodes[2],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[2],
                                                        edge_index=edge_index_list[2],
                                                    )  
        ## Load state dicts
        self.model_warmup_outflow.load_state_dict(torch.load("../models/" + model_path[2] + ".pth"))
        
        # DP outflow prediction GNN
        self.model_dp_trained_inflow = MultiPredictionNet5(n_input_periods=n_input_periods,
                                             n_output_periods=n_output_periods,
                                             node_features=node_features,
                                             n_nodes=n_nodes, 
                                             hidden_warmup=hidden_warmup,
                                             edge_index_list=edge_index_list,
                                             model_path=model_path,
                                           )
        ## Load state dicts
        self.model_dp_trained_inflow.load_state_dict(torch.load("../models/" + model_path[3] + ".pth"))
        
        self.linear = torch.nn.Linear(n_output_periods * sum(n_nodes), 
                                                n_output_periods * (n_nodes[0]+n_nodes[2])) 

    def forward(self, x):
        o_speed = self.model_warmup_speed(x[0])
        o_speed_flattened = o_speed.view(o_speed.shape[0], 1, 1, -1)
        
        o_inflow = self.model_dp_trained_inflow(x[0])
        o_inflow_flattened = o_inflow.view(o_inflow.shape[0], 1, 1, -1)
        
        o_outflow = self.model_warmup_outflow(x[1])
        o_outflow_flattened = o_outflow.view(o_outflow.shape[0], 1, 1, -1)
        
        input_fusion = torch.concat((o_speed_flattened, o_inflow_flattened, o_outflow_flattened), dim=-1)
        output = self.linear(F.relu(input_fusion))
        return output 
    
# MultiPredictionNet15 & MultiPredictionNet16
# Warning: MultiPredictionNet15 and MultiPredictionNet16 are already defined based on MultiPredictionNet9 and MultiPredictionNet10 so DONT use their names

#  Input: Speed + Inflow + Outflow, Prediction: inflow. Speed/Inflow/Outflow prediction GNNs have warmup. 
class MultiPredictionNet17(torch.nn.Module): 
    def __init__(self, 
                 n_input_periods,
                 n_output_periods,
                 node_features,
                 n_nodes, 
                 hidden_warmup,
                 edge_index_list,
                 model_path,
                ):
        super(MultiPredictionNet17, self).__init__()
        # Speed prediction GNN
        self.model_warmup_speed = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = node_features[0],
                                                        num_of_vertices = n_nodes[0],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[0],
                                                        edge_index=edge_index_list[0],
                                                    )  
        ## Load state dicts
        self.model_warmup_speed.load_state_dict(torch.load("../models/" + model_path[0] + ".pth"))
        
        # Inflow prediction GNN
        self.model_warmup_inflow = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = node_features[1],
                                                        num_of_vertices = n_nodes[1],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[1],
                                                        edge_index=edge_index_list[1],
                                                    )  
        ## Load state dicts
        self.model_warmup_inflow.load_state_dict(torch.load("../models/" + model_path[1] + ".pth"))
        
        # outflow prediction GNN
        self.model_warmup_outflow = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = node_features[2],
                                                        num_of_vertices = n_nodes[2],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[2],
                                                        edge_index=edge_index_list[2],
                                                    )  
        ## Load state dicts
        self.model_warmup_outflow.load_state_dict(torch.load("../models/" + model_path[2] + ".pth"))

        self.linear = torch.nn.Linear(n_output_periods * (n_nodes[0]+n_nodes[1]+n_nodes[2]), 
                                                n_output_periods * (n_nodes[1])) 
        

    def forward(self, x):
        o_speed = self.model_warmup_speed(x[0])
        o_speed_flattened = o_speed.view(o_speed.shape[0], 1, 1, -1)
        
        o_inflow = self.model_warmup_inflow(x[1])
        o_inflow_flattened = o_inflow.view(o_inflow.shape[0], 1, 1, -1)
        
        o_outflow = self.model_warmup_outflow(x[2])
        o_outflow_flattened = o_outflow.view(o_outflow.shape[0], 1, 1, -1)
        
        input_fusion = torch.concat((o_speed_flattened, o_inflow_flattened, o_outflow_flattened), dim=-1)
        output = self.linear(F.relu(input_fusion))
        return output 
    

#  Input: Speed + Inflow + Outflow, Prediction: inflow. Speed/Inflow/Outflow prediction GNNs have warmup. 
class MultiPredictionNet18(torch.nn.Module): 
    def __init__(self, 
                 n_input_periods,
                 n_output_periods,
                 node_features,
                 n_nodes, 
                 hidden_warmup,
                 edge_index_list,
                 model_path,
                ):
        super(MultiPredictionNet18, self).__init__()
        # Speed prediction GNN
        self.model_warmup_speed = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = node_features[0],
                                                        num_of_vertices = n_nodes[0],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[0],
                                                        edge_index=edge_index_list[0],
                                                    )  
        ## Load state dicts
        self.model_warmup_speed.load_state_dict(torch.load("../models/" + model_path[0] + ".pth"))
        
        # Inflow prediction GNN
        self.model_warmup_inflow = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = node_features[1],
                                                        num_of_vertices = n_nodes[1],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[1],
                                                        edge_index=edge_index_list[1],
                                                    )  
        ## Load state dicts
        self.model_warmup_inflow.load_state_dict(torch.load("../models/" + model_path[1] + ".pth"))
        
        # outflow prediction GNN
        self.model_warmup_outflow = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = node_features[2],
                                                        num_of_vertices = n_nodes[2],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[2],
                                                        edge_index=edge_index_list[2],
                                                    )  
        ## Load state dicts
        self.model_warmup_outflow.load_state_dict(torch.load("../models/" + model_path[2] + ".pth"))

        self.linear = torch.nn.Linear(n_output_periods * (n_nodes[0]+n_nodes[1]+n_nodes[2]), 
                                                n_output_periods * (n_nodes[2])) 
        

    def forward(self, x):
        o_speed = self.model_warmup_speed(x[0])
        o_speed_flattened = o_speed.view(o_speed.shape[0], 1, 1, -1)
        
        o_inflow = self.model_warmup_inflow(x[1])
        o_inflow_flattened = o_inflow.view(o_inflow.shape[0], 1, 1, -1)
        
        o_outflow = self.model_warmup_outflow(x[2])
        o_outflow_flattened = o_outflow.view(o_outflow.shape[0], 1, 1, -1)
        
        input_fusion = torch.concat((o_speed_flattened, o_inflow_flattened, o_outflow_flattened), dim=-1)
        output = self.linear(F.relu(input_fusion))
        return output 
    

#  Input: Speed + Inflow + Outflow, Prediction: Speed + Inflow. Speed/Inflow/Outflow prediction GNNs have warmup. 
class MultiPredictionNet19(torch.nn.Module): 
    def __init__(self, 
                 n_input_periods,
                 n_output_periods,
                 node_features,
                 n_nodes, 
                 hidden_warmup,
                 edge_index_list,
                 model_path,
                ):
        super(MultiPredictionNet19, self).__init__()
        # Speed prediction GNN
        self.model_warmup_speed = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = node_features[0],
                                                        num_of_vertices = n_nodes[0],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[0],
                                                        edge_index=edge_index_list[0],
                                                    )  
        ## Load state dicts
        self.model_warmup_speed.load_state_dict(torch.load("../models/" + model_path[0] + ".pth"))
        
        # Inflow prediction GNN
        self.model_warmup_inflow = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = node_features[1],
                                                        num_of_vertices = n_nodes[1],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[1],
                                                        edge_index=edge_index_list[1],
                                                    )  
        ## Load state dicts
        self.model_warmup_inflow.load_state_dict(torch.load("../models/" + model_path[1] + ".pth"))
        
        # DP outflow prediction GNN
        self.model_warmup_outflow = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = node_features[2],
                                                        num_of_vertices = n_nodes[2],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[2],
                                                        edge_index=edge_index_list[2],
                                                    )  
        ## Load state dicts
        self.model_warmup_outflow.load_state_dict(torch.load("../models/" + model_path[2] + ".pth"))

        self.linear = torch.nn.Linear(n_output_periods * sum(n_nodes), 
                                                n_output_periods * (n_nodes[0]+n_nodes[1])) 

    def forward(self, x):
        o_speed = self.model_warmup_speed(x[0])
        o_speed_flattened = o_speed.view(o_speed.shape[0], 1, 1, -1)
        
        o_inflow = self.model_warmup_inflow(x[1])
        o_inflow_flattened = o_inflow.view(o_inflow.shape[0], 1, 1, -1)
        
        o_outflow = self.model_warmup_outflow(x[2])
        o_outflow_flattened = o_outflow.view(o_outflow.shape[0], 1, 1, -1)
        
        input_fusion = torch.concat((o_speed_flattened, o_inflow_flattened, o_outflow_flattened), dim=-1)
        output = self.linear(F.relu(input_fusion))
        return output 
    
    

#  Input: Speed + Inflow + Outflow, Prediction: speed + inflow + outflow. Speed/Inflow/Outflow prediction GNNs have warmup. 
class MultiPredictionNet21(torch.nn.Module): 
    def __init__(self, 
                 n_input_periods,
                 n_output_periods,
                 node_features,
                 n_nodes, 
                 hidden_warmup,
                 edge_index_list,
                 model_path,
                ):
        super(MultiPredictionNet21, self).__init__()
        # Speed prediction GNN
        self.model_warmup_speed = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = node_features[0],
                                                        num_of_vertices = n_nodes[0],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[0],
                                                        edge_index=edge_index_list[0],
                                                    )  
        ## Load state dicts
        self.model_warmup_speed.load_state_dict(torch.load("../models/" + model_path[0] + ".pth"))
        
        # Inflow prediction GNN
        self.model_warmup_inflow = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = node_features[1],
                                                        num_of_vertices = n_nodes[1],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[1],
                                                        edge_index=edge_index_list[1],
                                                    )  
        ## Load state dicts
        self.model_warmup_inflow.load_state_dict(torch.load("../models/" + model_path[1] + ".pth"))
        
        # DP outflow prediction GNN
        self.model_warmup_outflow = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = node_features[2],
                                                        num_of_vertices = n_nodes[2],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[2],
                                                        edge_index=edge_index_list[2],
                                                    )  
        ## Load state dicts
        self.model_warmup_outflow.load_state_dict(torch.load("../models/" + model_path[2] + ".pth"))

        self.linear = torch.nn.Linear(n_output_periods * sum(n_nodes), 
                                                n_output_periods * sum(n_nodes)) 

    def forward(self, x):
        o_speed = self.model_warmup_speed(x[0])
        o_speed_flattened = o_speed.view(o_speed.shape[0], 1, 1, -1)
        
        o_inflow = self.model_warmup_inflow(x[1])
        o_inflow_flattened = o_inflow.view(o_inflow.shape[0], 1, 1, -1)
        
        o_outflow = self.model_warmup_outflow(x[2])
        o_outflow_flattened = o_outflow.view(o_outflow.shape[0], 1, 1, -1)
        
        input_fusion = torch.concat((o_speed_flattened, o_inflow_flattened, o_outflow_flattened), dim=-1)
        output = self.linear(F.relu(input_fusion))
        return output 
    
    

#  Input: Speed + Inflow + Outflow, Prediction: inflow + outflow. Speed/Inflow/Outflow prediction GNNs have warmup. 
class MultiPredictionNet23(torch.nn.Module): 
    def __init__(self, 
                 n_input_periods,
                 n_output_periods,
                 node_features,
                 n_nodes, 
                 hidden_warmup,
                 edge_index_list,
                 model_path,
                ):
        super(MultiPredictionNet23, self).__init__()
        # Speed prediction GNN
        self.model_warmup_speed = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = node_features[0],
                                                        num_of_vertices = n_nodes[0],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[0],
                                                        edge_index=edge_index_list[0],
                                                    )  
        ## Load state dicts
        self.model_warmup_speed.load_state_dict(torch.load("../models/" + model_path[0] + ".pth"))
        
        # Inflow prediction GNN
        self.model_warmup_inflow = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = node_features[1],
                                                        num_of_vertices = n_nodes[1],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[1],
                                                        edge_index=edge_index_list[1],
                                                    )  
        ## Load state dicts
        self.model_warmup_inflow.load_state_dict(torch.load("../models/" + model_path[1] + ".pth"))
        
        # DP outflow prediction GNN
        self.model_warmup_outflow = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = node_features[2],
                                                        num_of_vertices = n_nodes[2],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[2],
                                                        edge_index=edge_index_list[2],
                                                    )  
        ## Load state dicts
        self.model_warmup_outflow.load_state_dict(torch.load("../models/" + model_path[2] + ".pth"))

        self.linear = torch.nn.Linear(n_output_periods * sum(n_nodes), 
                                                n_output_periods * (n_nodes[1]+n_nodes[2])) 

    def forward(self, x):
        o_speed = self.model_warmup_speed(x[0])
        o_speed_flattened = o_speed.view(o_speed.shape[0], 1, 1, -1)
        
        o_inflow = self.model_warmup_inflow(x[1])
        o_inflow_flattened = o_inflow.view(o_inflow.shape[0], 1, 1, -1)
        
        o_outflow = self.model_warmup_outflow(x[2])
        o_outflow_flattened = o_outflow.view(o_outflow.shape[0], 1, 1, -1)
        
        input_fusion = torch.concat((o_speed_flattened, o_inflow_flattened, o_outflow_flattened), dim=-1)
        output = self.linear(F.relu(input_fusion))
        return output 
    
    

#  Input: Inflow + Outflow, Prediction: inflow + outflow. Inflow/Outflow prediction GNNs have warmup. 
class MultiPredictionNet24(torch.nn.Module): 
    def __init__(self, 
                 n_input_periods,
                 n_output_periods,
                 node_features,
                 n_nodes, 
                 hidden_warmup,
                 edge_index_list,
                 model_path,
                ):
        super(MultiPredictionNet24, self).__init__()
        
        # Inflow prediction GNN
        self.model_warmup_inflow = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = node_features[1],
                                                        num_of_vertices = n_nodes[1],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[1],
                                                        edge_index=edge_index_list[1],
                                                    )  
        ## Load state dicts
        self.model_warmup_inflow.load_state_dict(torch.load("../models/" + model_path[1] + ".pth"))
        
        # DP outflow prediction GNN
        self.model_warmup_outflow = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = node_features[2],
                                                        num_of_vertices = n_nodes[2],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[2],
                                                        edge_index=edge_index_list[2],
                                                    )  
        ## Load state dicts
        self.model_warmup_outflow.load_state_dict(torch.load("../models/" + model_path[2] + ".pth"))

        self.linear = torch.nn.Linear(n_output_periods * (n_nodes[1]+n_nodes[2]), 
                                                n_output_periods * (n_nodes[1]+n_nodes[2])) 

    def forward(self, x):
        o_inflow = self.model_warmup_inflow(x[0])
        o_inflow_flattened = o_inflow.view(o_inflow.shape[0], 1, 1, -1)
        
        o_outflow = self.model_warmup_outflow(x[1])
        o_outflow_flattened = o_outflow.view(o_outflow.shape[0], 1, 1, -1)
        
        input_fusion = torch.concat((o_inflow_flattened, o_outflow_flattened), dim=-1)
        output = self.linear(F.relu(input_fusion))
        return output 
    
    
#  Input: Inflow gy + Outflow gy , Prediction: inflow + outflow. Inflow/Outflow prediction GNNs have warmup. 
class MP24(torch.nn.Module): 
    def __init__(self, 
                 n_input_periods,
                 n_output_periods,
                 node_features,
                 n_nodes, 
                 hidden_warmup,
                 edge_index_list,
                 model_path,
                ):
        super(MP24, self).__init__()
        
        # Inflow prediction GNN
        self.model_warmup_inflow = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = node_features[3],
                                                        num_of_vertices = n_nodes[3],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[3],
                                                        edge_index=edge_index_list[3],
                                                    )  
        ## Load state dicts
        self.model_warmup_inflow.load_state_dict(torch.load("../models/" + model_path[3] + ".pth"))
        
        # DP outflow prediction GNN
        self.model_warmup_outflow = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = node_features[4],
                                                        num_of_vertices = n_nodes[4],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[4],
                                                        edge_index=edge_index_list[4],
                                                    )  
        ## Load state dicts
        self.model_warmup_outflow.load_state_dict(torch.load("../models/" + model_path[4] + ".pth"))

        self.linear = torch.nn.Linear(n_output_periods * (n_nodes[3]+n_nodes[4]), 
                                                n_output_periods * (n_nodes[1]+n_nodes[2])) 

    def forward(self, x):
        o_inflow = self.model_warmup_inflow(x[0])
#         print("o_inflow.shape: ", o_inflow.shape)
        o_inflow_flattened = o_inflow.view(o_inflow.shape[0], 1, 1, -1)
        
        o_outflow = self.model_warmup_outflow(x[1])
#         print("o_outflow.shape: ", o_outflow.shape)
        o_outflow_flattened = o_outflow.view(o_outflow.shape[0], 1, 1, -1)
        
        input_fusion = torch.concat((o_inflow_flattened, o_outflow_flattened), dim=-1)
        output = self.linear(F.relu(input_fusion))
        return output 
    
    
    
    
#  Input: Inflow gy + Outflow gy , Prediction: inflow + outflow. Inflow/Outflow prediction GNNs have warmup. 
class MP24PP(torch.nn.Module): 
    def __init__(self, 
                 n_input_periods,
                 n_output_periods,
                 node_features,
                 n_nodes, 
                 hidden_warmup,
                 edge_index_list,
                 model_path,
                ):
        super(MP24PP, self).__init__()

        # GNN with 2 features
        self.model_gnn = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = 2, # Combine two graph features to become 2 input features
                                                        num_of_vertices = n_nodes[3],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[3],
                                                        edge_index=edge_index_list[3],
                                                    )  
        

        self.linear = torch.nn.Linear(n_output_periods * (n_nodes[3]), 
                                                n_output_periods * (n_nodes[1]+n_nodes[2])) 

    def forward(self, x):
        inflow_outflow_concatenated = torch.cat((x[0],
                                                 x[1]
                                                ), 
                                                dim=2)
        gnn_output = self.model_gnn(inflow_outflow_concatenated)
        gnn_output_flattened = gnn_output.view(gnn_output.shape[0], 1, 1, -1)
        output = self.linear(F.relu(gnn_output_flattened))
        return output 
        

#  Input: Inflow + Outflow, Prediction: inflow. Inflow/Outflow prediction GNNs have warmup. 
class MultiPredictionNet25(torch.nn.Module): 
    def __init__(self, 
                 n_input_periods,
                 n_output_periods,
                 node_features,
                 n_nodes, 
                 hidden_warmup,
                 edge_index_list,
                 model_path,
                ):
        super(MultiPredictionNet25, self).__init__()
        
        # Inflow prediction GNN
        self.model_warmup_inflow = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = node_features[1],
                                                        num_of_vertices = n_nodes[1],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[1],
                                                        edge_index=edge_index_list[1],
                                                    )  
        ## Load state dicts
        self.model_warmup_inflow.load_state_dict(torch.load("../models/" + model_path[1] + ".pth"))
        
        # DP outflow prediction GNN
        self.model_warmup_outflow = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = node_features[2],
                                                        num_of_vertices = n_nodes[2],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[2],
                                                        edge_index=edge_index_list[2],
                                                    )  
        ## Load state dicts
        self.model_warmup_outflow.load_state_dict(torch.load("../models/" + model_path[2] + ".pth"))

        self.linear = torch.nn.Linear(n_output_periods * (n_nodes[1]+n_nodes[2]), 
                                                n_output_periods * (n_nodes[1])) 

    def forward(self, x):
        o_inflow = self.model_warmup_inflow(x[0])
        o_inflow_flattened = o_inflow.view(o_inflow.shape[0], 1, 1, -1)
        
        o_outflow = self.model_warmup_outflow(x[1])
        o_outflow_flattened = o_outflow.view(o_outflow.shape[0], 1, 1, -1)
        
        input_fusion = torch.concat((o_inflow_flattened, o_outflow_flattened), dim=-1)
        output = self.linear(F.relu(input_fusion))
        return output 

    
    
    
#  Input: Inflow, Prediction: inflow. Inflow prediction GNNs have warmup. 
class MP0_i(torch.nn.Module): 
    def __init__(self, 
                 n_input_periods,
                 n_output_periods,
                 node_features,
                 n_nodes, 
                 hidden_warmup,
                 edge_index_list,
                 model_path,
                 train_mode = True,
                ):
        super(MP0_i, self).__init__()
        self.n_node = n_nodes[3]
        
        # Inflow prediction GNN
        self.model_warmup_inflow = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = node_features[3],
                                                        num_of_vertices = n_nodes[3],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[3],
                                                        edge_index=edge_index_list[3],
                                                    )  
        ## Load state dicts
        if train_mode:
            self.model_warmup_inflow.load_state_dict(torch.load("../models/" + model_path[3] + ".pth"))

    def forward(self, x):
        return self.model_warmup_inflow(x[0])
    
    
    
#  Input: Outflow, Prediction: outflow. Outflow prediction GNNs have warmup. 
class MP0_o(torch.nn.Module): 
    def __init__(self, 
                 n_input_periods,
                 n_output_periods,
                 node_features,
                 n_nodes, 
                 hidden_warmup,
                 edge_index_list,
                 model_path,
                 train_mode = True,
                ):
        super(MP0_o, self).__init__()
        self.n_node = n_nodes[4]
        # Inflow prediction GNN
        self.model_warmup_outflow = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = node_features[4],
                                                        num_of_vertices = n_nodes[4],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[4],
                                                        edge_index=edge_index_list[4],
                                                    )  
        ## Load state dicts
        if train_mode:
            self.model_warmup_outflow.load_state_dict(torch.load("../models/" + model_path[4] + ".pth"))

    def forward(self, x):
        return self.model_warmup_outflow(x[0])
    

    
#  Input: Inflow, Prediction: inflow. Inflow prediction GNNs have warmup. 
class MP0_no_relu(torch.nn.Module): 
    def __init__(self, 
                 n_input_periods,
                 n_output_periods,
                 node_features,
                 n_nodes, 
                 hidden_warmup,
                 edge_index_list,
                 model_path,
                ):
        super(MP0_no_relu, self).__init__()
        self.n_node = n_nodes[3]
        
        # Inflow prediction GNN
        self.model_gnn = Customized_ASTGCN(len_input = n_input_periods,
                                        num_for_predict = n_output_periods,
                                        in_channels = node_features[3],
                                        num_of_vertices = n_nodes[3],
                                        nb_block = 2,
                                        K = 3,
                                        nb_chev_filter = 64,
                                        nb_time_filter = 64,
                                        time_strides = 1,
                                        edge_index=edge_index_list[3])

    def forward(self, x):
        return self.model_gnn(x[0])
    
    
#  Input: Inflow + Outflow, Prediction: inflow. Inflow/Outflow prediction GNNs have warmup. 
class MP25(torch.nn.Module): 
    def __init__(self, 
                 n_input_periods,
                 n_output_periods,
                 node_features,
                 n_nodes, 
                 hidden_warmup,
                 edge_index_list,
                 model_path,
                ):
        super(MP25, self).__init__()
        
        # Inflow prediction GNN
        self.model_warmup_inflow = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = node_features[3],
                                                        num_of_vertices = n_nodes[3],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[3],
                                                        edge_index=edge_index_list[3],
                                                    )  
        ## Load state dicts
        self.model_warmup_inflow.load_state_dict(torch.load("../models/" + model_path[3] + ".pth"))
        
        # DP outflow prediction GNN
        self.model_warmup_outflow = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = node_features[4],
                                                        num_of_vertices = n_nodes[4],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[4],
                                                        edge_index=edge_index_list[4],
                                                    )  
        ## Load state dicts
        self.model_warmup_outflow.load_state_dict(torch.load("../models/" + model_path[4] + ".pth"))

        assert n_nodes[1] == n_nodes[2], print('Different inflow and outflow graphs')
        self.linear = torch.nn.Linear(n_output_periods * (n_nodes[3]+n_nodes[4]), 
                                                n_output_periods * (n_nodes[1])) 

    def forward(self, x):
        o_inflow = self.model_warmup_inflow(x[0])
        o_inflow_flattened = o_inflow.view(o_inflow.shape[0], 1, 1, -1)
        
        o_outflow = self.model_warmup_outflow(x[1])
        o_outflow_flattened = o_outflow.view(o_outflow.shape[0], 1, 1, -1)
        
        input_fusion = torch.concat((o_inflow_flattened, o_outflow_flattened), dim=-1)
        output = self.linear(F.relu(input_fusion))
        return output 
    
    
#  Input: Inflow + Outflow, Prediction: inflow. Inflow/Outflow prediction GNNs have warmup. 
class MP25X(torch.nn.Module): 
    def __init__(self, 
                 n_input_periods,
                 n_output_periods,
                 node_features,
                 n_nodes, 
                 hidden_warmup,
                 edge_index_list,
                 model_path,
                ):
        super(MP25X, self).__init__()
        
        # Inflow prediction GNN
        self.model_warmup_inflow = Customized_ASTGCN(len_input = n_input_periods,
                                                    num_for_predict = n_output_periods,
                                                    in_channels = node_features[3],
                                                    num_of_vertices = n_nodes[3],
                                                    nb_block = 2,
                                                    K = 3,
                                                    nb_chev_filter = 64,
                                                    nb_time_filter = 64,
                                                    time_strides = 1,
                                                    edge_index=edge_index_list[3]) 
        
        
        # DP outflow prediction GNN
        self.model_warmup_outflow = Customized_ASTGCN(len_input = n_input_periods,
                                                    num_for_predict = n_output_periods,
                                                    in_channels = node_features[3],
                                                    num_of_vertices = n_nodes[3],
                                                    nb_block = 2,
                                                    K = 3,
                                                    nb_chev_filter = 64,
                                                    nb_time_filter = 64,
                                                    time_strides = 1,
                                                    edge_index=edge_index_list[3]) 

        assert n_nodes[1] == n_nodes[2], print('Different inflow and outflow graphs')
        self.linear = torch.nn.Linear(n_output_periods * (n_nodes[3]+n_nodes[4]), 
                                                n_output_periods * (n_nodes[1])) 

    def forward(self, x):
        o_inflow = self.model_warmup_inflow(x[0])
        o_inflow_flattened = o_inflow.reshape(o_inflow.shape[0], 1, 1, -1)
        
        o_outflow = self.model_warmup_outflow(x[1])
        o_outflow_flattened = o_outflow.reshape(o_outflow.shape[0], 1, 1, -1)
        
        input_fusion = torch.concat((o_inflow_flattened, o_outflow_flattened), dim=-1)
        output = self.linear(F.relu(input_fusion))
        return output 
    
    
#  Input: Inflow + Outflow, Prediction: inflow. Inflow/Outflow prediction GNNs have warmup. 
class MultiPredictionNet25pp(torch.nn.Module): 
    def __init__(self, 
                 n_input_periods,
                 n_output_periods,
                 node_features,
                 n_nodes, 
                 hidden_warmup,
                 edge_index_list,
                 model_path,
                ):
        super(MultiPredictionNet25pp, self).__init__()
        
        self.n_node = n_nodes[1]
        
        # Inflow prediction GNN
        self.model_warmup_inflow = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = node_features[1],
                                                        num_of_vertices = n_nodes[1],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[1],
                                                        edge_index=edge_index_list[1],
                                                    )  
        ## Load state dicts
        self.model_warmup_inflow.load_state_dict(torch.load("../models/" + model_path[1] + ".pth"))
        
        ## Freeze parameters of model_dp_trained_outflow
        for param in self.model_warmup_inflow.parameters():
            param.requires_grad = False
                    
        # DP outflow prediction GNN
        self.model_warmup_outflow = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = node_features[2],
                                                        num_of_vertices = n_nodes[2],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[2],
                                                        edge_index=edge_index_list[2],
                                                    )  
        ## Load state dicts
        self.model_warmup_outflow.load_state_dict(torch.load("../models/" + model_path[2] + ".pth"))
        
        ## Freeze parameters of model_dp_trained_outflow
        for param in self.model_warmup_outflow.parameters():
            param.requires_grad = False

        # DP inflow prediction GNN
        self.model_inflow = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = 2, # Combine two graph features to become 2 input features
                                                        num_of_vertices = n_nodes[1],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[1],
                                                        edge_index=edge_index_list[1],
                                                    )  
        
        self.linear = torch.nn.Linear(n_output_periods * (n_nodes[1]), 
                                                n_output_periods * (n_nodes[1])) 

    def forward(self, x):
        o_inflow = self.model_warmup_inflow(x[0])
        o_inflow_reshaped = o_inflow.reshape(o_inflow.shape[0], self.n_node, 1, -1)
        o_outflow = self.model_warmup_outflow(x[1])
        o_outflow_reshaped = o_outflow.reshape(o_outflow.shape[0], self.n_node, 1, -1)
        inflow_final = torch.cat((o_inflow_reshaped,
                                   o_outflow_reshaped), dim=2)
        inflow_final = self.model_inflow(inflow_final)
        inflow_final_flattened = inflow_final.view(inflow_final.shape[0], 1, 1, -1)
        inflow_final_flattened = self.linear(F.relu(inflow_final_flattened))
        return inflow_final_flattened 
    
    

#  Input: Inflow + Outflow, Prediction: inflow. Inflow/Outflow prediction GNNs have warmup. 
class MP25P(torch.nn.Module): 
    def __init__(self, 
                 n_input_periods,
                 n_output_periods,
                 node_features,
                 n_nodes, 
                 hidden_warmup,
                 edge_index_list,
                 model_path,
                ):
        super(MP25P, self).__init__()
        
        self.n_node = n_nodes[1]
        
        # Inflow prediction GNN
        self.model_warmup_inflow = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = node_features[3],
                                                        num_of_vertices = n_nodes[3],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[3],
                                                        edge_index=edge_index_list[3],
                                                    )  
        ## Load state dicts
        self.model_warmup_inflow.load_state_dict(torch.load("../models/" + model_path[3] + ".pth"))
        
        # DP outflow prediction GNN
        self.model_warmup_outflow = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = node_features[4],
                                                        num_of_vertices = n_nodes[4],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[4],
                                                        edge_index=edge_index_list[4],
                                                    )  
        ## Load state dicts
        self.model_warmup_outflow.load_state_dict(torch.load("../models/" + model_path[4] + ".pth"))
        
        # DP inflow prediction GNN
        self.model_inflow = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = 2, # Combine two graph features to become 2 input features
                                                        num_of_vertices = n_nodes[4],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[4],
                                                        edge_index=edge_index_list[4],
                                                    )  
        
    def forward(self, x):
        o_inflow = self.model_warmup_inflow(x[0])
#         o_inflow_unsqueezed = o_inflow.reshape(o_inflow.shape[0], self.n_node, 1, -1)
        o_inflow_unsqueezed = torch.unsqueeze(o_inflow, 2)
        
        o_outflow = self.model_warmup_outflow(x[1])
#         o_outflow_unsqueezed = o_outflow.reshape(o_outflow.shape[0], self.n_node, 1, -1)
        o_outflow_unsqueezed = torch.unsqueeze(o_outflow, 2)
        
        inflow_final = torch.cat((o_inflow_unsqueezed,
                                  o_outflow_unsqueezed,
                                 ), 
                                 dim=2)
        return self.model_inflow(inflow_final)
        
    

#  Input: Inflow + Outflow, Prediction: inflow. Inflow/Outflow prediction GNNs have warmup. 
class MP25P_2(torch.nn.Module): 
    def __init__(self, 
                 n_input_periods,
                 n_output_periods,
                 node_features,
                 n_nodes, 
                 hidden_warmup,
                 edge_index_list,
                 model_path,
                ):
        super(MP25P_2, self).__init__()
        
        self.n_node = n_nodes[1]
        
        # GNN with 2 inputs
        self.model_gnn = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = 2, # Combine two graph features to become 2 input features
                                                        num_of_vertices = n_nodes[4],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[4],
                                                        edge_index=edge_index_list[4],
                                                    )  
        
    def forward(self, x):
        inflow_outflow = torch.cat((x[0], x[1]), dim=2)
        return self.model_gnn(inflow_outflow)
        
    
    
#  Input: Inflow + Outflow, Prediction: inflow. Inflow/Outflow prediction GNNs have warmup. 
class MP25PP(torch.nn.Module): 
    def __init__(self, 
                 n_input_periods,
                 n_output_periods,
                 node_features,
                 n_nodes, 
                 hidden_warmup,
                 edge_index_list,
                 model_path,
                ):
        super(MP25PP, self).__init__()
        
        self.n_node = n_nodes[1]
        
        # Inflow prediction GNN
        self.model_warmup_inflow = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = node_features[3],
                                                        num_of_vertices = n_nodes[3],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[3],
                                                        edge_index=edge_index_list[3],
                                                    )  
        ## Load state dicts
        self.model_warmup_inflow.load_state_dict(torch.load("../models/" + model_path[3] + ".pth"))
        
        # DP outflow prediction GNN
        self.model_warmup_outflow = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = node_features[4],
                                                        num_of_vertices = n_nodes[4],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[4],
                                                        edge_index=edge_index_list[4],
                                                    )  
        ## Load state dicts
        self.model_warmup_outflow.load_state_dict(torch.load("../models/" + model_path[4] + ".pth"))
        
        # DP inflow prediction GNN
        self.model_inflow = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = 2, # Combine two graph features to become 2 input features
                                                        num_of_vertices = n_nodes[4],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[4],
                                                        edge_index=edge_index_list[4],
                                                    )  
        
        self.linear = torch.nn.Linear(n_output_periods * (n_nodes[4]), 
                                                n_output_periods * (n_nodes[1])) 

    def forward(self, x):
        o_inflow = self.model_warmup_inflow(x[0])
#         o_inflow_reshaped = o_inflow.reshape(o_inflow.shape[0], self.n_node, 1, -1)
        o_inflow_unsqueezed = torch.unsqueeze(o_inflow, 2)
        
        o_outflow = self.model_warmup_outflow(x[1])
#         o_outflow_reshaped = o_outflow.reshape(o_outflow.shape[0], self.n_node, 1, -1)
        o_outflow_unsqueezed = torch.unsqueeze(o_outflow, 2)
        
        inflow_final = torch.cat((o_inflow_unsqueezed,
                                  o_outflow_unsqueezed), 
                                 dim=2)
        inflow_final = self.model_inflow(inflow_final)
        inflow_final_flattened = inflow_final.view(inflow_final.shape[0], 1, 1, -1)
        inflow_final_flattened = self.linear(F.relu(inflow_final_flattened))
        return inflow_final_flattened 
    
    
    
    
#  Input: Inflow + Outflow, Prediction: inflow. Inflow/Outflow prediction GNNs have warmup. 
class MP25PPSmall(torch.nn.Module): 
    def __init__(self, 
                 n_input_periods,
                 n_output_periods,
                 node_features,
                 n_nodes, 
                 hidden_warmup,
                 edge_index_list,
                 model_path,
                ):
        super(MP25PPSmall, self).__init__()
        
        self.n_node = n_nodes[1]
        
        # Inflow prediction GNN
        self.model_warmup_inflow = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = node_features[3],
                                                        num_of_vertices = n_nodes[3],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[3],
                                                        edge_index=edge_index_list[3],
                                                    )  
        ## Load state dicts
        self.model_warmup_inflow.load_state_dict(torch.load("../models/" + model_path[3] + ".pth"))
                    
        # DP outflow prediction GNN
        self.model_warmup_outflow = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = node_features[4],
                                                        num_of_vertices = n_nodes[4],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[4],
                                                        edge_index=edge_index_list[4],
                                                    )  
        ## Load state dicts
        self.model_warmup_outflow.load_state_dict(torch.load("../models/" + model_path[4] + ".pth"))

        # DP inflow prediction GNN
        self.model_gnn = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = 2, # Combine two graph features to become 2 input features
                                                        num_of_vertices = n_nodes[4],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[4],
                                                        edge_index=edge_index_list[4],
                                                    )  
        
        self.linear = torch.nn.Linear(n_output_periods, n_output_periods)

    def forward(self, x):
        o_inflow = self.model_warmup_inflow(x[0])
#         o_inflow_reshaped = o_inflow.reshape(o_inflow.shape[0], self.n_node, 1, -1)
        o_inflow_unsqueezed = torch.unsqueeze(o_inflow, 2)
        
        o_outflow = self.model_warmup_outflow(x[1])
#         o_outflow_reshaped = o_outflow.reshape(o_outflow.shape[0], self.n_node, 1, -1)
        o_outflow_unsqueezed = torch.unsqueeze(o_outflow, 2)
        
        output_final = torch.cat((o_inflow_unsqueezed,
                                  o_outflow_unsqueezed), 
                                 dim=2)
        output_final = self.model_gnn(output_final)
        return  self.linear(F.relu(output_final))
        
    
#  Input: Speed + Inflow + Outflow, Prediction: inflow. Speed/Inflow/Outflow prediction GNNs have warmup. 
class MultiPredictionNet26(torch.nn.Module): 
    def __init__(self, 
                 n_input_periods,
                 n_output_periods,
                 node_features,
                 n_nodes, 
                 hidden_warmup,
                 edge_index_list,
                 model_path,
                ):
        super(MultiPredictionNet26, self).__init__()
        # Speed prediction GNN
        self.model_warmup_speed = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = node_features[0],
                                                        num_of_vertices = n_nodes[0],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[0],
                                                        edge_index=edge_index_list[0],
                                                    )  
        ## Load state dicts
        self.model_warmup_speed.load_state_dict(torch.load("../models/" + model_path[0] + ".pth"))
        
        # Inflow prediction GNN
        self.model_warmup_inflow = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = node_features[1],
                                                        num_of_vertices = n_nodes[1],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[1],
                                                        edge_index=edge_index_list[1],
                                                    )  
        ## Load state dicts
        self.model_warmup_inflow.load_state_dict(torch.load("../models/" + model_path[1] + ".pth"))
        
        # DP outflow prediction GNN
        self.model_dp_trained_outflow = MultiPredictionNet6(n_input_periods=n_input_periods,
                                             n_output_periods=n_output_periods,
                                             node_features=node_features,
                                             n_nodes=n_nodes, 
                                             hidden_warmup=hidden_warmup,
                                             edge_index_list=edge_index_list,
                                             model_path=model_path,
                                           )
        ## Load state dicts
        self.model_dp_trained_outflow.load_state_dict(torch.load("../models/" + model_path[3] + ".pth"))
        
        ## Freeze parameters of model_dp_trained_outflow
        for param in self.model_dp_trained_outflow.parameters():
            param.requires_grad = False
            
        self.linear = torch.nn.Linear(n_output_periods * sum(n_nodes), 
                                                n_output_periods * (n_nodes[1])) 
        

    def forward(self, x):
        o_speed = self.model_warmup_speed(x[0])
        o_speed_flattened = o_speed.view(o_speed.shape[0], 1, 1, -1)
        
        o_inflow = self.model_warmup_inflow(x[1])
        o_inflow_flattened = o_inflow.view(o_inflow.shape[0], 1, 1, -1)
        
        o_outflow = self.model_dp_trained_outflow(x[0])
        o_outflow_flattened = o_outflow.view(o_outflow.shape[0], 1, 1, -1)
        
        input_fusion = torch.concat((o_speed_flattened, o_inflow_flattened, o_outflow_flattened), dim=-1)
        output = self.linear(F.relu(input_fusion))
        return output 
    

    

#  Input: Speed + Inflow + Outflow, Prediction: outflow. Speed/Inflow/Outflow prediction GNNs have warmup. 
class MultiPredictionNet26_o(torch.nn.Module): 
    def __init__(self, 
                 n_input_periods,
                 n_output_periods,
                 node_features,
                 n_nodes, 
                 hidden_warmup,
                 edge_index_list,
                 model_path,
                ):
        super(MultiPredictionNet26_o, self).__init__()
        # Speed prediction GNN
        self.model_warmup_speed = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = node_features[0],
                                                        num_of_vertices = n_nodes[0],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[0],
                                                        edge_index=edge_index_list[0],
                                                    )  
        ## Load state dicts
        self.model_warmup_speed.load_state_dict(torch.load("../models/" + model_path[0] + ".pth"))
        
        # Inflow prediction GNN
        self.model_warmup_outflow = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = node_features[2],
                                                        num_of_vertices = n_nodes[2],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[2],
                                                        edge_index=edge_index_list[2],
                                                    )  
        ## Load state dicts
        self.model_warmup_outflow.load_state_dict(torch.load("../models/" + model_path[2] + ".pth"))
        
        # DP inflow prediction GNN
        self.model_dp_trained_inflow = MultiPredictionNet6(n_input_periods=n_input_periods,
                                             n_output_periods=n_output_periods,
                                             node_features=node_features,
                                             n_nodes=n_nodes, 
                                             hidden_warmup=hidden_warmup,
                                             edge_index_list=edge_index_list,
                                             model_path=model_path,
                                           )
        ## Load state dicts
        self.model_dp_trained_inflow.load_state_dict(torch.load("../models/" + model_path[3] + ".pth"))
        
        ## Freeze parameters of model_dp_trained_outflow
        for param in self.model_dp_trained_inflow.parameters():
            param.requires_grad = False
            
        self.linear = torch.nn.Linear(n_output_periods * sum(n_nodes), 
                                                n_output_periods * (n_nodes[2])) 
        

    def forward(self, x):
        o_speed = self.model_warmup_speed(x[0])
        o_speed_flattened = o_speed.view(o_speed.shape[0], 1, 1, -1)
        
        o_outflow = self.model_warmup_outflow(x[1])
        o_outflow_flattened = o_outflow.view(o_outflow.shape[0], 1, 1, -1)
        
        o_inflow = self.model_dp_trained_inflow(x[0])
        o_inflow_flattened = o_inflow.view(o_inflow.shape[0], 1, 1, -1)
        
        input_fusion = torch.concat((o_speed_flattened, o_outflow_flattened, o_inflow_flattened), dim=-1)
        output = self.linear(F.relu(input_fusion))
        return output 
    

    

#  Input: Inflow + Outflow, Prediction: inflow. Inflow/Outflow prediction GNNs have warmup. 
class MultiPredictionNet27(torch.nn.Module): 
    def __init__(self, 
                 n_input_periods,
                 n_output_periods,
                 node_features,
                 n_nodes, 
                 hidden_warmup,
                 edge_index_list,
                 model_path,
                ):
        super(MultiPredictionNet27, self).__init__()
        
        # Inflow prediction GNN
        self.model_warmup_inflow = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = node_features[1],
                                                        num_of_vertices = n_nodes[1],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[1],
                                                        edge_index=edge_index_list[1],
                                                    )  
        ## Load state dicts
        self.model_warmup_inflow.load_state_dict(torch.load("../models/" + model_path[1] + ".pth"))
        
        # DP outflow prediction GNN
        self.model_dp_trained_outflow = MultiPredictionNet6(n_input_periods=n_input_periods,
                                             n_output_periods=n_output_periods,
                                             node_features=node_features,
                                             n_nodes=n_nodes, 
                                             hidden_warmup=hidden_warmup,
                                             edge_index_list=edge_index_list,
                                             model_path=model_path,
                                           )
        ## Load state dicts
        self.model_dp_trained_outflow.load_state_dict(torch.load("../models/" + model_path[3] + ".pth"))
        
        ## Freeze parameters of model_dp_trained_outflow
        for param in self.model_dp_trained_outflow.parameters():
            param.requires_grad = False
            
        self.linear = torch.nn.Linear(n_output_periods * (n_nodes[1]+n_nodes[2]), 
                                                n_output_periods * (n_nodes[1])) 

    def forward(self, x):
        o_inflow = self.model_warmup_inflow(x[1])
        o_inflow_flattened = o_inflow.view(o_inflow.shape[0], 1, 1, -1)
        
        o_outflow = self.model_dp_trained_outflow(x[0])
        o_outflow_flattened = o_outflow.view(o_outflow.shape[0], 1, 1, -1)
        
        input_fusion = torch.concat((o_inflow_flattened, o_outflow_flattened), dim=-1)
        output = self.linear(F.relu(input_fusion))
        return output 
    
    

# Two-company scenario, Input: Inflow of one company + Outflow of one company + Speed, Prediction: inflow. Inflow/Outflow prediction GNNs have DP training. 
class MultiPredictionNet31_36(torch.nn.Module): 
    def __init__(self, 
                 n_input_periods,
                 n_output_periods,
                 node_features,
                 n_nodes, 
                 hidden_warmup,
                 edge_index_list,
                 model_path,
                ):
        super(MultiPredictionNet31_36, self).__init__()
        
        self.n_node = n_nodes[1]
        
        # DP inflow prediction GNN
        self.model_dp_trained_inflow = MultiPredictionNet5(n_input_periods=n_input_periods,
                                             n_output_periods=n_output_periods,
                                             node_features=node_features,
                                             n_nodes=n_nodes, 
                                             hidden_warmup=hidden_warmup,
                                             edge_index_list=edge_index_list,
                                             model_path=model_path,
                                           )
        ## Load state dicts
        self.model_dp_trained_inflow.load_state_dict(torch.load("../models/" + model_path[3] + ".pth"))
        
        ## Freeze parameters of model_dp_trained_outflow
        for param in self.model_dp_trained_inflow.parameters():
            param.requires_grad = False
            
        # DP outflow prediction GNN
        self.model_dp_trained_outflow = MultiPredictionNet6(n_input_periods=n_input_periods,
                                             n_output_periods=n_output_periods,
                                             node_features=node_features,
                                             n_nodes=n_nodes, 
                                             hidden_warmup=hidden_warmup,
                                             edge_index_list=edge_index_list,
                                             model_path=model_path,
                                           )
        ## Load state dicts
        self.model_dp_trained_outflow.load_state_dict(torch.load("../models/" + model_path[4] + ".pth"))
        
        ## Freeze parameters of model_dp_trained_outflow
        for param in self.model_dp_trained_outflow.parameters():
            param.requires_grad = False
            
        # Inflow prediction GNN
        self.model_inflow = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = 1, # inflow
                                                        num_of_vertices = n_nodes[1],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[1],
                                                        edge_index=edge_index_list[1],
                                                    )  
        ## Load state dicts
        self.model_inflow.load_state_dict(torch.load("../models/" + model_path[1] + ".pth"))
        
        ## Freeze parameters of model_dp_trained_outflow
        for param in self.model_inflow.parameters():
            param.requires_grad = False
            
            
        # Outflow prediction GNN
        self.model_outflow = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = 1, # outflow
                                                        num_of_vertices = n_nodes[2],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[2],
                                                        edge_index=edge_index_list[2],
                                                    )  
        ## Load state dicts
        self.model_outflow.load_state_dict(torch.load("../models/" + model_path[2] + ".pth"))
        
        ## Freeze parameters of model_dp_trained_outflow
        for param in self.model_outflow.parameters():
            param.requires_grad = False
        
        
    def forward(self, x):
        o_inflow_other_comapny = self.model_dp_trained_inflow(x[0])
        o_inflow_other_comapny_reshaped = o_inflow_other_comapny.reshape(o_inflow_other_comapny.shape[0], self.n_node, 1, -1)
    
        o_outflow_other_comapny = self.model_dp_trained_outflow(x[0])
        o_outflow_other_comapny_reshaped = o_outflow_other_comapny.reshape(o_outflow_other_comapny.shape[0], self.n_node, 1, -1)
        
        o_inflow_comapny = self.model_inflow(x[1])
        o_inflow_comapny_reshaped = o_inflow_comapny.reshape(o_inflow_comapny.shape[0], self.n_node, 1, -1)
    
        o_outflow_comapny = self.model_outflow(x[2])
        o_outflow_comapny_reshaped = o_outflow_comapny.reshape(o_outflow_comapny.shape[0], self.n_node, 1, -1)
        
        x_final = torch.cat((o_inflow_other_comapny_reshaped,
                               o_outflow_other_comapny_reshaped,
                               o_inflow_comapny_reshaped,
                               o_outflow_comapny_reshaped), dim=2)
        return x_final 
    

# Two-company scenario, Input: Inflow of one company + Outflow of one company + Speed, Prediction: inflow. Inflow/Outflow prediction GNNs of the other company have DP training. 
class MultiPredictionNet31(torch.nn.Module): 
    def __init__(self, 
                 n_input_periods,
                 n_output_periods,
                 node_features,
                 n_nodes, 
                 hidden_warmup,
                 edge_index_list,
                 model_path,
                ):
        super(MultiPredictionNet31, self).__init__()
        
        # DP inflow prediction GNN
        self.model_31_36 = MultiPredictionNet31_36(n_input_periods=n_input_periods,
                                             n_output_periods=n_output_periods,
                                             node_features=node_features,
                                             n_nodes=n_nodes, 
                                             hidden_warmup=hidden_warmup,
                                             edge_index_list=edge_index_list,
                                             model_path=model_path,
                                           )
            
        
        self.linear = torch.nn.Linear(n_output_periods * 4 * (n_nodes[1]), 
                                                n_output_periods * (n_nodes[1])) 
        
        
        
    def forward(self, x):
        x_inflow = self.model_31_36(x)
        x_inflow_flattened = x_inflow.view(x_inflow.shape[0], 1, 1, -1)
        x_inflow_flattened = self.linear(F.relu(x_inflow_flattened))
        return x_inflow_flattened 
    


# Two-company scenario, Input: Inflow of one company + Outflow of one company + Speed, Prediction: inflow. Inflow/Outflow prediction GNNs  of the other company have DP training. 
class MultiPredictionNet32(torch.nn.Module): 
    def __init__(self, 
                 n_input_periods,
                 n_output_periods,
                 node_features,
                 n_nodes, 
                 hidden_warmup,
                 edge_index_list,
                 model_path,
                ):
        super(MultiPredictionNet32, self).__init__()
        
        # DP inflow prediction GNN
        self.model_31_36 = MultiPredictionNet31_36(n_input_periods=n_input_periods,
                                             n_output_periods=n_output_periods,
                                             node_features=node_features,
                                             n_nodes=n_nodes, 
                                             hidden_warmup=hidden_warmup,
                                             edge_index_list=edge_index_list,
                                             model_path=model_path,
                                           )
        
        
        self.linear = torch.nn.Linear(n_output_periods * 4 * (n_nodes[2]), 
                                                n_output_periods * (n_nodes[2])) 
        
    def forward(self, x):
        x_outflow = self.model_31_36(x)
        x_outflow_flattened = x_outflow.view(x_outflow.shape[0], 1, 1, -1)
        x_outflow_flattened = self.linear(F.relu(x_outflow_flattened))
        return x_outflow_flattened 
    


# Two-company scenario, Input: Inflow of one company + Outflow of one company + Speed, Prediction: inflow. Inflow/Outflow prediction GNNs of the other company have DP training. 
class MultiPredictionNet33(torch.nn.Module): 
    def __init__(self, 
                 n_input_periods,
                 n_output_periods,
                 node_features,
                 n_nodes, 
                 hidden_warmup,
                 edge_index_list,
                 model_path,
                ):
        super(MultiPredictionNet33, self).__init__()
        
        # DP inflow prediction GNN
        self.model_31_36 = MultiPredictionNet31_36(n_input_periods=n_input_periods,
                                             n_output_periods=n_output_periods,
                                             node_features=node_features,
                                             n_nodes=n_nodes, 
                                             hidden_warmup=hidden_warmup,
                                             edge_index_list=edge_index_list,
                                             model_path=model_path,
                                           )
            
        # Final inflow prediction GNN
        self.model_inflow_final = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = 4, 
                                                        num_of_vertices = n_nodes[1],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = 0,
                                                        edge_index=edge_index_list[1],
                                                    )  
        
        
        
    def forward(self, x):
        x_inflow = self.model_31_36(x)
        x_inflow = self.model_inflow_final(F.relu(x_inflow))
        return x_inflow 
    


# Two-company scenario, Input: Inflow of one company + Outflow of one company + Speed, Prediction: inflow. Inflow/Outflow prediction GNNs  of the other company have DP training. 
class MultiPredictionNet34(torch.nn.Module): 
    def __init__(self, 
                 n_input_periods,
                 n_output_periods,
                 node_features,
                 n_nodes, 
                 hidden_warmup,
                 edge_index_list,
                 model_path,
                ):
        super(MultiPredictionNet34, self).__init__()
        
        # DP inflow prediction GNN
        self.model_31_36 = MultiPredictionNet31_36(n_input_periods=n_input_periods,
                                             n_output_periods=n_output_periods,
                                             node_features=node_features,
                                             n_nodes=n_nodes, 
                                             hidden_warmup=hidden_warmup,
                                             edge_index_list=edge_index_list,
                                             model_path=model_path,
                                           )
            
        # Final inflow prediction GNN
        self.model_outflow_final = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = 4, 
                                                        num_of_vertices = n_nodes[2],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = 0,
                                                        edge_index=edge_index_list[2],
                                                    )  
        
        
        
    def forward(self, x):
        x_outflow = self.model_31_36(x)
        x_outflow = self.model_outflow_final(x_inflow)
        return x_outflow 
    
    

# Two-company scenario, Input: Inflow of one company + Outflow of one company + Speed, Prediction: inflow. Inflow/Outflow prediction GNNs of the other company have DP training. 
class MultiPredictionNet35(torch.nn.Module): 
    def __init__(self, 
                 n_input_periods,
                 n_output_periods,
                 node_features,
                 n_nodes, 
                 hidden_warmup,
                 edge_index_list,
                 model_path,
                ):
        super(MultiPredictionNet35, self).__init__()
        
        # DP inflow prediction GNN
        self.model_31_36 = MultiPredictionNet31_36(n_input_periods=n_input_periods,
                                             n_output_periods=n_output_periods,
                                             node_features=node_features,
                                             n_nodes=n_nodes, 
                                             hidden_warmup=hidden_warmup,
                                             edge_index_list=edge_index_list,
                                             model_path=model_path,
                                           )
            
        # Final inflow prediction GNN
        self.model_inflow_final = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = 4, 
                                                        num_of_vertices = n_nodes[1],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = 0,
                                                        edge_index=edge_index_list[1],
                                                    )  
        
        self.linear = torch.nn.Linear(n_output_periods * (n_nodes[1]), 
                                                n_output_periods * (n_nodes[1])) 
        
        
    def forward(self, x):
        x_inflow = self.model_31_36(x)
        x_inflow = self.model_inflow_final(F.relu(x_inflow))
        x_inflow_flattened = x_inflow.view(x_inflow.shape[0], 1, 1, -1)
        x_inflow_flattened = self.linear(F.relu(x_inflow_flattened))
        return x_inflow_flattened 
    


# Two-company scenario, Input: Inflow of one company + Outflow of one company + Speed, Prediction: inflow. Inflow/Outflow prediction GNNs  of the other company have DP training. 
class MultiPredictionNet36(torch.nn.Module): 
    def __init__(self, 
                 n_input_periods,
                 n_output_periods,
                 node_features,
                 n_nodes, 
                 hidden_warmup,
                 edge_index_list,
                 model_path,
                ):
        super(MultiPredictionNet36, self).__init__()
        
        # DP inflow prediction GNN
        self.model_31_36 = MultiPredictionNet31_36(n_input_periods=n_input_periods,
                                             n_output_periods=n_output_periods,
                                             node_features=node_features,
                                             n_nodes=n_nodes, 
                                             hidden_warmup=hidden_warmup,
                                             edge_index_list=edge_index_list,
                                             model_path=model_path,
                                           )
            
        # Final inflow prediction GNN
        self.model_outflow_final = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = 4, 
                                                        num_of_vertices = n_nodes[2],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = 0,
                                                        edge_index=edge_index_list[2],
                                                    )  
        
        self.linear = torch.nn.Linear(n_output_periods * (n_nodes[2]), 
                                                n_output_periods * (n_nodes[2])) 
        
        
    def forward(self, x):
        x_outflow = self.model_31_36(x)
        x_outflow = self.model_outflow_final(F.relu(x_outflow))
        x_outflow_flattened = x_outflow.view(x_outflow.shape[0], 1, 1, -1)
        x_outflow_flattened = self.linear(F.relu(x_outflow_flattened))
        return x_outflow_flattened 
    
    
    
# Two-company scenario, Input: Inflow of one company + Outflow of one company + Speed, Prediction: inflow. Inflow/Outflow prediction GNNs have DP training. 
class MultiPredictionNet37(torch.nn.Module): 
    def __init__(self, 
                 n_input_periods,
                 n_output_periods,
                 node_features,
                 n_nodes, 
                 hidden_warmup,
                 edge_index_list,
                 model_path,
                ):
        super(MultiPredictionNet37, self).__init__()
        
        self.n_node = n_nodes[1]
        
        # DP inflow prediction GNN
        self.model_dp_trained_inflow = MultiPredictionNet5(n_input_periods=n_input_periods,
                                             n_output_periods=n_output_periods,
                                             node_features=node_features,
                                             n_nodes=n_nodes, 
                                             hidden_warmup=hidden_warmup,
                                             edge_index_list=edge_index_list,
                                             model_path=model_path,
                                           )
        ## Load state dicts
        self.model_dp_trained_inflow.load_state_dict(torch.load("../models/" + model_path[3] + ".pth"))
        
        ## Freeze parameters of model_dp_trained_outflow
        for param in self.model_dp_trained_inflow.parameters():
            param.requires_grad = False
            
        # DP outflow prediction GNN
        self.model_dp_trained_outflow = MultiPredictionNet6(n_input_periods=n_input_periods,
                                             n_output_periods=n_output_periods,
                                             node_features=node_features,
                                             n_nodes=n_nodes, 
                                             hidden_warmup=hidden_warmup,
                                             edge_index_list=edge_index_list,
                                             model_path=model_path,
                                           )
        ## Load state dicts
        self.model_dp_trained_outflow.load_state_dict(torch.load("../models/" + model_path[4] + ".pth"))
        
        ## Freeze parameters of model_dp_trained_outflow
        for param in self.model_dp_trained_outflow.parameters():
            param.requires_grad = False
            
        # Inflow prediction GNN
        self.model_inflow = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = 4, # two inflows and two outflows
                                                        num_of_vertices = n_nodes[1],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[1],
                                                        edge_index=edge_index_list[1],
                                                    )  

    def forward(self, x):
        o_inflow_other_comapny = self.model_dp_trained_inflow(x[0])
        o_inflow_other_comapny_reshaped = o_inflow_other_comapny.reshape(o_inflow_other_comapny.shape[0], self.n_node, 1, -1)
    
        o_outflow_other_comapny = self.model_dp_trained_outflow(x[0])
        o_outflow_other_comapny_reshaped = o_outflow_other_comapny.reshape(o_outflow_other_comapny.shape[0], self.n_node, 1, -1)
        
        x_final = torch.cat((o_inflow_other_comapny_reshaped,
                               o_outflow_other_comapny_reshaped,
                               x[1],
                               x[2]), dim=2)
        output = self.model_inflow(F.relu(x_final))
        return output 
    

# Two-company scenario, Input: Inflow of one company + Outflow of one company + Speed, Prediction: outflow. Inflow/Outflow prediction GNNs have DP training. 
class MultiPredictionNet38(torch.nn.Module): 
    def __init__(self, 
                 n_input_periods,
                 n_output_periods,
                 node_features,
                 n_nodes, 
                 hidden_warmup,
                 edge_index_list,
                 model_path,
                ):
        super(MultiPredictionNet38, self).__init__()
        
        self.n_node = n_nodes[2]
        
        # DP inflow prediction GNN
        self.model_dp_trained_inflow = MultiPredictionNet5(n_input_periods=n_input_periods,
                                             n_output_periods=n_output_periods,
                                             node_features=node_features,
                                             n_nodes=n_nodes, 
                                             hidden_warmup=hidden_warmup,
                                             edge_index_list=edge_index_list,
                                             model_path=model_path,
                                           )
        ## Load state dicts
        self.model_dp_trained_inflow.load_state_dict(torch.load("../models/" + model_path[3] + ".pth"))
        
        ## Freeze parameters of model_dp_trained_outflow
        for param in self.model_dp_trained_inflow.parameters():
            param.requires_grad = False
            
        # DP outflow prediction GNN
        self.model_dp_trained_outflow = MultiPredictionNet6(n_input_periods=n_input_periods,
                                             n_output_periods=n_output_periods,
                                             node_features=node_features,
                                             n_nodes=n_nodes, 
                                             hidden_warmup=hidden_warmup,
                                             edge_index_list=edge_index_list,
                                             model_path=model_path,
                                           )
        ## Load state dicts
        self.model_dp_trained_outflow.load_state_dict(torch.load("../models/" + model_path[4] + ".pth"))
        
        ## Freeze parameters of model_dp_trained_outflow
        for param in self.model_dp_trained_outflow.parameters():
            param.requires_grad = False
            
        # Inflow prediction GNN
        self.model_outflow = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = 4, # two inflows and two outflows
                                                        num_of_vertices = self.n_node,
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[1],
                                                        edge_index=edge_index_list[1],
                                                    )  
        

    def forward(self, x):
        o_inflow_other_comapny = self.model_dp_trained_inflow(x[0])
        o_inflow_other_comapny_reshaped = o_inflow_other_comapny.reshape(o_inflow_other_comapny.shape[0], self.n_node, 1, -1)
    
        o_outflow_other_comapny = self.model_dp_trained_outflow(x[0])
        o_outflow_other_comapny_reshaped = o_outflow_other_comapny.reshape(o_outflow_other_comapny.shape[0], self.n_node, 1, -1)
        
        x_final = torch.cat((o_inflow_other_comapny_reshaped,
                               o_outflow_other_comapny_reshaped,
                               x[1],
                               x[2]), dim=2)
        output = self.model_outflow(F.relu(x_final))
        return output 
    
    
# Two-company scenario, Input: Inflow of one company + Outflow of one company + Speed, Prediction: inflow. Inflow/Outflow prediction GNNs have DP training. 
class MultiPredictionNet39(torch.nn.Module): 
    def __init__(self, 
                 n_input_periods,
                 n_output_periods,
                 node_features,
                 n_nodes, 
                 hidden_warmup,
                 edge_index_list,
                 model_path,
                ):
        super(MultiPredictionNet39, self).__init__()
        
        # DP inflow prediction GNN
        self.model_37 = MultiPredictionNet37(n_input_periods=n_input_periods,
                                                             n_output_periods=n_output_periods,
                                                             node_features=node_features,
                                                             n_nodes=n_nodes, 
                                                             hidden_warmup=hidden_warmup,
                                                             edge_index_list=edge_index_list,
                                                             model_path=model_path,
                                                           )
            
        self.linear = torch.nn.Linear(n_output_periods * (n_nodes[1]), 
                                                n_output_periods * (n_nodes[1])) 

    def forward(self, x):
        o_37 = self.model_37(x)        
#         print("\no_37.shape: ", o_37.shape)
        o_37_flattened = o_37.view(o_37.shape[0], 1, 1, -1)
#         print("\no_37_flattened.shape: ", o_37_flattened.shape)
        output = self.linear(F.relu(o_37_flattened))
        return output 
    
# Two-company scenario, Input: Inflow of one company + Outflow of one company + Speed, Prediction: inflow. Inflow/Outflow prediction GNNs have DP training. 
class MultiPredictionNet40(torch.nn.Module): 
    def __init__(self, 
                 n_input_periods,
                 n_output_periods,
                 node_features,
                 n_nodes, 
                 hidden_warmup,
                 edge_index_list,
                 model_path,
                ):
        super(MultiPredictionNet40, self).__init__()
        
        # DP inflow prediction GNN
        self.model_38 = MultiPredictionNet38(n_input_periods=n_input_periods,
                                                             n_output_periods=n_output_periods,
                                                             node_features=node_features,
                                                             n_nodes=n_nodes, 
                                                             hidden_warmup=hidden_warmup,
                                                             edge_index_list=edge_index_list,
                                                             model_path=model_path,
                                                           )
            
        self.linear = torch.nn.Linear(n_output_periods * (n_nodes[2]), 
                                                n_output_periods * (n_nodes[2])) 

    def forward(self, x):
        o_38 = self.model_38(x)        
        o_38_flattened = o_38.view(o_38.shape[0], 1, 1, -1)
        output = self.linear(F.relu(o_38_flattened))
        return output 
    

    
    
# Two-company scenario, Input: Inflow of one company + Outflow of one company + Speed, Prediction: inflow. Inflow/Outflow prediction GNNs have DP training. 
class MultiPredictionNet47(torch.nn.Module): 
    def __init__(self, 
                 n_input_periods,
                 n_output_periods,
                 node_features,
                 n_nodes, 
                 hidden_warmup,
                 edge_index_list,
                 model_path,
                 no_backprop_nondp_models = False,
                 no_backprop_dp_models = False,
                ):
        super(MultiPredictionNet47, self).__init__()
        
        self.n_node = n_nodes[1]
        
        # DP inflow prediction GNN
        self.model_dp_trained_inflow = MultiPredictionNet5(n_input_periods=n_input_periods,
                                             n_output_periods=n_output_periods,
                                             node_features=node_features,
                                             n_nodes=n_nodes, 
                                             hidden_warmup=hidden_warmup,
                                             edge_index_list=edge_index_list,
                                             model_path=model_path,
                                           )
        ## Load state dicts
        self.model_dp_trained_inflow.load_state_dict(torch.load("../models/" + model_path[3] + ".pth"))
        
        ## Freeze parameters of model_dp_trained_outflow
        if no_backprop_dp_models:
            for param in self.model_dp_trained_inflow.parameters():
                param.requires_grad = False
            
        # DP outflow prediction GNN
        self.model_dp_trained_outflow = MultiPredictionNet6(n_input_periods=n_input_periods,
                                             n_output_periods=n_output_periods,
                                             node_features=node_features,
                                             n_nodes=n_nodes, 
                                             hidden_warmup=hidden_warmup,
                                             edge_index_list=edge_index_list,
                                             model_path=model_path,
                                           )
        ## Load state dicts
        self.model_dp_trained_outflow.load_state_dict(torch.load("../models/" + model_path[4] + ".pth"))
        
        ## Freeze parameters of model_dp_trained_outflow
        if no_backprop_dp_models:
            for param in self.model_dp_trained_outflow.parameters():
                param.requires_grad = False
            
        # Inflow prediction GNN
        self.model_inflow = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = 4, # two inflows and two outflows
                                                        num_of_vertices = n_nodes[1],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[1],
                                                        edge_index=edge_index_list[1],
                                                    )  
        
        # Model 17
        self.model_17 = MultiPredictionNet17(n_input_periods = n_input_periods,
                                             n_output_periods = n_output_periods,
                                             node_features = node_features,
                                             n_nodes = n_nodes, 
                                             hidden_warmup = hidden_warmup,
                                             edge_index_list = edge_index_list,
                                             model_path = model_path,
                                           )
        ## Load state dicts
        self.model_17.load_state_dict(torch.load("../models/" + model_path[5] + ".pth"))
        ## Remove the last layer (i.e., fcnn layer)
        # https://stackoverflow.com/questions/75988246/how-to-change-the-last-layer-of-pretrained-pytorch-model
        self.model_17_layers_without_fcnn = list(self.model_17.children()) 
        ## Freeze parameters of model_dp_trained_outflow
        if no_backprop_nondp_models:
            for layer in self.model_17_layers_without_fcnn[:-1]:
                for param in layer.parameters():
                    param.requires_grad = False
                    
        # Linear model
        self.linear = torch.nn.Linear(n_output_periods * (n_nodes[0] + 2*n_nodes[1] + 2*n_nodes[2]), 
                                                n_output_periods * (n_nodes[1])) 
        
    def forward(self, x):
        # S -> I DP model
        o_inflow_other_comapny = self.model_dp_trained_inflow(x[0])
        o_inflow_other_comapny_flattened = o_inflow_other_comapny.reshape(o_inflow_other_comapny.shape[0], 1, 1, -1)
        # S -> O DP model
        o_outflow_other_comapny = self.model_dp_trained_outflow(x[0])
        o_outflow_other_comapny_flattened = o_outflow_other_comapny.reshape(o_outflow_other_comapny.shape[0], 1, 1, -1)
        # S -> S layer of model 17
        o_speed = self.model_17_layers_without_fcnn[0](x[0])   
        o_speed_flattened = o_speed.view(o_speed.shape[0], 1, 1, -1)
        # I -> I layer of model 17
        o_inflow = self.model_17_layers_without_fcnn[1](x[1]) 
        o_inflow_flattened = o_inflow.view(o_inflow.shape[0], 1, 1, -1)        
        # O -> O layer of model 17
        o_outflow = self.model_17_layers_without_fcnn[2](x[2]) 
        o_outflow_flattened = o_outflow.view(o_outflow.shape[0], 1, 1, -1)
        
        output_cat = torch.concat((o_speed_flattened, 
                                    o_inflow_flattened, 
                                    o_outflow_flattened,
                                    o_inflow_other_comapny_flattened,
                                    o_outflow_other_comapny_flattened,
                                   ), dim=-1)
        
        output = self.linear(F.relu(output_cat))
        
        return output 
    
class MultiPredictionNet47_finetune(torch.nn.Module): 
    def __init__(self, 
                 n_input_periods,
                 n_output_periods,
                 node_features,
                 n_nodes, 
                 hidden_warmup,
                 edge_index_list,
                 model_path,
                 no_backprop_nondp_models = False,
                 no_backprop_dp_models = True,
                ):
        super(MultiPredictionNet47_finetune, self).__init__()
        
        self.model_47 = MultiPredictionNet47(n_input_periods = n_input_periods,
                                                     n_output_periods = n_output_periods,
                                                     node_features = node_features,
                                                     n_nodes = n_nodes, 
                                                     hidden_warmup = hidden_warmup,
                                                     edge_index_list = edge_index_list,
                                                     model_path = model_path,
                                                     no_backprop_nondp_models = no_backprop_nondp_models,
                                                     no_backprop_dp_models = no_backprop_dp_models,
                                                   )
        ## Load state dicts
        print(f"Load model \n{model_path[-1]}\n")
        self.model_47.load_state_dict(torch.load("../models/" + model_path[-1] + ".pth"))
        ## All parameters finetunable
        for layer in list(self.model_47.children()):
            print("THIS LAYER IS: \n", layer)
            for param in layer.parameters():
                param.requires_grad = True
                    
    def forward(self, x):
        output = self.model_47(x)
        return output 
    
    
    
    
class MultiPredictionNet55(torch.nn.Module): 
    def __init__(self, 
                 n_input_periods,
                 n_output_periods,
                 node_features,
                 n_nodes, 
                 hidden_warmup,
                 edge_index_list,
                 model_path,
                 no_backprop_nondp_models = False,
                 no_backprop_dp_models = False,
                ):
        super(MultiPredictionNet55, self).__init__()
        
        self.n_node = n_nodes[1]
        
#         # Speed prediction GNN
#         self.model_warmup_speed = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
#                                                         num_for_predict = n_output_periods,
#                                                         in_channels = node_features[0],
#                                                         num_of_vertices = n_nodes[0],
#                                                         nb_block = 2,
#                                                         K = 3,
#                                                         nb_chev_filter = 64,
#                                                         nb_time_filter = 64,
#                                         #                         time_strides = num_of_hours,
#                                                         time_strides = 1,
#                                                         hidden = hidden_warmup[0],
#                                                         edge_index=edge_index_list[0],
#                                                     )  
#         ## Load state dicts
#         self.model_warmup_speed.load_state_dict(torch.load("../models/" + model_path[0] + ".pth"))
        
#         # DP inflow prediction GNN
#         self.model_dp_trained_inflow = MultiPredictionNet5(n_input_periods=n_input_periods,
#                                              n_output_periods=n_output_periods,
#                                              node_features=node_features,
#                                              n_nodes=n_nodes, 
#                                              hidden_warmup=hidden_warmup,
#                                              edge_index_list=edge_index_list,
#                                              model_path=model_path,
#                                            )
#         ## Load state dicts
#         self.model_dp_trained_inflow.load_state_dict(torch.load("../models/" + model_path[3] + ".pth"))
        
#         ## Freeze parameters of model_dp_trained_outflow
#         if no_backprop_dp_models:
#             for param in self.model_dp_trained_inflow.parameters():
#                 param.requires_grad = False
            
        # DP outflow prediction GNN
        self.model_dp_trained_outflow = MultiPredictionNet6(n_input_periods=n_input_periods,
                                             n_output_periods=n_output_periods,
                                             node_features=node_features,
                                             n_nodes=n_nodes, 
                                             hidden_warmup=hidden_warmup,
                                             edge_index_list=edge_index_list,
                                             model_path=model_path[:3],
                                           )
        ## Load state dicts
        self.model_dp_trained_outflow.load_state_dict(torch.load("../models/" + model_path[4] + ".pth"))
        
        ## Freeze parameters of model_dp_trained_outflow
        if no_backprop_dp_models:
            for param in self.model_dp_trained_outflow.parameters():
                param.requires_grad = False
            
        # Model 7
        self.model_7 = MultiPredictionNet7(n_input_periods = n_input_periods,
                                             n_output_periods = n_output_periods,
                                             node_features = node_features,
                                             n_nodes = n_nodes, 
                                             hidden_warmup = hidden_warmup,
                                             edge_index_list = edge_index_list,
                                             model_path = model_path[:3],
                                           )
        ## Load state dicts
        self.model_7.load_state_dict(torch.load("../models/" + model_path[5] + ".pth"))
        ## Remove the last layer (i.e., fcnn layer)
        # https://stackoverflow.com/questions/75988246/how-to-change-the-last-layer-of-pretrained-pytorch-model
        self.model_7_layers_without_fcnn = list(self.model_7.children()) 
        ## Freeze parameters of model_dp_trained_outflow
        if no_backprop_nondp_models:
            for layer in self.model_7_layers_without_fcnn[:-1]:
                for param in layer.parameters():
                    param.requires_grad = False
                    
        # Linear model
        self.linear = torch.nn.Linear(n_output_periods * (n_nodes[0] + n_nodes[2] + n_nodes[1]), 
                                                n_output_periods * (n_nodes[1])) 
        
    def forward(self, x):
#         # S -> S model warmup
#         o_speed = self.model_warmup_speed(x[0])
#         o_speed_flattened = o_speed.view(o_speed.shape[0], 1, 1, -1)
#         # S -> I DP model
#         o_inflow_other_comapny = self.model_dp_trained_inflow(x[0])
#         o_inflow_other_comapny_flattened = o_inflow_other_comapny.reshape(o_inflow_other_comapny.shape[0], 1, 1, -1)
        # S -> O DP model
        o_outflow_other_comapny = self.model_dp_trained_outflow(x[0])
        o_outflow_other_comapny_flattened = o_outflow_other_comapny.reshape(o_outflow_other_comapny.shape[0], 1, 1, -1)
        # S -> S layer of model 7
        o_speed_model_7 = self.model_7_layers_without_fcnn[0](x[0])   
        o_speed_flattened_model_7 = o_speed.view(o_speed_model_7.shape[0], 1, 1, -1)
        # I -> I layer of model 7
        o_inflow_model_7 = self.model_7_layers_without_fcnn[1](x[1]) 
        o_inflow_flattened_model_7 = o_inflow.view(o_inflow_model_7.shape[0], 1, 1, -1)        
#         # O -> O layer of model 17
#         o_outflow = self.model_17_layers_without_fcnn[2](x[2]) 
#         o_outflow_flattened = o_outflow.view(o_outflow.shape[0], 1, 1, -1)
        
        output_cat = torch.concat((o_speed_flattened_model_7, 
#                                     o_inflow_flattened, 
                                   o_inflow_flattened_model_7,
#                                     o_inflow_other_comapny_flattened,
                                   o_outflow_other_comapny_flattened,
                                   ), 
                                  dim=-1)
        
        output = self.linear(F.relu(output_cat))
        
        return output 
    
    
    
class MultiPredictionNet56(torch.nn.Module): 
    def __init__(self, 
                 n_input_periods,
                 n_output_periods,
                 node_features,
                 n_nodes, 
                 hidden_warmup,
                 edge_index_list,
                 model_path,
                 no_backprop_nondp_models = False,
                 no_backprop_dp_models = False,
                ):
        super(MultiPredictionNet56, self).__init__()
        
        self.n_node = n_nodes[1]
        
#         # Speed prediction GNN
#         self.model_warmup_speed = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
#                                                         num_for_predict = n_output_periods,
#                                                         in_channels = node_features[0],
#                                                         num_of_vertices = n_nodes[0],
#                                                         nb_block = 2,
#                                                         K = 3,
#                                                         nb_chev_filter = 64,
#                                                         nb_time_filter = 64,
#                                         #                         time_strides = num_of_hours,
#                                                         time_strides = 1,
#                                                         hidden = hidden_warmup[0],
#                                                         edge_index=edge_index_list[0],
#                                                     )  
#         ## Load state dicts
#         self.model_warmup_speed.load_state_dict(torch.load("../models/" + model_path[0] + ".pth"))
        
        # DP inflow prediction GNN
        self.model_dp_trained_inflow = MultiPredictionNet5(n_input_periods=n_input_periods,
                                             n_output_periods=n_output_periods,
                                             node_features=node_features,
                                             n_nodes=n_nodes, 
                                             hidden_warmup=hidden_warmup,
                                             edge_index_list=edge_index_list,
                                             model_path=model_path,
                                           )
        ## Load state dicts
        self.model_dp_trained_inflow.load_state_dict(torch.load("../models/" + model_path[3] + ".pth"))
        
        ## Freeze parameters of model_dp_trained_outflow
        if no_backprop_dp_models:
            for param in self.model_dp_trained_inflow.parameters():
                param.requires_grad = False
            
#         # DP outflow prediction GNN
#         self.model_dp_trained_outflow = MultiPredictionNet6(n_input_periods=n_input_periods,
#                                              n_output_periods=n_output_periods,
#                                              node_features=node_features,
#                                              n_nodes=n_nodes, 
#                                              hidden_warmup=hidden_warmup,
#                                              edge_index_list=edge_index_list,
#                                              model_path=model_path,
#                                            )
#         ## Load state dicts
#         self.model_dp_trained_outflow.load_state_dict(torch.load("../models/" + model_path[4] + ".pth"))
        
#         ## Freeze parameters of model_dp_trained_outflow
#         if no_backprop_dp_models:
#             for param in self.model_dp_trained_outflow.parameters():
#                 param.requires_grad = False
            
        # Model 8
        self.model_8 = MultiPredictionNet8(n_input_periods = n_input_periods,
                                             n_output_periods = n_output_periods,
                                             node_features = node_features,
                                             n_nodes = n_nodes, 
                                             hidden_warmup = hidden_warmup,
                                             edge_index_list = edge_index_list,
                                             model_path = model_path,
                                           )
        ## Load state dicts
        self.model_8.load_state_dict(torch.load("../models/" + model_path[5] + ".pth"))
        ## Remove the last layer (i.e., fcnn layer)
        # https://stackoverflow.com/questions/75988246/how-to-change-the-last-layer-of-pretrained-pytorch-model
        self.model_8_layers_without_fcnn = list(self.model_8.children()) 
        ## Freeze parameters of model_dp_trained_outflow
        if no_backprop_nondp_models:
            for layer in self.model_8_layers_without_fcnn[:-1]:
                for param in layer.parameters():
                    param.requires_grad = False
                    
        # Linear model
        self.linear = torch.nn.Linear(n_output_periods * (n_nodes[0] + n_nodes[1] + n_nodes[2]), 
                                                n_output_periods * (n_nodes[2])) 
        
    def forward(self, x):
        # S -> I DP model
        o_inflow_other_comapny = self.model_dp_trained_inflow(x[0])
        o_inflow_other_comapny_flattened = o_inflow_other_comapny.reshape(o_inflow_other_comapny.shape[0], 1, 1, -1)
#         # S -> O DP model
#         o_outflow_other_comapny = self.model_dp_trained_outflow(x[0])
#         o_outflow_other_comapny_flattened = o_outflow_other_comapny.reshape(o_outflow_other_comapny.shape[0], 1, 1, -1)
        # S -> S layer of model 8
        o_speed_model_8 = self.model_8_layers_without_fcnn[0](x[0])   
        o_speed_flattened_model_8 = o_speed.view(o_speed_model_8.shape[0], 1, 1, -1)
#         # I -> I layer of model 17
#         o_inflow = self.model_17_layers_without_fcnn[1](x[1]) 
#         o_inflow_flattened = o_inflow.view(o_inflow.shape[0], 1, 1, -1)        
        # O -> O layer of model 7
        o_outflow_model_8 = self.model_8_layers_without_fcnn[1](x[1]) 
        o_outflow_flattened_model_8 = o_outflow.view(o_outflow_model_8.shape[0], 1, 1, -1)
        
        output_cat = torch.concat((o_speed_flattened_model_8, 
#                                     o_inflow_flattened, 
                                    o_outflow_flattened_model_8,
                                    o_inflow_other_comapny_flattened,
#                                     o_outflow_other_comapny_flattened,
                                   ), 
                                  dim=-1)
        
        output = self.linear(F.relu(output_cat))
        
        return output 
    
     
class DualCompany_1(torch.nn.Module): 
    def __init__(self, 
                 n_input_periods,
                 n_output_periods,
                 node_features,
                 n_nodes, 
                 hidden_warmup,
                 edge_index_list,
                 model_path,
                 backprop_loaded_model = True,
                ):
        super(DualCompany_1, self).__init__()
        
        self.n_node = n_nodes[1]
        assert n_nodes[1] == n_nodes[2], print("Inflow and outflow graph do not match")
        assert n_nodes[1] == n_nodes[3] or n_nodes[2] == n_nodes[4], print("Inflow or outflow graphs of companies do not match")
        # Prediction GNN of the other company
        self.model_other_company = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = node_features[3],
                                                        num_of_vertices = n_nodes[3],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[3],
                                                        edge_index=edge_index_list[3],
                                                    )  
        ## Load state dicts
        self.model_other_company.load_state_dict(torch.load("../models/" + model_path[3] + ".pth"))
        
        ## Freeze parameters of model_dp_trained_outflow
        if not backprop_loaded_model:
            for param in self.model_other_company.parameters():
                param.requires_grad = False
                
        # Prediction GNN of the company
        self.model_GNN_aggregate = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = 2, # two inflows and two outflows
                                                        num_of_vertices = n_nodes[1],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[1],
                                                        edge_index=edge_index_list[1],
                                                    )  

    def forward(self, x):
        # if model=dual_company_1_i, then x[1] is inflow of the other company
        # if model=dual_company_1_o, then x[1] is outflow of the other company
        o_other_comapny = self.model_other_company(x[1]) 
        o_other_comapny_reshaped = o_other_comapny.reshape(o_other_comapny.shape[0], self.n_node, 1, -1)
        
        # if model=dual_company_1_i, then x[0] is inflow of the company
        # if model=dual_company_1_o, then x[0] is outflow of the company
        o_concat = torch.cat((x[0],
                              o_other_comapny_reshaped,
                            ), 
                            dim=2)
        output = self.model_GNN_aggregate(F.relu(o_concat))
        return output 
    
    
    

class DualCompany_2(torch.nn.Module): 
    def __init__(self, 
                 n_input_periods,
                 n_output_periods,
                 node_features,
                 n_nodes, 
                 hidden_warmup,
                 edge_index_list,
                 model_path,
                 backprop_loaded_model_other_company = True,
                 backprop_loaded_model_company = True,
                 inflow_pred = True,
                ):
        super(DualCompany_2, self).__init__()
        
        self.n_node = n_nodes[1]
        assert n_nodes[1] == n_nodes[2], print("Inflow and outflow graph do not match")
        assert n_nodes[1] == n_nodes[3] or n_nodes[2] == n_nodes[4], print("Inflow or outflow graphs of companies do not match")
        # Prediction GNN of the other company
        self.model_other_company = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                                    num_for_predict = n_output_periods,
                                                                    in_channels = node_features[3],
                                                                    num_of_vertices = n_nodes[3],
                                                                    nb_block = 2,
                                                                    K = 3,
                                                                    nb_chev_filter = 64,
                                                                    nb_time_filter = 64,
                                                    #                         time_strides = num_of_hours,
                                                                    time_strides = 1,
                                                                    hidden = hidden_warmup[3],
                                                                    edge_index=edge_index_list[3],
                                                                 )  
        ## Load state dicts
        self.model_other_company.load_state_dict(torch.load("../models/" + model_path[3] + ".pth"))
        
        ## Freeze parameters of model_dp_trained_outflow
        if not backprop_loaded_model_other_company:
            for param in self.model_other_company.parameters():
                param.requires_grad = False
                
        # Prediction GNN of the company
        self.model_company = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                            num_for_predict = n_output_periods,
                                                            in_channels = node_features[1],
                                                            num_of_vertices = n_nodes[1],
                                                            nb_block = 2,
                                                            K = 3,
                                                            nb_chev_filter = 64,
                                                            nb_time_filter = 64,
                                            #                         time_strides = num_of_hours,
                                                            time_strides = 1,
                                                            hidden = hidden_warmup[1],
                                                            edge_index=edge_index_list[1],
                                                         ) 
        if inflow_pred:
            ## Load state dicts of inflow prediction
            self.model_company.load_state_dict(torch.load("../models/" + model_path[1] + ".pth"))
        else:
            ## Load state dicts of outflow prediction
            self.model_company.load_state_dict(torch.load("../models/" + model_path[2] + ".pth"))
            
        ## Freeze parameters of model_dp_trained_outflow
        if not backprop_loaded_model_company:
            for param in self.model_company.parameters():
                param.requires_grad = False
                
        # Linear model
        self.linear = torch.nn.Linear(n_output_periods * (n_nodes[1] + n_nodes[2]), 
                                                n_output_periods * (n_nodes[1])) 

    def forward(self, x):
        # if model=dual_company_2_i, then x[0] is inflow of the company
        # if model=dual_company_2_o, then x[0] is outflow of the company
        o_comapny = self.model_company(x[0]) 
        o_comapny_flattened = o_comapny.view(o_comapny.shape[0], 1, 1, -1)
        
        # if model=dual_company_2_i, then x[1] is inflow of the other company
        # if model=dual_company_2_o, then x[1] is outflow of the other company
        o_other_comapny = self.model_other_company(x[1]) 
        o_other_comapny_flattened = o_other_comapny.view(o_other_comapny.shape[0], 1, 1, -1)

        o_concat = torch.concat((o_comapny_flattened,
                                 o_other_comapny_flattened,
                                ), 
                                dim = -1)
        
        output = self.linear(F.relu(o_concat))
        return output 
    
    
    

class Surge_1(torch.nn.Module): 
    def __init__(self, 
                 n_input_periods,
                 n_output_periods,
                 node_features,
                 n_nodes, 
                 hidden_warmup,
                 edge_index_list,
                 model_path,
                 backprop_loaded_model_inflow,
                 backprop_loaded_model_outflow, 
                 surge_dp,
                ):
        super(Surge_1, self).__init__()
        
        self.n_node = n_nodes[1]
        assert n_nodes[1] == n_nodes[2], print("Inflow and outflow graph do not match")
        
        model_path_outflow = model_path.pop()
        model_path_inflow = model_path.pop()
        # Model inflow (17 for non-dp and 51 for dp)
        if surge_dp:
            # As we pass model_path to load models, we pop the model path of two models to remove them from model_path
            # model_path of models 51 and 52 includes models 5, 6, 17, and 18
            model_path_18 = model_path.pop()
            model_path_17 = model_path.pop()
            self.model_inflow = MultiPredictionNet47(n_input_periods=n_input_periods,
                                                     n_output_periods=n_output_periods,
                                                     node_features=node_features,
                                                     n_nodes=n_nodes, 
                                                     hidden_warmup=hidden_warmup,
                                                     edge_index_list=edge_index_list,
                                                     model_path=model_path + [model_path_17], # model path of speed, inflow, outflow, 5, 6, and 17
                                                     no_backprop_nondp_models = True,
                                                     no_backprop_dp_models = True
                                                   )
            self.model_outflow = MultiPredictionNet47(n_input_periods=n_input_periods,
                                                     n_output_periods=n_output_periods,
                                                     node_features=node_features,
                                                     n_nodes=n_nodes, 
                                                     hidden_warmup=hidden_warmup,
                                                     edge_index_list=edge_index_list,
                                                     model_path=model_path + [model_path_18],  # model path of speed, inflow, outflow, 5, 6, and 18
                                                     no_backprop_nondp_models = True,
                                                     no_backprop_dp_models = True
                                                   )
            
        else:
            self.model_inflow = MultiPredictionNet17(n_input_periods = n_input_periods,
                                                     n_output_periods = n_output_periods,
                                                     node_features = node_features,
                                                     n_nodes = n_nodes,
                                                     hidden_warmup = hidden_warmup,
                                                     edge_index_list = edge_index_list,
                                                     model_path = model_path,
                                                   )
            self.model_outflow = MultiPredictionNet18(n_input_periods = n_input_periods,
                                                     n_output_periods = n_output_periods,
                                                     node_features = node_features,
                                                     n_nodes = n_nodes,
                                                     hidden_warmup = hidden_warmup,
                                                     edge_index_list = edge_index_list,
                                                     model_path = model_path,
                                                   )
        ## Load state dicts
        self.model_inflow.load_state_dict(torch.load("../models/" + model_path_inflow + ".pth"))
        self.model_outflow.load_state_dict(torch.load("../models/" + model_path_outflow + ".pth"))
        ## Remove the last layer (i.e., fcnn layer)
        # https://stackoverflow.com/questions/75988246/how-to-change-the-last-layer-of-pretrained-pytorch-model
        self.model_inflow_layers_without_fcnn = list(self.model_inflow.children())[:-1]
        self.model_outflow_layers_without_fcnn = list(self.model_outflow.children())[:-1]
        ## Freeze parameters 
        if backprop_loaded_model_inflow:
            for layer in self.model_inflow_layers_without_fcnn:
                for param in layer.parameters():
                    param.requires_grad = False
        if backprop_loaded_model_outflow:
            for layer in self.model_outflow_layers_without_fcnn:
                for param in layer.parameters():
                    param.requires_grad = False
                    
        # Linear model
        self.linear = torch.nn.Linear(2 * n_output_periods * (n_nodes[0] + n_nodes[1] + n_nodes[2]), 
                                                n_output_periods * (n_nodes[1])) 

    def forward(self, x):
        # S -> S layer of model inflow
        o_speed_inflow = self.model_inflow_layers_without_fcnn[0](x[0])   
        o_speed_inflow_flattened = o_speed_inflow.view(o_speed_inflow.shape[0], 1, 1, -1)
        print(f"o_speed_inflow_flattened.shape: ", o_speed_inflow_flattened.shape)
        # I -> I layer of model inflow
        o_inflow_inflow = self.model_inflow_layers_without_fcnn[1](x[1]) 
        o_inflow_inflow_flattened = o_inflow_inflow.view(o_inflow_inflow.shape[0], 1, 1, -1)   
        print(f"o_inflow_inflow_flattened.shape: ", o_inflow_inflow_flattened.shape)     
        # O -> O layer of model inflow
        o_outflow_inflow = self.model_inflow_layers_without_fcnn[2](x[2]) 
        o_outflow_inflow_flattened = o_outflow_inflow.view(o_outflow_inflow.shape[0], 1, 1, -1)
        print(f"o_outflow_inflow_flattened.shape: ", o_outflow_inflow_flattened.shape)
        
        # S -> S layer of model outflow
        o_speed_outflow = self.model_outflow_layers_without_fcnn[0](x[0])   
        o_speed_outflow_flattened = o_speed_outflow.view(o_speed_outflow.shape[0], 1, 1, -1)
        print(f"o_speed_outflow_flattened.shape: ", o_speed_outflow_flattened.shape)
        # I -> I layer of model outflow
        o_inflow_outflow = self.model_outflow_layers_without_fcnn[1](x[1]) 
        o_inflow_outflow_flattened = o_inflow_outflow.view(o_inflow_outflow.shape[0], 1, 1, -1)   
        print(f"o_inflow_outflow_flattened.shape: ", o_inflow_outflow_flattened.shape)     
        # O -> O layer of model outflow
        o_outflow_outflow = self.model_outflow_layers_without_fcnn[2](x[2]) 
        o_outflow_outflow_flattened = o_outflow_outflow.view(o_outflow_outflow.shape[0], 1, 1, -1)
        print(f"o_outflow_outflow_flattened.shape: ", o_outflow_outflow_flattened.shape)
        
        output_cat = torch.concat((o_speed_inflow_flattened, 
                                    o_inflow_inflow_flattened, 
                                    o_outflow_inflow_flattened,
                                    o_speed_outflow_flattened, 
                                    o_inflow_outflow_flattened, 
                                    o_outflow_outflow_flattened,
                                   ), dim=-1)
        print(f"output_cat.shape: ", output_cat.shape)
        
        # Pass the linear layer with ReLy activation
        output = self.linear(F.relu(output_cat))
        print(f"output.shape: ", output.shape)
        
        return output 
    
    

class Surge_2(torch.nn.Module): 
    def __init__(self, 
                 n_input_periods,
                 n_output_periods,
                 node_features,
                 n_nodes, 
                 hidden_warmup,
                 edge_index_list,
                 model_path,
                 backprop_loaded_model_inflow,
                 backprop_loaded_model_outflow, 
                 surge_dp,
                ):
        super(Surge_2, self).__init__()
        
        self.n_node = n_nodes[1]
        assert n_nodes[1] == n_nodes[2], print("Inflow and outflow graph do not match")
        
        model_path_outflow = model_path.pop()
        model_path_inflow = model_path.pop()
        # Model inflow (17 for non-dp and 51 for dp)
        if surge_dp:
            # As we pass model_path to load models, we pop the model path of two models to remove them from model_path
            # model_path of models 51 and 52 includes models 5, 6, 17, and 18
            model_path_18 = model_path.pop()
            model_path_17 = model_path.pop()
            self.model_inflow = MultiPredictionNet47(n_input_periods=n_input_periods,
                                                     n_output_periods=n_output_periods,
                                                     node_features=node_features,
                                                     n_nodes=n_nodes, 
                                                     hidden_warmup=hidden_warmup,
                                                     edge_index_list=edge_index_list,
                                                     model_path=model_path + [model_path_17], # model path of speed, inflow, outflow, 5, 6, and 17
                                                     no_backprop_nondp_models = True,
                                                     no_backprop_dp_models = True
                                                   )
            self.model_outflow = MultiPredictionNet47(n_input_periods=n_input_periods,
                                                     n_output_periods=n_output_periods,
                                                     node_features=node_features,
                                                     n_nodes=n_nodes, 
                                                     hidden_warmup=hidden_warmup,
                                                     edge_index_list=edge_index_list,
                                                     model_path=model_path + [model_path_18],  # model path of speed, inflow, outflow, 5, 6, and 18
                                                     no_backprop_nondp_models = True,
                                                     no_backprop_dp_models = True
                                                   )
            
            model_path.append(model_path_17)
            model_path.append(model_path_18)
            model_path.append(model_path_inflow)
            model_path.append(model_path_outflow)
            
        else:
            self.model_inflow = MultiPredictionNet17(n_input_periods = n_input_periods,
                                                     n_output_periods = n_output_periods,
                                                     node_features = node_features,
                                                     n_nodes = n_nodes,
                                                     hidden_warmup = hidden_warmup,
                                                     edge_index_list = edge_index_list,
                                                     model_path = model_path,
                                                   )
            self.model_outflow = MultiPredictionNet18(n_input_periods = n_input_periods,
                                                     n_output_periods = n_output_periods,
                                                     node_features = node_features,
                                                     n_nodes = n_nodes,
                                                     hidden_warmup = hidden_warmup,
                                                     edge_index_list = edge_index_list,
                                                     model_path = model_path,
                                                   )
        ## Load state dicts
        self.model_inflow.load_state_dict(torch.load("../models/" + model_path_inflow + ".pth"))
        self.model_outflow.load_state_dict(torch.load("../models/" + model_path_outflow + ".pth"))
        ## Freeze parameters
        if backprop_loaded_model_inflow:
            for layer in list(self.model_inflow.children()):
                for param in layer.parameters():
                    param.requires_grad = False
        if backprop_loaded_model_outflow:
            for layer in list(self.model_inflow.children()):
                for param in layer.parameters():
                    param.requires_grad = False
                    
        # Linear model
        self.linear = torch.nn.Linear(n_output_periods * (n_nodes[1] + n_nodes[2]), 
                                                n_output_periods * (n_nodes[1])) 

    def forward(self, x):
        # S, I, O -> I
        o_inflow = self.model_inflow(x)
#         print(f"o_inflow shape: ", o_inflow.shape)
        # S, I, O -> O
        o_outflow = self.model_outflow(x)
#         print(f"o_outflow shape: ", o_outflow.shape)
        # Concatenate output of two models
        output_cat = torch.concat((o_inflow, 
                                    o_outflow, 
                                   ), dim=-1)
#         print(f"output_cat shape: ", output_cat.shape)
        # Pass the linear layer with ReLy activation
        output = self.linear(F.relu(output_cat))
#         print(f"output shape: ", output.shape)
        
        return output 
    
    
    
    
#  Input: Inflow + Outflow, Prediction: inflow. Inflow/Outflow prediction GNNs have warmup. 
class M60A(torch.nn.Module): 
    def __init__(self, 
                 n_input_periods,
                 n_output_periods,
                 node_features,
                 n_nodes, 
                 hidden_warmup,
                 edge_index_list,
                 model_path,
                ):
        super(M60A, self).__init__()
        
        # Inflow prediction GNN
        self.model_warmup_inflow = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = node_features[1],
                                                        num_of_vertices = n_nodes[1],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[1],
                                                        edge_index=edge_index_list[1],
                                                    )  
        ## Load state dicts
        self.model_warmup_inflow.load_state_dict(torch.load("../models/" + model_path[1] + ".pth"))
        
        # DP outflow prediction GNN, GY
        self.model_warmup_outflow = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = node_features[2],
                                                        num_of_vertices = n_nodes[2],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[2],
                                                        edge_index=edge_index_list[2],
                                                    )  
        ## Load state dicts
        self.model_warmup_outflow.load_state_dict(torch.load("../models/" + model_path[2] + ".pth"))
        
        # Inflow prediction GNN, GY
        self.model_warmup_inflow_gy = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = node_features[3],
                                                        num_of_vertices = n_nodes[3],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[3],
                                                        edge_index=edge_index_list[3],
                                                    )  
        ## Load state dicts
        self.model_warmup_inflow_gy.load_state_dict(torch.load("../models/" + model_path[3] + ".pth"))
        
        # DP outflow prediction GNN, GY
        self.model_warmup_outflow_gy = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = node_features[4],
                                                        num_of_vertices = n_nodes[4],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[4],
                                                        edge_index=edge_index_list[4],
                                                    )  
        ## Load state dicts
        self.model_warmup_outflow_gy.load_state_dict(torch.load("../models/" + model_path[4] + ".pth"))

        assert n_nodes[1] == n_nodes[2], print('Different inflow and outflow graphs of company')
        assert n_nodes[3] == n_nodes[4], print('Different inflow and outflow graphs of gy')
        assert n_nodes[1] == n_nodes[3], print('Different graphs of company and gy')
        self.linear = torch.nn.Linear(n_output_periods * (n_nodes[1]+n_nodes[2]+n_nodes[3]+n_nodes[4]), 
                                                n_output_periods * (n_nodes[1])) 

    def forward(self, x):
        o_inflow = self.model_warmup_inflow(x[0])
        o_inflow_flattened = o_inflow.view(o_inflow.shape[0], 1, 1, -1)
        
        o_outflow = self.model_warmup_outflow(x[1])
        o_outflow_flattened = o_outflow.view(o_outflow.shape[0], 1, 1, -1)
        
        o_inflow_gy = self.model_warmup_inflow_gy(x[2])
        o_inflow_gy_flattened = o_inflow_gy.view(o_inflow_gy.shape[0], 1, 1, -1)
        
        o_outflow_gy = self.model_warmup_outflow_gy(x[3])
        o_outflow_gy_flattened = o_outflow_gy.view(o_outflow_gy.shape[0], 1, 1, -1)
        
        input_fusion = torch.concat((o_inflow_flattened, 
                                     o_outflow_flattened,
                                     o_inflow_gy_flattened,
                                     o_outflow_gy_flattened,
                                    ), dim=-1)
        output = self.linear(F.relu(input_fusion))
        return output 
    
#  Input: Inflow + Outflow, Prediction: inflow. Inflow/Outflow prediction GNNs have warmup. 
class M60B(torch.nn.Module): 
    def __init__(self, 
                 n_input_periods,
                 n_output_periods,
                 node_features,
                 n_nodes, 
                 hidden_warmup,
                 edge_index_list,
                 model_path,
                ):
        super(M60B, self).__init__()
        
        assert n_nodes[1] == n_nodes[2], print('Different inflow and outflow graphs of company')
        assert n_nodes[3] == n_nodes[4], print('Different inflow and outflow graphs of gy')
        assert n_nodes[1] == n_nodes[3], print('Different graphs of company and gy')
        # GNN with 4 inputs
        self.model_gnn = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = 4,
                                                        num_of_vertices = n_nodes[1],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[1],
                                                        edge_index=edge_index_list[1],
                                                    )  

    def forward(self, x):
        input_inflows_outflows = torch.cat((x[0],
                                            x[1],
                                            x[2],
                                            x[3],
                                           ),
                                           dim=2)
        return self.model_gnn(input_inflows_outflows)
                                 

        

    
#  Input: Inflow + Outflow, Prediction: inflow. Inflow/Outflow prediction GNNs have warmup. 
class M60C(torch.nn.Module): 
    def __init__(self, 
                 n_input_periods,
                 n_output_periods,
                 node_features,
                 n_nodes, 
                 hidden_warmup,
                 edge_index_list,
                 model_path,
                ):
        super(M60C, self).__init__()
        
        # Inflow prediction GNN
        self.model_warmup_inflow = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = node_features[1],
                                                        num_of_vertices = n_nodes[1],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[1],
                                                        edge_index=edge_index_list[1],
                                                    )  
        ## Load state dicts
        self.model_warmup_inflow.load_state_dict(torch.load("../models/" + model_path[1] + ".pth"))
        
        # DP outflow prediction GNN, GY
        self.model_warmup_outflow = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = node_features[2],
                                                        num_of_vertices = n_nodes[2],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[2],
                                                        edge_index=edge_index_list[2],
                                                    )  
        ## Load state dicts
        self.model_warmup_outflow.load_state_dict(torch.load("../models/" + model_path[2] + ".pth"))
        
        # Inflow prediction GNN, GY
        self.model_warmup_inflow_gy = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = node_features[3],
                                                        num_of_vertices = n_nodes[3],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[3],
                                                        edge_index=edge_index_list[3],
                                                    )  
        ## Load state dicts
        self.model_warmup_inflow_gy.load_state_dict(torch.load("../models/" + model_path[3] + ".pth"))
        
        # DP outflow prediction GNN, GY
        self.model_warmup_outflow_gy = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = node_features[4],
                                                        num_of_vertices = n_nodes[4],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[4],
                                                        edge_index=edge_index_list[4],
                                                    )  
        ## Load state dicts
        self.model_warmup_outflow_gy.load_state_dict(torch.load("../models/" + model_path[4] + ".pth"))

        assert n_nodes[1] == n_nodes[2], print('Different inflow and outflow graphs of company')
        assert n_nodes[3] == n_nodes[4], print('Different inflow and outflow graphs of gy')
        assert n_nodes[1] == n_nodes[3], print('Different graphs of company and gy')
        # GNN with 4 inputs
        self.model_gnn = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = 4,
                                                        num_of_vertices = n_nodes[1],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[1],
                                                        edge_index=edge_index_list[1],
                                                    )  
        

    def forward(self, x):
        o_inflow = self.model_warmup_inflow(x[0])
        o_inflow_unsqueezed = torch.unsqueeze(o_inflow, 2)
        
        o_outflow = self.model_warmup_outflow(x[1])
        o_outflow_unsqueezed = torch.unsqueeze(o_outflow, 2)
        
        o_inflow_gy = self.model_warmup_inflow_gy(x[2])
        o_inflow_gy_unsqueezed = torch.unsqueeze(o_inflow_gy, 2)
        
        o_outflow_gy = self.model_warmup_outflow_gy(x[3])
        o_outflow_gy_unsqueezed = torch.unsqueeze(o_outflow_gy, 2)
        
        o_inflows_outflows = torch.concat((o_inflow_unsqueezed, 
                                             o_outflow_unsqueezed,
                                             o_inflow_gy_unsqueezed,
                                             o_outflow_gy_unsqueezed,
                                            ), dim=2)

        return  self.model_gnn(o_inflows_outflows)
        
        
    
    
    
#  Input: Inflow + Outflow, Prediction: inflow. Inflow/Outflow prediction GNNs have warmup. 
class M60D(torch.nn.Module): 
    def __init__(self, 
                 n_input_periods,
                 n_output_periods,
                 node_features,
                 n_nodes, 
                 hidden_warmup,
                 edge_index_list,
                 model_path,
                ):
        super(M60D, self).__init__()
        
        assert n_nodes[1] == n_nodes[2], print('Different inflow and outflow graphs of company')
        assert n_nodes[3] == n_nodes[4], print('Different inflow and outflow graphs of gy')
        assert n_nodes[1] == n_nodes[3], print('Different graphs of company and gy')
        # GNN with 4 inputs
        self.model_gnn = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = 4,
                                                        num_of_vertices = n_nodes[1],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[1],
                                                        edge_index=edge_index_list[1],
                                                    )  
        
        self.linear = torch.nn.Linear(n_output_periods * (n_nodes[1]), 
                                                n_output_periods * (n_nodes[1]))

    def forward(self, x):
        input_inflows_outflows = torch.cat((x[0],
                                            x[1],
                                            x[2],
                                            x[3],
                                           ),
                                           dim=2)
        o_gnn = self.model_gnn(input_inflows_outflows)
        
        o_gnn_flattened = o_gnn.view(o_gnn.shape[0], 1, 1, -1)
        
        return self.linear(F.relu(o_gnn_flattened))
                                 

        
    
#  Input: Inflow + Outflow, Prediction: inflow. Inflow/Outflow prediction GNNs have warmup. 
class M60E(torch.nn.Module): 
    def __init__(self, 
                 n_input_periods,
                 n_output_periods,
                 node_features,
                 n_nodes, 
                 hidden_warmup,
                 edge_index_list,
                 model_path,
                ):
        super(M60E, self).__init__()
        
        # Inflow prediction GNN
        self.model_warmup_inflow = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = node_features[1],
                                                        num_of_vertices = n_nodes[1],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[1],
                                                        edge_index=edge_index_list[1],
                                                    )  
        ## Load state dicts
        self.model_warmup_inflow.load_state_dict(torch.load("../models/" + model_path[1] + ".pth"))
        
        # DP outflow prediction GNN, GY
        self.model_warmup_outflow = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = node_features[2],
                                                        num_of_vertices = n_nodes[2],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[2],
                                                        edge_index=edge_index_list[2],
                                                    )  
        ## Load state dicts
        self.model_warmup_outflow.load_state_dict(torch.load("../models/" + model_path[2] + ".pth"))
        
        # Inflow prediction GNN, GY
        self.model_warmup_inflow_gy = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = node_features[3],
                                                        num_of_vertices = n_nodes[3],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[3],
                                                        edge_index=edge_index_list[3],
                                                    )  
        ## Load state dicts
        self.model_warmup_inflow_gy.load_state_dict(torch.load("../models/" + model_path[3] + ".pth"))
        
        # DP outflow prediction GNN, GY
        self.model_warmup_outflow_gy = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = node_features[4],
                                                        num_of_vertices = n_nodes[4],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[4],
                                                        edge_index=edge_index_list[4],
                                                    )  
        ## Load state dicts
        self.model_warmup_outflow_gy.load_state_dict(torch.load("../models/" + model_path[4] + ".pth"))

        assert n_nodes[1] == n_nodes[2], print('Different inflow and outflow graphs of company')
        assert n_nodes[3] == n_nodes[4], print('Different inflow and outflow graphs of gy')
        assert n_nodes[1] == n_nodes[3], print('Different graphs of company and gy')
        # GNN with 4 inputs
        self.model_gnn = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = 4,
                                                        num_of_vertices = n_nodes[1],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[1],
                                                        edge_index=edge_index_list[1],
                                                    )  
        
        self.linear = torch.nn.Linear(n_output_periods * (n_nodes[1]), 
                                                n_output_periods * (n_nodes[1])) 

    def forward(self, x):
        o_inflow = self.model_warmup_inflow(x[0])
        o_inflow_unsqueezed = torch.unsqueeze(o_inflow, 2)
        
        o_outflow = self.model_warmup_outflow(x[1])
        o_outflow_unsqueezed = torch.unsqueeze(o_outflow, 2)
        
        o_inflow_gy = self.model_warmup_inflow_gy(x[2])
        o_inflow_gy_unsqueezed = torch.unsqueeze(o_inflow_gy, 2)
        
        o_outflow_gy = self.model_warmup_outflow_gy(x[3])
        o_outflow_gy_unsqueezed = torch.unsqueeze(o_outflow_gy, 2)
        
        o_inflows_outflows = torch.concat((o_inflow_unsqueezed, 
                                             o_outflow_unsqueezed,
                                             o_inflow_gy_unsqueezed,
                                             o_outflow_gy_unsqueezed,
                                            ), dim=2)

        o_gnn = self.model_gnn(o_inflows_outflows)
        o_gnn_flattened = o_gnn.view(o_gnn.shape[0], 1, 1, -1)
        return self.linear(F.relu(o_gnn_flattened))
    

    
#  Input: Inflow gy + Outflow gy , Prediction: inflow + outflow. Inflow/Outflow prediction GNNs have warmup. 
class M61A(torch.nn.Module): 
    def __init__(self, 
                 n_input_periods,
                 n_output_periods,
                 node_features,
                 n_nodes, 
                 hidden_warmup,
                 edge_index_list,
                 model_path,
                ):
        super(M61A, self).__init__()
        
        # Inflow prediction GNN
        self.model_warmup_inflow = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = node_features[1],
                                                        num_of_vertices = n_nodes[1],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[1],
                                                        edge_index=edge_index_list[1],
                                                    )  
        ## Load state dicts
        self.model_warmup_inflow.load_state_dict(torch.load("../models/" + model_path[1] + ".pth"))
        
        # DP outflow prediction GNN, GY
        self.model_warmup_outflow = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = node_features[2],
                                                        num_of_vertices = n_nodes[2],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[2],
                                                        edge_index=edge_index_list[2],
                                                    )  
        ## Load state dicts
        self.model_warmup_outflow.load_state_dict(torch.load("../models/" + model_path[2] + ".pth"))
        
        # Inflow prediction GNN, GY
        self.model_warmup_inflow_gy = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = node_features[3],
                                                        num_of_vertices = n_nodes[3],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[3],
                                                        edge_index=edge_index_list[3],
                                                    )  
        ## Load state dicts
        self.model_warmup_inflow_gy.load_state_dict(torch.load("../models/" + model_path[3] + ".pth"))
        
        # DP outflow prediction GNN, GY
        self.model_warmup_outflow_gy = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = node_features[4],
                                                        num_of_vertices = n_nodes[4],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[4],
                                                        edge_index=edge_index_list[4],
                                                    )  
        ## Load state dicts
        self.model_warmup_outflow_gy.load_state_dict(torch.load("../models/" + model_path[4] + ".pth"))
        
        assert n_nodes[1] == n_nodes[2], print('Different inflow and outflow graphs of company')
        assert n_nodes[3] == n_nodes[4], print('Different inflow and outflow graphs of gy')
        assert n_nodes[1] == n_nodes[3], print('Different graphs of company and gy')
        self.linear = torch.nn.Linear(n_output_periods * (n_nodes[1]+n_nodes[2]+n_nodes[3]+n_nodes[4]), 
                                                n_output_periods * (n_nodes[1] + n_nodes[2])) 

    def forward(self, x):
        o_inflow = self.model_warmup_inflow(x[0])
        o_inflow_flattened = o_inflow.view(o_inflow.shape[0], 1, 1, -1)
        
        o_outflow = self.model_warmup_outflow(x[1])
        o_outflow_flattened = o_outflow.view(o_outflow.shape[0], 1, 1, -1)
        
        o_inflow_gy = self.model_warmup_inflow_gy(x[2])
        o_inflow_gy_flattened = o_inflow_gy.view(o_inflow_gy.shape[0], 1, 1, -1)
        
        o_outflow_gy = self.model_warmup_outflow_gy(x[3])
        o_outflow_gy_flattened = o_outflow_gy.view(o_outflow_gy.shape[0], 1, 1, -1)
        
        input_fusion = torch.concat((o_inflow_flattened, 
                                     o_outflow_flattened,
                                     o_inflow_gy_flattened,
                                     o_outflow_gy_flattened,
                                    ), dim=-1)
        output = self.linear(F.relu(input_fusion))
        return output 
    

    

    
    
#  Input: Inflow + Outflow, Prediction: inflow. Inflow/Outflow prediction GNNs have warmup. 
class M61B(torch.nn.Module): 
    def __init__(self, 
                 n_input_periods,
                 n_output_periods,
                 node_features,
                 n_nodes, 
                 hidden_warmup,
                 edge_index_list,
                 model_path,
                ):
        super(M61B, self).__init__()
        
        assert n_nodes[1] == n_nodes[2], print('Different inflow and outflow graphs of company')
        assert n_nodes[3] == n_nodes[4], print('Different inflow and outflow graphs of gy')
        assert n_nodes[1] == n_nodes[3], print('Different graphs of company and gy')
        # GNN with 4 inputs
        self.model_gnn = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = 4,
                                                        num_of_vertices = n_nodes[1],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[1],
                                                        edge_index=edge_index_list[1],
                                                    )  
        
        self.linear = torch.nn.Linear(n_output_periods * (n_nodes[1]), 
                                                n_output_periods * (n_nodes[1]+n_nodes[2]))

    def forward(self, x):
        input_inflows_outflows = torch.cat((x[0],
                                            x[1],
                                            x[2],
                                            x[3],
                                           ),
                                           dim=2)
        o_gnn = self.model_gnn(input_inflows_outflows)
        
        o_gnn_flattened = o_gnn.view(o_gnn.shape[0], 1, 1, -1)
        
        return self.linear(F.relu(o_gnn_flattened))
    
    
    
    

    
#  Input: Inflow + Outflow, Prediction: inflow. Inflow/Outflow prediction GNNs have warmup. 
class M61C(torch.nn.Module): 
    def __init__(self, 
                 n_input_periods,
                 n_output_periods,
                 node_features,
                 n_nodes, 
                 hidden_warmup,
                 edge_index_list,
                 model_path,
                ):
        super(M61C, self).__init__()
        
        # Inflow prediction GNN
        self.model_warmup_inflow = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = node_features[1],
                                                        num_of_vertices = n_nodes[1],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[1],
                                                        edge_index=edge_index_list[1],
                                                    )  
        ## Load state dicts
        self.model_warmup_inflow.load_state_dict(torch.load("../models/" + model_path[1] + ".pth"))
        
        # DP outflow prediction GNN, GY
        self.model_warmup_outflow = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = node_features[2],
                                                        num_of_vertices = n_nodes[2],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[2],
                                                        edge_index=edge_index_list[2],
                                                    )  
        ## Load state dicts
        self.model_warmup_outflow.load_state_dict(torch.load("../models/" + model_path[2] + ".pth"))
        
        # Inflow prediction GNN, GY
        self.model_warmup_inflow_gy = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = node_features[3],
                                                        num_of_vertices = n_nodes[3],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[3],
                                                        edge_index=edge_index_list[3],
                                                    )  
        ## Load state dicts
        self.model_warmup_inflow_gy.load_state_dict(torch.load("../models/" + model_path[3] + ".pth"))
        
        # DP outflow prediction GNN, GY
        self.model_warmup_outflow_gy = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = node_features[4],
                                                        num_of_vertices = n_nodes[4],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[4],
                                                        edge_index=edge_index_list[4],
                                                    )  
        ## Load state dicts
        self.model_warmup_outflow_gy.load_state_dict(torch.load("../models/" + model_path[4] + ".pth"))

        assert n_nodes[1] == n_nodes[2], print('Different inflow and outflow graphs of company')
        assert n_nodes[3] == n_nodes[4], print('Different inflow and outflow graphs of gy')
        assert n_nodes[1] == n_nodes[3], print('Different graphs of company and gy')
        # GNN with 4 inputs
        self.model_gnn = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = 4,
                                                        num_of_vertices = n_nodes[1],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[1],
                                                        edge_index=edge_index_list[1],
                                                    )  
        
        self.linear = torch.nn.Linear(n_output_periods * (n_nodes[1]), 
                                                n_output_periods * (n_nodes[1]+n_nodes[2])) 

    def forward(self, x):
        o_inflow = self.model_warmup_inflow(x[0])
        o_inflow_unsqueezed = torch.unsqueeze(o_inflow, 2)
        
        o_outflow = self.model_warmup_outflow(x[1])
        o_outflow_unsqueezed = torch.unsqueeze(o_outflow, 2)
        
        o_inflow_gy = self.model_warmup_inflow_gy(x[2])
        o_inflow_gy_unsqueezed = torch.unsqueeze(o_inflow_gy, 2)
        
        o_outflow_gy = self.model_warmup_outflow_gy(x[3])
        o_outflow_gy_unsqueezed = torch.unsqueeze(o_outflow_gy, 2)
        
        o_inflows_outflows = torch.concat((o_inflow_unsqueezed, 
                                             o_outflow_unsqueezed,
                                             o_inflow_gy_unsqueezed,
                                             o_outflow_gy_unsqueezed,
                                            ), dim=2)

        o_gnn = self.model_gnn(o_inflows_outflows)
        o_gnn_flattened = o_gnn.view(o_gnn.shape[0], 1, 1, -1)
        return self.linear(F.relu(o_gnn_flattened))
    


    
    
#  Input: Inflow + Outflow, Prediction: inflow. Inflow/Outflow prediction GNNs have warmup. 
class M64A(torch.nn.Module): 
    def __init__(self, 
                 n_input_periods,
                 n_output_periods,
                 node_features,
                 n_nodes, 
                 hidden_warmup,
                 edge_index_list,
                 model_path,
                ):
        super(M64A, self).__init__()
        
        # Speed prediction GNN
        self.model_warmup_speed = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = node_features[0],
                                                        num_of_vertices = n_nodes[0],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[0],
                                                        edge_index=edge_index_list[0],
                                                    )  
        ## Load state dicts
        self.model_warmup_speed.load_state_dict(torch.load("../models/" + model_path[0] + ".pth"))
        
        # Inflow prediction GNN
        self.model_warmup_inflow = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = node_features[1],
                                                        num_of_vertices = n_nodes[1],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[1],
                                                        edge_index=edge_index_list[1],
                                                    )  
        ## Load state dicts
        self.model_warmup_inflow.load_state_dict(torch.load("../models/" + model_path[1] + ".pth"))
        
        # DP outflow prediction GNN, GY
        self.model_warmup_outflow = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = node_features[2],
                                                        num_of_vertices = n_nodes[2],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[2],
                                                        edge_index=edge_index_list[2],
                                                    )  
        ## Load state dicts
        self.model_warmup_outflow.load_state_dict(torch.load("../models/" + model_path[2] + ".pth"))
        
        # Inflow prediction GNN, GY
        self.model_warmup_inflow_gy = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = node_features[3],
                                                        num_of_vertices = n_nodes[3],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[3],
                                                        edge_index=edge_index_list[3],
                                                    )  
        ## Load state dicts
        self.model_warmup_inflow_gy.load_state_dict(torch.load("../models/" + model_path[3] + ".pth"))
        
        # DP outflow prediction GNN, GY
        self.model_warmup_outflow_gy = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = node_features[4],
                                                        num_of_vertices = n_nodes[4],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[4],
                                                        edge_index=edge_index_list[4],
                                                    )  
        ## Load state dicts
        self.model_warmup_outflow_gy.load_state_dict(torch.load("../models/" + model_path[4] + ".pth"))

        assert n_nodes[1] == n_nodes[2], print('Different inflow and outflow graphs of company')
        assert n_nodes[3] == n_nodes[4], print('Different inflow and outflow graphs of gy')
        assert n_nodes[1] == n_nodes[3], print('Different graphs of company and gy')
        self.linear = torch.nn.Linear(n_output_periods * (n_nodes[0]+n_nodes[1]+n_nodes[2]+n_nodes[3]+n_nodes[4]), 
                                                n_output_periods * (n_nodes[1])) 

    def forward(self, x):
        o_speed = self.model_warmup_speed(x[0])
        o_speed_flattened = o_speed.view(o_speed.shape[0], 1, 1, -1)
        
        o_inflow = self.model_warmup_inflow(x[1])
        o_inflow_flattened = o_inflow.view(o_inflow.shape[0], 1, 1, -1)
        
        o_outflow = self.model_warmup_outflow(x[2])
        o_outflow_flattened = o_outflow.view(o_outflow.shape[0], 1, 1, -1)
        
        o_inflow_gy = self.model_warmup_inflow_gy(x[3])
        o_inflow_gy_flattened = o_inflow_gy.view(o_inflow_gy.shape[0], 1, 1, -1)
        
        o_outflow_gy = self.model_warmup_outflow_gy(x[4])
        o_outflow_gy_flattened = o_outflow_gy.view(o_outflow_gy.shape[0], 1, 1, -1)
        
        input_fusion = torch.concat((o_speed_flattened, 
                                     o_inflow_flattened, 
                                     o_outflow_flattened,
                                     o_inflow_gy_flattened,
                                     o_outflow_gy_flattened,
                                    ), dim=-1)
        output = self.linear(F.relu(input_fusion))
        return output 
    
        
    
#  Input: Inflow + Outflow, Prediction: inflow. Inflow/Outflow prediction GNNs have warmup. 
class M64B(torch.nn.Module): 
    def __init__(self, 
                 n_input_periods,
                 n_output_periods,
                 node_features,
                 n_nodes, 
                 hidden_warmup,
                 edge_index_list,
                 model_path,
                ):
        super(M64B, self).__init__()
        
        # Speed prediction GNN
        self.model_warmup_speed = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = node_features[0],
                                                        num_of_vertices = n_nodes[0],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[0],
                                                        edge_index=edge_index_list[0],
                                                    )  
        ## Load state dicts
        self.model_warmup_speed.load_state_dict(torch.load("../models/" + model_path[0] + ".pth"))
        
        # Inflow prediction GNN
        self.model_warmup_inflow = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = node_features[1],
                                                        num_of_vertices = n_nodes[1],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[1],
                                                        edge_index=edge_index_list[1],
                                                    )  
        ## Load state dicts
        self.model_warmup_inflow.load_state_dict(torch.load("../models/" + model_path[1] + ".pth"))
        
        # DP outflow prediction GNN, GY
        self.model_warmup_outflow = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = node_features[2],
                                                        num_of_vertices = n_nodes[2],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[2],
                                                        edge_index=edge_index_list[2],
                                                    )  
        ## Load state dicts
        self.model_warmup_outflow.load_state_dict(torch.load("../models/" + model_path[2] + ".pth"))
        
        # Inflow prediction GNN, GY
        self.model_warmup_inflow_gy = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = node_features[3],
                                                        num_of_vertices = n_nodes[3],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[3],
                                                        edge_index=edge_index_list[3],
                                                    )  
        ## Load state dicts
        self.model_warmup_inflow_gy.load_state_dict(torch.load("../models/" + model_path[3] + ".pth"))
        
        # DP outflow prediction GNN, GY
        self.model_warmup_outflow_gy = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = node_features[4],
                                                        num_of_vertices = n_nodes[4],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[4],
                                                        edge_index=edge_index_list[4],
                                                    )  
        ## Load state dicts
        self.model_warmup_outflow_gy.load_state_dict(torch.load("../models/" + model_path[4] + ".pth"))

        assert n_nodes[1] == n_nodes[2], print('Different inflow and outflow graphs of company')
        assert n_nodes[3] == n_nodes[4], print('Different inflow and outflow graphs of gy')
        assert n_nodes[1] == n_nodes[3], print('Different graphs of company and gy')
        # GNN with 4 inputs
        self.model_gnn = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = 4,
                                                        num_of_vertices = n_nodes[1],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[1],
                                                        edge_index=edge_index_list[1],
                                                    )  
        
        self.linear = torch.nn.Linear(n_output_periods * (n_nodes[0] + n_nodes[1]), 
                                                n_output_periods * (n_nodes[1])) 

    def forward(self, x):
        o_speed = self.model_warmup_speed(x[0])
        o_speed_flattened = o_speed.view(o_speed.shape[0], 1, 1, -1)
        
        o_inflow = self.model_warmup_inflow(x[1])
        o_inflow_unsqueezed = torch.unsqueeze(o_inflow, 2)
        
        o_outflow = self.model_warmup_outflow(x[2])
        o_outflow_unsqueezed = torch.unsqueeze(o_outflow, 2)
        
        o_inflow_gy = self.model_warmup_inflow_gy(x[3])
        o_inflow_gy_unsqueezed = torch.unsqueeze(o_inflow_gy, 2)
        
        o_outflow_gy = self.model_warmup_outflow_gy(x[4])
        o_outflow_gy_unsqueezed = torch.unsqueeze(o_outflow_gy, 2)
        
        o_inflows_outflows = torch.concat((o_inflow_unsqueezed, 
                                             o_outflow_unsqueezed,
                                             o_inflow_gy_unsqueezed,
                                             o_outflow_gy_unsqueezed,
                                            ), dim=2)

        o_gnn = self.model_gnn(o_inflows_outflows)
        o_gnn_flattened = o_gnn.view(o_gnn.shape[0], 1, 1, -1)
        
        input_fusion = torch.concat((o_speed_flattened, o_gnn_flattened), dim=-1)
        
        return self.linear(F.relu(input_fusion))
                                 
    

    
#  Input: Inflow + Outflow, Prediction: inflow. Inflow/Outflow prediction GNNs have warmup. 
class M64C(torch.nn.Module): 
    def __init__(self, 
                 n_input_periods,
                 n_output_periods,
                 node_features,
                 n_nodes, 
                 hidden_warmup,
                 edge_index_list,
                 model_path,
                ):
        super(M64C, self).__init__()
        
        # Speed prediction GNN
        self.model_warmup_speed = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = node_features[0],
                                                        num_of_vertices = n_nodes[0],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[0],
                                                        edge_index=edge_index_list[0],
                                                    )  
        ## Load state dicts
        self.model_warmup_speed.load_state_dict(torch.load("../models/" + model_path[0] + ".pth"))
        
        
        assert n_nodes[1] == n_nodes[2], print('Different inflow and outflow graphs of company')
        assert n_nodes[3] == n_nodes[4], print('Different inflow and outflow graphs of gy')
        assert n_nodes[1] == n_nodes[3], print('Different graphs of company and gy')
        # GNN with 4 inputs
        self.model_gnn = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = 4,
                                                        num_of_vertices = n_nodes[1],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[1],
                                                        edge_index=edge_index_list[1],
                                                    )  
        
        self.linear = torch.nn.Linear(n_output_periods * (n_nodes[0] + n_nodes[1]), 
                                                n_output_periods * (n_nodes[1]))

    def forward(self, x):
        o_speed = self.model_warmup_speed(x[0])
        o_speed_flattened = o_speed.view(o_speed.shape[0], 1, 1, -1)
        
        input_inflows_outflows = torch.cat((x[1],
                                            x[2],
                                            x[3],
                                            x[4],
                                           ),
                                           dim=2)
        o_gnn = self.model_gnn(input_inflows_outflows)
        o_gnn_flattened = o_gnn.view(o_gnn.shape[0], 1, 1, -1)
        
        input_fusion = torch.concat((o_speed_flattened, o_gnn_flattened), dim=-1)
        
        return self.linear(F.relu(input_fusion))
                                 



    
#  Input: Inflow + Outflow, Prediction: inflow. Inflow/Outflow prediction GNNs have warmup. 
class M65A(torch.nn.Module): 
    def __init__(self, 
                 n_input_periods,
                 n_output_periods,
                 node_features,
                 n_nodes, 
                 hidden_warmup,
                 edge_index_list,
                 model_path,
                ):
        super(M65A, self).__init__()
        
        # Speed prediction GNN
        self.model_warmup_speed = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = node_features[0],
                                                        num_of_vertices = n_nodes[0],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[0],
                                                        edge_index=edge_index_list[0],
                                                    )  
        ## Load state dicts
        self.model_warmup_speed.load_state_dict(torch.load("../models/" + model_path[0] + ".pth"))
        
        # Inflow prediction GNN
        self.model_warmup_inflow = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = node_features[1],
                                                        num_of_vertices = n_nodes[1],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[1],
                                                        edge_index=edge_index_list[1],
                                                    )  
        ## Load state dicts
        self.model_warmup_inflow.load_state_dict(torch.load("../models/" + model_path[1] + ".pth"))
        
        # DP outflow prediction GNN, GY
        self.model_warmup_outflow = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = node_features[2],
                                                        num_of_vertices = n_nodes[2],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[2],
                                                        edge_index=edge_index_list[2],
                                                    )  
        ## Load state dicts
        self.model_warmup_outflow.load_state_dict(torch.load("../models/" + model_path[2] + ".pth"))
        
        # Inflow prediction GNN, GY
        self.model_warmup_inflow_gy = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = node_features[3],
                                                        num_of_vertices = n_nodes[3],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[3],
                                                        edge_index=edge_index_list[3],
                                                    )  
        ## Load state dicts
        self.model_warmup_inflow_gy.load_state_dict(torch.load("../models/" + model_path[3] + ".pth"))
        
        # DP outflow prediction GNN, GY
        self.model_warmup_outflow_gy = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = node_features[4],
                                                        num_of_vertices = n_nodes[4],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[4],
                                                        edge_index=edge_index_list[4],
                                                    )  
        ## Load state dicts
        self.model_warmup_outflow_gy.load_state_dict(torch.load("../models/" + model_path[4] + ".pth"))

        assert n_nodes[1] == n_nodes[2], print('Different inflow and outflow graphs of company')
        assert n_nodes[3] == n_nodes[4], print('Different inflow and outflow graphs of gy')
        assert n_nodes[1] == n_nodes[3], print('Different graphs of company and gy')
        self.linear = torch.nn.Linear(n_output_periods * (n_nodes[0]+n_nodes[1]+n_nodes[2]+n_nodes[3]+n_nodes[4]), 
                                                n_output_periods * (n_nodes[1] + n_nodes[2])) 

    def forward(self, x):
        o_speed = self.model_warmup_speed(x[0])
        o_speed_flattened = o_speed.view(o_speed.shape[0], 1, 1, -1)
        
        o_inflow = self.model_warmup_inflow(x[1])
        o_inflow_flattened = o_inflow.view(o_inflow.shape[0], 1, 1, -1)
        
        o_outflow = self.model_warmup_outflow(x[2])
        o_outflow_flattened = o_outflow.view(o_outflow.shape[0], 1, 1, -1)
        
        o_inflow_gy = self.model_warmup_inflow_gy(x[3])
        o_inflow_gy_flattened = o_inflow_gy.view(o_inflow_gy.shape[0], 1, 1, -1)
        
        o_outflow_gy = self.model_warmup_outflow_gy(x[4])
        o_outflow_gy_flattened = o_outflow_gy.view(o_outflow_gy.shape[0], 1, 1, -1)
        
        input_fusion = torch.concat((o_speed_flattened, 
                                     o_inflow_flattened, 
                                     o_outflow_flattened,
                                     o_inflow_gy_flattened,
                                     o_outflow_gy_flattened,
                                    ), dim=-1)
        output = self.linear(F.relu(input_fusion))
        return output 
    
        
    
#  Input: Inflow + Outflow, Prediction: inflow. Inflow/Outflow prediction GNNs have warmup. 
class M65B(torch.nn.Module): 
    def __init__(self, 
                 n_input_periods,
                 n_output_periods,
                 node_features,
                 n_nodes, 
                 hidden_warmup,
                 edge_index_list,
                 model_path,
                ):
        super(M65B, self).__init__()
        
        # Speed prediction GNN
        self.model_warmup_speed = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = node_features[0],
                                                        num_of_vertices = n_nodes[0],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[0],
                                                        edge_index=edge_index_list[0],
                                                    )  
        ## Load state dicts
        self.model_warmup_speed.load_state_dict(torch.load("../models/" + model_path[0] + ".pth"))
        
        # Inflow prediction GNN
        self.model_warmup_inflow = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = node_features[1],
                                                        num_of_vertices = n_nodes[1],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[1],
                                                        edge_index=edge_index_list[1],
                                                    )  
        ## Load state dicts
        self.model_warmup_inflow.load_state_dict(torch.load("../models/" + model_path[1] + ".pth"))
        
        # DP outflow prediction GNN, GY
        self.model_warmup_outflow = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = node_features[2],
                                                        num_of_vertices = n_nodes[2],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[2],
                                                        edge_index=edge_index_list[2],
                                                    )  
        ## Load state dicts
        self.model_warmup_outflow.load_state_dict(torch.load("../models/" + model_path[2] + ".pth"))
        
        # Inflow prediction GNN, GY
        self.model_warmup_inflow_gy = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = node_features[3],
                                                        num_of_vertices = n_nodes[3],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[3],
                                                        edge_index=edge_index_list[3],
                                                    )  
        ## Load state dicts
        self.model_warmup_inflow_gy.load_state_dict(torch.load("../models/" + model_path[3] + ".pth"))
        
        # DP outflow prediction GNN, GY
        self.model_warmup_outflow_gy = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = node_features[4],
                                                        num_of_vertices = n_nodes[4],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[4],
                                                        edge_index=edge_index_list[4],
                                                    )  
        ## Load state dicts
        self.model_warmup_outflow_gy.load_state_dict(torch.load("../models/" + model_path[4] + ".pth"))

        assert n_nodes[1] == n_nodes[2], print('Different inflow and outflow graphs of company')
        assert n_nodes[3] == n_nodes[4], print('Different inflow and outflow graphs of gy')
        assert n_nodes[1] == n_nodes[3], print('Different graphs of company and gy')
        # GNN with 4 inputs
        self.model_gnn = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = 4,
                                                        num_of_vertices = n_nodes[1],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[1],
                                                        edge_index=edge_index_list[1],
                                                    )  
        
        self.linear = torch.nn.Linear(n_output_periods * (n_nodes[0] + n_nodes[1]), 
                                                n_output_periods * (n_nodes[1] + n_nodes[2])) 

    def forward(self, x):
        o_speed = self.model_warmup_speed(x[0])
        o_speed_flattened = o_speed.view(o_speed.shape[0], 1, 1, -1)
        
        o_inflow = self.model_warmup_inflow(x[1])
        o_inflow_unsqueezed = torch.unsqueeze(o_inflow, 2)
        
        o_outflow = self.model_warmup_outflow(x[2])
        o_outflow_unsqueezed = torch.unsqueeze(o_outflow, 2)
        
        o_inflow_gy = self.model_warmup_inflow_gy(x[3])
        o_inflow_gy_unsqueezed = torch.unsqueeze(o_inflow_gy, 2)
        
        o_outflow_gy = self.model_warmup_outflow_gy(x[4])
        o_outflow_gy_unsqueezed = torch.unsqueeze(o_outflow_gy, 2)
        
        o_inflows_outflows = torch.concat((o_inflow_unsqueezed, 
                                             o_outflow_unsqueezed,
                                             o_inflow_gy_unsqueezed,
                                             o_outflow_gy_unsqueezed,
                                            ), dim=2)

        o_gnn = self.model_gnn(o_inflows_outflows)
        o_gnn_flattened = o_gnn.view(o_gnn.shape[0], 1, 1, -1)
        
        input_fusion = torch.concat((o_speed_flattened, o_gnn_flattened), dim=-1)
        
        return self.linear(F.relu(input_fusion))
                                 
    

    
#  Input: Inflow + Outflow, Prediction: inflow. Inflow/Outflow prediction GNNs have warmup. 
class M65C(torch.nn.Module): 
    def __init__(self, 
                 n_input_periods,
                 n_output_periods,
                 node_features,
                 n_nodes, 
                 hidden_warmup,
                 edge_index_list,
                 model_path,
                ):
        super(M65C, self).__init__()
        
        # Speed prediction GNN
        self.model_warmup_speed = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = node_features[0],
                                                        num_of_vertices = n_nodes[0],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[0],
                                                        edge_index=edge_index_list[0],
                                                    )  
        ## Load state dicts
        self.model_warmup_speed.load_state_dict(torch.load("../models/" + model_path[0] + ".pth"))
        
        
        assert n_nodes[1] == n_nodes[2], print('Different inflow and outflow graphs of company')
        assert n_nodes[3] == n_nodes[4], print('Different inflow and outflow graphs of gy')
        assert n_nodes[1] == n_nodes[3], print('Different graphs of company and gy')
        # GNN with 4 inputs
        self.model_gnn = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = 4,
                                                        num_of_vertices = n_nodes[1],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[1],
                                                        edge_index=edge_index_list[1],
                                                    )  
        
        self.linear = torch.nn.Linear(n_output_periods * (n_nodes[0] + n_nodes[1]), 
                                                n_output_periods * (n_nodes[1] + n_nodes[2]))

    def forward(self, x):
        o_speed = self.model_warmup_speed(x[0])
        o_speed_flattened = o_speed.view(o_speed.shape[0], 1, 1, -1)
        
        input_inflows_outflows = torch.cat((x[1],
                                            x[2],
                                            x[3],
                                            x[4],
                                           ),
                                           dim=2)
        o_gnn = self.model_gnn(input_inflows_outflows)
        o_gnn_flattened = o_gnn.view(o_gnn.shape[0], 1, 1, -1)
        
        input_fusion = torch.concat((o_speed_flattened, o_gnn_flattened), dim=-1)
        
        return self.linear(F.relu(input_fusion))
                           

    
# Two-company scenario, Input: Inflow of one company + Outflow of one company + Speed, Prediction: inflow. Inflow/Outflow prediction GNNs have DP training. 
class M68(torch.nn.Module): 
    def __init__(self, 
                 n_input_periods,
                 n_output_periods,
                 node_features,
                 n_nodes, 
                 hidden_warmup,
                 edge_index_list,
                 model_path,
                 dp_model_i = 'MP25PP_i',
                 dp_model_o = 'MP25PP_o',
                 model_company = 'multi17',
                 no_backprop_nondp_models = False,
                 no_backprop_dp_models = False,
                ):
        super(M68, self).__init__()
        
        dp_model_func_dict = {'MP25_i': MP25,
                              'MP25_o': MP25,
                              'MP25PP_i': MP25PP,
                              'MP25PP_o': MP25PP,
                             }
        model_company_func_dict = {'multi17': MultiPredictionNet17,
                                   'multi18': MultiPredictionNet18,
                                  }
        self.n_node = n_nodes[1]
        
        # DP inflow prediction GNN
        self.model_dp_trained_inflow = dp_model_func_dict[dp_model_i](n_input_periods=n_input_periods,
                                                                     n_output_periods=n_output_periods,
                                                                     node_features=node_features,
                                                                     n_nodes=n_nodes, 
                                                                     hidden_warmup=hidden_warmup,
                                                                     edge_index_list=edge_index_list,
                                                                     model_path=model_path[:5],
                                                                   )
        ## Load state dicts
        self.model_dp_trained_inflow.load_state_dict(torch.load("../models/" + model_path[5] + ".pth"))
        ## Freeze parameters of model_dp_trained_inflow
        if no_backprop_dp_models:
            for param in self.model_dp_trained_inflow.parameters():
                param.requires_grad = False
            
        # DP outflow prediction GNN
        self.model_dp_trained_outflow = dp_model_func_dict[dp_model_o](n_input_periods=n_input_periods,
                                                                     n_output_periods=n_output_periods,
                                                                     node_features=node_features,
                                                                     n_nodes=n_nodes, 
                                                                     hidden_warmup=hidden_warmup,
                                                                     edge_index_list=edge_index_list,
                                                                     model_path=model_path[:5],
                                                                   )
        ## Load state dicts
        self.model_dp_trained_outflow.load_state_dict(torch.load("../models/" + model_path[6] + ".pth"))
        ## Freeze parameters of model_dp_trained_outflow
        if no_backprop_dp_models:
            for param in self.model_dp_trained_outflow.parameters():
                param.requires_grad = False
            
        # Model of the company itself
        self.model_company = model_company_func_dict[model_company](n_input_periods = n_input_periods,
                                                                     n_output_periods = n_output_periods,
                                                                     node_features = node_features,
                                                                     n_nodes = n_nodes, 
                                                                     hidden_warmup = hidden_warmup,
                                                                     edge_index_list = edge_index_list,
                                                                     model_path = model_path,
                                                                   )
        ## Load state dicts
        self.model_company.load_state_dict(torch.load("../models/" + model_path[7] + ".pth"))
        ## Remove the last layer (i.e., fcnn layer)
        # https://stackoverflow.com/questions/75988246/how-to-change-the-last-layer-of-pretrained-pytorch-model
        self.model_company_layers_without_fcnn = list(self.model_company.children())[:-1]
        ## Freeze parameters of model_company
        if no_backprop_nondp_models:
            for layer in self.model_company_layers_without_fcnn:
                for param in layer.parameters():
                    param.requires_grad = False
                    
        # Linear model
        self.linear = torch.nn.Linear(n_output_periods * (n_nodes[0] + 2*n_nodes[1] + 2*n_nodes[2]), 
                                                n_output_periods * (n_nodes[1])) 
        
    def forward(self, x):
        # GY data -> I DP model
        o_inflow_other_comapny = self.model_dp_trained_inflow([x[3], x[4]])
        o_inflow_other_comapny_flattened = o_inflow_other_comapny.reshape(o_inflow_other_comapny.shape[0], 1, 1, -1)
        # GY data -> O DP model
        o_outflow_other_comapny = self.model_dp_trained_outflow([x[3], x[4]])
        o_outflow_other_comapny_flattened = o_outflow_other_comapny.reshape(o_outflow_other_comapny.shape[0], 1, 1, -1)
        # S -> S layer of DP model 
        o_speed = self.model_company_layers_without_fcnn[0](x[0])   
        o_speed_flattened = o_speed.view(o_speed.shape[0], 1, 1, -1)
        # I -> I layer of DP model 
        o_inflow = self.model_company_layers_without_fcnn[1](x[1]) 
        o_inflow_flattened = o_inflow.view(o_inflow.shape[0], 1, 1, -1)        
        # O -> O layer of DP model 
        o_outflow = self.model_company_layers_without_fcnn[2](x[2]) 
        o_outflow_flattened = o_outflow.view(o_outflow.shape[0], 1, 1, -1)
        
        output_cat = torch.concat((o_speed_flattened, 
                                    o_inflow_flattened, 
                                    o_outflow_flattened,
                                    o_inflow_other_comapny_flattened,
                                    o_outflow_other_comapny_flattened,
                                   ), dim=-1)
        
        output = self.linear(F.relu(output_cat))
        
        return output 
    
    

# Two-company scenario, Input: Inflow of one company + Outflow of one company + Speed, Prediction: inflow. Inflow/Outflow prediction GNNs have DP training. 
class M69(torch.nn.Module): 
    def __init__(self, 
                 n_input_periods,
                 n_output_periods,
                 node_features,
                 n_nodes, 
                 hidden_warmup,
                 edge_index_list,
                 model_path,
                 dp_model_i = 'MP25PP_i',
                 dp_model_o = 'MP25PP_o',
                 model_company = 'multi17',
                 no_backprop_nondp_models = False,
                 no_backprop_dp_models = False,
                ):
        super(M69, self).__init__()
        
        dp_model_func_dict = {'MP25_i': MP25,
                              'MP25_o': MP25,
                              'MP25PP_i': MP25PP,
                              'MP25PP_o': MP25PP,
                             }
        model_company_func_dict = {'multi17': MultiPredictionNet17,
                                   'multi18': MultiPredictionNet18,
                                  }
        self.n_node = n_nodes[1]
        
        # DP inflow prediction GNN
        self.model_dp_trained_inflow = dp_model_func_dict[dp_model_i](n_input_periods=n_input_periods,
                                                                     n_output_periods=n_output_periods,
                                                                     node_features=node_features,
                                                                     n_nodes=n_nodes, 
                                                                     hidden_warmup=hidden_warmup,
                                                                     edge_index_list=edge_index_list,
                                                                     model_path=model_path[:5],
                                                                   )
        ## Load state dicts
        self.model_dp_trained_inflow.load_state_dict(torch.load("../models/" + model_path[5] + ".pth"))
        ## Freeze parameters of model_dp_trained_inflow
        if no_backprop_dp_models:
            for param in self.model_dp_trained_inflow.parameters():
                param.requires_grad = False
            
        # DP outflow prediction GNN
        self.model_dp_trained_outflow = dp_model_func_dict[dp_model_o](n_input_periods=n_input_periods,
                                                                     n_output_periods=n_output_periods,
                                                                     node_features=node_features,
                                                                     n_nodes=n_nodes, 
                                                                     hidden_warmup=hidden_warmup,
                                                                     edge_index_list=edge_index_list,
                                                                     model_path=model_path[:5],
                                                                   )
        ## Load state dicts
        self.model_dp_trained_outflow.load_state_dict(torch.load("../models/" + model_path[6] + ".pth"))
        ## Freeze parameters of model_dp_trained_outflow
        if no_backprop_dp_models:
            for param in self.model_dp_trained_outflow.parameters():
                param.requires_grad = False
            
        # Model of the company itself
        self.model_company = model_company_func_dict[model_company](n_input_periods = n_input_periods,
                                                                     n_output_periods = n_output_periods,
                                                                     node_features = node_features,
                                                                     n_nodes = n_nodes, 
                                                                     hidden_warmup = hidden_warmup,
                                                                     edge_index_list = edge_index_list,
                                                                     model_path = model_path,
                                                                   )
        ## Load state dicts
        self.model_company.load_state_dict(torch.load("../models/" + model_path[7] + ".pth"))
        ## Freeze parameters of model_company
        if no_backprop_nondp_models:
            for layer in self.model_company:
                for param in layer.parameters():
                    param.requires_grad = False
                    
        # Linear model
        self.linear = torch.nn.Linear(n_output_periods * (n_nodes[0] + 2*n_nodes[1] + 2*n_nodes[2]), 
                                                n_output_periods * (n_nodes[1])) 
        
    def forward(self, x):
        # GY data -> I DP model
        o_inflow_other_comapny = self.model_dp_trained_inflow([x[3], x[4]])
        o_inflow_other_comapny_flattened = o_inflow_other_comapny.reshape(o_inflow_other_comapny.shape[0], 1, 1, -1)
        # GY data -> O DP model
        o_outflow_other_comapny = self.model_dp_trained_outflow([x[3], x[4]])
        o_outflow_other_comapny_flattened = o_outflow_other_comapny.reshape(o_outflow_other_comapny.shape[0], 1, 1, -1)
        # DP model: Inputs = S, I, and O.   
        output_s_i_o = self.model_company([x[0], x[1], x[2]])   
        output_s_i_o_flattened = output_s_i_o.view(output_s_i_o.shape[0], 1, 1, -1)
        
        output_cat = torch.concat((output_s_i_o_flattened, 
                                    o_inflow_other_comapny_flattened,
                                    o_outflow_other_comapny_flattened,
                                   ), dim=-1)
        
        output = self.linear(F.relu(output_cat))
        
        return output 
    
    
    
    

# Two-company scenario, Input: Inflow of one company + Outflow of one company + Speed, Prediction: inflow. Inflow/Outflow prediction GNNs have DP training. 
class M72(torch.nn.Module): 
    def __init__(self, 
                 n_input_periods,
                 n_output_periods,
                 node_features,
                 n_nodes, 
                 hidden_warmup,
                 edge_index_list,
                 model_path,
                 dp_model_i = 'MP25P_i',
                 dp_model_o = 'MP25P_o',
                 no_backprop_nondp_models = False,
                 no_backprop_dp_models = False,
                ):
        super(M72, self).__init__()
        
        dp_model_func_dict = {'MP25P_i': MP25P,
                              'MP25P_o': MP25P,
#                               'MP25_i': MP25,
#                               'MP25_o': MP25,
#                               'MP25PP_i': MP25PP,
#                               'MP25PP_o': MP25PP,
                             }
        model_company_func_dict = {'multi17': MultiPredictionNet17,
                                   'multi18': MultiPredictionNet18,
                                  }
        self.n_node = n_nodes[1]
        
        # DP inflow prediction GNN
        self.model_dp_trained_inflow = dp_model_func_dict[dp_model_i](n_input_periods=n_input_periods,
                                                                     n_output_periods=n_output_periods,
                                                                     node_features=node_features,
                                                                     n_nodes=n_nodes, 
                                                                     hidden_warmup=hidden_warmup,
                                                                     edge_index_list=edge_index_list,
                                                                     model_path=model_path[:5],
                                                                   )
        ## Load state dicts
        self.model_dp_trained_inflow.load_state_dict(torch.load("../models/" + model_path[5] + ".pth"))
        ## Freeze parameters of model_dp_trained_inflow
        if no_backprop_dp_models:
            for param in self.model_dp_trained_inflow.parameters():
                param.requires_grad = False
            
        # DP outflow prediction GNN
        self.model_dp_trained_outflow = dp_model_func_dict[dp_model_o](n_input_periods=n_input_periods,
                                                                     n_output_periods=n_output_periods,
                                                                     node_features=node_features,
                                                                     n_nodes=n_nodes, 
                                                                     hidden_warmup=hidden_warmup,
                                                                     edge_index_list=edge_index_list,
                                                                     model_path=model_path[:5],
                                                                   )
        ## Load state dicts
        self.model_dp_trained_outflow.load_state_dict(torch.load("../models/" + model_path[6] + ".pth"))
        ## Freeze parameters of model_dp_trained_outflow
        if no_backprop_dp_models:
            for param in self.model_dp_trained_outflow.parameters():
                param.requires_grad = False
                    
        # DP inflow prediction GNN from Speed
        self.model_dp_trained_inflow_from_speed = MultiPredictionNet5(n_input_periods=n_input_periods,
                                                                     n_output_periods=n_output_periods,
                                                                     node_features=node_features,
                                                                     n_nodes=n_nodes, 
                                                                     hidden_warmup=hidden_warmup,
                                                                     edge_index_list=edge_index_list,
                                                                     model_path=model_path,
                                                                   )
        ## Load state dicts
        self.model_dp_trained_inflow.load_state_dict(torch.load("../models/" + model_path[3] + ".pth"))
        
        ## Freeze parameters of model_dp_trained_outflow
        if no_backprop_dp_models:
            for param in self.model_dp_trained_inflow.parameters():
                param.requires_grad = False
            
        # DP outflow prediction GNN from Speed
        self.model_dp_trained_outflow = MultiPredictionNet6(n_input_periods=n_input_periods,
                                                             n_output_periods=n_output_periods,
                                                             node_features=node_features,
                                                             n_nodes=n_nodes, 
                                                             hidden_warmup=hidden_warmup,
                                                             edge_index_list=edge_index_list,
                                                             model_path=model_path,
                                                           )
        ## Load state dicts
        self.model_dp_trained_outflow.load_state_dict(torch.load("../models/" + model_path[4] + ".pth"))
        
        ## Freeze parameters of model_dp_trained_outflow
        if no_backprop_dp_models:
            for param in self.model_dp_trained_outflow.parameters():
                param.requires_grad = False
            
    def forward(self, x):
        input_gnn = torch.cat((x[1],
                                x[2],
                                x[3],
                                x[4],
                               ),
                               dim=2)
        
        # GY data -> I DP model
        o_inflow_other_comapny = self.model_dp_trained_inflow([x[3], x[4]])
        o_inflow_other_comapny_unsqueezed = torch.unsqueeze(o_inflow_other_comapny, 2)
        # GY data -> O DP model
        o_outflow_other_comapny = self.model_dp_trained_outflow([x[3], x[4]])
        o_outflow_other_comapny_unsqueezed = torch.unsqueeze(o_outflow_other_comapny, 2)
        
        x_final = torch.cat((x[1],
                             x[2],
                             x[3],
                             x[4],
                             o_inflow_other_comapny_unsqueezed,
                             o_outflow_other_comapny_unsqueezed,
                            ),
                            dim=2)
        
        output = self.linear(F.relu(output_cat))
        
        return output 
    
    
# Two-company scenario, Input: Inflow of one company + Outflow of one company + Speed, Prediction: inflow. Inflow/Outflow prediction GNNs have DP training. 
class MA_1in(torch.nn.Module): 
    def __init__(self, 
                 n_input_periods,
                 n_output_periods,
                 node_features,
                 n_nodes, 
                 hidden_warmup,
                 edge_index_list,
                 model_path,
                ):
        super(MA_1in, self).__init__()
        
        # GNN
        self.model_gnn = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = 1, # No. of features
                                                        num_of_vertices = n_nodes[1],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[1],
                                                        edge_index=edge_index_list[1],
                                                    ) 
    def forward(self, x):
        return self.model_gnn(x[0])
    
    
# Two-company scenario, Input: Inflow of one company + Outflow of one company + Speed, Prediction: inflow. Inflow/Outflow prediction GNNs have DP training. 
class MM_1in(torch.nn.Module): 
    def __init__(self, 
                 n_input_periods,
                 n_output_periods,
                 node_features,
                 n_nodes, 
                 hidden_warmup,
                 edge_index_list,
                 model_path,
                ):
        super(MM_1in, self).__init__()
        # GNN
        self.model_gnn = MSTGCN_temp(len_input = n_input_periods,
                                     num_for_predict = n_output_periods,
                                     in_channels = 1, # No. of features
                                     nb_block = 2,
                                     K = 3,
                                     nb_chev_filter = 64,
                                     nb_time_filter = 64,
                                     time_strides = 1,
                                     edge_index=edge_index_list[1],
                                    ) 
    def forward(self, x):
        return self.model_gnn(x[0])

# Two-company scenario, Input: Inflow of one company + Outflow of one company + Speed, Prediction: inflow. Inflow/Outflow prediction GNNs have DP training. 
class MA_2in(torch.nn.Module): 
    def __init__(self, 
                 n_input_periods,
                 n_output_periods,
                 node_features,
                 n_nodes, 
                 hidden_warmup,
                 edge_index_list,
                 model_path,
                ):
        super(MA_2in, self).__init__()
        
        # GNN
        self.model_gnn = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = 2, # No. of features
                                                        num_of_vertices = n_nodes[1],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[1],
                                                        edge_index=edge_index_list[1],
                                                    ) 
    def forward(self, x):
        input_gnn = torch.cat((x[0],
                               x[1],
                              ),
                              dim=2)
        return self.model_gnn(input_gnn)
        
        

# Two-company scenario, Input: Inflow of one company + Outflow of one company + Speed, Prediction: inflow. Inflow/Outflow prediction GNNs have DP training. 
class MM_2in(torch.nn.Module): 
    def __init__(self, 
                 n_input_periods,
                 n_output_periods,
                 node_features,
                 n_nodes, 
                 hidden_warmup,
                 edge_index_list,
                 model_path,
                ):
        super(MM_2in, self).__init__()
        
        # GNN
        self.model_gnn = MSTGCN_temp(len_input = n_input_periods,
                                     num_for_predict = n_output_periods,
                                     in_channels = 2, # No. of features
                                     nb_block = 2,
                                     K = 3,
                                     nb_chev_filter = 64,
                                     nb_time_filter = 64,
                                     time_strides = 1,
                                     edge_index=edge_index_list[1],
                                    ) 
    def forward(self, x):
        input_gnn = torch.cat((x[0],
                               x[1],
                              ),
                              dim=2)
        return self.model_gnn(input_gnn)
        
        

# Two-company scenario, Input: Inflow of one company + Outflow of one company + Speed, Prediction: inflow. Inflow/Outflow prediction GNNs have DP training. 
class MA_2in_linear(torch.nn.Module): 
    def __init__(self, 
                 n_input_periods,
                 n_output_periods,
                 node_features,
                 n_nodes, 
                 hidden_warmup,
                 edge_index_list,
                 model_path,
                ):
        super(MA_2in_linear, self).__init__()
        
        # GNN 1
        self.model_gnn_1 = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = 1, # No. of features
                                                        num_of_vertices = n_nodes[1],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[1],
                                                        edge_index=edge_index_list[1],
                                                    ) 
        
        
        # GNN 2
        self.model_gnn_2 = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = 1, # No. of features
                                                        num_of_vertices = n_nodes[1],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[1],
                                                        edge_index=edge_index_list[1],
                                                    ) 
        
        # Linear model
        self.linear = torch.nn.Linear(n_output_periods * (n_nodes[1] * 2), 
                                                n_output_periods * (n_nodes[1])) 
        
    def forward(self, x):
        o1 = self.model_gnn_1(x[0])
        o1_flattened = o1.view(o1.shape[0], 1, 1, -1)
        o2 = self.model_gnn_2(x[1])
        o2_flattened = o2.view(o2.shape[0], 1, 1, -1)
        o_cat = torch.concat((o1_flattened, 
                              o2_flattened,
                             ),
                             dim = -1)
        return self.linear(F.relu(o_cat))
    
    
    
    
# Two-company scenario, Input: Inflow of one company + Outflow of one company + Speed, Prediction: inflow. Inflow/Outflow prediction GNNs have DP training. 
class MM_2in_linear(torch.nn.Module): 
    def __init__(self, 
                 n_input_periods,
                 n_output_periods,
                 node_features,
                 n_nodes, 
                 hidden_warmup,
                 edge_index_list,
                 model_path,
                ):
        super(MM_2in_linear, self).__init__()
        
        # GNN 1
        self.model_gnn_1 = MSTGCN_temp(len_input = n_input_periods,
                                     num_for_predict = n_output_periods,
                                     in_channels = 1, # No. of features
                                     nb_block = 2,
                                     K = 3,
                                     nb_chev_filter = 64,
                                     nb_time_filter = 64,
                                     time_strides = 1,
                                     edge_index=edge_index_list[1],
                                    ) 
        
        
        # GNN 2
        self.model_gnn_2 = MSTGCN_temp(len_input = n_input_periods,
                                     num_for_predict = n_output_periods,
                                     in_channels = 1, # No. of features
                                     nb_block = 2,
                                     K = 3,
                                     nb_chev_filter = 64,
                                     nb_time_filter = 64,
                                     time_strides = 1,
                                     edge_index=edge_index_list[1],
                                    ) 
        
        # Linear model
        self.linear = torch.nn.Linear(n_output_periods * (n_nodes[1] * 2), 
                                                n_output_periods * (n_nodes[1])) 
        
    def forward(self, x):
        o1 = self.model_gnn_1(x[0])
        o1_flattened = o1.view(o1.shape[0], 1, 1, -1)
        o2 = self.model_gnn_2(x[1])
        o2_flattened = o2.view(o2.shape[0], 1, 1, -1)
        o_cat = torch.concat((o1_flattened, 
                              o2_flattened,
                             ),
                             dim = -1)
        return self.linear(F.relu(o_cat))
    
    
# Two-company scenario, Input: Inflow of one company + Outflow of one company + Speed, Prediction: inflow. Inflow/Outflow prediction GNNs have DP training. 
class MA_3in_linear(torch.nn.Module): 
    def __init__(self, 
                 n_input_periods,
                 n_output_periods,
                 node_features,
                 n_nodes, 
                 hidden_warmup,
                 edge_index_list,
                 model_path,
                ):
        super(MA_3in_linear, self).__init__()
        
        # GNN
        self.model_gnn = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = 1, # No. of features
                                                        num_of_vertices = n_nodes[1],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[1],
                                                        edge_index=edge_index_list[1],
                                                    ) 
        
        # Linear model
        self.linear = torch.nn.Linear(n_output_periods * (n_nodes[1] * 2), 
                                                n_output_periods * (n_nodes[1])) 
        
    def forward(self, x):
        o1 = self.model_gnn([x[0], x[1]])
        o1_flattened = o1.view(o1.shape[0], 1, 1, -1)
        x2_flattened = x2.view(x[2].shape[0], 1, 1, -1)
        o_cat = torch.concat((o1_flattened, 
                              x2_flattened,
                             ),
                             dim = -1)
        return self.linear(F.relu(o_cat))
    
    
    
# Two-company scenario, Input: Inflow of one company + Outflow of one company + Speed, Prediction: inflow. Inflow/Outflow prediction GNNs have DP training. 
class MM_3in_linear(torch.nn.Module): 
    def __init__(self, 
                 n_input_periods,
                 n_output_periods,
                 node_features,
                 n_nodes, 
                 hidden_warmup,
                 edge_index_list,
                 model_path,
                ):
        super(MM_3in_linear, self).__init__()
        
        # GNN
        self.model_gnn = MSTGCN_temp(len_input = n_input_periods,
                                     num_for_predict = n_output_periods,
                                     in_channels = 1, # No. of features
                                     nb_block = 2,
                                     K = 3,
                                     nb_chev_filter = 64,
                                     nb_time_filter = 64,
                                     time_strides = 1,
                                     edge_index=edge_index_list[1],
                                    ) 
        
        # Linear model
        self.linear = torch.nn.Linear(n_output_periods * (n_nodes[1] * 2), 
                                                n_output_periods * (n_nodes[1])) 
        
    def forward(self, x):
        o1 = self.model_gnn([x[0], x[1]])
        o1_flattened = o1.view(o1.shape[0], 1, 1, -1)
        x2_flattened = x2.view(x[2].shape[0], 1, 1, -1)
        o_cat = torch.concat((o1_flattened, 
                              x2_flattened,
                             ),
                             dim = -1)
        return self.linear(F.relu(o_cat))
    
    
# # Two-company scenario, Input: Inflow of one company + Outflow of one company + Speed, Prediction: inflow. Inflow/Outflow prediction GNNs have DP training. 
# class MA_3in_speed(torch.nn.Module): 
#     def __init__(self, 
#                  n_input_periods,
#                  n_output_periods,
#                  node_features,
#                  n_nodes, 
#                  hidden_warmup,
#                  edge_index_list,
#                  model_path,
#                 ):
#         super(MA_3in_speed, self).__init__()
        
#         # GNN
#         self.model_gnn_final = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
#                                                         num_for_predict = n_output_periods,
#                                                         in_channels = 3, # No. of features
#                                                         num_of_vertices = n_nodes[1],
#                                                         nb_block = 2,
#                                                         K = 3,
#                                                         nb_chev_filter = 64,
#                                                         nb_time_filter = 64,
#                                         #                         time_strides = num_of_hours,
#                                                         time_strides = 1,
#                                                         hidden = hidden_warmup[1],
#                                                         edge_index=edge_index_list[1],
#                                                     ) 
        
#     def forward(self, x):
#         input_gnn = torch.cat((x[0],
#                                x[1],
#                                o1,
#                               ),
#                               dim=2)
#         output = self.model_gnn_final(input_gnn)
#         return output
    
    
# Two-company scenario, Input: Inflow of one company + Outflow of one company + Speed, Prediction: inflow. Inflow/Outflow prediction GNNs have DP training. 
class MA_4in(torch.nn.Module): 
    def __init__(self, 
                 n_input_periods,
                 n_output_periods,
                 node_features,
                 n_nodes, 
                 hidden_warmup,
                 edge_index_list,
                 model_path,
                ):
        super(MA_4in, self).__init__()
        
        # GNN
        self.model_gnn = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = 4, # No. of features
                                                        num_of_vertices = n_nodes[1],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[1],
                                                        edge_index=edge_index_list[1],
                                                    ) 
        
    def forward(self, x):
        input_gnn = torch.cat((x[0],
                               x[1],
                               x[2],
                               x[3],
                              ),
                              dim=2)
        output = self.model_gnn(input_gnn)
        return output
    
# Two-company scenario, Input: Inflow of one company + Outflow of one company + Speed, Prediction: inflow. Inflow/Outflow prediction GNNs have DP training. 
class MA_6in(torch.nn.Module): 
    def __init__(self, 
                 n_input_periods,
                 n_output_periods,
                 node_features,
                 n_nodes, 
                 hidden_warmup,
                 edge_index_list,
                 model_path,
                ):
        super(MA_6in, self).__init__()
        
        # GNN
        self.model_gnn = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = 6, # No. of features
                                                        num_of_vertices = n_nodes[1],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[1],
                                                        edge_index=edge_index_list[1],
                                                    ) 
        
    def forward(self, x):
        input_gnn = torch.cat((x[0],
                               x[1],
                               x[2],
                               x[3],
                               x[4],
                               x[5],
                              ),
                              dim=2)
        output = self.model_gnn(input_gnn)
        return output
    
    
# Two-company scenario, Input: Inflow of one company + Outflow of one company + Speed, Prediction: inflow. Inflow/Outflow prediction GNNs have DP training. 
class MA_4in_linear(torch.nn.Module): 
    def __init__(self, 
                 n_input_periods,
                 n_output_periods,
                 node_features,
                 n_nodes, 
                 hidden_warmup,
                 edge_index_list,
                 model_path,
                ):
        super(MA_4in_linear, self).__init__()
        
        # GNN
        self.model_gnn = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = 2, # No. of features
                                                        num_of_vertices = n_nodes[1],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[1],
                                                        edge_index=edge_index_list[1],
                                                    ) 
        
        
        # Linear model
        self.linear = torch.nn.Linear(n_output_periods * (n_nodes[1] * 3), 
                                                n_output_periods * (n_nodes[1])) 
        
    def forward(self, x):
        input_gnn = torch.cat((x[0],
                               x[1],
                              ),
                              dim=2)
        o1 = self.model_gnn(input_gnn)
        o1_flattened = o1.view(o1.shape[0], 1, 1, -1)
        x2_flattened = x[2].reshape(x[2].shape[0], 1, 1, -1)
        x3_flattened = x[3].reshape(x[3].shape[0], 1, 1, -1)
        o_cat = torch.concat((o1_flattened, 
                              x2_flattened,
                              x3_flattened,
                             ),
                             dim = -1)
        return self.linear(F.relu(o_cat))
    
# Two-company scenario, Input: Inflow of one company + Outflow of one company + Speed, Prediction: inflow. Inflow/Outflow prediction GNNs have DP training. 
class MM_4in_linear(torch.nn.Module): 
    def __init__(self, 
                 n_input_periods,
                 n_output_periods,
                 node_features,
                 n_nodes, 
                 hidden_warmup,
                 edge_index_list,
                 model_path,
                ):
        super(MM_4in_linear, self).__init__()
        
        # GNN
        self.model_gnn = MSTGCN_temp(len_input = n_input_periods,
                                     num_for_predict = n_output_periods,
                                     in_channels = 2, # No. of features
                                     nb_block = 2,
                                     K = 3,
                                     nb_chev_filter = 64,
                                     nb_time_filter = 64,
                                     time_strides = 1,
                                     edge_index=edge_index_list[1],
                                    ) 
        
        
        # Linear model
        self.linear = torch.nn.Linear(n_output_periods * (n_nodes[1] * 4), 
                                                n_output_periods * (n_nodes[1])) 
        
    def forward(self, x):
        input_gnn = torch.cat((x[0],
                               x[1],
                              ),
                              dim=2)
        o1 = self.model_gnn(input_gnn)
        o1_flattened = o1.view(o1.shape[0], 1, 1, -1)
        x2_flattened = x2.view(x[2].shape[0], 1, 1, -1)
        x3_flattened = x3.view(x[3].shape[0], 1, 1, -1)
        o_cat = torch.concat((o1_flattened, 
                              x2_flattened,
                              x3_flattened,
                             ),
                             dim = -1)
        return self.linear(F.relu(o_cat))
    
# Two-company scenario, Input: Inflow of one company + Outflow of one company + Speed, Prediction: inflow. Inflow/Outflow prediction GNNs have DP training. 
class MA_3in_speed_linear(torch.nn.Module): 
    def __init__(self, 
                 n_input_periods,
                 n_output_periods,
                 node_features,
                 n_nodes, 
                 hidden_warmup,
                 edge_index_list,
                 model_path,
                ):
        super(MA_3in_speed_linear, self).__init__()
                    
        # DP inflow prediction GNN from Speed
        self.model_inflow_from_speed = MultiPredictionNet5(n_input_periods=n_input_periods,
                                                                     n_output_periods=n_output_periods,
                                                                     node_features=node_features,
                                                                     n_nodes=n_nodes, 
                                                                     hidden_warmup=hidden_warmup,
                                                                     edge_index_list=edge_index_list,
                                                                     model_path=model_path,
                                                                   )
        ## Load state dicts
        self.model_inflow_from_speed.load_state_dict(torch.load("../models/" + model_path[7] + ".pth"))
        ## Freeze parameters of model_dp_trained_outflow
        for param in self.model_inflow_from_speed.parameters():
            param.requires_grad = False
            
        # DP outflow prediction GNN from Speed
        self.model_outflow_from_speed = MultiPredictionNet6(n_input_periods=n_input_periods,
                                                             n_output_periods=n_output_periods,
                                                             node_features=node_features,
                                                             n_nodes=n_nodes, 
                                                             hidden_warmup=hidden_warmup,
                                                             edge_index_list=edge_index_list,
                                                             model_path=model_path,
                                                           )
        ## Load state dicts
        self.model_outflow_from_speed.load_state_dict(torch.load("../models/" + model_path[8] + ".pth"))
        ## Freeze parameters of model_dp_trained_outflow
        for param in self.model_outflow_from_speed.parameters():
            param.requires_grad = False
            
        # GNN of the company itself 
        self.model_gnn = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = 2, # No. of features
                                                        num_of_vertices = n_nodes[1],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[1],
                                                        edge_index=edge_index_list[1],
                                                    ) 
        
                
        # Linear model
        self.linear = torch.nn.Linear(n_output_periods * (n_nodes[1] * 3), 
                                                n_output_periods * (n_nodes[1])) 
        
    def forward(self, x):
        # S -> I, DP model
        o_inflow_other_comapny = self.model_inflow_from_speed(x[0])
        o_inflow_other_comapny_flattened = o_inflow_other_comapny.view(o_inflow_other_comapny.shape[0], 1, 1, -1)
        # S -> O, DP model
        o_outflow_other_comapny = self.model_outflow_from_speed(x[0])
        o_outflow_other_comapny_flattened = o_outflow_other_comapny.view(o_outflow_other_comapny.shape[0], 1, 1, -1)
        # GNN of company itself
        input_gnn = torch.cat((x[1],
                               x[2],
                              ),
                              dim = 2)
        o1 = self.model_gnn(input_gnn)
        o1_flattened = o1.view(o1.shape[0], 1, 1, -1)
        # Concatenate inputs of the linear layer
        o_cat = torch.concat((o1_flattened, 
                              o_inflow_other_comapny_flattened,
                              o_outflow_other_comapny_flattened,
                             ),
                             dim = -1)
        output = self.linear(F.relu(o_cat))
        return output 
    
    
    
# Two-company scenario, Input: Inflow of one company + Outflow of one company + Speed, Prediction: inflow. Inflow/Outflow prediction GNNs have DP training. 
class MA_3in_speed(torch.nn.Module): 
    def __init__(self, 
                 n_input_periods,
                 n_output_periods,
                 node_features,
                 n_nodes, 
                 hidden_warmup,
                 edge_index_list,
                 model_path,
                 no_backprop_dp_models,
                ):
        super(MA_3in_speed, self).__init__()
                    
        # DP inflow prediction GNN from Speed
        self.model_inflow_from_speed = MultiPredictionNet5(n_input_periods=n_input_periods,
                                                                     n_output_periods=n_output_periods,
                                                                     node_features=node_features,
                                                                     n_nodes=n_nodes, 
                                                                     hidden_warmup=hidden_warmup,
                                                                     edge_index_list=edge_index_list,
                                                                     model_path=model_path,
                                                                   )
        ## Load state dicts
        self.model_inflow_from_speed.load_state_dict(torch.load("../models/" + model_path[7] + ".pth"))
        ## Freeze parameters of model_dp_trained_outflow
        if no_backprop_dp_models:
            for param in self.model_inflow_from_speed.parameters():
                param.requires_grad = False
            
        # DP outflow prediction GNN from Speed
        self.model_outflow_from_speed = MultiPredictionNet6(n_input_periods=n_input_periods,
                                                             n_output_periods=n_output_periods,
                                                             node_features=node_features,
                                                             n_nodes=n_nodes, 
                                                             hidden_warmup=hidden_warmup,
                                                             edge_index_list=edge_index_list,
                                                             model_path=model_path,
                                                           )
        ## Load state dicts
        self.model_outflow_from_speed.load_state_dict(torch.load("../models/" + model_path[8] + ".pth"))
        ## Freeze parameters of model_dp_trained_outflow
        if no_backprop_dp_models:
            for param in self.model_outflow_from_speed.parameters():
                param.requires_grad = False
            
        # GNN of the company itself 
        self.model_gnn = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = 4, # No. of features
                                                        num_of_vertices = n_nodes[1],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[1],
                                                        edge_index=edge_index_list[1],
                                                    ) 
        self.n_node = n_nodes[1]
        
    def forward(self, x):
        # S -> I, DP model
        o_inflow_other_comapny = self.model_inflow_from_speed(x[0])
        o_inflow_other_comapny_reshaped = o_inflow_other_comapny.reshape(o_inflow_other_comapny.shape[0], self.n_node, 1, -1)
        # S -> O, DP model
        o_outflow_other_comapny = self.model_outflow_from_speed(x[0])
        o_outflow_other_comapny_reshaped = o_outflow_other_comapny.reshape(o_outflow_other_comapny.shape[0], self.n_node, 1, -1)
        # GNN
        input_gnn = torch.cat((o_inflow_other_comapny_reshaped,
                               o_outflow_other_comapny_reshaped,
                               x[1],
                               x[2],
                              ),
                              dim = 2)
        output_gnn = self.model_gnn(input_gnn)
        return output_gnn 
    
    
    
# Two-company scenario, Input: Inflow of one company + Outflow of one company + Speed, Prediction: inflow. Inflow/Outflow prediction GNNs have DP training. 
class MA_3in_speed_baseline(torch.nn.Module): 
    def __init__(self, 
                 n_input_periods,
                 n_output_periods,
                 node_features,
                 n_nodes, 
                 hidden_warmup,
                 edge_index_list,
                 model_path,
                 no_backprop_s_models,
                ):
        super(MA_3in_speed_baseline, self).__init__()
                    
        # Warmed up speed prediction GNN
        self.model_speed = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                            num_for_predict = n_output_periods,
                                                            in_channels = 1, # No. of features
                                                            num_of_vertices = n_nodes[0],
                                                            nb_block = 2,
                                                            K = 3,
                                                            nb_chev_filter = 64,
                                                            nb_time_filter = 64,
                                                            time_strides = 1,
                                                            hidden = hidden_warmup[0],
                                                            edge_index=edge_index_list[0],
                                                         ) 
        ## Load state dicts of speed prediction
        self.model_speed.load_state_dict(torch.load("../models/" + model_path[0] + ".pth"))
        ## Freeze parameters of model_dp_trained_outflow
        if no_backprop_s_models:
            for param in self.model_speed.parameters():
                param.requires_grad = False
        
        # Linear model, Map speed graph to inflow/outflow graph
        self.linear = torch.nn.Linear(n_output_periods * n_nodes[0], n_output_periods * n_nodes[1]) 
            
        # GNN AS FINAL LAYER
        self.model_gnn = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = 3, # No. of features
                                                        num_of_vertices = n_nodes[1],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[1],
                                                        edge_index=edge_index_list[1],
                                                       ) 
        
        self.n_node = n_nodes[1]
        
    def forward(self, x):
        # S 
        o_speed = self.model_speed(x[0])
        o_speed_flattened = o_speed.view(o_speed.shape[0], 1, 1, -1)
        o_speed_mapped = self.linear(o_speed_flattened)
        o_speed_reshaped = o_speed_mapped.reshape(o_speed_mapped.shape[0], self.n_node, 1, -1)
        
        # GNN
        input_gnn = torch.cat((o_speed_reshaped,
                               x[1],
                               x[2],
                              ),
                              dim = 2)
        output_gnn = self.model_gnn(input_gnn)
        return output_gnn 
    
    
    
    
# Two-company scenario, Input: Inflow of one company + Outflow of one company + Speed, Prediction: inflow. Inflow/Outflow prediction GNNs have DP training. 
class MA_3in_speed_linear1_baseline(torch.nn.Module): 
    def __init__(self, 
                 n_input_periods,
                 n_output_periods,
                 node_features,
                 n_nodes, 
                 hidden_warmup,
                 edge_index_list,
                 model_path,
                 no_backprop_s_models,
                ):
        super(MA_3in_speed_linear1_baseline, self).__init__()
        
        # GNN for inflow and outflow 
        self.model_gnn_inflow_outflow = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = 2, # No. of features
                                                        num_of_vertices = n_nodes[1],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[1],
                                                        edge_index=edge_index_list[1],
                                                    ) 
                    
        # Warmed up speed prediction GNN
        self.model_speed = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                            num_for_predict = n_output_periods,
                                                            in_channels = 1, # No. of features
                                                            num_of_vertices = n_nodes[0],
                                                            nb_block = 2,
                                                            K = 3,
                                                            nb_chev_filter = 64,
                                                            nb_time_filter = 64,
                                                            time_strides = 1,
                                                            hidden = hidden_warmup[0],
                                                            edge_index=edge_index_list[0],
                                                         ) 
        ## Load state dicts of speed prediction
        self.model_speed.load_state_dict(torch.load("../models/" + model_path[0] + ".pth"))
        ## Freeze parameters of model_dp_trained_outflow
        if no_backprop_s_models:
            for param in self.model_speed.parameters():
                param.requires_grad = False
        
        # Linear model, Map speed graph to inflow/outflow graph
        self.linear_map_graph = torch.nn.Linear(n_output_periods * n_nodes[0], n_output_periods * n_nodes[1]) 
            
        # Linear AS FINAL LAYER
        self.linear_final = torch.nn.Linear(n_output_periods * (n_nodes[1] * 3), 
                                                            n_output_periods * (n_nodes[1])) 
        
        self.n_node = n_nodes[1]
        
        
    def forward(self, x):
        # S 
        o_speed = self.model_speed(x[0])
        o_speed_flattened = o_speed.view(o_speed.shape[0], 1, 1, -1)
        o_speed_mapped = self.linear_map_graph(o_speed_flattened)
        o_speed_flattened2 = o_speed_mapped.reshape(o_speed_mapped.shape[0], 1, 1, -1)
        
        # GNN
        input_gnn_inflow_outflow = torch.cat((x[1],
                                              x[2],
                                             ),
                                             dim = 2)
        output_gnn_inflow_outflow = self.model_gnn_inflow_outflow(input_gnn_inflow_outflow)
        output_gnn_inflow_outflow_flattened = output_gnn_inflow_outflow.view(output_gnn_inflow_outflow.shape[0], 1, 1, -1)
        
        # Concatenate inputs of the linear layer
        o_cat = torch.concat((o_speed_flattened2, 
                              output_gnn_inflow_outflow,
                             ),
                             dim = -1)
        output = self.linear_final(F.relu(o_cat))
        
        return output_gnn 
    
    
    
    
# Two-company scenario, Input: Inflow of one company + Outflow of one company + Speed, Prediction: inflow. Inflow/Outflow prediction GNNs have DP training. 
class MA_3in_speed_linear2_baseline(torch.nn.Module): 
    def __init__(self, 
                 n_input_periods,
                 n_output_periods,
                 node_features,
                 n_nodes, 
                 hidden_warmup,
                 edge_index_list,
                 model_path,
                 no_backprop_s_models,
                ):
        super(MA_3in_speed_linear2_baseline, self).__init__()
        
        # GNN for inflow and outflow 
        self.model_gnn_inflow_outflow = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = 2, # No. of features
                                                        num_of_vertices = n_nodes[1],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[1],
                                                        edge_index=edge_index_list[1],
                                                    ) 
                    
        # Warmed up speed prediction GNN
        self.model_speed = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                            num_for_predict = n_output_periods,
                                                            in_channels = 1, # No. of features
                                                            num_of_vertices = n_nodes[0],
                                                            nb_block = 2,
                                                            K = 3,
                                                            nb_chev_filter = 64,
                                                            nb_time_filter = 64,
                                                            time_strides = 1,
                                                            hidden = hidden_warmup[0],
                                                            edge_index=edge_index_list[0],
                                                         ) 
        ## Load state dicts of speed prediction
        self.model_speed.load_state_dict(torch.load("../models/" + model_path[0] + ".pth"))
        ## Freeze parameters of model_dp_trained_outflow
        if no_backprop_s_models:
            for param in self.model_speed.parameters():
                param.requires_grad = False

        # Linear AS FINAL LAYER
        self.linear_final = torch.nn.Linear(n_output_periods * (n_nodes[0] + n_nodes[1]), 
                                                            n_output_periods * (n_nodes[1])) 
        
        self.n_node = n_nodes[1]
        
    def forward(self, x):
        # S 
        o_speed = self.model_speed(x[0])
#         print(f"\no_speed.shape: {o_speed.shape}")
        o_speed_flattened = o_speed.view(o_speed.shape[0], 1, 1, -1)
#         print(f"o_speed_flattened.shape: {o_speed_flattened.shape}")
        
        # GNN
#         print(f"x[1].shape: {x[1].shape}")
#         print(f"x[2].shape: {x[2].shape}")
        input_gnn_inflow_outflow = torch.cat((x[1],
                                              x[2],
                                             ),
                                             dim = 2)
#         print(f"input_gnn_inflow_outflow.shape: {input_gnn_inflow_outflow.shape}")
        output_gnn_inflow_outflow = self.model_gnn_inflow_outflow(input_gnn_inflow_outflow)
#         print(f"output_gnn_inflow_outflow.shape: {output_gnn_inflow_outflow.shape}")
        output_gnn_inflow_outflow_flattened = output_gnn_inflow_outflow.view(output_gnn_inflow_outflow.shape[0], 1, 1, -1)
#         print(f"output_gnn_inflow_outflow_flattened.shape: {output_gnn_inflow_outflow_flattened.shape}")
        
        # Concatenate inputs of the linear layer
        o_cat = torch.concat((o_speed_flattened, 
                              output_gnn_inflow_outflow_flattened,
                             ),
                             dim = -1)
#         print(f"o_cat.shape: {o_cat.shape}")
        output = self.linear_final(F.relu(o_cat))
#         print(f"output.shape: {output.shape}")
        
        return output 
    
# Two-company scenario, Input: Inflow of one company + Outflow of one company + Speed, Prediction: inflow. Inflow/Outflow prediction GNNs have DP training. 
class MA_3in_speed_extra(torch.nn.Module): 
    def __init__(self, 
                 n_input_periods,
                 n_output_periods,
                 node_features,
                 n_nodes, 
                 hidden_warmup,
                 edge_index_list,
                 model_path,
                 no_backprop_s_models,
                 no_backprop_dp_models,
                ):
        super(MA_3in_speed_extra, self).__init__()
                    
        # DP inflow prediction GNN from Speed
        self.model_inflow_from_speed = MultiPredictionNet5(n_input_periods=n_input_periods,
                                                                     n_output_periods=n_output_periods,
                                                                     node_features=node_features,
                                                                     n_nodes=n_nodes, 
                                                                     hidden_warmup=hidden_warmup,
                                                                     edge_index_list=edge_index_list,
                                                                     model_path=model_path,
                                                                   )
        ## Load state dicts
        self.model_inflow_from_speed.load_state_dict(torch.load("../models/" + model_path[7] + ".pth"))
        ## Freeze parameters of model_dp_trained_outflow
        if no_backprop_dp_models:
            for param in self.model_inflow_from_speed.parameters():
                param.requires_grad = False
            
        # DP outflow prediction GNN from Speed
        self.model_outflow_from_speed = MultiPredictionNet6(n_input_periods=n_input_periods,
                                                             n_output_periods=n_output_periods,
                                                             node_features=node_features,
                                                             n_nodes=n_nodes, 
                                                             hidden_warmup=hidden_warmup,
                                                             edge_index_list=edge_index_list,
                                                             model_path=model_path,
                                                           )
        ## Load state dicts
        self.model_outflow_from_speed.load_state_dict(torch.load("../models/" + model_path[8] + ".pth"))
        ## Freeze parameters of model_dp_trained_outflow
        if no_backprop_dp_models:
            for param in self.model_outflow_from_speed.parameters():
                param.requires_grad = False
            
        # Warmed up speed prediction GNN
        self.model_speed = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                            num_for_predict = n_output_periods,
                                                            in_channels = 1, # No. of features
                                                            num_of_vertices = n_nodes[0],
                                                            nb_block = 2,
                                                            K = 3,
                                                            nb_chev_filter = 64,
                                                            nb_time_filter = 64,
                                                            time_strides = 1,
                                                            hidden = hidden_warmup[0],
                                                            edge_index=edge_index_list[0],
                                                         ) 
        ## Load state dicts of speed prediction
        self.model_speed.load_state_dict(torch.load("../models/" + model_path[0] + ".pth"))
        ## Freeze parameters of model_dp_trained_outflow
        if no_backprop_s_models:
            for param in self.model_speed.parameters():
                param.requires_grad = False
        # Linear model, Map speed graph to inflow/outflow graph
        self.linear = torch.nn.Linear(n_output_periods * n_nodes[0], n_output_periods * n_nodes[1]) 
        
        # GNN of the company itself 
        self.model_gnn = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = 5, # No. of input features
                                                        num_of_vertices = n_nodes[1],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[1],
                                                        edge_index=edge_index_list[1],
                                                    ) 
        self.n_node = n_nodes[1]
        
    def forward(self, x):
        # S -> I, DP model
        o_inflow_other_comapny = self.model_inflow_from_speed(x[0])
        o_inflow_other_comapny_reshaped = o_inflow_other_comapny.reshape(o_inflow_other_comapny.shape[0], self.n_node, 1, -1)
        # S -> O, DP model
        o_outflow_other_comapny = self.model_outflow_from_speed(x[0])
        o_outflow_other_comapny_reshaped = o_outflow_other_comapny.reshape(o_outflow_other_comapny.shape[0], self.n_node, 1, -1)
        # S 
        o_speed = self.model_speed(x[0])
        o_speed_flattened = o_speed.view(o_speed.shape[0], 1, 1, -1)
        o_speed_mapped = self.linear(o_speed_flattened)
        o_speed_reshaped = o_speed_mapped.reshape(o_speed_mapped.shape[0], self.n_node, 1, -1)
        # GNN
        input_gnn = torch.cat((x[1],
                               x[2],
                               o_speed_reshaped,
                               o_inflow_other_comapny_reshaped,
                               o_outflow_other_comapny_reshaped,
                              ),
                              dim = 2)
        output_gnn = self.model_gnn(input_gnn)
        return output_gnn 

# Two-company scenario, Input: Inflow of one company + Outflow of one company + Speed, Prediction: inflow. Inflow/Outflow prediction GNNs have DP training. 
class MA_3in_speed_extra_dp(torch.nn.Module): 
    def __init__(self, 
                 n_input_periods,
                 n_output_periods,
                 node_features,
                 n_nodes, 
                 hidden_warmup,
                 edge_index_list,
                 model_path,
                 no_backprop_s_models,
                ):
        super(MA_3in_speed_extra_dp, self).__init__()
                    
        # Warmed up speed prediction GNN
        self.model_speed = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                            num_for_predict = n_output_periods,
                                                            in_channels = 1, # No. of features
                                                            num_of_vertices = n_nodes[0],
                                                            nb_block = 2,
                                                            K = 3,
                                                            nb_chev_filter = 64,
                                                            nb_time_filter = 64,
                                                            time_strides = 1,
                                                            hidden = hidden_warmup[0],
                                                            edge_index=edge_index_list[0],
                                                         ) 
        ## Load state dicts of speed prediction
        self.model_speed.load_state_dict(torch.load("../models/" + model_path[0] + ".pth"))
        ## Freeze parameters of model_dp_trained_outflow
        if no_backprop_s_models:
            for param in self.model_speed.parameters():
                param.requires_grad = False
        # Linear model, Map speed graph to inflow/outflow graph
        self.linear = torch.nn.Linear(n_output_periods * n_nodes[0], n_output_periods * n_nodes[1]) 
        
        # GNN of the company itself 
        self.model_gnn = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = 5, # No. of input features
                                                        num_of_vertices = n_nodes[1],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[1],
                                                        edge_index=edge_index_list[1],
                                                    ) 
        self.n_node = n_nodes[1]
        
    def forward(self, x):
        # S 
        o_speed = self.model_speed(x[0])
        o_speed_flattened = o_speed.view(o_speed.shape[0], 1, 1, -1)
        o_speed_mapped = self.linear(o_speed_flattened)
        o_speed_reshaped = o_speed_mapped.reshape(o_speed_mapped.shape[0], self.n_node, 1, -1)
        # GNN
        input_gnn = torch.cat((x[1],
                               x[2],
                               o_speed_reshaped,
                               x[3],
                               x[4],
                              ),
                              dim = 2)
        output_gnn = self.model_gnn(input_gnn)
        return output_gnn 
        
# Two-company scenario, Input: Inflow of one company + Outflow of one company + Speed, Prediction: inflow. Inflow/Outflow prediction GNNs have DP training. 
class MA_3in_speed_extra_pool_sum(torch.nn.Module): 
    def __init__(self, 
                 n_input_periods,
                 n_output_periods,
                 node_features,
                 n_nodes, 
                 hidden_warmup,
                 edge_index_list,
                 model_path,
                 no_backprop_models,
                ):
        super(MA_3in_speed_extra_pool_sum, self).__init__()
                    
        # DP inflow prediction GNN from Speed
        self.model_inflow_from_speed = MultiPredictionNet5(n_input_periods=n_input_periods,
                                                                     n_output_periods=n_output_periods,
                                                                     node_features=node_features,
                                                                     n_nodes=n_nodes, 
                                                                     hidden_warmup=hidden_warmup,
                                                                     edge_index_list=edge_index_list,
                                                                     model_path=model_path,
                                                                   )
        ## Load state dicts
        self.model_inflow_from_speed.load_state_dict(torch.load("../models/" + model_path[7] + ".pth"))
        ## Freeze parameters of model_dp_trained_outflow
        if no_backprop_models:
            for param in self.model_inflow_from_speed.parameters():
                param.requires_grad = False
            
        # DP outflow prediction GNN from Speed
        self.model_outflow_from_speed = MultiPredictionNet6(n_input_periods=n_input_periods,
                                                             n_output_periods=n_output_periods,
                                                             node_features=node_features,
                                                             n_nodes=n_nodes, 
                                                             hidden_warmup=hidden_warmup,
                                                             edge_index_list=edge_index_list,
                                                             model_path=model_path,
                                                           )
        ## Load state dicts
        self.model_outflow_from_speed.load_state_dict(torch.load("../models/" + model_path[8] + ".pth"))
        ## Freeze parameters of model_dp_trained_outflow
        if no_backprop_models:
            for param in self.model_outflow_from_speed.parameters():
                param.requires_grad = False
            
        # Warmed up speed prediction GNN
        self.model_speed = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                            num_for_predict = n_output_periods,
                                                            in_channels = 1, # No. of features
                                                            num_of_vertices = n_nodes[0],
                                                            nb_block = 2,
                                                            K = 3,
                                                            nb_chev_filter = 64,
                                                            nb_time_filter = 64,
                                                            time_strides = 1,
                                                            hidden = hidden_warmup[0],
                                                            edge_index=edge_index_list[0],
                                                         ) 
        ## Load state dicts of speed prediction
        self.model_speed.load_state_dict(torch.load("../models/" + model_path[0] + ".pth"))
        ## Freeze parameters of model_dp_trained_outflow
        if no_backprop_models:
            for param in self.model_speed.parameters():
                param.requires_grad = False
        # Linear model, Map speed graph to inflow/outflow graph
        self.linear = torch.nn.Linear(n_output_periods * n_nodes[0], n_output_periods * n_nodes[1]) 
        
        # GNN of the company itself 
        self.model_gnn = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = 3, # No. of input features
                                                        num_of_vertices = n_nodes[1],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[1],
                                                        edge_index=edge_index_list[1],
                                                    ) 
        self.n_node = n_nodes[1]
        
    def forward(self, x):
        # S -> I, DP model
        o_inflow_other_comapny = self.model_inflow_from_speed(x[0])
        o_inflow_other_comapny_reshaped = o_inflow_other_comapny.reshape(o_inflow_other_comapny.shape[0], self.n_node, 1, -1)
        # S -> O, DP model
        o_outflow_other_comapny = self.model_outflow_from_speed(x[0])
        o_outflow_other_comapny_reshaped = o_outflow_other_comapny.reshape(o_outflow_other_comapny.shape[0], self.n_node, 1, -1)
        # S 
        o_speed = self.model_speed(x[0])
        o_speed_flattened = o_speed.view(o_speed.shape[0], 1, 1, -1)
        o_speed_mapped = self.linear(o_speed_flattened)
        o_speed_reshaped = o_speed_mapped.reshape(o_speed_mapped.shape[0], self.n_node, 1, -1)
        # GNN
        pool_inflow = x[1] + o_inflow_other_comapny_reshaped
        pool_outflow = x[2] + o_outflow_other_comapny_reshaped
        input_gnn = torch.cat((pool_inflow,
                               pool_outflow,
                               o_speed_reshaped,
                              ),
                              dim = 2)
        output_gnn = self.model_gnn(input_gnn)
#         for node in range(59):
#             print(f"\n*---------------------------- NODE = {node} ---------------------------------*")
#             print(f"inflow other = {o_inflow_other_comapny_reshaped[0, node, :, :]}")
#             print(f"inflow company = {x[1][0, node, :, :]}")
#             print(f"pool_inflow = {pool_inflow[0, node, :, :]}")
#             print(f"outflow other = {o_outflow_other_comapny_reshaped[0, node, :, :]}")
#             print(f"outflow company = {x[2][0, node, :, :]}")
#             print(f"pool_outflow = {pool_outflow[0, node, :, :]}")
#             print(f"o_speed_reshaped[0, {node}, :, :] = {o_speed_reshaped[0, node, :, :]}")
#             print(f"output_gnn = {output_gnn[0, node, :]}")
#         assert 1 == 2, print("End")
        return output_gnn 
    
    
# Two-company scenario, Input: Inflow of one company + Outflow of one company + Speed, Prediction: inflow. Inflow/Outflow prediction GNNs have DP training. 
class MA_3in_pool_sum(torch.nn.Module): 
    def __init__(self, 
                 n_input_periods,
                 n_output_periods,
                 node_features,
                 n_nodes, 
                 hidden_warmup,
                 edge_index_list,
                 model_path,
                 no_backprop_models,
                ):
        super(MA_3in_pool_sum, self).__init__()
                    
        # DP inflow prediction GNN from Speed
        self.model_inflow_from_speed = MultiPredictionNet5(n_input_periods=n_input_periods,
                                                                     n_output_periods=n_output_periods,
                                                                     node_features=node_features,
                                                                     n_nodes=n_nodes, 
                                                                     hidden_warmup=hidden_warmup,
                                                                     edge_index_list=edge_index_list,
                                                                     model_path=model_path,
                                                                   )
        ## Load state dicts
        self.model_inflow_from_speed.load_state_dict(torch.load("../models/" + model_path[7] + ".pth"))
        ## Freeze parameters of model_dp_trained_outflow
        if no_backprop_models:
            for param in self.model_inflow_from_speed.parameters():
                param.requires_grad = False
            
        # DP outflow prediction GNN from Speed
        self.model_outflow_from_speed = MultiPredictionNet6(n_input_periods=n_input_periods,
                                                             n_output_periods=n_output_periods,
                                                             node_features=node_features,
                                                             n_nodes=n_nodes, 
                                                             hidden_warmup=hidden_warmup,
                                                             edge_index_list=edge_index_list,
                                                             model_path=model_path,
                                                           )
        ## Load state dicts
        self.model_outflow_from_speed.load_state_dict(torch.load("../models/" + model_path[8] + ".pth"))
        ## Freeze parameters of model_dp_trained_outflow
        if no_backprop_models:
            for param in self.model_outflow_from_speed.parameters():
                param.requires_grad = False
            
        # GNN of the company itself 
        self.model_gnn = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = 2, # No. of input features
                                                        num_of_vertices = n_nodes[1],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[1],
                                                        edge_index=edge_index_list[1],
                                                    ) 
        self.n_node = n_nodes[1]
        
    def forward(self, x):
        # S -> I, DP model
        o_inflow_other_comapny = self.model_inflow_from_speed(x[0])
        o_inflow_other_comapny_reshaped = o_inflow_other_comapny.reshape(o_inflow_other_comapny.shape[0], self.n_node, 1, -1)
        # S -> O, DP model
        o_outflow_other_comapny = self.model_outflow_from_speed(x[0])
        o_outflow_other_comapny_reshaped = o_outflow_other_comapny.reshape(o_outflow_other_comapny.shape[0], self.n_node, 1, -1)
        # GNN
        pool_inflow = x[1] + o_inflow_other_comapny_reshaped
        pool_outflow = x[2] + o_outflow_other_comapny_reshaped
        input_gnn = torch.cat((pool_inflow,
                               pool_outflow,
                              ),
                              dim = 2)
        output_gnn = self.model_gnn(input_gnn)
#         for node in range(59):
#             print(f"\n*---------------------------- NODE = {node} ---------------------------------*")
#             print(f"o_inflow_other_comapny_reshaped[0, {node}, :, :] = {o_inflow_other_comapny_reshaped[0, node, :, :]}")
#             print(f"o_outflow_other_comapny_reshaped[0, {node}, :, :] = {o_outflow_other_comapny_reshaped[0, node, :, :]}")
#             print(f"x[1][0, {node}, :, :] = {x[1][0, node, :, :]}")
#             print(f"x[2][0, {node}, :, :] = {x[2][0, node, :, :]}")
#             print(f"pool_inflow[0, {node}, :, :] = {pool_inflow[0, 0, :, :]}")
#             print(f"pool_outflow[0, {node}, :, :] = {pool_outflow[0, 0, :, :]}")
#             print(f"output_gnn[0, {node}, :] = {output_gnn[0, node, :]}")
#         assert 1 == 2, print("End")
        return output_gnn 
    
    
# Two-company scenario, Input: Inflow of one company + Outflow of one company + Speed, Prediction: inflow. Inflow/Outflow prediction GNNs have DP training. 
class MA_5in_speed_pool_sum_baseline(torch.nn.Module): 
    def __init__(self, 
                 n_input_periods,
                 n_output_periods,
                 node_features,
                 n_nodes, 
                 hidden_warmup,
                 edge_index_list,
                 model_path,
                 no_backprop_s_models,
                ):
        super(MA_5in_speed_pool_sum_baseline, self).__init__()
                    
        # Warmed up speed prediction GNN
        self.model_speed = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                            num_for_predict = n_output_periods,
                                                            in_channels = 1, # No. of features
                                                            num_of_vertices = n_nodes[0],
                                                            nb_block = 2,
                                                            K = 3,
                                                            nb_chev_filter = 64,
                                                            nb_time_filter = 64,
                                                            time_strides = 1,
                                                            hidden = hidden_warmup[0],
                                                            edge_index=edge_index_list[0],
                                                         ) 
        ## Load state dicts of speed prediction
        self.model_speed.load_state_dict(torch.load("../models/" + model_path[0] + ".pth"))
        ## Freeze parameters of model_dp_trained_outflow
        if no_backprop_s_models:
            for param in self.model_speed.parameters():
                param.requires_grad = False
        # Linear model, Map speed graph to inflow/outflow graph
        self.linear = torch.nn.Linear(n_output_periods * n_nodes[0], n_output_periods * n_nodes[1]) 
        
        # GNN of the company itself 
        self.model_gnn = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = 3, # No. of input features
                                                        num_of_vertices = n_nodes[1],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[1],
                                                        edge_index=edge_index_list[1],
                                                    ) 
        self.n_node = n_nodes[1]
        
    def forward(self, x):
        # S 
        o_speed = self.model_speed(x[0])
        o_speed_flattened = o_speed.view(o_speed.shape[0], 1, 1, -1)
        o_speed_mapped = self.linear(o_speed_flattened)
        o_speed_reshaped = o_speed_mapped.reshape(o_speed_mapped.shape[0], self.n_node, 1, -1)
        # GNN
        pool_inflow = x[1] + x[3]
        pool_outflow = x[2] + x[4]
        input_gnn = torch.cat((pool_inflow,
                               pool_outflow,
                               o_speed_reshaped,
                              ),
                              dim = 2)
        output_gnn = self.model_gnn(input_gnn)
        return output_gnn 
    
    

# Two-company scenario, Input: Inflow of one company + Outflow of one company + Speed, Prediction: inflow. Inflow/Outflow prediction GNNs have DP training. 
class MA_5in_pool_sum_baseline(torch.nn.Module): 
    def __init__(self, 
                 n_input_periods,
                 n_output_periods,
                 node_features,
                 n_nodes, 
                 hidden_warmup,
                 edge_index_list,
                 model_path,
                ):
        super(MA_5in_speed_pool_sum_baseline, self).__init__()
        
        # GNN of the company itself 
        self.model_gnn = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = 2, # No. of input features
                                                        num_of_vertices = n_nodes[1],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[1],
                                                        edge_index=edge_index_list[1],
                                                    ) 
        self.n_node = n_nodes[1]
        
    def forward(self, x):
        # GNN
        pool_inflow = x[1] + x[3]
        pool_outflow = x[2] + x[4]
        input_gnn = torch.cat((pool_inflow,
                               pool_outflow,
                              ),
                              dim = 2)
        output_gnn = self.model_gnn(input_gnn)
        return output_gnn 
    
    
# Two-company scenario, Input: Inflow of one company + Outflow of one company + Speed, Prediction: inflow. Inflow/Outflow prediction GNNs have DP training. 
class MA_3in_speed_extra_pool_mean(torch.nn.Module): 
    def __init__(self, 
                 n_input_periods,
                 n_output_periods,
                 node_features,
                 n_nodes, 
                 hidden_warmup,
                 edge_index_list,
                 model_path,
                 no_backprop_models,
                ):
        super(MA_3in_speed_extra_pool_mean, self).__init__()
                    
        # DP inflow prediction GNN from Speed
        self.model_inflow_from_speed = MultiPredictionNet5(n_input_periods=n_input_periods,
                                                                     n_output_periods=n_output_periods,
                                                                     node_features=node_features,
                                                                     n_nodes=n_nodes, 
                                                                     hidden_warmup=hidden_warmup,
                                                                     edge_index_list=edge_index_list,
                                                                     model_path=model_path,
                                                                   )
        ## Load state dicts
        self.model_inflow_from_speed.load_state_dict(torch.load("../models/" + model_path[7] + ".pth"))
        ## Freeze parameters of model_dp_trained_outflow
        if no_backprop_models:
            for param in self.model_inflow_from_speed.parameters():
                param.requires_grad = False
            
        # DP outflow prediction GNN from Speed
        self.model_outflow_from_speed = MultiPredictionNet6(n_input_periods=n_input_periods,
                                                             n_output_periods=n_output_periods,
                                                             node_features=node_features,
                                                             n_nodes=n_nodes, 
                                                             hidden_warmup=hidden_warmup,
                                                             edge_index_list=edge_index_list,
                                                             model_path=model_path,
                                                           )
        ## Load state dicts
        self.model_outflow_from_speed.load_state_dict(torch.load("../models/" + model_path[8] + ".pth"))
        ## Freeze parameters of model_dp_trained_outflow
        if no_backprop_models:
            for param in self.model_outflow_from_speed.parameters():
                param.requires_grad = False
            
        # Warmed up speed prediction GNN
        self.model_speed = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                            num_for_predict = n_output_periods,
                                                            in_channels = 1, # No. of features
                                                            num_of_vertices = n_nodes[0],
                                                            nb_block = 2,
                                                            K = 3,
                                                            nb_chev_filter = 64,
                                                            nb_time_filter = 64,
                                                            time_strides = 1,
                                                            hidden = hidden_warmup[0],
                                                            edge_index=edge_index_list[0],
                                                         ) 
        ## Load state dicts of speed prediction
        self.model_speed.load_state_dict(torch.load("../models/" + model_path[0] + ".pth"))
        ## Freeze parameters of model_dp_trained_outflow
        if no_backprop_models:
            for param in self.model_speed.parameters():
                param.requires_grad = False
        # Linear model, Map speed graph to inflow/outflow graph
        self.linear = torch.nn.Linear(n_output_periods * n_nodes[0], n_output_periods * n_nodes[1]) 
        
        # GNN of the company itself 
        self.model_gnn = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = 3, # No. of input features
                                                        num_of_vertices = n_nodes[1],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[1],
                                                        edge_index=edge_index_list[1],
                                                    ) 
        self.n_node = n_nodes[1]
        
    def forward(self, x):
        # S -> I, DP model
        o_inflow_other_comapny = self.model_inflow_from_speed(x[0])
        o_inflow_other_comapny_reshaped = o_inflow_other_comapny.reshape(o_inflow_other_comapny.shape[0], self.n_node, 1, -1)
        # S -> O, DP model
        o_outflow_other_comapny = self.model_outflow_from_speed(x[0])
        o_outflow_other_comapny_reshaped = o_outflow_other_comapny.reshape(o_outflow_other_comapny.shape[0], self.n_node, 1, -1)
        # S 
        o_speed = self.model_speed(x[0])
        o_speed_flattened = o_speed.view(o_speed.shape[0], 1, 1, -1)
        o_speed_mapped = self.linear(o_speed_flattened)
        o_speed_reshaped = o_speed_mapped.reshape(o_speed_mapped.shape[0], self.n_node, 1, -1)
        # GNN
        pool_inflow = (x[1] + o_inflow_other_comapny_reshaped)/2
        pool_outflow = (x[2] + o_outflow_other_comapny_reshaped)/2
        input_gnn = torch.cat((pool_inflow,
                               pool_outflow,
                               o_speed_reshaped,
                              ),
                              dim = 2)
        output_gnn = self.model_gnn(input_gnn)
        return output_gnn 
    
# Two-company scenario, Input: Inflow of one company + Outflow of one company + Speed, Prediction: inflow. Inflow/Outflow prediction GNNs have DP training. 
class MA_3in_speed_extra_plus(torch.nn.Module): 
    def __init__(self, 
                 n_input_periods,
                 n_output_periods,
                 node_features,
                 n_nodes, 
                 hidden_warmup,
                 edge_index_list,
                 model_path,
                 no_backprop_dp_models,
                ):
        super(MA_3in_speed_extra_plus, self).__init__()
                    
        # DP inflow prediction GNN from Speed
        self.model_inflow_from_speed = MultiPredictionNet5(n_input_periods=n_input_periods,
                                                                     n_output_periods=n_output_periods,
                                                                     node_features=node_features,
                                                                     n_nodes=n_nodes, 
                                                                     hidden_warmup=hidden_warmup,
                                                                     edge_index_list=edge_index_list,
                                                                     model_path=model_path,
                                                                   )
        ## Load state dicts
        self.model_inflow_from_speed.load_state_dict(torch.load("../models/" + model_path[7] + ".pth"))
        ## Freeze parameters of model_dp_trained_outflow
        if no_backprop_dp_models:
            for param in self.model_inflow_from_speed.parameters():
                param.requires_grad = False
            
        # DP outflow prediction GNN from Speed
        self.model_outflow_from_speed = MultiPredictionNet6(n_input_periods=n_input_periods,
                                                             n_output_periods=n_output_periods,
                                                             node_features=node_features,
                                                             n_nodes=n_nodes, 
                                                             hidden_warmup=hidden_warmup,
                                                             edge_index_list=edge_index_list,
                                                             model_path=model_path,
                                                           )
        ## Load state dicts
        self.model_outflow_from_speed.load_state_dict(torch.load("../models/" + model_path[8] + ".pth"))
        ## Freeze parameters of model_dp_trained_outflow
        if no_backprop_dp_models:
            for param in self.model_outflow_from_speed.parameters():
                param.requires_grad = False
                    
        # DP inflow prediction GNN from Speed
        self.model_inflow_from_speed2 = MultiPredictionNet5(n_input_periods=n_input_periods,
                                                                     n_output_periods=n_output_periods,
                                                                     node_features=node_features,
                                                                     n_nodes=n_nodes, 
                                                                     hidden_warmup=hidden_warmup,
                                                                     edge_index_list=edge_index_list,
                                                                     model_path=model_path,
                                                                   )
        ## Load state dicts
        self.model_inflow_from_speed2.load_state_dict(torch.load("../models/" + model_path[9] + ".pth"))
        ## Freeze parameters of model_dp_trained_outflow
        if no_backprop_dp_models:
            for param in self.model_inflow_from_speed2.parameters():
                param.requires_grad = False
            
        # DP outflow prediction GNN from Speed
        self.model_outflow_from_speed2 = MultiPredictionNet6(n_input_periods=n_input_periods,
                                                             n_output_periods=n_output_periods,
                                                             node_features=node_features,
                                                             n_nodes=n_nodes, 
                                                             hidden_warmup=hidden_warmup,
                                                             edge_index_list=edge_index_list,
                                                             model_path=model_path,
                                                           )
        ## Load state dicts
        self.model_outflow_from_speed2.load_state_dict(torch.load("../models/" + model_path[10] + ".pth"))
        ## Freeze parameters of model_dp_trained_outflow
        if no_backprop_dp_models:
            for param in self.model_outflow_from_speed2.parameters():
                param.requires_grad = False
            
        # GNN of the company itself 
        self.model_gnn = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = 6, # No. of features
                                                        num_of_vertices = n_nodes[1],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[1],
                                                        edge_index=edge_index_list[1],
                                                    ) 
        self.n_node = n_nodes[1]
        
    def forward(self, x):
        # S -> I, DP model
        o_inflow_other_comapny = self.model_inflow_from_speed(x[0])
        o_inflow_other_comapny_reshaped = o_inflow_other_comapny.reshape(o_inflow_other_comapny.shape[0], self.n_node, 1, -1)
        # S -> O, DP model
        o_outflow_other_comapny = self.model_outflow_from_speed(x[0])
        o_outflow_other_comapny_reshaped = o_outflow_other_comapny.reshape(o_outflow_other_comapny.shape[0], self.n_node, 1, -1)
        # S -> I, DP model
        o_inflow_comapny = self.model_inflow_from_speed2(x[0])
        o_inflow_comapny_reshaped = o_inflow_comapny.reshape(o_inflow_comapny.shape[0], self.n_node, 1, -1)
        # S -> O, DP model
        o_outflow_comapny = self.model_outflow_from_speed2(x[0])
        o_outflow_comapny_reshaped = o_outflow_comapny.reshape(o_outflow_comapny.shape[0], self.n_node, 1, -1)
        # GNN
        input_gnn = torch.cat((o_inflow_other_comapny_reshaped,
                               o_outflow_other_comapny_reshaped,
                               o_inflow_comapny_reshaped,
                               o_outflow_comapny_reshaped,
                               x[1],
                               x[2],
                              ),
                              dim = 2)
        output_gnn = self.model_gnn(input_gnn)
        return output_gnn 
    
    
    
    
    
# Two-company scenario, Input: Inflow of one company + Outflow of one company + Speed, Prediction: inflow. Inflow/Outflow prediction GNNs have DP training. 
class MA_3in_warm_speed_linear(torch.nn.Module): 
    def __init__(self, 
                 n_input_periods,
                 n_output_periods,
                 node_features,
                 n_nodes, 
                 hidden_warmup,
                 edge_index_list,
                 model_path,
                 no_backprop_nondp_models,
                 no_backprop_dp_models,
                ):
        super(MA_3in_warm_speed_linear, self).__init__()
                    
        # DP inflow prediction GNN from Speed
        self.model_inflow_from_speed = MultiPredictionNet5(n_input_periods=n_input_periods,
                                                                     n_output_periods=n_output_periods,
                                                                     node_features=node_features,
                                                                     n_nodes=n_nodes, 
                                                                     hidden_warmup=hidden_warmup,
                                                                     edge_index_list=edge_index_list,
                                                                     model_path=model_path,
                                                                   )
        ## Load state dicts
        self.model_inflow_from_speed.load_state_dict(torch.load("../models/" + model_path[7] + ".pth"))
        ## Freeze parameters of model_dp_trained_outflow
        if no_backprop_dp_models:
            for param in self.model_inflow_from_speed.parameters():
                param.requires_grad = False
            
        # DP outflow prediction GNN from Speed
        self.model_outflow_from_speed = MultiPredictionNet6(n_input_periods=n_input_periods,
                                                             n_output_periods=n_output_periods,
                                                             node_features=node_features,
                                                             n_nodes=n_nodes, 
                                                             hidden_warmup=hidden_warmup,
                                                             edge_index_list=edge_index_list,
                                                             model_path=model_path,
                                                           )
        ## Load state dicts
        self.model_outflow_from_speed.load_state_dict(torch.load("../models/" + model_path[8] + ".pth"))
        ## Freeze parameters of model_dp_trained_outflow
        if no_backprop_dp_models:
            for param in self.model_outflow_from_speed.parameters():
                param.requires_grad = False
            
        # GNN of the company itself 
        self.model_gnn = MA_2in(n_input_periods,
                                n_output_periods,
                                node_features,
                                n_nodes, 
                                hidden_warmup,
                                edge_index_list,
                                model_path,
                               ) 
        ## Load state dicts
        self.model_gnn.load_state_dict(torch.load("../models/" + model_path[9] + ".pth"))
        ## Freeze parameters of model_dp_trained_outflow
        if no_backprop_nondp_models:
            for param in self.model_gnn.parameters():
                param.requires_grad = False
        
                
        # Linear model
        self.linear = torch.nn.Linear(n_output_periods * (n_nodes[1] * 3), 
                                                n_output_periods * (n_nodes[1])) 
        
    def forward(self, x):
        # S -> I, DP model
        o_inflow_other_comapny = self.model_inflow_from_speed(x[0])
        o_inflow_other_comapny_flattened = o_inflow_other_comapny.view(o_inflow_other_comapny.shape[0], 1, 1, -1)
        # S -> O, DP model
        o_outflow_other_comapny = self.model_outflow_from_speed(x[0])
        o_outflow_other_comapny_flattened = o_outflow_other_comapny.view(o_outflow_other_comapny.shape[0], 1, 1, -1)
        # GNN of company itself
        o1 = self.model_gnn([x[1], x[2]])
        o1_flattened = o1.view(o1.shape[0], 1, 1, -1)
        # Concatenate inputs of the linear layer
        o_cat = torch.concat((o1_flattened, 
                              o_inflow_other_comapny_flattened,
                              o_outflow_other_comapny_flattened,
                             ),
                             dim = -1)
        output = self.linear(F.relu(o_cat))
        return output 
    
    
    
# Two-company scenario, Input: Inflow of one company + Outflow of one company + Speed, Prediction: inflow. Inflow/Outflow prediction GNNs have DP training. 
class MA_3in_warm_speed(torch.nn.Module): 
    def __init__(self, 
                 n_input_periods,
                 n_output_periods,
                 node_features,
                 n_nodes, 
                 hidden_warmup,
                 edge_index_list,
                 model_path,
                 no_backprop_nondp_models,
                 no_backprop_dp_models,
                ):
        super(MA_3in_warm_speed, self).__init__()
                    
        # DP inflow prediction GNN from Speed
        self.model_inflow_from_speed = MultiPredictionNet5(n_input_periods=n_input_periods,
                                                                     n_output_periods=n_output_periods,
                                                                     node_features=node_features,
                                                                     n_nodes=n_nodes, 
                                                                     hidden_warmup=hidden_warmup,
                                                                     edge_index_list=edge_index_list,
                                                                     model_path=model_path,
                                                                   )
        ## Load state dicts
        self.model_inflow_from_speed.load_state_dict(torch.load("../models/" + model_path[7] + ".pth"))
        ## Freeze parameters of model_dp_trained_outflow
        if no_backprop_dp_models:
            for param in self.model_inflow_from_speed.parameters():
                param.requires_grad = False
            
        # DP outflow prediction GNN from Speed
        self.model_outflow_from_speed = MultiPredictionNet6(n_input_periods=n_input_periods,
                                                             n_output_periods=n_output_periods,
                                                             node_features=node_features,
                                                             n_nodes=n_nodes, 
                                                             hidden_warmup=hidden_warmup,
                                                             edge_index_list=edge_index_list,
                                                             model_path=model_path,
                                                           )
        ## Load state dicts
        self.model_outflow_from_speed.load_state_dict(torch.load("../models/" + model_path[8] + ".pth"))
        ## Freeze parameters of model_dp_trained_outflow
        if no_backprop_dp_models:
            for param in self.model_outflow_from_speed.parameters():
                param.requires_grad = False
            
        # GNN of the company itself 
        self.model_gnn = MA_2in(n_input_periods,
                                n_output_periods,
                                node_features,
                                n_nodes, 
                                hidden_warmup,
                                edge_index_list,
                                model_path,
                               ) 
        ## Load state dicts
        self.model_gnn.load_state_dict(torch.load("../models/" + model_path[9] + ".pth"))
        ## Freeze parameters of model_dp_trained_outflow
        if no_backprop_nondp_models:
            for param in self.model_gnn.parameters():
                param.requires_grad = False
                
                
        # GNN as the final layer 
        self.model_gnn_final_layer = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = 3, # No. of features
                                                        num_of_vertices = n_nodes[1],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[1],
                                                        edge_index=edge_index_list[1],
                                                    ) 
        self.n_node = n_nodes[1]
        
    def forward(self, x):
        # S -> I, DP model
        o_inflow_other_comapny = self.model_inflow_from_speed(x[0])
#         o_inflow_other_comapny_unsqueezed = torch.unsqueeze(o_inflow_other_comapny, 2)
        o_inflow_other_comapny_reshaped = o_inflow_other_comapny.reshape(o_inflow_other_comapny.shape[0], self.n_node, 1, -1)
#         print("\n\n\n\n")
#         print("o_inflow_other_comapny:", o_inflow_other_comapny.shape)
        # S -> O, DP model
        o_outflow_other_comapny = self.model_outflow_from_speed(x[0])
#         o_outflow_other_comapny_unsqueezed = torch.unsqueeze(o_outflow_other_comapny, 2)
        o_outflow_other_comapny_reshaped = o_inflow_other_comapny.reshape(o_outflow_other_comapny.shape[0], self.n_node, 1, -1)
#         print("o_outflow_other_comapny_unsqueezed:", o_outflow_other_comapny_unsqueezed.shape)
        # GNN of company itself
        o1 = self.model_gnn([x[1], x[2]])
        o1_unsqueezed = torch.unsqueeze(o1, 2)
#         o1_other_comapny_reshaped = o1.reshape(o1.shape[0], self.n_node, 1, -1)
#         print("o1:", o1.shape)
        # Concatenate inputs of the linear layer
        o_cat = torch.concat((o1_unsqueezed, 
                              o_inflow_other_comapny_reshaped,
                              o_outflow_other_comapny_reshaped,
                             ),
                             dim = 2)
        output = self.model_gnn_final_layer(o_cat)
        return output 
    
    
# Two-company scenario, Input: Inflow of one company + Outflow of one company + Speed, Prediction: inflow. Inflow/Outflow prediction GNNs have DP training. 
class MA_4in_gy_linear(torch.nn.Module): 
    def __init__(self, 
                 n_input_periods,
                 n_output_periods,
                 node_features,
                 n_nodes, 
                 hidden_warmup,
                 edge_index_list,
                 model_path,
                ):
        super(MA_4in_gy_linear, self).__init__()
                    
        # DP inflow prediction GNN from gy inflow
        self.model_inflow_from_gy_inflow = MP0_i(n_input_periods=n_input_periods,
                                             n_output_periods=n_output_periods,
                                             node_features=node_features,
                                             n_nodes=n_nodes, 
                                             hidden_warmup=hidden_warmup,
                                             edge_index_list=edge_index_list,
                                             model_path=model_path,
                                             train_mode = False,
                                            )
        ## Load state dicts
        self.model_inflow_from_gy_inflow.load_state_dict(torch.load("../models/" + model_path[7] + ".pth"))
        ## Freeze parameters of model_dp_trained_outflow
        for param in self.model_inflow_from_gy_inflow.parameters():
            param.requires_grad = False
            
        # DP outflow prediction GNN from Speed
        self.model_outflow_from_gy_outflow = MP0_o(n_input_periods=n_input_periods,
                                                  n_output_periods=n_output_periods,
                                                  node_features=node_features,
                                                  n_nodes=n_nodes, 
                                                  hidden_warmup=hidden_warmup,
                                                  edge_index_list=edge_index_list,
                                                  model_path=model_path,
                                                  train_mode = False,
                                                 )
        ## Load state dicts
        self.model_outflow_from_gy_outflow.load_state_dict(torch.load("../models/" + model_path[8] + ".pth"))
        ## Freeze parameters of model_dp_trained_outflow
        for param in self.model_outflow_from_gy_outflow.parameters():
            param.requires_grad = False
            
        # GNN of the company itself 
        self.model_gnn = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = 2, # No. of features
                                                        num_of_vertices = n_nodes[1],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[1],
                                                        edge_index=edge_index_list[1],
                                                    ) 
        
                
        # Linear model
        self.linear = torch.nn.Linear(n_output_periods * (n_nodes[1] * 3), 
                                                n_output_periods * (n_nodes[1])) 
        
    def forward(self, x):
        # GNN of company itself
        input_gnn = torch.cat((x[0],
                               x[1],
                              ),
                              dim = 2)
        o1 = self.model_gnn(input_gnn)
        o1_flattened = o1.view(o1.shape[0], 1, 1, -1)
        # GY Inflow -> I, DP model
        o_inflow_other_comapny = self.model_inflow_from_gy_inflow([x[2]])
        o_inflow_other_comapny_flattened = o_inflow_other_comapny.view(o_inflow_other_comapny.shape[0], 1, 1, -1)
        # GY Outflow -> O, DP model
        o_outflow_other_comapny = self.model_outflow_from_gy_outflow([x[3]])
        o_outflow_other_comapny_flattened = o_outflow_other_comapny.view(o_outflow_other_comapny.shape[0], 1, 1, -1)
        # Concatenate inputs of the linear layer
        o_cat = torch.concat((o1_flattened, 
                              o_inflow_other_comapny_flattened,
                              o_outflow_other_comapny_flattened,
                             ),
                             dim = -1)
        output = self.linear(F.relu(o_cat))
        return output 
    
    
    
    
    
# Two-company scenario, Input: Inflow of one company + Outflow of one company + Speed, Prediction: inflow. Inflow/Outflow prediction GNNs have DP training. 
class MA_4in_warm_gy_linear(torch.nn.Module): 
    def __init__(self, 
                 n_input_periods,
                 n_output_periods,
                 node_features,
                 n_nodes, 
                 hidden_warmup,
                 edge_index_list,
                 model_path,
                 no_backprop_nondp_models,
                 no_backprop_dp_models,
                ):
        super(MA_4in_warm_gy_linear, self).__init__()
                    
        # DP inflow prediction GNN from gy inflow
        self.model_inflow_from_gy_inflow = MP0_i(n_input_periods=n_input_periods,
                                             n_output_periods=n_output_periods,
                                             node_features=node_features,
                                             n_nodes=n_nodes, 
                                             hidden_warmup=hidden_warmup,
                                             edge_index_list=edge_index_list,
                                             model_path=model_path,
                                             train_mode = False,
                                            )
        ## Load state dicts
        self.model_inflow_from_gy_inflow.load_state_dict(torch.load("../models/" + model_path[7] + ".pth"))
        ## Freeze parameters of model_dp_trained_outflow
        if no_backprop_dp_models:
            for param in self.model_inflow_from_gy_inflow.parameters():
                param.requires_grad = False
            
        # DP outflow prediction GNN from Speed
        self.model_outflow_from_gy_outflow = MP0_o(n_input_periods=n_input_periods,
                                                  n_output_periods=n_output_periods,
                                                  node_features=node_features,
                                                  n_nodes=n_nodes, 
                                                  hidden_warmup=hidden_warmup,
                                                  edge_index_list=edge_index_list,
                                                  model_path=model_path,
                                                  train_mode = False,
                                                 )
        ## Load state dicts
        self.model_outflow_from_gy_outflow.load_state_dict(torch.load("../models/" + model_path[8] + ".pth"))
        ## Freeze parameters of model_dp_trained_outflow
        if no_backprop_dp_models:
            for param in self.model_outflow_from_gy_outflow.parameters():
                param.requires_grad = False
            
        # GNN of the company itself 
        self.model_gnn = MA_2in(n_input_periods,
                                n_output_periods,
                                node_features,
                                n_nodes, 
                                hidden_warmup,
                                edge_index_list,
                                model_path,
                               ) 
        ## Load state dicts
        self.model_gnn.load_state_dict(torch.load("../models/" + model_path[9] + ".pth"))
        ## Freeze parameters of model_dp_trained_outflow
        if no_backprop_nondp_models:
            for param in self.model_gnn.parameters():
                param.requires_grad = False
        
        
                
        # Linear model
        self.linear = torch.nn.Linear(n_output_periods * (n_nodes[1] * 3), 
                                                n_output_periods * (n_nodes[1])) 
        
    def forward(self, x):
        # GNN of company itself
        o1 = self.model_gnn([x[0], x[1]])
        o1_flattened = o1.view(o1.shape[0], 1, 1, -1)
        # GY Inflow -> I, DP model
        o_inflow_other_comapny = self.model_inflow_from_gy_inflow([x[2]])
        o_inflow_other_comapny_flattened = o_inflow_other_comapny.view(o_inflow_other_comapny.shape[0], 1, 1, -1)
        # GY Outflow -> O, DP model
        o_outflow_other_comapny = self.model_outflow_from_gy_outflow([x[3]])
        o_outflow_other_comapny_flattened = o_outflow_other_comapny.view(o_outflow_other_comapny.shape[0], 1, 1, -1)
        # Concatenate inputs of the linear layer
        o_cat = torch.concat((o1_flattened, 
                              o_inflow_other_comapny_flattened,
                              o_outflow_other_comapny_flattened,
                             ),
                             dim = -1)
        output = self.linear(F.relu(o_cat))
        return output 
    
    
    
    
# Two-company scenario, Input: Inflow of one company + Outflow of one company + Speed, Prediction: inflow. Inflow/Outflow prediction GNNs have DP training. 
class MA_4in_warm_gy(torch.nn.Module): 
    def __init__(self, 
                 n_input_periods,
                 n_output_periods,
                 node_features,
                 n_nodes, 
                 hidden_warmup,
                 edge_index_list,
                 model_path,
                 no_backprop_nondp_models,
                 no_backprop_dp_models,
                ):
        super(MA_4in_warm_gy, self).__init__()
                    
        # DP inflow prediction GNN from gy inflow
        self.model_inflow_from_gy_inflow = MP0_i(n_input_periods=n_input_periods,
                                             n_output_periods=n_output_periods,
                                             node_features=node_features,
                                             n_nodes=n_nodes, 
                                             hidden_warmup=hidden_warmup,
                                             edge_index_list=edge_index_list,
                                             model_path=model_path,
                                             train_mode = False,
                                            )
        ## Load state dicts
        self.model_inflow_from_gy_inflow.load_state_dict(torch.load("../models/" + model_path[7] + ".pth"))
        ## Freeze parameters of model_dp_trained_outflow
        if no_backprop_dp_models:
            for param in self.model_inflow_from_gy_inflow.parameters():
                param.requires_grad = False
            
        # DP outflow prediction GNN from Speed
        self.model_outflow_from_gy_outflow = MP0_o(n_input_periods=n_input_periods,
                                                  n_output_periods=n_output_periods,
                                                  node_features=node_features,
                                                  n_nodes=n_nodes, 
                                                  hidden_warmup=hidden_warmup,
                                                  edge_index_list=edge_index_list,
                                                  model_path=model_path,
                                                  train_mode = False,
                                                 )
        ## Load state dicts
        self.model_outflow_from_gy_outflow.load_state_dict(torch.load("../models/" + model_path[8] + ".pth"))
        ## Freeze parameters of model_dp_trained_outflow
        if no_backprop_dp_models:
            for param in self.model_outflow_from_gy_outflow.parameters():
                param.requires_grad = False
            
        # GNN of the company itself 
        self.model_gnn = MA_2in(n_input_periods,
                                n_output_periods,
                                node_features,
                                n_nodes, 
                                hidden_warmup,
                                edge_index_list,
                                model_path,
                               ) 
        ## Load state dicts
        self.model_gnn.load_state_dict(torch.load("../models/" + model_path[9] + ".pth"))
        ## Freeze parameters of model_dp_trained_outflow
        if no_backprop_nondp_models:
            for param in self.model_gnn.parameters():
                param.requires_grad = False
        
                
        # GNN as the final layer 
        self.model_gnn_final_layer = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = 3, # No. of features
                                                        num_of_vertices = n_nodes[1],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[1],
                                                        edge_index=edge_index_list[1],
                                                    ) 
         
        
    def forward(self, x):
        # GNN of company itself
        o1 = self.model_gnn([x[0], x[1]])
        o1_unsqueezed = torch.unsqueeze(o1, 2)
        # GY Inflow -> I, DP model
        o_inflow_other_comapny = self.model_inflow_from_gy_inflow([x[2]])
        o_inflow_other_comapny_unsqueezed = torch.unsqueeze(o_inflow_other_comapny, 2)
        # GY Outflow -> O, DP model
        o_outflow_other_comapny = self.model_outflow_from_gy_outflow([x[3]])
        o_outflow_other_comapny_unsqueezed = torch.unsqueeze(o_outflow_other_comapny, 2)
        # Concatenate inputs of the linear layer
        o_cat = torch.concat((o1_unsqueezed, 
                              o_inflow_other_comapny_unsqueezed,
                              o_outflow_other_comapny_unsqueezed,
                             ),
                             dim = 2)
        output = self.model_gnn_final_layer(o_cat)
        return output 
    
    
# Two-company scenario, Input: Inflow of one company + Outflow of one company + Speed, Prediction: inflow. Inflow/Outflow prediction GNNs have DP training. 
class MA_4in_gy(torch.nn.Module): 
    def __init__(self, 
                 n_input_periods,
                 n_output_periods,
                 node_features,
                 n_nodes, 
                 hidden_warmup,
                 edge_index_list,
                 model_path,
                 no_backprop_dp_models,
                ):
        super(MA_4in_gy, self).__init__()
        # DP inflow prediction GNN from gy inflow
        self.model_inflow_from_gy_inflow = MP0_i(n_input_periods=n_input_periods,
                                             n_output_periods=n_output_periods,
                                             node_features=node_features,
                                             n_nodes=n_nodes, 
                                             hidden_warmup=hidden_warmup,
                                             edge_index_list=edge_index_list,
                                             model_path=model_path,
                                             train_mode = False,
                                            )
        ## Load state dicts
        self.model_inflow_from_gy_inflow.load_state_dict(torch.load("../models/" + model_path[7] + ".pth"))
        ## Freeze parameters of model_dp_trained_outflow
        if no_backprop_dp_models:
            for param in self.model_inflow_from_gy_inflow.parameters():
                param.requires_grad = False
            
        # DP outflow prediction GNN from Speed
        self.model_outflow_from_gy_outflow = MP0_o(n_input_periods=n_input_periods,
                                                  n_output_periods=n_output_periods,
                                                  node_features=node_features,
                                                  n_nodes=n_nodes, 
                                                  hidden_warmup=hidden_warmup,
                                                  edge_index_list=edge_index_list,
                                                  model_path=model_path,
                                                  train_mode = False,
                                                 )
        ## Load state dicts
        self.model_outflow_from_gy_outflow.load_state_dict(torch.load("../models/" + model_path[8] + ".pth"))
        ## Freeze parameters of model_dp_trained_outflow
        if no_backprop_dp_models:
            for param in self.model_outflow_from_gy_outflow.parameters():
                param.requires_grad = False

        # GNN of the company itself 
        self.model_gnn = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = 4, # No. of features
                                                        num_of_vertices = n_nodes[1],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[1],
                                                        edge_index=edge_index_list[1],
                                                    ) 
        
        
    def forward(self, x):
        # GY Inflow -> I, DP model
        o_inflow_other_comapny = self.model_inflow_from_gy_inflow([x[2]])
        o_inflow_other_comapny_unsqueezed = torch.unsqueeze(o_inflow_other_comapny, 2)
        # GY Outflow -> O, DP model
        o_outflow_other_comapny = self.model_outflow_from_gy_outflow([x[3]])
        o_outflow_other_comapny_unsqueezed = torch.unsqueeze(o_outflow_other_comapny, 2)
        # Concatenate inputs of the linear layer
        input_gnn = torch.concat((o_inflow_other_comapny_unsqueezed,
                                  o_outflow_other_comapny_unsqueezed,
                                  x[0],
                                  x[1],
                                 ),
                             dim=2)
        # GNN
        output_gnn = self.model_gnn(input_gnn)
        return output_gnn 
    
    
    
    
# Two-company scenario, Input: Inflow of one company + Outflow of one company + Speed, Prediction: inflow. Inflow/Outflow prediction GNNs have DP training. 
class MA_4in_gy_extra(torch.nn.Module): 
    def __init__(self, 
                 n_input_periods,
                 n_output_periods,
                 node_features,
                 n_nodes, 
                 hidden_warmup,
                 edge_index_list,
                 model_path,
                 no_backprop_dp_models,
                ):
        super(MA_4in_gy_extra, self).__init__()
        # DP inflow prediction GNN from gy inflow
        self.model_inflow_from_gy_inflow = MP0_i(n_input_periods=n_input_periods,
                                             n_output_periods=n_output_periods,
                                             node_features=node_features,
                                             n_nodes=n_nodes, 
                                             hidden_warmup=hidden_warmup,
                                             edge_index_list=edge_index_list,
                                             model_path=model_path,
                                             train_mode = False,
                                            )
        ## Load state dicts
        self.model_inflow_from_gy_inflow.load_state_dict(torch.load("../models/" + model_path[7] + ".pth"))
        ## Freeze parameters of model_dp_trained_outflow
        if no_backprop_dp_models:
            for param in self.model_inflow_from_gy_inflow.parameters():
                param.requires_grad = False
            
        # DP outflow prediction GNN from Speed
        self.model_outflow_from_gy_outflow = MP0_o(n_input_periods=n_input_periods,
                                                  n_output_periods=n_output_periods,
                                                  node_features=node_features,
                                                  n_nodes=n_nodes, 
                                                  hidden_warmup=hidden_warmup,
                                                  edge_index_list=edge_index_list,
                                                  model_path=model_path,
                                                  train_mode = False,
                                                 )
        ## Load state dicts
        self.model_outflow_from_gy_outflow.load_state_dict(torch.load("../models/" + model_path[8] + ".pth"))
        ## Freeze parameters of model_dp_trained_outflow
        if no_backprop_dp_models:
            for param in self.model_outflow_from_gy_outflow.parameters():
                param.requires_grad = False

        # GNN of the company itself 
        self.model_gnn = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = 6, # No. of features
                                                        num_of_vertices = n_nodes[1],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[1],
                                                        edge_index=edge_index_list[1],
                                                    ) 
        
        
    def forward(self, x):
        # GY Inflow -> I, DP model
        o_inflow_other_comapny = self.model_inflow_from_gy_inflow([x[2]])
        o_inflow_other_comapny_unsqueezed = torch.unsqueeze(o_inflow_other_comapny, 2)
        # GY Outflow -> O, DP model
        o_outflow_other_comapny = self.model_outflow_from_gy_outflow([x[3]])
        o_outflow_other_comapny_unsqueezed = torch.unsqueeze(o_outflow_other_comapny, 2)
        # Concatenate inputs of the linear layer
        input_gnn = torch.concat((x[0],
                                  x[1],
                                  o_inflow_other_comapny_unsqueezed,
                                  o_outflow_other_comapny_unsqueezed,
                                  x[2],
                                  x[3],
                                 ),
                             dim=2)
        # GNN
        output_gnn = self.model_gnn(input_gnn)
        return output_gnn 
    
    
    
    
# Two-company scenario, Input: Inflow of one company + Outflow of one company + Speed, Prediction: inflow. Inflow/Outflow prediction GNNs have DP training. 
class MA_5in_speed(torch.nn.Module): 
    def __init__(self, 
                 n_input_periods,
                 n_output_periods,
                 node_features,
                 n_nodes, 
                 hidden_warmup,
                 edge_index_list,
                 model_path,
                 no_backprop_s_models,
                ):
        super(MA_5in_speed, self).__init__()
                    
        # Warmed up speed prediction GNN
        self.model_speed = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                            num_for_predict = n_output_periods,
                                                            in_channels = 1, # No. of features
                                                            num_of_vertices = n_nodes[0],
                                                            nb_block = 2,
                                                            K = 3,
                                                            nb_chev_filter = 64,
                                                            nb_time_filter = 64,
                                                            time_strides = 1,
                                                            hidden = hidden_warmup[0],
                                                            edge_index=edge_index_list[0],
                                                         ) 
        ## Load state dicts of speed prediction
        self.model_speed.load_state_dict(torch.load("../models/" + model_path[0] + ".pth"))
        ## Freeze parameters of model_dp_trained_outflow
        if no_backprop_s_models:
            for param in self.model_speed.parameters():
                param.requires_grad = False
        # Linear model, Map speed graph to inflow/outflow graph
        self.linear = torch.nn.Linear(n_output_periods * n_nodes[0], n_output_periods * n_nodes[1]) 
            
        # GNN 
        self.model_gnn = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = 5, # No. of features
                                                        num_of_vertices = n_nodes[1],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[1],
                                                        edge_index=edge_index_list[1],
                                                       ) 
        
        self.n_node = n_nodes[1]
        
    def forward(self, x):
        # S 
        o_speed = self.model_speed(x[0])
        o_speed_flattened = o_speed.view(o_speed.shape[0], 1, 1, -1)
        o_speed_mapped = self.linear(o_speed_flattened)
        o_speed_reshaped = o_speed_mapped.reshape(o_speed_mapped.shape[0], self.n_node, 1, -1)
        
        # GNN
        input_gnn = torch.cat((o_speed_reshaped,
                               x[1],
                               x[2],
                               x[3],
                               x[4],
                              ),
                              dim = 2)
        output_gnn = self.model_gnn(input_gnn)
        return output_gnn 
    
    
# Two-company scenario, Input: Inflow of one company + Outflow of one company + Speed, Prediction: inflow. Inflow/Outflow prediction GNNs have DP training. 
class MA_5in_speed2(torch.nn.Module): 
    def __init__(self, 
                 n_input_periods,
                 n_output_periods,
                 node_features,
                 n_nodes, 
                 hidden_warmup,
                 edge_index_list,
                 model_path,
                 no_backprop_s_baseline_models,
                 no_backprop_baseline_models,
                ):
        super(MA_5in_speed2, self).__init__()
                    
        # Trained baseline inflow prediction model
        self.model_baseline_inflow = MA_3in_speed_baseline(n_input_periods,
                                                             n_output_periods,
                                                             node_features,
                                                             n_nodes, 
                                                             hidden_warmup,
                                                             edge_index_list,
                                                             model_path,
                                                             no_backprop_s_baseline_models,
                                                            )
        ## Load state dicts of speed prediction
        self.model_baseline_inflow.load_state_dict(torch.load("../models/" + model_path[7] + ".pth"))
        ## Freeze parameters of model_dp_trained_outflow
        if no_backprop_baseline_models:
            for param in self.model_baseline_inflow.parameters():
                param.requires_grad = False
                    
        # Trained baseline inflow prediction model
        self.model_baseline_outflow = MA_3in_speed_baseline(n_input_periods,
                                                             n_output_periods,
                                                             node_features,
                                                             n_nodes, 
                                                             hidden_warmup,
                                                             edge_index_list,
                                                             model_path,
                                                             no_backprop_s_baseline_models,
                                                            )
        ## Load state dicts of speed prediction
        self.model_baseline_outflow.load_state_dict(torch.load("../models/" + model_path[8] + ".pth"))
        ## Freeze parameters of model_dp_trained_outflow
        if no_backprop_baseline_models:
            for param in self.model_baseline_outflow.parameters():
                param.requires_grad = False
        
        # GNN 
        self.model_gnn = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = 4, # No. of features
                                                        num_of_vertices = n_nodes[1],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[1],
                                                        edge_index=edge_index_list[1],
                                                       ) 
        
    def forward(self, x):
        # Inflow baseline
        o_baseline_inflow = self.model_baseline_inflow([x[0], x[1], x[2]])
        o_baseline_inflow_unsqueezed = torch.unsqueeze(o_baseline_inflow, 2)
        # Outflow baseline
        o_baseline_outflow = self.model_baseline_outflow([x[0], x[1], x[2]])
        o_baseline_outflow_unsqueezed = torch.unsqueeze(o_baseline_outflow, 2)
        # GNN
        input_gnn = torch.cat((o_baseline_inflow_unsqueezed,
                               o_baseline_outflow_unsqueezed,
                               x[3],
                               x[4],
                              ),
                              dim = 2)
        output_gnn = self.model_gnn(input_gnn)
        return output_gnn 
    
    
    
# Two-company scenario, Input: Inflow of one company + Outflow of one company + Speed, Prediction: inflow. Inflow/Outflow prediction GNNs have DP training. 
class MA_5in_speed3(torch.nn.Module): 
    def __init__(self, 
                 n_input_periods,
                 n_output_periods,
                 node_features,
                 n_nodes, 
                 hidden_warmup,
                 edge_index_list,
                 model_path,
                 no_backprop_s_baseline_models,
                 no_backprop_baseline_models,
                ):
        super(MA_5in_speed3, self).__init__()
                    
        # Trained baseline inflow prediction model
        self.model_baseline_inflow = MA_3in_speed_baseline(n_input_periods,
                                                     n_output_periods,
                                                     node_features,
                                                     n_nodes, 
                                                     hidden_warmup,
                                                     edge_index_list,
                                                     model_path,
                                                     no_backprop_s_baseline_models,
                                                    )
        ## Load state dicts of speed prediction
        self.model_baseline_inflow.load_state_dict(torch.load("../models/" + model_path[7] + ".pth"))
        ## Freeze parameters of model_dp_trained_outflow
        if no_backprop_baseline_models:
            for param in self.model_baseline_inflow.parameters():
                param.requires_grad = False
                    
        # Trained baseline inflow prediction model
        self.model_baseline_outflow = MA_3in_speed_baseline(n_input_periods,
                                                     n_output_periods,
                                                     node_features,
                                                     n_nodes, 
                                                     hidden_warmup,
                                                     edge_index_list,
                                                     model_path,
                                                     no_backprop_s_baseline_models,
                                                    )
        ## Load state dicts of speed prediction
        self.model_baseline_outflow.load_state_dict(torch.load("../models/" + model_path[8] + ".pth"))
        ## Freeze parameters of model_dp_trained_outflow
        if no_backprop_baseline_models:
            for param in self.model_baseline_outflow.parameters():
                param.requires_grad = False
        
        # GNN 
        self.model_gnn = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = 6, # No. of features
                                                        num_of_vertices = n_nodes[1],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[1],
                                                        edge_index=edge_index_list[1],
                                                       ) 
        
    def forward(self, x):
        # Inflow baseline
        o_baseline_inflow = self.model_baseline_inflow([x[0], x[1], x[2]])
        o_baseline_inflow_unsqueezed = torch.unsqueeze(o_baseline_inflow, 2)
        # Outflow baseline
        o_baseline_outflow = self.model_baseline_outflow([x[0], x[1], x[2]])
        o_baseline_outflow_unsqueezed = torch.unsqueeze(o_baseline_outflow, 2)
        # GNN
        input_gnn = torch.cat((o_baseline_inflow_unsqueezed,
                               o_baseline_outflow_unsqueezed,
                               x[1],
                               x[2],
                               x[3],
                               x[4],
                              ),
                              dim = 2)
        output_gnn = self.model_gnn(input_gnn)
        return output_gnn 
    
    
    
# Two-company scenario, Input: Inflow of one company + Outflow of one company + Speed, Prediction: inflow. Inflow/Outflow prediction GNNs have DP training. 
class MA_5in_speed4(torch.nn.Module): 
    def __init__(self, 
                 n_input_periods,
                 n_output_periods,
                 node_features,
                 n_nodes, 
                 hidden_warmup,
                 edge_index_list,
                 model_path,
                 no_backprop_s_baseline_models,
                 no_backprop_baseline_models,
                 no_backprop_s_models,
                ):
        super(MA_5in_speed4, self).__init__()
                    
        # Trained baseline inflow prediction model
        self.model_baseline_inflow = MA_3in_speed_baseline(n_input_periods,
                                                             n_output_periods,
                                                             node_features,
                                                             n_nodes, 
                                                             hidden_warmup,
                                                             edge_index_list,
                                                             model_path,
                                                             no_backprop_s_baseline_models,
                                                            )
        ## Load state dicts of speed prediction
        self.model_baseline_inflow.load_state_dict(torch.load("../models/" + model_path[7] + ".pth"))
        ## Freeze parameters of model_dp_trained_outflow
        if no_backprop_baseline_models:
            for param in self.model_baseline_inflow.parameters():
                param.requires_grad = False
                    
        # Trained baseline inflow prediction model
        self.model_baseline_outflow = MA_3in_speed_baseline(n_input_periods,
                                                             n_output_periods,
                                                             node_features,
                                                             n_nodes, 
                                                             hidden_warmup,
                                                             edge_index_list,
                                                             model_path,
                                                             no_backprop_s_baseline_models,
                                                            )
        ## Load state dicts of speed prediction
        self.model_baseline_outflow.load_state_dict(torch.load("../models/" + model_path[8] + ".pth"))
        ## Freeze parameters of model_dp_trained_outflow
        if no_backprop_baseline_models:
            for param in self.model_baseline_outflow.parameters():
                param.requires_grad = False
        
                
        # Warmed up speed prediction GNN
        self.model_speed = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                          num_for_predict = n_output_periods,
                                                          in_channels = 1, # No. of features
                                                          num_of_vertices = n_nodes[0],
                                                          nb_block = 2,
                                                          K = 3,
                                                          nb_chev_filter = 64,
                                                          nb_time_filter = 64,
                                                          time_strides = 1,
                                                          hidden = hidden_warmup[0],
                                                          edge_index=edge_index_list[0],
                                                         ) 
        ## Load state dicts of speed prediction
        self.model_speed.load_state_dict(torch.load("../models/" + model_path[0] + ".pth"))
        ## Freeze parameters of model_dp_trained_outflow
        if no_backprop_s_models:
            for param in self.model_speed.parameters():
                param.requires_grad = False
        # Linear model, Map speed graph to inflow/outflow graph
        self.linear = torch.nn.Linear(n_output_periods * n_nodes[0], n_output_periods * n_nodes[1]) 
        
        # GNN 
        self.model_gnn = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = 7, # No. of features
                                                        num_of_vertices = n_nodes[1],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[1],
                                                        edge_index=edge_index_list[1],
                                                       ) 
        
        self.n_node = n_nodes[1]
        
    def forward(self, x):
        # Inflow baseline
        o_baseline_inflow = self.model_baseline_inflow(x)
        o_baseline_inflow_unsqueezed = torch.unsqueeze(o_baseline_inflow, 2)
        # Outflow baseline
        o_baseline_outflow = self.model_baseline_outflow(x)
        o_baseline_outflow_unsqueezed = torch.unsqueeze(o_baseline_outflow, 2)
        # S 
        o_speed = self.model_speed(x[0])
        o_speed_flattened = o_speed.view(o_speed.shape[0], 1, 1, -1)
        o_speed_mapped = self.linear(o_speed_flattened)
        o_speed_reshaped = o_speed_mapped.reshape(o_speed_mapped.shape[0], self.n_node, 1, -1)
        # GNN
        input_gnn = torch.cat((o_baseline_inflow_unsqueezed,
                               o_baseline_outflow_unsqueezed,
                               x[1],
                               x[2],
                               x[3],
                               x[4],
                               o_speed_reshaped
                              ),
                              dim = 2)
        output_gnn = self.model_gnn(input_gnn)
        return output_gnn 
    
    
    
    
# Two-company scenario, Input: Inflow of one company + Outflow of one company + Speed, Prediction: inflow. Inflow/Outflow prediction GNNs have DP training. 
class MA_5in_speed5(torch.nn.Module): 
    def __init__(self, 
                 n_input_periods,
                 n_output_periods,
                 node_features,
                 n_nodes, 
                 hidden_warmup,
                 edge_index_list,
                 model_path,
                 no_backprop_s_baseline_models,
                 no_backprop_baseline_models,
                 no_backprop_s_models,
                ):
        super(MA_5in_speed5, self).__init__()
                    
        # Trained baseline inflow prediction model
        self.model_baseline_inflow = MA_3in_speed_baseline(n_input_periods,
                                                             n_output_periods,
                                                             node_features,
                                                             n_nodes, 
                                                             hidden_warmup,
                                                             edge_index_list,
                                                             model_path,
                                                             no_backprop_s_baseline_models,
                                                            )
        ## Load state dicts of speed prediction
        self.model_baseline_inflow.load_state_dict(torch.load("../models/" + model_path[7] + ".pth"))
        ## Freeze parameters of model_dp_trained_outflow
        if no_backprop_baseline_models:
            for param in self.model_baseline_inflow.parameters():
                param.requires_grad = False
                    
        # Trained baseline inflow prediction model
        self.model_baseline_outflow = MA_3in_speed_baseline(n_input_periods,
                                                             n_output_periods,
                                                             node_features,
                                                             n_nodes, 
                                                             hidden_warmup,
                                                             edge_index_list,
                                                             model_path,
                                                             no_backprop_s_baseline_models,
                                                            )
        ## Load state dicts of speed prediction
        self.model_baseline_outflow.load_state_dict(torch.load("../models/" + model_path[8] + ".pth"))
        ## Freeze parameters of model_dp_trained_outflow
        if no_backprop_baseline_models:
            for param in self.model_baseline_outflow.parameters():
                param.requires_grad = False
        
                
        # Warmed up speed prediction GNN
        self.model_speed = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                          num_for_predict = n_output_periods,
                                                          in_channels = 1, # No. of features
                                                          num_of_vertices = n_nodes[0],
                                                          nb_block = 2,
                                                          K = 3,
                                                          nb_chev_filter = 64,
                                                          nb_time_filter = 64,
                                                          time_strides = 1,
                                                          hidden = hidden_warmup[0],
                                                          edge_index=edge_index_list[0],
                                                         ) 
        ## Load state dicts of speed prediction
        self.model_speed.load_state_dict(torch.load("../models/" + model_path[0] + ".pth"))
        ## Freeze parameters of model_dp_trained_outflow
        if no_backprop_s_models:
            for param in self.model_speed.parameters():
                param.requires_grad = False
        # Linear model, Map speed graph to inflow/outflow graph
        self.linear = torch.nn.Linear(n_output_periods * n_nodes[0], n_output_periods * n_nodes[1]) 
        
        # GNN 
        self.model_gnn = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = 5, # No. of features
                                                        num_of_vertices = n_nodes[1],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[1],
                                                        edge_index=edge_index_list[1],
                                                       ) 
        
        self.n_node = n_nodes[1]
        
    def forward(self, x):
        # Inflow baseline
        o_baseline_inflow = self.model_baseline_inflow(x)
        o_baseline_inflow_unsqueezed = torch.unsqueeze(o_baseline_inflow, 2)
        # Outflow baseline
        o_baseline_outflow = self.model_baseline_outflow(x)
        o_baseline_outflow_unsqueezed = torch.unsqueeze(o_baseline_outflow, 2)
        # S 
        o_speed = self.model_speed(x[0])
        o_speed_flattened = o_speed.view(o_speed.shape[0], 1, 1, -1)
        o_speed_mapped = self.linear(o_speed_flattened)
        o_speed_reshaped = o_speed_mapped.reshape(o_speed_mapped.shape[0], self.n_node, 1, -1)
        # GNN
        input_gnn = torch.cat((o_baseline_inflow_unsqueezed,
                               o_baseline_outflow_unsqueezed,
                               x[3],
                               x[4],
                               o_speed_reshaped
                              ),
                              dim = 2)
        output_gnn = self.model_gnn(input_gnn)
        return output_gnn 
    
    
    
    
# Two-company scenario, Input: Inflow of one company + Outflow of one company + Speed, Prediction: inflow. Inflow/Outflow prediction GNNs have DP training. 
class MA_5in_speed5(torch.nn.Module): 
    def __init__(self, 
                 n_input_periods,
                 n_output_periods,
                 node_features,
                 n_nodes, 
                 hidden_warmup,
                 edge_index_list,
                 model_path,
                 no_backprop_s_baseline_models,
                 no_backprop_baseline_models,
                 no_backprop_s_models,
                ):
        super(MA_5in_speed5, self).__init__()
                    
        # Trained baseline inflow prediction model
        self.model_baseline_inflow = MA_3in_speed_baseline(n_input_periods,
                                                             n_output_periods,
                                                             node_features,
                                                             n_nodes, 
                                                             hidden_warmup,
                                                             edge_index_list,
                                                             model_path,
                                                             no_backprop_s_baseline_models,
                                                            )
        ## Load state dicts of speed prediction
        self.model_baseline_inflow.load_state_dict(torch.load("../models/" + model_path[7] + ".pth"))
        ## Freeze parameters of model_dp_trained_outflow
        if no_backprop_baseline_models:
            for param in self.model_baseline_inflow.parameters():
                param.requires_grad = False
                    
        # Trained baseline inflow prediction model
        self.model_baseline_outflow = MA_3in_speed_baseline(n_input_periods,
                                                             n_output_periods,
                                                             node_features,
                                                             n_nodes, 
                                                             hidden_warmup,
                                                             edge_index_list,
                                                             model_path,
                                                             no_backprop_s_baseline_models,
                                                            )
        ## Load state dicts of speed prediction
        self.model_baseline_outflow.load_state_dict(torch.load("../models/" + model_path[8] + ".pth"))
        ## Freeze parameters of model_dp_trained_outflow
        if no_backprop_baseline_models:
            for param in self.model_baseline_outflow.parameters():
                param.requires_grad = False
        
                
        # Warmed up speed prediction GNN
        self.model_speed = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                          num_for_predict = n_output_periods,
                                                          in_channels = 1, # No. of features
                                                          num_of_vertices = n_nodes[0],
                                                          nb_block = 2,
                                                          K = 3,
                                                          nb_chev_filter = 64,
                                                          nb_time_filter = 64,
                                                          time_strides = 1,
                                                          hidden = hidden_warmup[0],
                                                          edge_index=edge_index_list[0],
                                                         ) 
        ## Load state dicts of speed prediction
        self.model_speed.load_state_dict(torch.load("../models/" + model_path[0] + ".pth"))
        ## Freeze parameters of model_dp_trained_outflow
        if no_backprop_s_models:
            for param in self.model_speed.parameters():
                param.requires_grad = False
        # Linear model, Map speed graph to inflow/outflow graph
        self.linear = torch.nn.Linear(n_output_periods * n_nodes[0], n_output_periods * n_nodes[1]) 
        
        # GNN 
        self.model_gnn = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = 5, # No. of features
                                                        num_of_vertices = n_nodes[1],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[1],
                                                        edge_index=edge_index_list[1],
                                                       ) 
        
        self.n_node = n_nodes[1]
        
    def forward(self, x):
        # Inflow baseline
        o_baseline_inflow = self.model_baseline_inflow(x)
        o_baseline_inflow_unsqueezed = torch.unsqueeze(o_baseline_inflow, 2)
        # Outflow baseline
        o_baseline_outflow = self.model_baseline_outflow(x)
        o_baseline_outflow_unsqueezed = torch.unsqueeze(o_baseline_outflow, 2)
        # S 
        o_speed = self.model_speed(x[0])
        o_speed_flattened = o_speed.view(o_speed.shape[0], 1, 1, -1)
        o_speed_mapped = self.linear(o_speed_flattened)
        o_speed_reshaped = o_speed_mapped.reshape(o_speed_mapped.shape[0], self.n_node, 1, -1)
        # GNN
        input_gnn = torch.cat((o_baseline_inflow_unsqueezed,
                               o_baseline_outflow_unsqueezed,
                               x[3],
                               x[4],
                               o_speed_reshaped
                              ),
                              dim = 2)
        output_gnn = self.model_gnn(input_gnn)
        return output_gnn 
    
    
    
    
# Two-company scenario, Input: Inflow of one company + Outflow of one company + Speed, Prediction: inflow. Inflow/Outflow prediction GNNs have DP training. 
class MA_5in_speed6(torch.nn.Module): 
    def __init__(self, 
                 n_input_periods,
                 n_output_periods,
                 node_features,
                 n_nodes, 
                 hidden_warmup,
                 edge_index_list,
                 model_path,
                 no_backprop_s_baseline_models,
                 no_backprop_baseline_models,
                 no_backprop_dp_i_models,
                 no_backprop_dp_o_models,
                ):
        super(MA_5in_speed6, self).__init__()
                    
        # Trained baseline inflow prediction model
        self.model_baseline_inflow = MA_3in_speed_baseline(n_input_periods,
                                                             n_output_periods,
                                                             node_features,
                                                             n_nodes, 
                                                             hidden_warmup,
                                                             edge_index_list,
                                                             model_path,
                                                             no_backprop_s_baseline_models,
                                                            )
        ## Load state dicts of speed prediction
        self.model_baseline_inflow.load_state_dict(torch.load("../models/" + model_path[7] + ".pth"))
        ## Freeze parameters of model_dp_trained_outflow
        if no_backprop_baseline_models:
            for param in self.model_baseline_inflow.parameters():
                param.requires_grad = False
                    
        # Trained baseline inflow prediction model
        self.model_baseline_outflow = MA_3in_speed_baseline(n_input_periods,
                                                             n_output_periods,
                                                             node_features,
                                                             n_nodes, 
                                                             hidden_warmup,
                                                             edge_index_list,
                                                             model_path,
                                                             no_backprop_s_baseline_models,
                                                            )
        ## Load state dicts of speed prediction
        self.model_baseline_outflow.load_state_dict(torch.load("../models/" + model_path[8] + ".pth"))
        ## Freeze parameters of model_dp_trained_outflow
        if no_backprop_baseline_models:
            for param in self.model_baseline_outflow.parameters():
                param.requires_grad = False
        
        # DP inflow prediction GNN from Speed
        self.model_inflow_from_speed = MultiPredictionNet5(n_input_periods=n_input_periods,
                                                                     n_output_periods=n_output_periods,
                                                                     node_features=node_features,
                                                                     n_nodes=n_nodes, 
                                                                     hidden_warmup=hidden_warmup,
                                                                     edge_index_list=edge_index_list,
                                                                     model_path=model_path,
                                                                   )
        ## Load state dicts
        self.model_inflow_from_speed.load_state_dict(torch.load("../models/" + model_path[9] + ".pth"))
        ## Freeze parameters of model_dp_trained_outflow
        if no_backprop_dp_i_models:
            for param in self.model_inflow_from_speed.parameters():
                param.requires_grad = False
            
        # DP outflow prediction GNN from Speed
        self.model_outflow_from_speed = MultiPredictionNet6(n_input_periods=n_input_periods,
                                                             n_output_periods=n_output_periods,
                                                             node_features=node_features,
                                                             n_nodes=n_nodes, 
                                                             hidden_warmup=hidden_warmup,
                                                             edge_index_list=edge_index_list,
                                                             model_path=model_path,
                                                           )
        ## Load state dicts
        self.model_outflow_from_speed.load_state_dict(torch.load("../models/" + model_path[10] + ".pth"))
        ## Freeze parameters of model_dp_trained_outflow
        if no_backprop_dp_o_models:
            for param in self.model_outflow_from_speed.parameters():
                param.requires_grad = False
                
        # GNN 
        self.model_gnn = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = 4, # No. of features
                                                        num_of_vertices = n_nodes[1],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[1],
                                                        edge_index=edge_index_list[1],
                                                       ) 
        
        self.n_node = n_nodes[1]
        
    def forward(self, x):
        # Inflow baseline
        o_baseline_inflow = self.model_baseline_inflow(x)
        o_baseline_inflow_unsqueezed = torch.unsqueeze(o_baseline_inflow, 2)
        # Outflow baseline
        o_baseline_outflow = self.model_baseline_outflow(x)
        o_baseline_outflow_unsqueezed = torch.unsqueeze(o_baseline_outflow, 2)
        # S -> I, DP model
        o_inflow_other_comapny = self.model_inflow_from_speed(x[0])
        o_inflow_other_comapny_reshaped = o_inflow_other_comapny.reshape(o_inflow_other_comapny.shape[0], self.n_node, 1, -1)
        # S -> O, DP model
        o_outflow_other_comapny = self.model_outflow_from_speed(x[0])
        o_outflow_other_comapny_reshaped = o_outflow_other_comapny.reshape(o_outflow_other_comapny.shape[0], self.n_node, 1, -1)
        # GNN
        input_gnn = torch.cat((o_baseline_inflow_unsqueezed,
                               o_baseline_outflow_unsqueezed,
                               o_inflow_other_comapny_reshaped,
                               o_outflow_other_comapny_reshaped,
                              ),
                              dim = 2)
        output_gnn = self.model_gnn(input_gnn)
        return output_gnn 
    
    
    
# Two-company scenario, Input: Inflow of one company + Outflow of one company + Speed, Prediction: inflow. Inflow/Outflow prediction GNNs have DP training. 
class MA_5in_speed7(torch.nn.Module): 
    def __init__(self, 
                 n_input_periods,
                 n_output_periods,
                 node_features,
                 n_nodes, 
                 hidden_warmup,
                 edge_index_list,
                 model_path,
                 no_backprop_s_baseline_models,
                 no_backprop_baseline_models,
                 no_backprop_s_models,
                 no_backprop_dp_i_models,
                 no_backprop_dp_o_models,
                ):
        super(MA_5in_speed7, self).__init__()
                    
        # Trained baseline inflow prediction model
        self.model_baseline_inflow = MA_3in_speed_baseline(n_input_periods,
                                                             n_output_periods,
                                                             node_features,
                                                             n_nodes, 
                                                             hidden_warmup,
                                                             edge_index_list,
                                                             model_path,
                                                             no_backprop_s_baseline_models,
                                                            )
        ## Load state dicts of speed prediction
        self.model_baseline_inflow.load_state_dict(torch.load("../models/" + model_path[7] + ".pth"))
        ## Freeze parameters of model_dp_trained_outflow
        if no_backprop_baseline_models:
            for param in self.model_baseline_inflow.parameters():
                param.requires_grad = False
                    
        # Trained baseline inflow prediction model
        self.model_baseline_outflow = MA_3in_speed_baseline(n_input_periods,
                                                             n_output_periods,
                                                             node_features,
                                                             n_nodes, 
                                                             hidden_warmup,
                                                             edge_index_list,
                                                             model_path,
                                                             no_backprop_s_baseline_models,
                                                            )
        ## Load state dicts of speed prediction
        self.model_baseline_outflow.load_state_dict(torch.load("../models/" + model_path[8] + ".pth"))
        ## Freeze parameters of model_dp_trained_outflow
        if no_backprop_baseline_models:
            for param in self.model_baseline_outflow.parameters():
                param.requires_grad = False
        
        # DP inflow prediction GNN from Speed
        self.model_inflow_from_speed = MultiPredictionNet5(n_input_periods=n_input_periods,
                                                                     n_output_periods=n_output_periods,
                                                                     node_features=node_features,
                                                                     n_nodes=n_nodes, 
                                                                     hidden_warmup=hidden_warmup,
                                                                     edge_index_list=edge_index_list,
                                                                     model_path=model_path,
                                                                   )
        ## Load state dicts
        self.model_inflow_from_speed.load_state_dict(torch.load("../models/" + model_path[9] + ".pth"))
        ## Freeze parameters of model_dp_trained_outflow
        if no_backprop_dp_i_models:
            for param in self.model_inflow_from_speed.parameters():
                param.requires_grad = False
            
        # DP outflow prediction GNN from Speed
        self.model_outflow_from_speed = MultiPredictionNet6(n_input_periods=n_input_periods,
                                                             n_output_periods=n_output_periods,
                                                             node_features=node_features,
                                                             n_nodes=n_nodes, 
                                                             hidden_warmup=hidden_warmup,
                                                             edge_index_list=edge_index_list,
                                                             model_path=model_path,
                                                           )
        ## Load state dicts
        self.model_outflow_from_speed.load_state_dict(torch.load("../models/" + model_path[10] + ".pth"))
        ## Freeze parameters of model_dp_trained_outflow
        if no_backprop_dp_o_models:
            for param in self.model_outflow_from_speed.parameters():
                param.requires_grad = False
                
        # Warmed up speed prediction GNN
        self.model_speed = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                          num_for_predict = n_output_periods,
                                                          in_channels = 1, # No. of features
                                                          num_of_vertices = n_nodes[0],
                                                          nb_block = 2,
                                                          K = 3,
                                                          nb_chev_filter = 64,
                                                          nb_time_filter = 64,
                                                          time_strides = 1,
                                                          hidden = hidden_warmup[0],
                                                          edge_index=edge_index_list[0],
                                                         ) 
        ## Load state dicts of speed prediction
        self.model_speed.load_state_dict(torch.load("../models/" + model_path[0] + ".pth"))
        ## Freeze parameters of model_dp_trained_outflow
        if no_backprop_s_models:
            for param in self.model_speed.parameters():
                param.requires_grad = False
        # Linear model, Map speed graph to inflow/outflow graph
        self.linear = torch.nn.Linear(n_output_periods * n_nodes[0], n_output_periods * n_nodes[1]) 
        
        # GNN 
        self.model_gnn = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = 5, # No. of features
                                                        num_of_vertices = n_nodes[1],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[1],
                                                        edge_index=edge_index_list[1],
                                                       ) 
        
        self.n_node = n_nodes[1]
        
    def forward(self, x):
        # Inflow baseline
        o_baseline_inflow = self.model_baseline_inflow(x)
        o_baseline_inflow_unsqueezed = torch.unsqueeze(o_baseline_inflow, 2)
        # Outflow baseline
        o_baseline_outflow = self.model_baseline_outflow(x)
        o_baseline_outflow_unsqueezed = torch.unsqueeze(o_baseline_outflow, 2)
        # S -> I, DP model
        o_inflow_other_comapny = self.model_inflow_from_speed(x[0])
        o_inflow_other_comapny_reshaped = o_inflow_other_comapny.reshape(o_inflow_other_comapny.shape[0], self.n_node, 1, -1)
        # S -> O, DP model
        o_outflow_other_comapny = self.model_outflow_from_speed(x[0])
        o_outflow_other_comapny_reshaped = o_outflow_other_comapny.reshape(o_outflow_other_comapny.shape[0], self.n_node, 1, -1)
        # S 
        o_speed = self.model_speed(x[0])
        o_speed_flattened = o_speed.view(o_speed.shape[0], 1, 1, -1)
        o_speed_mapped = self.linear(o_speed_flattened)
        o_speed_reshaped = o_speed_mapped.reshape(o_speed_mapped.shape[0], self.n_node, 1, -1)
        # GNN
        input_gnn = torch.cat((o_baseline_inflow_unsqueezed,
                               o_baseline_outflow_unsqueezed,
                               o_inflow_other_comapny_reshaped,
                               o_outflow_other_comapny_reshaped,
                               o_speed_reshaped,
                              ),
                              dim = 2)
        output_gnn = self.model_gnn(input_gnn)
        return output_gnn 
    
    
    
# Two-company scenario, Input: Inflow of one company + Outflow of one company + Speed, Prediction: inflow. Inflow/Outflow prediction GNNs have DP training. 
class MA_5in_speed8(torch.nn.Module): 
    def __init__(self, 
                 n_input_periods,
                 n_output_periods,
                 node_features,
                 n_nodes, 
                 hidden_warmup,
                 edge_index_list,
                 model_path,
                 no_backprop_s_baseline_models,
                 no_backprop_baseline_models,
                 no_backprop_dp_i_models,
                 no_backprop_dp_o_models,
                ):
        super(MA_5in_speed8, self).__init__()
                    
        # Trained baseline inflow prediction model
        self.model_baseline_inflow = MA_3in_speed_baseline(n_input_periods,
                                                     n_output_periods,
                                                     node_features,
                                                     n_nodes, 
                                                     hidden_warmup,
                                                     edge_index_list,
                                                     model_path,
                                                     no_backprop_s_baseline_models,
                                                    )
        ## Load state dicts of speed prediction
        self.model_baseline_inflow.load_state_dict(torch.load("../models/" + model_path[7] + ".pth"))
        ## Freeze parameters of model_dp_trained_outflow
        if no_backprop_baseline_models:
            for param in self.model_baseline_inflow.parameters():
                param.requires_grad = False
                    
        # Trained baseline inflow prediction model
        self.model_baseline_outflow = MA_3in_speed_baseline(n_input_periods,
                                                     n_output_periods,
                                                     node_features,
                                                     n_nodes, 
                                                     hidden_warmup,
                                                     edge_index_list,
                                                     model_path,
                                                     no_backprop_s_baseline_models,
                                                    )
        ## Load state dicts of speed prediction
        self.model_baseline_outflow.load_state_dict(torch.load("../models/" + model_path[8] + ".pth"))
        ## Freeze parameters of model_dp_trained_outflow
        if no_backprop_baseline_models:
            for param in self.model_baseline_outflow.parameters():
                param.requires_grad = False
                
        # DP inflow prediction GNN from Speed
        self.model_inflow_from_speed = MultiPredictionNet5(n_input_periods=n_input_periods,
                                                                     n_output_periods=n_output_periods,
                                                                     node_features=node_features,
                                                                     n_nodes=n_nodes, 
                                                                     hidden_warmup=hidden_warmup,
                                                                     edge_index_list=edge_index_list,
                                                                     model_path=model_path,
                                                                   )
        ## Load state dicts
        self.model_inflow_from_speed.load_state_dict(torch.load("../models/" + model_path[9] + ".pth"))
        ## Freeze parameters of model_dp_trained_outflow
        if no_backprop_dp_i_models:
            for param in self.model_inflow_from_speed.parameters():
                param.requires_grad = False
            
        # DP outflow prediction GNN from Speed
        self.model_outflow_from_speed = MultiPredictionNet6(n_input_periods=n_input_periods,
                                                             n_output_periods=n_output_periods,
                                                             node_features=node_features,
                                                             n_nodes=n_nodes, 
                                                             hidden_warmup=hidden_warmup,
                                                             edge_index_list=edge_index_list,
                                                             model_path=model_path,
                                                           )
        ## Load state dicts
        self.model_outflow_from_speed.load_state_dict(torch.load("../models/" + model_path[10] + ".pth"))
        ## Freeze parameters of model_dp_trained_outflow
        if no_backprop_dp_o_models:
            for param in self.model_outflow_from_speed.parameters():
                param.requires_grad = False
                
        
        # GNN 
        self.model_gnn = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = 6, # No. of features
                                                        num_of_vertices = n_nodes[1],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[1],
                                                        edge_index=edge_index_list[1],
                                                       ) 
        
        self.n_node = n_nodes[1]
        
    def forward(self, x):
        # Inflow baseline
        o_baseline_inflow = self.model_baseline_inflow([x[0], x[1], x[2]])
        o_baseline_inflow_unsqueezed = torch.unsqueeze(o_baseline_inflow, 2)
        # Outflow baseline
        o_baseline_outflow = self.model_baseline_outflow([x[0], x[1], x[2]])
        o_baseline_outflow_unsqueezed = torch.unsqueeze(o_baseline_outflow, 2)
        # S -> I, DP model
        o_inflow_other_comapny = self.model_inflow_from_speed(x[0])
        o_inflow_other_comapny_reshaped = o_inflow_other_comapny.reshape(o_inflow_other_comapny.shape[0], self.n_node, 1, -1)
        # S -> O, DP model
        o_outflow_other_comapny = self.model_outflow_from_speed(x[0])
        o_outflow_other_comapny_reshaped = o_outflow_other_comapny.reshape(o_outflow_other_comapny.shape[0], self.n_node, 1, -1)
        # GNN
        input_gnn = torch.cat((o_baseline_inflow_unsqueezed,
                               o_baseline_outflow_unsqueezed,
                               x[1],
                               x[2],
                               o_inflow_other_comapny_reshaped,
                               o_outflow_other_comapny_reshaped,
                              ),
                              dim = 2)
        output_gnn = self.model_gnn(input_gnn)
        return output_gnn 
    
    
# Two-company scenario, Input: Inflow of one company + Outflow of one company + Speed, Prediction: inflow. Inflow/Outflow prediction GNNs have DP training. 
class MSA_2in(torch.nn.Module): 
    def __init__(self, 
                 n_input_periods,
                 n_output_periods,
                 node_features,
                 n_nodes, 
                 hidden_warmup,
                 edge_index_list,
                 model_path,
                 no_backprop_models,
                ):
        super(MSA_2in, self).__init__()
        
        # DP inflow prediction GNN from gy inflow
        self.model_inflow = MA_2in(n_input_periods=n_input_periods,
                                             n_output_periods=n_output_periods,
                                             node_features=node_features,
                                             n_nodes=n_nodes, 
                                             hidden_warmup=hidden_warmup,
                                             edge_index_list=edge_index_list,
                                             model_path=model_path,
                                            )
        ## Load state dicts
        self.model_inflow.load_state_dict(torch.load("../models/" + model_path[9] + ".pth"))
        ## Freeze parameters of model_dp_trained_outflow
        if no_backprop_models:
            for param in self.model_inflow.parameters():
                param.requires_grad = False
            
        # DP outflow prediction GNN from Speed
        self.model_outflow = MA_2in(n_input_periods=n_input_periods,
                                                  n_output_periods=n_output_periods,
                                                  node_features=node_features,
                                                  n_nodes=n_nodes, 
                                                  hidden_warmup=hidden_warmup,
                                                  edge_index_list=edge_index_list,
                                                  model_path=model_path,
                                                 )
        ## Load state dicts
        self.model_outflow.load_state_dict(torch.load("../models/" + model_path[10] + ".pth"))
        ## Freeze parameters of model_dp_trained_outflow
        if no_backprop_models:
            for param in self.model_outflow.parameters():
                param.requires_grad = False
                
        # GNN as the final layer 
        self.model_gnn_final_layer = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = 2, # No. of features
                                                        num_of_vertices = n_nodes[1],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[1],
                                                        edge_index=edge_index_list[1],
                                                    ) 
         
        
    def forward(self, x):
        # GY Inflow -> I, DP model
        o_inflow = self.model_inflow([x[0], x[1]])
        o_inflow_unsqueezed = torch.unsqueeze(o_inflow, 2)
        # GY Outflow -> O, DP model
        o_outflow = self.model_outflow([x[1], x[0]])
        o_outflow_unsqueezed = torch.unsqueeze(o_outflow, 2)
        # Concatenate inputs of the linear layer
        o_cat = torch.concat((o_inflow_unsqueezed,
                              o_outflow_unsqueezed,
                             ),
                             dim = 2)
        output = self.model_gnn_final_layer(o_cat)
        return output 
    
    
    
    
    
# Two-company scenario, Input: Inflow of one company + Outflow of one company + Speed, Prediction: inflow. Inflow/Outflow prediction GNNs have DP training. 
class MSA_4in(torch.nn.Module): 
    def __init__(self, 
                 n_input_periods,
                 n_output_periods,
                 node_features,
                 n_nodes, 
                 hidden_warmup,
                 edge_index_list,
                 model_path,
                 no_backprop_models,
                ):
        super(MSA_4in, self).__init__()
        
        # DP inflow prediction GNN from gy inflow
        self.model_inflow = MA_4in(n_input_periods=n_input_periods,
                                             n_output_periods=n_output_periods,
                                             node_features=node_features,
                                             n_nodes=n_nodes, 
                                             hidden_warmup=hidden_warmup,
                                             edge_index_list=edge_index_list,
                                             model_path=model_path,
                                            )
        ## Load state dicts
        self.model_inflow.load_state_dict(torch.load("../models/" + model_path[9] + ".pth"))
        ## Freeze parameters of model_dp_trained_outflow
        if no_backprop_models:
            for param in self.model_inflow.parameters():
                param.requires_grad = False
            
        # DP outflow prediction GNN from Speed
        self.model_outflow = MA_4in(n_input_periods=n_input_periods,
                                                  n_output_periods=n_output_periods,
                                                  node_features=node_features,
                                                  n_nodes=n_nodes, 
                                                  hidden_warmup=hidden_warmup,
                                                  edge_index_list=edge_index_list,
                                                  model_path=model_path,
                                                 )
        ## Load state dicts
        self.model_outflow.load_state_dict(torch.load("../models/" + model_path[10] + ".pth"))
        ## Freeze parameters of model_dp_trained_outflow
        if no_backprop_models:
            for param in self.model_outflow.parameters():
                param.requires_grad = False
                
        # GNN as the final layer 
        self.model_gnn_final_layer = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = 2, # No. of features
                                                        num_of_vertices = n_nodes[1],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[1],
                                                        edge_index=edge_index_list[1],
                                                    ) 
         
        
    def forward(self, x):
        # GY Inflow -> I, DP model
        o_inflow = self.model_inflow(x)
        o_inflow_unsqueezed = torch.unsqueeze(o_inflow, 2)
        # GY Outflow -> O, DP model
        o_outflow = self.model_outflow(x)
        o_outflow_unsqueezed = torch.unsqueeze(o_outflow, 2)
        # Concatenate inputs of the linear layer
        o_cat = torch.concat((o_inflow_unsqueezed,
                              o_outflow_unsqueezed,
                             ),
                             dim = 2)
        output = self.model_gnn_final_layer(o_cat)
        return output 
    
    
    
    
# Two-company scenario, Input: Inflow of one company + Outflow of one company + Speed, Prediction: inflow. Inflow/Outflow prediction GNNs have DP training. 
class MSA_5in_1(torch.nn.Module): 
    def __init__(self, 
                 n_input_periods,
                 n_output_periods,
                 node_features,
                 n_nodes, 
                 hidden_warmup,
                 edge_index_list,
                 model_path,
                 no_backprop_models,
                 no_backprop_s_models,
                ):
        super(MSA_5in_1, self).__init__()
        
        # Inflow prediction 
        self.model_inflow = MA_5in_speed(n_input_periods,
                                         n_output_periods,
                                         node_features,
                                         n_nodes, 
                                         hidden_warmup,
                                         edge_index_list,
                                         model_path[:7],
                                         no_backprop_s_models,
                                        )
        ## Load state dicts
        self.model_inflow.load_state_dict(torch.load("../models/" + model_path[9] + ".pth"))
        ## Freeze parameters of model_dp_trained_outflow
        if no_backprop_models:
            for param in self.model_inflow.parameters():
                param.requires_grad = False
            
        # Outflow prediction
        self.model_outflow = MA_5in_speed(n_input_periods,
                                         n_output_periods,
                                         node_features,
                                         n_nodes, 
                                         hidden_warmup,
                                         edge_index_list,
                                         model_path[:7],
                                         no_backprop_s_models,
                                        )
        ## Load state dicts
        self.model_outflow.load_state_dict(torch.load("../models/" + model_path[10] + ".pth"))
        ## Freeze parameters of model_dp_trained_outflow
        if no_backprop_models:
            for param in self.model_outflow.parameters():
                param.requires_grad = False
                
        # GNN as the final layer 
        self.model_gnn_final_layer = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = 2, # No. of features
                                                        num_of_vertices = n_nodes[1],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[1],
                                                        edge_index=edge_index_list[1],
                                                    ) 
         
        
    def forward(self, x):
        # Inflow
        o_inflow = self.model_inflow(x)
        o_inflow_unsqueezed = torch.unsqueeze(o_inflow, 2)
        # Outflow
        o_outflow = self.model_outflow(x)
        o_outflow_unsqueezed = torch.unsqueeze(o_outflow, 2)
        # Concatenate  
        o_cat = torch.concat((o_inflow_unsqueezed,
                              o_outflow_unsqueezed,
                             ),
                             dim = 2)
        output = self.model_gnn_final_layer(o_cat)
        return output 
    
    
# Two-company scenario, Input: Inflow of one company + Outflow of one company + Speed, Prediction: inflow. Inflow/Outflow prediction GNNs have DP training. 
class MSA_5in_1_L(torch.nn.Module): 
    def __init__(self, 
                 n_input_periods,
                 n_output_periods,
                 node_features,
                 n_nodes, 
                 hidden_warmup,
                 edge_index_list,
                 model_path,
                 no_backprop_models,
                 no_backprop_s_models,
                ):
        super(MSA_5in_1_L, self).__init__()
        
        # Inflow prediction 
        self.model_inflow = MA_5in_speed(n_input_periods,
                                         n_output_periods,
                                         node_features,
                                         n_nodes, 
                                         hidden_warmup,
                                         edge_index_list,
                                         model_path[:7],
                                         no_backprop_s_models,
                                        )
        ## Load state dicts
        self.model_inflow.load_state_dict(torch.load("../models/" + model_path[9] + ".pth"))
        ## Freeze parameters of model_dp_trained_outflow
        if no_backprop_models:
            for param in self.model_inflow.parameters():
                param.requires_grad = False
            
        # Outflow prediction
        self.model_outflow = MA_5in_speed(n_input_periods,
                                         n_output_periods,
                                         node_features,
                                         n_nodes, 
                                         hidden_warmup,
                                         edge_index_list,
                                         model_path[:7],
                                         no_backprop_s_models,
                                        )
        ## Load state dicts
        self.model_outflow.load_state_dict(torch.load("../models/" + model_path[10] + ".pth"))
        ## Freeze parameters of model_dp_trained_outflow
        if no_backprop_models:
            for param in self.model_outflow.parameters():
                param.requires_grad = False
                
        # Linear model
        self.linear = torch.nn.Linear(n_output_periods * (n_nodes[1] * 2), 
                                                n_output_periods * (n_nodes[1])) 
         
        
    def forward(self, x):
        # GY Inflow -> I, DP model
        o_inflow = self.model_inflow(x)
        o1_flattened = o_inflow.view(o_inflow.shape[0], 1, 1, -1)
        # GY Outflow -> O, DP model
        o_outflow = self.model_outflow(x)
        o2_flattened = o_outflow.view(o_outflow.shape[0], 1, 1, -1)
        # Concatenate inputs of the linear layer
        o_cat = torch.concat((o1_flattened,
                              o2_flattened,
                             ),
                             dim = -1)
        output = self.linear(o_cat)
        return output 
    
    
    

    
    
# Two-company scenario, Input: Inflow of one company + Outflow of one company + Speed, Prediction: inflow. Inflow/Outflow prediction GNNs have DP training. 
class MSA_3in_1(torch.nn.Module): 
    def __init__(self, 
                 n_input_periods,
                 n_output_periods,
                 node_features,
                 n_nodes, 
                 hidden_warmup,
                 edge_index_list,
                 model_path,
                 no_backprop_models,
                 no_backprop_s_models,
                 no_backprop_dp_models,
                ):
        super(MSA_3in_1, self).__init__()
        
        # Inflow prediction 
        self.model_inflow = MA_3in_speed_extra(n_input_periods,
                                                 n_output_periods,
                                                 node_features,
                                                 n_nodes, 
                                                 hidden_warmup,
                                                 edge_index_list,
                                                 model_path[:7] + [model_path[9], model_path[10]],
                                                 no_backprop_s_models,
                                                 no_backprop_dp_models,
                                                )
        ## Load state dicts
        self.model_inflow.load_state_dict(torch.load("../models/" + model_path[11] + ".pth"))
        ## Freeze parameters of model_dp_trained_outflow
        if no_backprop_models:
            for param in self.model_inflow.parameters():
                param.requires_grad = False
            
        # Outflow prediction
        self.model_outflow = MA_3in_speed_extra(n_input_periods,
                                                 n_output_periods,
                                                 node_features,
                                                 n_nodes, 
                                                 hidden_warmup,
                                                 edge_index_list,
                                                 model_path[:7] + [model_path[9], model_path[10]],
                                                 no_backprop_s_models,
                                                 no_backprop_dp_models,
                                                )
        ## Load state dicts
        self.model_outflow.load_state_dict(torch.load("../models/" + model_path[12] + ".pth"))
        ## Freeze parameters of model_dp_trained_outflow
        if no_backprop_models:
            for param in self.model_outflow.parameters():
                param.requires_grad = False
                
        # GNN as the final layer 
        self.model_gnn_final_layer = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                        num_for_predict = n_output_periods,
                                                        in_channels = 2, # No. of features
                                                        num_of_vertices = n_nodes[1],
                                                        nb_block = 2,
                                                        K = 3,
                                                        nb_chev_filter = 64,
                                                        nb_time_filter = 64,
                                        #                         time_strides = num_of_hours,
                                                        time_strides = 1,
                                                        hidden = hidden_warmup[1],
                                                        edge_index=edge_index_list[1],
                                                    ) 
         
        
    def forward(self, x):
        # GY Inflow -> I, DP model
        o_inflow = self.model_inflow(x)
        o_inflow_unsqueezed = torch.unsqueeze(o_inflow, 2)
        # GY Outflow -> O, DP model
        o_outflow = self.model_outflow(x)
        o_outflow_unsqueezed = torch.unsqueeze(o_outflow, 2)
        # Concatenate inputs of the linear layer
        o_cat = torch.concat((o_inflow_unsqueezed,
                              o_outflow_unsqueezed,
                             ),
                             dim = 2)
        output = self.model_gnn_final_layer(o_cat)
        return output 
    
    
# Two-company scenario, Input: Inflow of one company + Outflow of one company + Speed, Prediction: inflow. Inflow/Outflow prediction GNNs have DP training. 
class MSA_3in_1_L(torch.nn.Module): 
    def __init__(self, 
                 n_input_periods,
                 n_output_periods,
                 node_features,
                 n_nodes, 
                 hidden_warmup,
                 edge_index_list,
                 model_path,
                 no_backprop_models,
                 no_backprop_s_models,
                 no_backprop_dp_models,
                ):
        super(MSA_3in_1_L, self).__init__()
        
        # Inflow prediction 
        self.model_inflow = MA_3in_speed_extra(n_input_periods,
                                                 n_output_periods,
                                                 node_features,
                                                 n_nodes, 
                                                 hidden_warmup,
                                                 edge_index_list,
                                                 model_path[:7] + [model_path[9], model_path[10]],
                                                 no_backprop_s_models,
                                                 no_backprop_dp_models,
                                                )
        ## Load state dicts
        self.model_inflow.load_state_dict(torch.load("../models/" + model_path[11] + ".pth"))
        ## Freeze parameters of model_dp_trained_outflow
        if no_backprop_models:
            for param in self.model_inflow.parameters():
                param.requires_grad = False
            
        # Outflow prediction
        self.model_outflow = MA_3in_speed_extra(n_input_periods,
                                                 n_output_periods,
                                                 node_features,
                                                 n_nodes, 
                                                 hidden_warmup,
                                                 edge_index_list,
                                                 model_path[:7] + [model_path[9], model_path[10]],
                                                 no_backprop_s_models,
                                                 no_backprop_dp_models,
                                                )
        ## Load state dicts
        self.model_outflow.load_state_dict(torch.load("../models/" + model_path[12] + ".pth"))
        ## Freeze parameters of model_dp_trained_outflow
        if no_backprop_models:
            for param in self.model_outflow.parameters():
                param.requires_grad = False
                
        # Linear model
        self.linear = torch.nn.Linear(n_output_periods * (n_nodes[1] * 2), 
                                                n_output_periods * (n_nodes[1])) 
         
        
    def forward(self, x):
        # GY Inflow -> I, DP model
        o_inflow = self.model_inflow(x)
        o1_flattened = o_inflow.view(o_inflow.shape[0], 1, 1, -1)
        # GY Outflow -> O, DP model
        o_outflow = self.model_outflow(x)
        o2_flattened = o_outflow.view(o_outflow.shape[0], 1, 1, -1)
        # Concatenate inputs of the linear layer
        o_cat = torch.concat((o1_flattened,
                              o2_flattened,
                             ),
                             dim = -1)
        output = self.linear(o_cat)
        return output 
    
    
    
# ########################################## barrier ###############################################

# ##################################################################################################

# ##################################################################################################

# ##################################################################################################

# ##################################################################################################

# ##################################################################################################

# ##################################################################################################

# ##################################################################################################

# ##################################################################################################

# ##################################################################################################

# ##################################################################################################

# ##################################################################################################

# ##################################################################################################

# ENSEMBLE 6: predict speed, inflow, outflow + add historical speed as a residual information 
class EnsembleNetResidual(torch.nn.Module): 
    def __init__(self, modelA, modelB, modelC, hidden, periods, n_nodes):
        super(EnsembleNetResidual, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.modelC = modelC
#         self.linear_1 = torch.nn.Linear(periods*sum(n_nodes), hidden)
#         self.linear_2 = torch.nn.Linear(hidden, periods*n_nodes[0])
        self.hidden = hidden
        self.linear_1 = torch.nn.Linear(periods*sum(n_nodes)*2, hidden)
        self.linear_2 = torch.nn.Linear(hidden, periods*sum(n_nodes))
        self.linear_single = torch.nn.Linear(periods*sum(n_nodes)*2, periods*sum(n_nodes))
        
    def forward(self, x, edge_index):
        x1 = self.modelA(x[0], edge_index[0])
        x2 = self.modelB(x[1], edge_index[1])
        x3 = self.modelC(x[2], edge_index[2])
#         x = torch.concat((x1, x2, x3), dim=0).T.reshape(1, -1)
#         y = torch.concat((x1, x2, x3), dim=-1).unsqueeze(2)
        x1 = x1.reshape(x1.shape[0], 1, 1, -1)
        x2 = x2.reshape(x2.shape[0], 1, 1, -1)
        x3 = x3.reshape(x3.shape[0], 1, 1, -1)
        x_speed = x[0].reshape(x[0].shape[0], 1, 1, -1)
        x_inflow = x[1].reshape(x[1].shape[0], 1, 1, -1)
        x_outflow = x[2].reshape(x[2].shape[0], 1, 1, -1)
        y = torch.concat((x1, x2, x3, x_speed, x_inflow, x_outflow), dim=-1)
        if self.hidden==0:
            y = self.linear_single(F.relu(y))
        else:
            y = self.linear_1(F.relu(y))
            y = self.linear_2(F.relu(y))
            
        return y
    
# ENSEMBLE 2 : predict speed
class EnsembleNetSpeed(torch.nn.Module):
    def __init__(self, modelA, modelB, modelC, hidden, periods, n_nodes):
        super(EnsembleNetSpeed, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.modelC = modelC
        self.hidden = hidden
        
        self.linear_1 = torch.nn.Linear(periods*sum(n_nodes), hidden)
        self.linear_2 = torch.nn.Linear(hidden, periods*n_nodes[0])
        
        self.linear_single = torch.nn.Linear(periods*sum(n_nodes), periods*n_nodes[0])
        
    def forward(self, x, edge_index):
        x1 = self.modelA(x[0], edge_index[0])
        x2 = self.modelB(x[1], edge_index[1])
        x3 = self.modelC(x[2], edge_index[2])
#         x = torch.concat((x1, x2, x3), dim=0).T.reshape(1, -1)
#         y = torch.concat((x1, x2, x3), dim=-1).unsqueeze(2)
#         x1 = torch.transpose(x1, 2, 1)
        x1 = x1.reshape(x1.shape[0], 1, 1, -1)
        x2 = x2.reshape(x2.shape[0], 1, 1, -1)
        x3 = x3.reshape(x3.shape[0], 1, 1, -1)
        y = torch.concat((x1, x2, x3), dim=-1)
        if self.hidden==0:
            y = self.linear_single(F.relu(y))
        else:
            y = self.linear_1(F.relu(y))
            y = self.linear_2(F.relu(y))
            
        return y
    
    
    
# ENSEMBLE 3: predict speed + add historical speed as a residual information 
class EnsembleNetSpeedResidual(torch.nn.Module):
    def __init__(self, modelA, modelB, modelC, hidden, periods, n_nodes):
        super(EnsembleNetSpeedResidual, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.modelC = modelC
        self.hidden = hidden
        
        self.linear_1 = torch.nn.Linear(periods*(sum(n_nodes)+n_nodes[0]), hidden)
        self.linear_2 = torch.nn.Linear(hidden, periods*n_nodes[0])
        
        self.linear_single = torch.nn.Linear(periods*(sum(n_nodes)+n_nodes[0]), periods*n_nodes[0])
        
    def forward(self, x, edge_index):
        x1 = self.modelA(x[0], edge_index[0])
        x2 = self.modelB(x[1], edge_index[1])
        x3 = self.modelC(x[2], edge_index[2])
#         x = torch.concat((x1, x2, x3), dim=0).T.reshape(1, -1)
#         y = torch.concat((x1, x2, x3), dim=-1).unsqueeze(2)
#         x1 = torch.transpose(x1, 2, 1)
        x_speed = x[0].reshape(x[0].shape[0], 1, 1, -1)
        x1 = x1.reshape(x1.shape[0], 1, 1, -1)
        x2 = x2.reshape(x2.shape[0], 1, 1, -1)
        x3 = x3.reshape(x3.shape[0], 1, 1, -1)
        y = torch.concat((x_speed, x1, x2, x3), dim=-1)
        if self.hidden==0:
            y = self.linear_single(F.relu(y))
        else:
            y = self.linear_1(F.relu(y))
            y = self.linear_2(F.relu(y))
            
        return y

    
    
# ENSEMBLE 4: predict speed, add a graph layer upon model
class EnsembleNetSpeedGraphEnd(torch.nn.Module):
    def __init__(self, modelA, modelB, modelC, hidden, periods, n_nodes):
        super(EnsembleNetSpeedGraphEnd, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.modelC = modelC
        self.linear_1 = torch.nn.Linear(periods*sum(n_nodes), periods*n_nodes[0])
        self.astgcn_layer = ASTGCN_AttentionGCN(len_input = periods, 
                                                num_for_predict = periods,
                                                in_channels = 1, 
                                                num_of_vertices = n_nodes[0],
                                                nb_block = 2, 
                                                K = 3, 
                                                nb_chev_filter = 64, 
                                                nb_time_filter = 64,
                                                # time_strides = num_of_hours,
                                                time_strides = 1,
                                                hidden = hidden)
#         self.linear_2 = torch.nn.Linear(periods, periods)
        
    def forward(self, x, edge_index):
        x1 = self.modelA(x[0], edge_index[0])
        x2 = self.modelB(x[1], edge_index[1])
        x3 = self.modelC(x[2], edge_index[2])
#         x = torch.concat((x1, x2, x3), dim=0).T.reshape(1, -1)
#         y = torch.concat((x1, x2, x3), dim=-1).unsqueeze(2)
#         x1 = torch.transpose(x1, 2, 1)
        x1 = x1.reshape(x1.shape[0], 1, 1, -1)
        x2 = x2.reshape(x2.shape[0], 1, 1, -1)
        x3 = x3.reshape(x3.shape[0], 1, 1, -1)
        y = torch.concat((x1, x2, x3), dim=-1)
        y = self.linear_1(F.relu(y))
        y = y.reshape(x[0].shape)
        y = self.astgcn_layer(y, edge_index[0])
#         y = self.linear_2(F.relu(y))
            
        return y
    
    
# ENSEMBLE 5: predict speed, add a graph layer upon model + add historical speed as a residual information 
class EnsembleNetSpeedResidualGraphEnd(torch.nn.Module):
    def __init__(self, modelA, modelB, modelC, hidden, periods, n_nodes):
        super(EnsembleNetSpeedResidualGraphEnd, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.modelC = modelC
        self.linear_1 = torch.nn.Linear(periods*(sum(n_nodes)+n_nodes[0]), periods*n_nodes[0])
        self.astgcn_layer = ASTGCN_AttentionGCN(len_input = periods, 
                                                num_for_predict = periods,
                                                in_channels = 1, 
                                                num_of_vertices = n_nodes[0],
                                                nb_block = 2, 
                                                K = 3, 
                                                nb_chev_filter = 64, 
                                                nb_time_filter = 64,
                                                # time_strides = num_of_hours,
                                                time_strides = 1,
                                                hidden = hidden)
#         self.linear_2 = torch.nn.Linear(periods, periods)
        
    def forward(self, x, edge_index):
        x1 = self.modelA(x[0], edge_index[0])
        x2 = self.modelB(x[1], edge_index[1])
        x3 = self.modelC(x[2], edge_index[2])
#         x = torch.concat((x1, x2, x3), dim=0).T.reshape(1, -1)
#         y = torch.concat((x1, x2, x3), dim=-1).unsqueeze(2)
#         x1 = torch.transpose(x1, 2, 1)
        x_speed = x[0].reshape(x[0].shape[0], 1, 1, -1)
        x1 = x1.reshape(x1.shape[0], 1, 1, -1)
        x2 = x2.reshape(x2.shape[0], 1, 1, -1)
        x3 = x3.reshape(x3.shape[0], 1, 1, -1)
        y = torch.concat((x_speed, x1, x2, x3), dim=-1)
        y = self.linear_1(F.relu(y))
        y = y.reshape(x[0].shape)
        y = self.astgcn_layer(y, edge_index[0])
#         y = self.linear_2(F.relu(y))
            
        return y
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
class DCRNN_RecurrentGCN(torch.nn.Module):
    def __init__(self, node_features, out_channels: int = 32):
        super(RecurrentGCN, self).__init__()
        self.recurrent = DCRNN(node_features, out_channels, 1)
        self.linear = torch.nn.Linear(out_channels, 1)

    def forward(self, x, edge_index, edge_weight):
        # TODO: the DCRNN class implements a DC recurrent unit. we probably need to use
        # it in a recurrent fashion for it to be used properly
        h = self.recurrent(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.linear(h)
        return h

# Model source: https://colab.research.google.com/drive/132hNQ0voOtTVk3I4scbD3lgmPTQub0KR?usp=sharing#scrollTo=gQB8MPV0sU4K
class TemporalGNN(torch.nn.Module):
    def __init__(self, node_features, periods, out_channels: int = 32):
        super(TemporalGNN, self).__init__()
        # Attention Temporal Graph Convolutional Cell
        self.tgnn = A3TGCN(in_channels=node_features, 
                           out_channels=out_channels, 
                           periods=periods)
        # Equals single-shot prediction
        self.linear = torch.nn.Linear(out_channels, periods)

    def forward(self, x, edge_index):
        """
        x = Node features for T time steps
        edge_index = Graph edge indices
        """
        h = self.tgnn(x, edge_index)
        h = F.relu(h)
        h = self.linear(h)
        return h

# a3tgcn source code: https://pytorch-geometric-temporal.readthedocs.io/en/latest/_modules/torch_geometric_temporal/nn/recurrent/attentiontemporalgcn.html#A3TGCN
# a3tgcn_example: https://github.com/benedekrozemberczki/pytorch_geometric_temporal/blob/master/examples/recurrent/a3tgcn_example.py 
class A3TGCN_RecurrentGCN(torch.nn.Module):
    def __init__(self, node_features, periods, batch_size, edge_index, out_channels: int = 32):
        super(A3TGCN_RecurrentGCN, self).__init__()
        # Attention Temporal Graph Convolutional Cell
        self.tgnn = A3TGCN2(in_channels=node_features, out_channels=out_channels, periods=periods, batch_size=batch_size)
        self.edge_index = edge_index
        # Equals single-shot prediction
        self.linear = torch.nn.Linear(out_channels, periods)

    def forward(self, x, ):
        """
        x = Node features for T time steps
        edge_index = Graph edge indices
        """
        h = self.tgnn(x, self.edge_index) # x [b, n_nodes, n_features, periods]  returns h [b, n_nodes, periods]
        h = F.relu(h) 
        h = self.linear(h)
        return h
    
    
# https://github.com/benedekrozemberczki/pytorch_geometric_temporal/blob/master/examples/recurrent/agcrn_example.py
class AGCRN_RecurrentGCN(torch.nn.Module):
    def __init__(self, node_features):
        super(RecurrentGCN, self).__init__()
        self.recurrent = AGCRN(number_of_nodes = 20,
                              in_channels = node_features,
                              out_channels = 2,
                              K = 2,
                              embedding_dimensions = 4)
        self.linear = torch.nn.Linear(2, 1)

    def forward(self, x, e, h):
        h_0 = self.recurrent(x, e, h)
        y = F.relu(h_0)
        y = self.linear(y)
        return y, h_0
        
