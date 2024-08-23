import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn.conv import ChebConv
from torch_geometric.nn import BatchNorm, global_mean_pool
from torch_geometric.data import Data, Batch

class EEGGraphChebConvNet(nn.Module):

    def __init__(self, num_feats=4, k=2):
        super(EEGGraphChebConvNet, self).__init__()

        self.cheb1 = ChebConv(num_feats,30, k)
        self.bn1 = BatchNorm(30)
        
        self.cheb2 = ChebConv(30,128, k)
        self.bn2 = BatchNorm(128)
        
        self.cheb3 = ChebConv(128,128,k)
        self.bn3 = BatchNorm(128)
        
        
        self.cheb4 = ChebConv(128, 256, k)
        self.bn4 = BatchNorm(256)
        
        self.cheb5 = ChebConv(256, 64, k)
        self.bn5 = BatchNorm(64)
        
        self.fc_block1 = nn.Linear(256, 128)
        self.fc_block2 = nn.Linear(128, 64)
        self.fc_block3 = nn.Linear(64, 32)
        self.fc_block4 = nn.Linear(32, 1)
        
    def forward(self, x, batch):

        x = torch.squeeze(x,dim=1)
        batch_data = batch.to_data_list()

        for i in range(len(batch_data)): 
            batch_data[i].x = x[i,:,:]
        x = Batch.from_data_list(batch_data).x #mandatory to call "from_data_list" to reconstruct
        
        edge_index = batch.edge_index
        edge_weights = batch.edge_attr
        
        x = self.cheb1(x, edge_index, edge_weight=edge_weights)
        x = self.bn1(x)
        x = F.leaky_relu(x)
        
        x = self.cheb2(x, edge_index, edge_weight=edge_weights)
        x = self.bn2(x)
        x = F.leaky_relu(x)
        
        x = self.cheb3(x, edge_index, edge_weight=edge_weights)
        x = self.bn3(x)
        x = F.leaky_relu(x)

        x = self.cheb4(x, edge_index, edge_weight=edge_weights)
        x = self.bn4(x)
        x = F.leaky_relu(global_mean_pool(x, batch=batch.batch))

        
#         x = self.cheb5(x, edge_index, edge_weight=edge_weights)
#         x = self.bn5(x)
#         x = x.view(-1, 1, x.size(0), x.size(1))
        
        out = F.dropout(x, p=0.3)
        out_fc1 = F.relu(self.fc_block1(out))
        out = F.dropout(out_fc1, p=0.3)
        out_fc2 = F.relu(self.fc_block2(out))
        out = F.dropout(out_fc2, p=0.3)
        out_fc3 = F.relu(self.fc_block3(out))
        out = torch.sigmoid(self.fc_block4(out_fc3))
            
        return out
        
        
                    
                

                    
             
        
    