import math
from itertools import product

import random
import params
import numpy as np
import pandas as pd
import torch
import pickle
from torch_geometric.data import Data,Dataset
from statsmodels.tsa.stattools import grangercausalitytests

########################################################################### features ###########################################################################
def compute_window_ppa(window):
    return np.max(window, axis=0) - np.min(window, axis=0)

def compute_window_upslope(window):
    return np.max(np.diff(window, axis=0), axis=0)

def compute_window_std(window):
    return np.std(window, axis=0)

def compute_window_average_slope(window):
    abs_slopes = np.abs(np.diff(window, axis=0))
    return np.max((abs_slopes[:-1] + abs_slopes[1:]) / 2, axis=0)

list_of_features = ["ppa", "upslope", "std", "average_slope"]
compute_features = {"ppa" : compute_window_ppa, "upslope" : compute_window_upslope, "std": compute_window_std, "average_slope": compute_window_average_slope}

class EEGGraphDataset(Dataset):
    """Build graph, treat all nodes as the same type
    Parameters
    ----------
    x: edge weights of 8-node complete graph                                
        There are 1 x 64 edges
    y: labels (diseased/healthy)
    num_nodes: the number of nodes of the graph. In our case, it is 8.          
    indices: Patient level indices. They are used to generate edge weights.

    Output
    ------
    a complete 8-node DGLGraph with node features and edge weights
    """

    def __init__(self, X_ids, dim, path, subjects=[], aug=False):
        # CAUTION - x and labels are memory-mapped, used as if they are in RAM.
        self.x = X_ids
        self.path = path
        self.dim = dim

        self.num_nodes = 274

        # NOTE: this order decides the node index, keep consistent!
        self.ch_names = [ 'MLC11-2805', 'MLC12-2805', 'MLC13-2805', 'MLC14-2805', 'MLC15-2805', 'MLC16-2805', 'MLC17-2805', 'MLC21-2805', 'MLC22-2805', 'MLC23-2805', 'MLC24-2805', 'MLC25-2805', 'MLC31-2805', 'MLC32-2805', 'MLC41-2805', 'MLC42-2805', 'MLC51-2805', 'MLC52-2805', 'MLC53-2805', 'MLC54-2805', 'MLC55-2805', 'MLC61-2805', 'MLC62-2805', 'MLC63-2805', 'MLF11-2805', 'MLF12-2805', 'MLF13-2805', 'MLF14-2805', 'MLF21-2805', 'MLF22-2805', 'MLF23-2805', 'MLF24-2805', 'MLF25-2805', 'MLF31-2805', 'MLF32-2805', 'MLF33-2805', 'MLF34-2805', 'MLF35-2805', 'MLF41-2805', 'MLF42-2805', 'MLF43-2805', 'MLF44-2805', 'MLF45-2805', 'MLF46-2805', 'MLF51-2805', 'MLF52-2805', 'MLF53-2805', 'MLF54-2805', 'MLF55-2805', 'MLF56-2805', 'MLF61-2805', 'MLF62-2805', 'MLF63-2805', 'MLF64-2805', 'MLF65-2805', 'MLF66-2805', 'MLF67-2805', 'MLO11-2805', 'MLO12-2805', 'MLO13-2805', 'MLO14-2805', 'MLO21-2805', 'MLO22-2805', 'MLO23-2805', 'MLO24-2805', 'MLO31-2805', 'MLO32-2805', 'MLO33-2805', 'MLO34-2805', 'MLO41-2805', 'MLO42-2805', 'MLO43-2805', 'MLO44-2805', 'MLO51-2805', 'MLO52-2805', 'MLO53-2805', 'MLP11-2805', 'MLP12-2805', 'MLP21-2805', 'MLP22-2805', 'MLP23-2805', 'MLP31-2805', 'MLP32-2805', 'MLP33-2805', 'MLP34-2805', 'MLP35-2805', 'MLP41-2805', 'MLP42-2805', 'MLP43-2805', 'MLP44-2805', 'MLP45-2805', 'MLP51-2805', 'MLP52-2805', 'MLP53-2805', 'MLP54-2805', 'MLP55-2805', 'MLP56-2805', 'MLP57-2805', 'MLT11-2805', 'MLT12-2805', 'MLT13-2805', 'MLT14-2805', 'MLT15-2805', 'MLT16-2805', 'MLT21-2805', 'MLT22-2805', 'MLT23-2805', 'MLT24-2805', 'MLT25-2805', 'MLT26-2805', 'MLT27-2805', 'MLT31-2805', 'MLT32-2805', 'MLT33-2805', 'MLT34-2805', 'MLT35-2805', 'MLT36-2805', 'MLT37-2805', 'MLT41-2805', 'MLT42-2805', 'MLT43-2805', 'MLT44-2805', 'MLT45-2805', 'MLT46-2805', 'MLT47-2805', 'MLT51-2805', 'MLT52-2805', 'MLT53-2805', 'MLT54-2805', 'MLT55-2805', 'MLT56-2805', 'MLT57-2805', 'MRC11-2805', 'MRC12-2805', 'MRC13-2805', 'MRC14-2805', 'MRC15-2805', 'MRC16-2805', 'MRC17-2805', 'MRC21-2805', 'MRC22-2805', 'MRC23-2805', 'MRC24-2805', 'MRC25-2805', 'MRC31-2805', 'MRC32-2805', 'MRC41-2805', 'MRC42-2805', 'MRC51-2805', 'MRC52-2805', 'MRC53-2805', 'MRC54-2805', 'MRC55-2805', 'MRC61-2805', 'MRC62-2805', 'MRC63-2805', 'MRF11-2805', 'MRF12-2805', 'MRF13-2805', 'MRF14-2805', 'MRF21-2805', 'MRF22-2805', 'MRF23-2805', 'MRF24-2805', 'MRF25-2805', 'MRF31-2805', 'MRF32-2805', 'MRF33-2805', 'MRF34-2805', 'MRF35-2805', 'MRF41-2805', 'MRF42-2805', 'MRF43-2805', 'MRF44-2805', 'MRF45-2805', 'MRF46-2805', 'MRF51-2805', 'MRF52-2805', 'MRF53-2805', 'MRF54-2805', 'MRF55-2805', 'MRF56-2805', 'MRF61-2805', 'MRF62-2805', 'MRF63-2805', 'MRF64-2805', 'MRF65-2805', 'MRF66-2805', 'MRF67-2805', 'MRO11-2805', 'MRO12-2805', 'MRO13-2805', 'MRO14-2805', 'MRO21-2805', 'MRO22-2805', 'MRO23-2805', 'MRO24-2805', 'MRO31-2805', 'MRO32-2805', 'MRO33-2805', 'MRO34-2805', 'MRO41-2805', 'MRO42-2805', 'MRO43-2805', 'MRO44-2805', 'MRO51-2805', 'MRO52-2805', 'MRO53-2805', 'MRP11-2805', 'MRP12-2805', 'MRP21-2805', 'MRP22-2805', 'MRP23-2805', 'MRP31-2805', 'MRP32-2805', 'MRP33-2805', 'MRP34-2805', 'MRP35-2805', 'MRP41-2805', 'MRP42-2805', 'MRP43-2805', 'MRP44-2805', 'MRP45-2805', 'MRP51-2805', 'MRP53-2805', 'MRP54-2805', 'MRP55-2805', 'MRP56-2805', 'MRP57-2805', 'MRT11-2805', 'MRT12-2805', 'MRT13-2805', 'MRT14-2805', 'MRT15-2805', 'MRT16-2805', 'MRT21-2805', 'MRT22-2805', 'MRT23-2805', 'MRT24-2805', 'MRT25-2805', 'MRT26-2805', 'MRT27-2805', 'MRT31-2805', 'MRT32-2805','MRT33-2805','MRT34-2805','MRT35-2805','MRT36-2805','MRT37-2805','MRT41-2805','MRT42-2805','MRT43-2805','MRT44-2805','MRT45-2805','MRT46-2805','MRT47-2805','MRT51-2805','MRT52-2805','MRT53-2805','MRT54-2805','MRT55-2805','MRT56-2805','MRT57-2805','MZC01-2805','MZC02-2805','MZC03-2805','MZC04-2805','MZF01-2805','MZF02-2805','MZF03-2805','MZO01-2805','MZO02-2805','MZO03-2805','MZP01-2805']
        # edge indices source to target - 2 x E = 2 x 64
        # fully connected undirected graph so 8*8=64 edges
        self.node_ids = range(len(self.ch_names))
        self.edge_index = (
            torch.tensor(
                [[a, b] for a, b in product(self.node_ids, self.node_ids)],
                dtype=torch.long,
            )
            .t()
            .contiguous()
        )

        # edge attributes - E x 1
        # only the spatial distance between electrodes for now - standardize between 0 and 1
        distances = self.load_obj('dist_matrix.pkl', "")
#         print(distances)
#         dist_min, dist_max = np.min(distances), np.max(distances)
#         normalized_distances = 1 - ((distances - dist_min) / (dist_max - dist_min))
#         self.distances = normalized_distances
#         mask = self.distances > 0.9
#         self.distances = self.distances[mask]
#         self.edge_index = self.edge_index[:, mask]

        # Granger causality
        corr = {}
        for s in subjects:
            corr_matrix = np.array(self.load_obj('correlation_'+str(s)+'.pkl','/sps/crnl/ademasson/data/correlation/'))
            corr_min, corr_max = np.min(corr_matrix, axis=-1), np.max(corr_matrix, axis=-1)
            corr_matrix = (corr_matrix - corr_min) / (corr_max - corr_min)
            corr_tensor = torch.tensor([corr_matrix[a, b].item() for a, b in product(self.node_ids, self.node_ids)],  dtype=torch.float)
            corr[s] = corr_tensor
        self.distances = corr

#         print("Vecteur distances :", self.distances, self.distances.shape)

    def __len__(self):
        return len(self.x)
    
    def name(self):
        return "EEGGraphDataset"
    
    def load_obj(self,name, path):
        with open(path+ name, 'rb') as f:
            return pickle.load(f)

    # retrieve one sample from the dataset after applying all transforms
    def __getitem__(self, index):

        win = np.array(self.x[index])[0]
        sub = np.array(self.x[index])[1]
        
        prefixe = 'data_raw_'
        #suffixe = '_b3_windows_bi'
        suffixe = '_b3_windows_bi'

        path_data = self.path 
        f = open(path_data+prefixe+str(sub).zfill(3)+suffixe)
        # Set cursor position to 30 (nb time points)*274 (nb channels)*windows_id*4 because data is stored as float32 and dtype.itemsize = 4
        f.seek(self.dim[0]*self.dim[1]*win*4)
        # From cursor location, get data from 1 window
        sample = np.fromfile(f, dtype='float32', count=self.dim[0]*self.dim[1])
        # Reshape to create a 2D array (data from the binary file is just a vector)
        sample = sample.reshape(self.dim[0],self.dim[1])

        # Calculate features for the sample
        features = np.empty((self.dim[1], len(list_of_features)))
        for i, feature_name in enumerate(list_of_features):
            feature_func = compute_features[feature_name]
            feature_value = feature_func(sample)
            min_val = np.min(feature_value, axis=0)
            max_val = np.max(feature_value, axis=0)
            features[:,i]= (feature_value - min_val) / (max_val - min_val)
        
#         gc = np.empty((self.dim[1]**2))
#         for i, (a,b) in enumerate(product(range(self.dim[1]), range(self.dim[2]))):
#             channels = np.vstack([sample[:,a], sample[:,b]]).T
#             gc[i] = grangercausalitytests(channels, 1, verbose=False)[1][0]['ssr_ftest'][1]
        # Concatenate all features into a single feature vector
        
        _x = features
        _y = np.array(self.x[index])[2]

        """suffixe2 = '_b3_corr_matrix_bi'

        g = open('/mnt/data/pmouches/data/MEG_PHRC_2006_preprocessedNEWGraphs/'+prefixe+str(sub).zfill(3)+suffixe2)
        g.seek(self.dim[1]*self.dim[1]*     win*4)
        matrix = np.fromfile(g,dtype = 'float32',count=self.dim[1]*self.dim[1])
        matrix_min, matrix_max = np.min(matrix), np.max(matrix)
        normalized_corr = 1 - ((matrix - matrix_min) / (matrix_max - matrix_min)) """ 
        
#         edge_weights = gc
        edge_weights = self.distances[sub]
#         edge_weights = self.distances
        #*normalized_corr#(self.distances + normalized_corr)/2
        #edge_weights = edge_weights/2
        mask = edge_weights > 0.9
#         edge_weights = torch.tensor(edge_weights,dtype = torch.float32)
        edge_index = self.edge_index.long()
        edge_index = edge_index[:,mask]    
        edge_weights = edge_weights[mask]
        
        data = Data(x=torch.tensor(_x,dtype = torch.float32), y=torch.tensor(_y, dtype = torch.float32), edge_index = edge_index, edge_attr=edge_weights)
        return data
        


