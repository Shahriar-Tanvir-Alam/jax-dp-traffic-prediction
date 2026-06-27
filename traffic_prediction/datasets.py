import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch_geometric_temporal.signal import StaticGraphTemporalSignal
import googlemaps
from collections import defaultdict
import geopy.distance
from traffic_prediction.aggregate import *


# https://pytorch-geometric-temporal.readthedocs.io/en/latest/_modules/torch_geometric_temporal/dataset/pems_bay.html
def normalize_data(X):
    # Normalise as in DCRNN paper (via Z-Score Method)
    # Normalize across features
    means = torch.mean(X, axis=(0, 1, 3), keepdim=True) # Mean of features
    X = X - means
    stds = torch.std(X, axis=(0, 1, 3), keepdim=True) # STD of features
    X = X / stds

    return X, means, stds


def minmax_scale(X):
    # Get the shape of the original tensor
    original_shape = X.shape
    
    # Initialize the scaled tensor
    scaled_tensor = torch.zeros_like(X)
    mins = []
    maxs = []
    # Apply min-max scaling for each feature independently
    for i in range(original_shape[2]):
        feature_tensor = X[:, :, i, :]
        min_val = feature_tensor.min()
        max_val = feature_tensor.max()
        scaled_feature_tensor = (feature_tensor - min_val) / (max_val - min_val)
        scaled_tensor[:, :, i, :] = scaled_feature_tensor
        mins.append(min_val)
        maxs.append(max_val)
    return scaled_tensor, mins, maxs


def normalize_series(series: pd.Series):
    # z-score normalization on the series (what they did in the DCRNN paper)
    return (series - series.mean()) / series.std()

def adj_mtx_to_edge_index(adj_mtx: torch.Tensor):
    """ must be a torch tensor because numpy nonzero() returns a tuple instead
        and this method doesn't work as expected
    """
    return adj_mtx.nonzero().t().contiguous()

def time_interval_to_integer_idx(df):
    time_bucket_int_idx, _ = pd.factorize(df.index.levels[0])

    new_index = pd.MultiIndex.from_product([time_bucket_int_idx, df.index.levels[1]], names=['temporal_id', 'spatial_id'])

    return df.set_index(new_index)

# Note that it is necessary that the elements in edge_index only hold indices in the range { 0, ..., num_nodes - 1}. This is needed as we want our final data representation to be as compact as possible, e.g., we want to index the source and destination node features of the first edge (0, 1) via x[0] and x[1], respectively. (source: https://pytorch-geometric.readthedocs.io/en/latest/get_started/introduction.html)
def pyg_edge_representation(edges):
    # Extract unique number of nodes and sort them
    unique_numbers = sorted(set(x for edge in edges for x in edge))

    # Create a mapping from unique node number to its index in the list
    mapping = {num: idx for idx, num in enumerate(unique_numbers)}

    # Create a new list of edges with the mapped values
    new_edges = [(mapping[start], mapping[end]) for start, end in edges]

    return new_edges, mapping
    
class RealtimeNYCDatasetLoader:    
#     def __init__(self, freq, start_datetime='02/1/2019', end_datetime='02/28/2019', data_path='../data/realtime_nyc/Y2019/M2/clean_aggregate_tlc_feb_10min.parquet'):
    def __init__(self, data_path, edges_data_path):
        # Load the aggregate data or aggregate the loaded data
        # Only aggregating data means skipping handling the missing values
#         df = pd.read_parquet(data_path)
#         feature_df = aggregate_realtime_speed(df, start_datetime, end_datetime, freq)
        feature_df = pd.read_parquet(data_path)
        self.data = time_interval_to_integer_idx(feature_df)
        self.n_features = len(self.data.columns)
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         self.device = torch.device('cpu')

        self.edges_data_path = edges_data_path
    
    def _get_edges(self):
        df_loaded = pd.read_csv(self.edges_data_path)
        edges_original_id = list(zip(df_loaded['Source'], df_loaded['Target']))

        edges, self.mapping = pyg_edge_representation(edges_original_id)
        
        G = nx.Graph()
        G.add_edges_from(edges)
        self.nodelist = list(G.nodes)
        self.nodelist.sort()
        adj_mtx = nx.to_numpy_array(G, self.nodelist)
        self.edge_index = adj_mtx_to_edge_index(torch.tensor(adj_mtx))

    def _make_feature_mtx(self, t):
        X_t = np.zeros((len(self.nodelist), self.n_features))

        for node, row in self.data.loc[t, :].iterrows():
            if node in self.mapping:
                idx = self.mapping[node]
            else:
                continue
            X_t[idx, :] = row.values

        return X_t

    def _get_feature_mtxes(self, time_snapshots):
        feature_mtxes = []
        for t in range(time_snapshots):
            feature_mtxes.append(self._make_feature_mtx(t))
        # [temporal dim, spatial dim, feature dim]
        return np.array(feature_mtxes)

    def _get_targets_and_features(self, feature_mtxes, time_snapshots, total_timesteps, num_timesteps_in):
        indices = [(i, i + (total_timesteps))
                   for i in range(time_snapshots - total_timesteps + 1)]

        features, targets = [], []
        for i, j in indices:
            features.append(feature_mtxes[i:i+num_timesteps_in, ...])
            targets.append(feature_mtxes[i+num_timesteps_in:j, ...])
        print(len(features))
        print(features[0].shape)
        self.features = np.array(features).transpose((0, 2, 3, 1))
        self.targets = np.array(targets).transpose((0, 2, 3, 1))
        
    def _get_data(self, num_timesteps_in, num_timesteps_out):
        if getattr(self, "nodelist") is None:
            raise RuntimeError("must run _get_edges() before calling this function")
        total_timesteps = num_timesteps_in + num_timesteps_out
        time_snapshots = self.data.index.get_level_values(0).nunique()
        print("total_timesteps: ", total_timesteps)
        print("time_snapshots: ", time_snapshots)
        # 1) loop over all the timesteps and make a list of feature matrices
        feature_mtxes = self._get_feature_mtxes(time_snapshots)

        # 2) generate (x,y) pairs with sliding temporal window
        self._get_targets_and_features(feature_mtxes, time_snapshots, total_timesteps, num_timesteps_in)

    def get_dataset(self, num_timesteps_in: int = 12, num_timesteps_out: int = 12) -> StaticGraphTemporalSignal:
        self._get_edges()
        self._get_data(num_timesteps_in, num_timesteps_out)
        dataset = StaticGraphTemporalSignal(edge_index=self.edge_index, edge_weight=None, features=self.features, targets=self.targets)

        return dataset


class TLCNYCDatasetLoader:
#     def __init__(self, freq, start_datetime='02/1/2019', end_datetime='02/28/2019', zones_metadata_path='../data/tlc_nyc/taxi_zones/zones_with_location_id.shp', data_path='../data/tlc_nyc/Y2019/M2/clean_aggregate_realtime_10min.parquet') -> None:
    def __init__(self, data_path, zones_metadata_path=None, edges_data_path=None) -> None:
        # zones_metadata_path: SHP file
        # edges_data_path: CSV file
        assert zones_metadata_path != None or edges_data_path != None, "Provide zones_metadata_path or edges_data_path!"
        assert not(zones_metadata_path is None and edges_data_path is None), "Provide only one of zones_metadata_path or edges_data_path!"
        if zones_metadata_path:
            self.zones_metadata_path = zones_metadata_path
        else: 
            self.edges_data_path = edges_data_path
            
        # Load the aggregate data or aggregate the loaded data
        # Only aggregating data means skipping handling the missing values
#         df = pd.read_parquet(data_path)
#         feature_df = aggregate_tlc(df, start_datetime, end_datetime, self.neighbors, freq)        
        feature_df = pd.read_parquet(data_path)

        self.data = time_interval_to_integer_idx(feature_df)
        self.n_features = len(self.data.columns)

    def _get_edges(self):
        if self.zones_metadata_path:
            zones_gdf = gpd.read_file(self.zones_metadata_path)
            neighbors = find_neighbors(zones_gdf, neighbor_function='intersects')
            edges_original_id = []
    #         for source_zone, neighbor_list in self.neighbors.items():
    #             edges_original_id.extend([(source_zone, neighbor_zone) for neighbor_zone in neighbor_list])
            self.data_nodelist = self.data.index.get_level_values(1).unique()
            for source_zone, neighbor_list in neighbors.items():
                if source_zone not in self.data_nodelist:
                    continue
                for neighbor_zone in neighbor_list:
                    if neighbor_zone in self.data_nodelist:
                        edges_original_id.append((source_zone, neighbor_zone))
        else:
            df_loaded = pd.read_csv(self.edges_data_path)
            edges_original_id = list(zip(df_loaded['Source'], df_loaded['Target']))
        
        edges, self.mapping = pyg_edge_representation(edges_original_id)
        
        G = nx.Graph()
        G.add_edges_from(edges)
        self.nodelist = list(G.nodes)
        self.nodelist.sort()
        adj_mtx = nx.to_numpy_array(G, self.nodelist)
        self.edge_index = adj_mtx_to_edge_index(torch.tensor(adj_mtx))

    def _make_feature_mtx(self, t):
        X_t = np.zeros((len(self.nodelist), self.n_features))

#         for node, row in self.data.loc[t, :].iterrows():
#             if node not in self.nodelist: continue
#             idx = self.nodelist.index(node)
#             X_t[idx, :] = row.values
        for node, row in self.data.loc[t, :].iterrows():
            if node in self.mapping:
                idx = self.mapping[node]
            else:
                continue
            X_t[idx, :] = row.values
            
        return X_t

    def _get_feature_mtxes(self, time_snapshots):
        feature_mtxes = []
        for t in range(time_snapshots):
            feature_mtxes.append(self._make_feature_mtx(t))
        # [temporal dim, spatial dim, feature dim]
        return np.array(feature_mtxes)

    def _get_targets_and_features(self, feature_mtxes, time_snapshots, total_timesteps, num_timesteps_in):
        indices = [(i, i + (total_timesteps))
                   for i in range(time_snapshots - total_timesteps + 1)]

        features, targets = [], []
        for i, j in indices:
            features.append(feature_mtxes[i:i+num_timesteps_in, ...])
            targets.append(feature_mtxes[i+num_timesteps_in:j, ...])

        self.features = np.array(features).transpose((0, 2, 3, 1))
        self.targets = np.array(targets).transpose((0, 2, 3, 1))
        
    def _get_data(self, num_timesteps_in, num_timesteps_out):
        if getattr(self, "nodelist") is None:
            raise RuntimeError("must run _get_edges() before calling this function")

        total_timesteps = num_timesteps_in + num_timesteps_out

        time_snapshots = self.data.index.get_level_values(0).nunique()

        print("total_timesteps: ", total_timesteps)
        print("time_snapshots: ", time_snapshots)
        
        # 1) loop over all the timesteps and make a list of feature matrices
        feature_mtxes = self._get_feature_mtxes(time_snapshots)

        # 2) generate (x,y) pairs with sliding temporal window
        self._get_targets_and_features(feature_mtxes, time_snapshots, total_timesteps, num_timesteps_in)
        
    def get_dataset(self, num_timesteps_in: int = 6, num_timesteps_out: int = 3):
        self._get_edges()
        self._get_data(num_timesteps_in, num_timesteps_out)
        # StaticGraphTemporalSignal: https://pytorch-geometric-temporal.readthedocs.io/en/latest/_modules/torch_geometric_temporal/signal/static_graph_temporal_signal.html
        dataset = StaticGraphTemporalSignal(self.edge_index, None, self.features, self.targets)

        return dataset

def get_graph_connections(lat_lon_dict_nodes, neighborhood_threshold):  
    graph = []   
    ids_list = list(lat_lon_dict_nodes.keys())
    for idx_id_source in range(len(ids_list)):
        for idx_id_target in range(idx_id_source+1, len(ids_list)):
            id_source = ids_list[idx_id_source]
            id_target = ids_list[idx_id_target]
            lat_lon_source_list = lat_lon_dict_nodes[id_source]
            lat_lon_target_list = lat_lon_dict_nodes[id_target]
            min_dist = float('inf')
            for lat_lon_source in lat_lon_source_list:
                for lat_lon_target in lat_lon_target_list:
                    miles_dist_iter = geopy.distance.geodesic(lat_lon_source, lat_lon_target).miles
                    min_dist = min(min_dist, miles_dist_iter)
            if min_dist<=neighborhood_threshold:
                graph.append([id_source, id_target])
    return graph

def remove_and_connect(edges, node_to_remove):
    # Find neighbors of the node to remove
    in_neighbors = [source for source, target in edges if target == node_to_remove]
    out_neighbors = [target for source, target in edges if source == node_to_remove]

    # Create new edges that connect in-neighbors to out-neighbors
    new_edges = [(in_node, out_node) for in_node in in_neighbors for out_node in out_neighbors]

    # Remove the edges that contain the node to remove
    edges = [edge for edge in edges if node_to_remove not in edge]

    # Add the new edges to the list and return
    return edges + new_edges

def get_decoded_coordinates(df_real_time):
    decoded_coordinates_dict = defaultdict(int)

    # https://github.com/googlemaps/google-maps-services-python/tree/645e07de5a27c4c858b2c0673f0dd6f23ca62d28
    # https://github.com/googlemaps/google-maps-services-python/blob/645e07de5a27c4c858b2c0673f0dd6f23ca62d28/googlemaps/convert.py#L289
    # https://developers.google.com/maps/documentation/utilities/polylineutility?csw=1
    for index_row, row in df_real_time.iterrows():
        encoded_polyline = row.encoded_poly_line
        if row.id in decoded_coordinates_dict:
            continue
        else:
            try:
                decoded_coordinates_dict[row.id] = googlemaps.convert.decode_polyline(encoded_polyline)
            except:
    #             print(f"Zone {row.ID} at row {index_row} did not work.")
                pass
   
    return decoded_coordinates_dict

def borough_mapping(borough):
    borough_mapping_dict = {'Staten Island': 'SI', 'Manhattan': "M", 'Bronx': "BRX", 'Brooklyn': "BK", \
                   'Staten island': 'SI', 'Queens': "Q"}
    return borough_mapping_dict[borough]

if __name__ == '__main__':
    print('Running dataset.py ...')
#     loader = TLCNYCDatasetLoader('15min')
#     dataset = loader.get_dataset()
#     print(dataset)