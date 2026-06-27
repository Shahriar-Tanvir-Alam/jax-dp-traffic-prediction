
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import STL
from train_private import build_network, build_criterion, get_model_name, compute_val_loss, get_data_params, get_data_object, \
                        get_data_loader_path
import torch
from torch_geometric_temporal.signal import StaticGraphTemporalSignal
from traffic_prediction.datasets import normalize_data, minmax_scale
from traffic_prediction.utils import set_seed
import scipy.stats as stats
import math

#########################################################################################################
########################################### Get bootstrap data ##########################################
#########################################################################################################
def get_boot_input_data(config, 
                        data_params_list,
                        n_boot_series, 
                        period, 
                        block_size, 
                        overlapping,
                        robust,
                        device,
                        nonstationarity_alpha,
                        verbose = False,
                        train_val_test = 'test',
                       ):
    print("\nPrepare feature matrcies for bootstrapping")
    # Get the feature matrices of the datasets
    print("\nGet the feature matrices of the datasets (get_feature_mtxes_dict).")
    loader_dict, feature_mtxes_dict, nodes_dict, n_nodes_list, node_features_list = get_feature_mtxes_dict(
                                                                                         data_params_list=data_params_list,
                                                                                         config=config,
                                                                                         )
    # Get the ratio of train/validation/test split to separate the test data accordingly
    print("\nGet the ratio of train/validation/test split to separate the test data accordingly (get_n_train_val_test_dict).")
    n_train_val_test_dict = get_n_train_val_test_dict(feature_mtxes_dict=feature_mtxes_dict,
                                                      n_input_periods=config.n_input_periods,
                                                      n_output_periods=config.n_output_periods,
                                                      train_ratio=config.train_ratio, 
                                                      val_ratio=config.val_ratio,
                                                     )
    # Get the test data portion from feature matrices
    print("\nGet the test data portion from feature matrices (get_train_val_test_feature_mtxes).")
    _, _, feature_mtxes_test_dict = get_train_val_test_feature_mtxes(feature_mtxes_dict = feature_mtxes_dict,
                                                                    n_train_val_test_dict = n_train_val_test_dict,
                                                                    n_input_periods=config.n_input_periods,
                                                                    n_output_periods=config.n_output_periods,
                                                                    verbose = verbose,
                                                                    )
    if train_val_test == 'test':
        feature_mtxes_dict = feature_mtxes_test_dict
    else:
        assert 1 == 2, print('Develop pending!')
        
    # STL decomposition of feature matrices
    print("\nIs data nonstainary? (check_feature_mtxes_stationarity).")
    nonstationary_data = check_feature_mtxes_stationarity(alpha = nonstationarity_alpha, 
                                                          feature_mtxes_dict = feature_mtxes_dict,
                                                          verbose = False,
                                                         )
    
    if nonstationary_data:
        print("\nNonstationary data >> STL decomposition of feature matrices (get_feature_mtxes_stl_decomposition).")
        node_data_decomposed_dict = get_feature_mtxes_stl_decomposition(feature_mtxes_dict=feature_mtxes_dict, 
                                                                    nodes_dict=nodes_dict,
                                                                    n_train_val_test_dict=n_train_val_test_dict,
                                                                    period=period,
                                                                    robust=robust,
                                                                   )
    else:
        print("\nStainary data >> (node_level_data_decomposition).")
        node_data_decomposed_dict = node_level_data_decomposition(feature_mtxes_dict=feature_mtxes_dict, 
                                                                    nodes_dict=nodes_dict,
                                                                   )
        
        
    # Get bootsrapped series from decomposed data
    print("\nBootstrapping data")
    bootstrapped_series_final_dict = {}
    for dataset_name in node_data_decomposed_dict.keys():
        if nonstationary_data:
            residuals = node_data_decomposed_dict[dataset_name]['residuals']
        else:
            residuals = node_data_decomposed_dict[dataset_name]
            
        bootstrapped_master_time_series = bootstrapping_residuals(
                                                              residuals = residuals, \
                                                              nodes = nodes_dict[dataset_name], \
                                                              n_boot_series = n_boot_series, \
                                                              block_size = block_size,
                                                              overlapping=overlapping,
                                                                    verbose = verbose,
                                                             )
        bootstrapped_series = build_bootstrapped_df(bootstrapped_series=bootstrapped_master_time_series, \
                                                    nodes = nodes_dict[dataset_name], \
                                                    residuals = residuals, \
                                                    block_size = block_size,
                                                                    verbose = verbose,)
        
        if nonstationary_data:
            bootstrapped_series_final_dict[dataset_name] = add_trend_seasonality(
                                                  bootstrapped_series = bootstrapped_series, \
                                                  node_data_decomposed = node_data_decomposed_dict[dataset_name], \
                                                  n_boot_series = n_boot_series,
                                                  nodes = nodes_dict[dataset_name], 
                                                                    verbose = verbose,)
        else:
            bootstrapped_series_final_dict[dataset_name] = bootstrapped_series
        
    print("\nPrepare bootstrapped data for evaluation")
    # Get features matrices from the bootsrapped samples
    boot_feature_mtxes_dict = get_feature_mtxes_from_boot_samples(bootstrapped_series_final_dict=bootstrapped_series_final_dict,
                                                             n_boot_series = n_boot_series,
                                                             )
    # Create StaticGraphTemporalSignal from feature matrices
    dataset_StaticGraphTemporalSignal_list = []
    for boot_idx in range(len(boot_feature_mtxes_dict.keys())):
        print("\nboot_idx: ", boot_idx)
        dataset_StaticGraphTemporalSignal_dict, static_edge_index_list = get_StaticGraphTemporalSignal(
                                                      data_params_list = data_params_list, 
                                                      loader_dict = loader_dict,
                                                      feature_mtxes_dict = boot_feature_mtxes_dict[boot_idx], 
                                                      n_input_periods = config.n_input_periods,
                                                      n_output_periods = config.n_output_periods,
                                                      device=device
                                                     )
        dataset_StaticGraphTemporalSignal_list.append(dataset_StaticGraphTemporalSignal_dict)
    
    # Create DataLoader for evaluation based on StaticGraphTemporalSignals
    boot_DataLoader_list = []
    boot_param_norm_x_target_dict_list = []
    for boot_idx, dataset_StaticGraphTemporalSignal_dict in enumerate(dataset_StaticGraphTemporalSignal_list):
        print("\nboot_idx: ", boot_idx)
        DataLoader_list, static_edge_index_list, param_norm_x_target_dict = \
                                               get_DataLoader_dict(
                                               dataset_StaticGraphTemporalSignal_dict=dataset_StaticGraphTemporalSignal_dict, 
                                               data_params_list=data_params_list, 
                                               normalize = config.normalize,
                                               device = device,
                                               batch_size = 1)
        boot_DataLoader_list.append(DataLoader_list)
        # Customize the param_norm_x_target_dict so that it has mean and std for all trian-val-test parts
        param_norm_x_target_dict_customized_for_compute_val = {}
        for dataname_param_norm, param_norm_x_target in param_norm_x_target_dict.items():
            # Focus is only one part here >> mu, std of other parts are empty
            if train_val_test == 'test':
#                 train_mu_std = []
#                 test_mu_std = param_norm_x_target
#                 val_mu_std = []
#                 param_norm_list = [train_mu_std, test_mu_std, val_mu_std]
                param_norm_x_target_dict_customized_for_compute_val[dataname_param_norm] = [[], param_norm_x_target, []]
            else:
                assert 1 == 2, print('Develop pending!')
        boot_param_norm_x_target_dict_list.append(param_norm_x_target_dict_customized_for_compute_val)
    return boot_DataLoader_list, n_nodes_list, node_features_list, static_edge_index_list, boot_param_norm_x_target_dict_list


######################################## Graph Data Preparation #########################################
def get_feature_mtxes_dict(data_params_list, config):
    # Get feature_mtxes of datasets
    loader_dict = {}
    feature_mtxes_dict = {}
    nodes_dict = {}
    n_nodes_list = []
    node_features_list = []
    for dataset_name, datapath, params_iter, config_iter in data_params_list:
        print("dataset_name: ", dataset_name)
        if dataset_name != "realtime":
            company = params_iter.company.lower()
        else:
            company = ""
        # Get data object instance
#         loader = get_data_object(config.year, config.start_month, config.end_month, 
#                                  config.boroughs, config.bucket_size, dataset_name, 
#                                  datapath, config.graph_version, config.company)
        datapath_data, datapath_graph = get_data_loader_path(params_iter.year, params_iter.start_month, params_iter.end_month, 
                                 params_iter.boroughs, params_iter.bucket_size, dataset_name, 
                                 datapath, params_iter.graph_version, company)
        loader = get_data_object(dataset_name, datapath_data, datapath_graph)

        # Create graph
        loader._get_edges()
        # Create feature_mtxes
#         total_timesteps = config.n_input_periods + config.n_output_periods
        time_snapshots = loader.data.index.get_level_values(0).nunique()
        # 1) loop over all the timesteps and make a list of feature matrices
        feature_mtxes = loader._get_feature_mtxes(time_snapshots)
        loader_dict[dataset_name] = loader
        feature_mtxes_dict[dataset_name] = feature_mtxes
        nodes_dict[dataset_name] = loader.nodelist
        n_nodes_list.append(len(loader.nodelist)) # the order is globally used during all the phases of training and bootstrapping
        node_features_list.append(loader.n_features)
    return loader_dict, feature_mtxes_dict, nodes_dict, n_nodes_list, node_features_list


def get_n_train_val_test(feature_mtxes, n_input_periods, n_output_periods, train_ratio, val_ratio):
    print(f"feature_mtxes.shape = {feature_mtxes.shape}")
    total_samples_features_targets_split = feature_mtxes.shape[0] - \
                                            (n_input_periods + n_output_periods) + 1 
    print("total_samples: ", total_samples_features_targets_split)
    n_train = int(total_samples_features_targets_split * train_ratio)
    n_validation_test = total_samples_features_targets_split - n_train
    test_ratio_total = 1.0 - (train_ratio + val_ratio)
    test_ratio = val_ratio/(1-train_ratio)
    n_validation = int(n_validation_test * test_ratio)
    n_test = int(n_validation_test - n_validation)
    print("Train/Validation/Test: ", train_ratio, "/", val_ratio, "/",  test_ratio_total)
    print("Train/Validation/Test: ", n_train, "/", n_validation, "/",  n_test)
#     temporal_id_train, temporal_id_validation, temporal_id_test = (0, n_train-1)
    return [n_train, n_validation, n_test]
    
def get_n_train_val_test_dict(feature_mtxes_dict, n_input_periods, n_output_periods, train_ratio, val_ratio):
    n_train_val_test_dict = {}
    for dataset_name, feature_mtxes in feature_mtxes_dict.items():
        print(f"\n{dataset_name}: ")
        n_train_val_test = get_n_train_val_test(feature_mtxes, 
                                                n_input_periods, 
                                                n_output_periods, 
                                                train_ratio, 
                                                val_ratio)
        n_train_val_test_dict[dataset_name] = n_train_val_test
    return n_train_val_test_dict


def get_test_feature_mtxes(feature_mtxes_dict,
                            n_train_val_test_dict,
                          ):
    feature_mtxes_test_dict = {}
    for dataset_name, feature_mtxes in feature_mtxes_dict.items():
        print(f"Get feature matrices of test dataset *{dataset_name}*")
        feature_mtxes_test_dict[dataset_name] = feature_mtxes[-n_train_val_test_dict[dataset_name][-1]:]
    return feature_mtxes_test_dict

def get_train_val_test_feature_mtxes(feature_mtxes_dict,
                                     n_train_val_test_dict,
                                     n_input_periods,
                                     n_output_periods,
                                     verbose=False,
                                      ):
    feature_mtxes_train_dict = {}
    feature_mtxes_val_dict = {}
    feature_mtxes_test_dict = {}
    correcting_count = n_input_periods + n_output_periods - 1
    for dataset_name, feature_mtxes in feature_mtxes_dict.items():
        if verbose: print(f"\nGet feature matrices of dataset *{dataset_name}*")
        n_train, n_val, n_test = n_train_val_test_dict[dataset_name][0], n_train_val_test_dict[dataset_name][1], n_train_val_test_dict[dataset_name][2]
        idx_train_start = 0 
        idx_train_end = idx_train_start + (n_train - 1) + correcting_count +1 # +1 for excluding idx
        if verbose: print(f"(idx_train_start, idx_train_end) = ({idx_train_start}, {idx_train_end})")
        feature_mtxes_train_dict[dataset_name] = feature_mtxes[:idx_train_end]
        
        idx_val_start = n_train
        idx_val_end = idx_val_start + (n_val - 1) + correcting_count +1 # +1 for excluding idx
        if verbose: print(f"(idx_val_start, idx_val_end) = ({idx_val_start}, {idx_val_end})")
        if verbose: print(f"idx_val_end-idx_val_start = {idx_val_end-idx_val_start}")
        feature_mtxes_val_dict[dataset_name] = feature_mtxes[idx_val_start:idx_val_end]
        
        idx_test_start = n_train + n_val
        idx_test_end = idx_test_start + (n_test - 1) + correcting_count +1 # +1 for excluding idx
        if verbose: print(f"(idx_test_start, idx_test_end) = ({idx_test_start}, {idx_test_end})")
        if verbose: print(f"idx_test_end-idx_test_start = {idx_test_end-idx_test_start}")
        feature_mtxes_test_dict[dataset_name] = feature_mtxes[idx_test_start:idx_test_end]
        
        if verbose: print(f"train shapes: {feature_mtxes_train_dict[dataset_name].shape} where feature_mtxes_train = feature_mtxes[0:{idx_train_end}]")
        if verbose: print(f"validation shapes: {feature_mtxes_val_dict[dataset_name].shape} where feature_mtxes_val = feature_mtxes[{idx_val_start}:{idx_val_end}]")
        if verbose: print(f"test shapes: {feature_mtxes_test_dict[dataset_name].shape} where feature_mtxes_test_dict = feature_mtxes[{idx_test_start}:{idx_test_end}]")
    
#     feature_mtxes_test_dict = {}
#     for dataset_name, feature_mtxes in feature_mtxes_dict.items():
#         if verbose: print(f"Get feature matrices of test dataset *{dataset_name}*")
#         feature_mtxes_test_dict[dataset_name] = feature_mtxes[-n_train_val_test_dict[dataset_name][-1]:]
    return feature_mtxes_train_dict, feature_mtxes_val_dict, feature_mtxes_test_dict


def get_feature_mtxes_stl_decomposition(feature_mtxes_dict,
                                        nodes_dict,
                                        n_train_val_test_dict, 
                                        period, 
                                        robust = True):
    node_data_decomposed_dict = {}
    for dataset_name, feature_mtxes in feature_mtxes_dict.items():
        print(f"STL decomposition of test data of dataset *{dataset_name}*")
        node_data_decomposed_dict[dataset_name] = STL_decomposition(feature_mtxes = feature_mtxes, 
                                                                     nodelist = nodes_dict[dataset_name], 
                                                                     period = period, 
                                                                     robust = robust,
                                                                    )
    return node_data_decomposed_dict

def node_level_data_decomposition(feature_mtxes_dict, 
                                  nodes_dict,
                                 ):
    node_level_data_decomposed = {}
    for dataset_name, feature_mtxes in feature_mtxes_dict.items():
        node_level_data_decomposed[dataset_name] = {}
        for node in nodes_dict[dataset_name]: # iterate over all the nodes
            node_level_data_decomposed[dataset_name][node] = np.squeeze(feature_mtxes[:, node], 1)
    return node_level_data_decomposed


def get_feature_mtxes_from_boot_samples(bootstrapped_series_final_dict,
                                       n_boot_series,
                                       ):
    boot_feature_mtxes_dict = {}
    for boot_idx in range(n_boot_series):
        boot_feature_mtxes_dict[boot_idx] = {}
        for dataset_name, bootstrapped_series_final in bootstrapped_series_final_dict.items():
            # Get data size
            n_node = len(bootstrapped_series_final_dict[dataset_name].keys())
            assert n_node > 0, "n_node is not positive"
            n_sample = len(bootstrapped_series_final_dict[dataset_name][0][0])
            assert n_sample > 0, "n_sample is not positive"
            shape_data = (n_sample, n_node, 1)

            # Create the boot_feature_mtxes
            boot_feature_mtxes_dict[boot_idx][dataset_name] = np.zeros(shape_data)
            for node_idx, node in enumerate(bootstrapped_series_final_dict[dataset_name].keys()):
                boot_feature_mtxes_dict[boot_idx][dataset_name][:, node_idx, :] = \
                        np.expand_dims(bootstrapped_series_final_dict[dataset_name][node][boot_idx], axis=1)
    return boot_feature_mtxes_dict

def get_StaticGraphTemporalSignal(data_params_list, 
                                  loader_dict,
                                  feature_mtxes_dict, 
                                  n_input_periods,
                                  n_output_periods,
                                  device,
                                 ):
    dataset_StaticGraphTemporalSignal_dict = {}
    static_edge_index_list = []
    for dataset_name, datapath, params_iter, config in data_params_list:
        print("dataset_name: ", dataset_name)
        total_timesteps = n_input_periods + n_output_periods
        # loader_dict[dataset_name].data does not match with feature_mtxes_dict[dataset_name] in 
        # bootstrapping test data
        time_snapshots = feature_mtxes_dict[dataset_name].shape[0] 
        print("time_snapshots: ", time_snapshots)
        # 2) generate (x,y) pairs with sliding temporal window
        print("feature_mtxes_dict[dataset_name].shape: ", feature_mtxes_dict[dataset_name].shape)
        loader_dict[dataset_name]._get_targets_and_features(feature_mtxes_dict[dataset_name], 
                                         time_snapshots, 
                                         total_timesteps, 
                                         n_input_periods)
#         loader_dict[dataset_name]._get_data(n_input_periods, n_output_periods)
        # StaticGraphTemporalSignal: https://pytorch-geometric-temporal.readthedocs.io/en/latest/_modules/torch_geometric_temporal/signal/static_graph_temporal_signal.html
        dataset_StaticGraphTemporalSignal = StaticGraphTemporalSignal(loader_dict[dataset_name].edge_index, None, 
                                            loader_dict[dataset_name].features, 
                                            loader_dict[dataset_name].targets)
        print("loader_dict[dataset_name].features.shape: ", loader_dict[dataset_name].features.shape)
        print("loader_dict[dataset_name].targets.shape: ", loader_dict[dataset_name].targets.shape)
        dataset_StaticGraphTemporalSignal_dict[dataset_name] = dataset_StaticGraphTemporalSignal
        
        # Loading the graph once because it's a static graph
        for snapshot in dataset_StaticGraphTemporalSignal:
            static_edge_index = snapshot.edge_index.to(device)
            break 
        static_edge_index_list.append(static_edge_index)
        
    return dataset_StaticGraphTemporalSignal_dict, static_edge_index_list


def get_single_DataLoader(dataset, normalize, device, batch_size = 1, shuffle = 0):
    # Fix randomization
    set_seed(5)
    
    shuffle = False if shuffle==0 else True
    # NumPy to Torch 
    x_tensor = torch.from_numpy(dataset.features).type(torch.FloatTensor).to(device)  # (B, N, F, T)
    target_tensor = torch.from_numpy(dataset.targets).type(torch.FloatTensor).to(device) # (B, N, F, T)
    # normalize features and target separately
    if normalize == 1:    
        x_tensor, param_norm_x_1, param_norm_x_2 = normalize_data(x_tensor)
        target_tensor, param_norm_target_1, param_norm_target_2 = normalize_data(target_tensor)
    elif normalize == 2:
        x_tensor, param_norm_x_1, param_norm_x_2 = minmax_scale(x_tensor)
        target_tensor, param_norm_target_1, param_norm_target_2 = minmax_scale(target_tensor)
        # Assumption: data has only 1 feature, otherwise, there is a min and a max for each feature
        param_norm_x_1 = param_norm_x_1[0]
        param_norm_x_2 = param_norm_x_2[0]
        param_norm_target_1 = param_norm_target_1[0]
        param_norm_target_2 = param_norm_target_2[0]
        
    # Reshape target tensor
    target_tensor = torch.squeeze(target_tensor, 2)  # (B, N, T)
    print(f"x_tensor.shape: {x_tensor.shape}, target_tensor.shape: {target_tensor.shape}")
    # Get DataLoader
    dataset_new = torch.utils.data.TensorDataset(x_tensor, target_tensor)
    loader = torch.utils.data.DataLoader(dataset_new, batch_size=batch_size, shuffle=shuffle) 
    
    # Loading the graph once because it's a static graph
    for snapshot in dataset:
        static_edge_index = snapshot.edge_index.to(device)
        break 
    
    param_norm_x_target = [param_norm_x_1.item(), param_norm_x_2.item(), param_norm_target_1.item(), param_norm_target_2.item()]
    return x_tensor, target_tensor, dataset_new, loader, static_edge_index, param_norm_x_target

def get_DataLoader_dict(dataset_StaticGraphTemporalSignal_dict, 
                               data_params_list, 
                               normalize, 
                               device, 
                               batch_size = 1,
                               shuffle = 0):
    DataLoader_list = []
    static_edge_index_list = []
    param_norm_x_target_dict = {}
    for dataset_name, datapath, params_iter, config in data_params_list:
        print(f"\n{dataset_name}: ")
        dataset_StaticGraphTemporalSignal = dataset_StaticGraphTemporalSignal_dict[dataset_name]
        x_tensor, target_tensor, dataset_new, DataLoader_temp, static_edge_index, param_norm_x_target = get_single_DataLoader(
                                                              dataset=dataset_StaticGraphTemporalSignal, 
                                                              normalize = normalize,
                                                              device = device,
                                                              batch_size = 1, # evalutation with batch size of 1
                                                               shuffle = 0)
        
#         data[dataset_name] = [x_tensor, target_tensor, dataset_new, DataLoader_temp]
        DataLoader_list.append(DataLoader_temp)
        static_edge_index_list.append(static_edge_index)
        param_norm_x_target_dict[dataset_name] = param_norm_x_target # Mean and STD of data portion (train or val or test)
    return DataLoader_list, static_edge_index_list, param_norm_x_target_dict


########################################### STL Decomposition ##########################################
# - STL paper: https://www.wessa.net/download/stl.pdf
# - Lecture on STL: https://cbergmeir.com/talks/bergmeir2014ISF_slides.pdf
        
# Period [https://stackoverflow.com/questions/75792509/understanding-period-parameter-in-statsmodel-tsa-seasonal]:
# - Period could be defined as: Expected samples in a full cycle / repetition of the seasonality component.
    
# STL parameters [https://otexts.com/fpp2/stl.html]:
# - The two main parameters to be chosen when using STL are the trend-cycle window (t.window) and the seasonal window (s.window). 
# - These control how rapidly the trend-cycle and seasonal components can change. 
# - Smaller values allow for more rapid changes. 
# - Both trend window and seasonality window should be odd numbers; 
#     - Trend window is the number of consecutive observations to be used when estimating the trend-cycle; 
#     - Seasonality window is the number of consecutive years to be used in estimating each value in the seasonal component. 
# - The user must specify seasonality window as there is no default. Setting it to be infinite is equivalent to forcing the seasonal component to be periodic (i.e., identical across years). 
# - Specifying trend window is optional, and a default value will be used if it is omitted.

# https://www.statsmodels.org/dev/examples/notebooks/generated/stl_decomposition.html#LOESS-degree
def STL_decomposition(feature_mtxes, nodelist, period, robust=True):
    node_level_data_decomposed = {}
    node_level_data_decomposed['STL'] = {}
    node_level_data_decomposed['residuals'] = {}
    node_level_data_decomposed['trend'] = {}
    node_level_data_decomposed['seasonality'] = {}
    for node in nodelist: # iterate over all the nodes
        node_level_data_decomposed['STL'][node] = STL(feature_mtxes[:, node], 
                                                      period = period, 
                                                      robust = robust,
                                                     ).fit()
        node_level_data_decomposed['residuals'][node] = node_level_data_decomposed['STL'][node].resid
        node_level_data_decomposed['trend'][node] = node_level_data_decomposed['STL'][node].trend
        node_level_data_decomposed['seasonality'][node] = node_level_data_decomposed['STL'][node].seasonal
    return node_level_data_decomposed


######################################## Moving Block Bootstrap #########################################
# This function creates the population matrix from which we will draw randomly blocks of the size=432 (default).
# If intervals are 10-minute, 432 means 3 days
# Each row represents one block of the size block_size
# Matrix shape: # number of blocks x block size 
# block_size = period length (e.g., 365 for days in a year, 12 for months in a year, 7 for days in a week)
def build_matrix_overlapping(residuals, block_size, verbose=False):
    n_residuals = len(residuals)
    if verbose: print("\n\n\n<< build_matrix_overlapping >>")
    if verbose: print("n_residuals: ", n_residuals)
    maximum_number_of_blocks = n_residuals - block_size + 1
    if verbose: print("maximum_number_of_blocks: ", maximum_number_of_blocks)
    block_candidates = np.zeros(shape=(block_size, maximum_number_of_blocks), dtype=float)
    for i in range(maximum_number_of_blocks):
        block_candidates[:, i] = residuals[i:i+block_size]
    block_candidates = block_candidates.T
    if verbose: print("block_candidates created with the size of: ", block_candidates.shape)
    return block_candidates


def build_matrix_nonoverlapping(residuals, block_size, verbose=False):
#     set_seed(5)
    n_residuals = len(residuals)
    if verbose: print("\n\n\n<< build_matrix nonoverlapping>>")
    if verbose: print("n_residuals: ", n_residuals)
    maximum_number_of_blocks = n_residuals//block_size
    if verbose: print("maximum_number_of_blocks: ", maximum_number_of_blocks)
    
#     n_remained_residuals = n_residuals - maximum_number_of_blocks*block_size
#     print("n_remained_residuals: ", n_remained_residuals)
#     random_idx_start = np.random.randint(0, n_remained_residuals) # exclusive range
#     print("random_idx_start: ", random_idx_start)
    
        
    n_remained_residuals = n_residuals - maximum_number_of_blocks*block_size
#     print("n_remained_residuals: ", n_remained_residuals)
    block_candidates_dict = {}
    for block_candidate_start in range(n_remained_residuals):
#         print("block_candidate_start: ", block_candidate_start)
        block_candidates = np.zeros(shape=(block_size, maximum_number_of_blocks), dtype=float)
        for i in range(maximum_number_of_blocks):
            block_candidates[:, i] = residuals[block_candidate_start+i*block_size:block_candidate_start+(i+1)*block_size]
#             print("block_candidate_start+i*block_size, block_candidate_start+(i+1)*block_size: ", block_candidate_start+i*block_size, block_candidate_start+(i+1)*block_size)
    #         print(residuals[i*block_size:(i+1)*block_size])
    #         print("residuals.shape: ", residuals.shape)
    #         print("i*block_size, (i+1)*block_size ", i*block_size, (i+1)*block_size)
        block_candidates = block_candidates.T
        if verbose: print("block_candidates created with the size of: ", block_candidates.shape)
        block_candidates_dict[block_candidate_start] = block_candidates
    return block_candidates_dict


# This function draws randomly from the population matrix and returns the desired number of 
# bootstrapped residuals time series
def bootstrapping_residuals(residuals, nodes, overlapping, n_boot_series=3, block_size=432, verbose=False):
    if verbose: print("\n\n\n<< bootstrapping_residuals >>")
    set_seed(5)
    n_residuals = len(residuals[0]) # len of residuals of node 0. It's similar in all the nodes.
    if verbose: print("residuals[0]: ", residuals[0])
    if verbose: print("n_residuals = residuals[0]: ", len(residuals[0]))
    if verbose: print("block_size: ", block_size)
    if overlapping:
        maximum_number_of_blocks = n_residuals - block_size + 1
        if verbose: print("maximum_number_of_blocks (n_residuals-block_size+1): ", maximum_number_of_blocks)
    else:
        maximum_number_of_blocks = n_residuals // block_size
        if verbose: print("maximum_number_of_blocks (n_residuals//block_size): ", maximum_number_of_blocks)
        n_remained_residuals = n_residuals - maximum_number_of_blocks*block_size
        if verbose: print("n_remained_residuals: ", n_remained_residuals)
    # Number of selected blocks
    number_of_drafts = int(n_residuals/block_size)
    if verbose: print("number_of_drafts = int(n_residuals/block_size): ", number_of_drafts)
    
    # Initialize bootstrapped_collection_time_series
    bootstrapped_collection_time_series = {}
    for node in nodes:
        bootstrapped_collection_time_series[node] = []

    # Create the block_candidates matrix of nodes
    block_candidates = {}
    for node in nodes:
        if overlapping:
            block_candidates[node] = build_matrix_overlapping(residuals[node], \
                                                              block_size = block_size, \
                                                              verbose = verbose)
        else:
            block_candidates[node] = build_matrix_nonoverlapping(residuals[node], \
                                                                  block_size = block_size, \
                                                                  verbose = verbose)
            
        if verbose: print(f"block_candidates[{node}].shape: ", block_candidates[node].shape)
        if verbose: print(f"block_candidates[{node}]: ", block_candidates[node])
        
    # number_of_drafts = # of Drafts for each bootstrapped time series
    # n_boot_series = # of series that will be bootstrapped
    # Bootstrap the series
    for bootstrapped_block_idx in range(n_boot_series):
        # Initialize bootstrapped_single_time_series
        bootstrapped_single_time_series = {}
        for node in nodes:
            bootstrapped_single_time_series[node] = []
        if verbose: print("\n\n\nbootstrapped_block_idx: ", bootstrapped_block_idx)

        if not overlapping:
            random_idx_start = np.random.randint(0, n_remained_residuals) # exclusive range
#             print("random_idx_start: ", random_idx_start)
        # Take drafts to build a single bootstraped series
        for draft_idx in range(number_of_drafts+2):
        # inner loop to draft blocks for time series boot_idx
        # here we draft each draft draft_idx a random block from the population matrix
            random_series_draft = np.random.randint(0, maximum_number_of_blocks)
            if verbose: print("\nrandom_series_draft: ", random_series_draft)
            for node in nodes:
                if verbose: print("node: ", node)
                    
                if overlapping:
                    sample_block = block_candidates[node][random_series_draft, :]
                else:
                    sample_block = block_candidates[node][random_idx_start][random_series_draft, :]
                    
                if verbose: print(f"sample_block.shape: ", sample_block.shape)
#                 if verbose: print(f"sample_block[:2]: ", sample_block[:2])
                if verbose: print(f"sample_block: ", sample_block)
                # After the last iteration in the inner loop 
                # bootstrapped_single_time_series is a bootstrapped time series that 
                bootstrapped_single_time_series[node].append(sample_block)
        if verbose: 
              for node in nodes:
                    print(f"\t> bootstrapped_single_time_series of node {node}: ", \
                              bootstrapped_single_time_series[node])
        if verbose: print("\n\n")
        for node in nodes:
            bootstrapped_collection_time_series[node].append(bootstrapped_single_time_series[node])
            if verbose: print(f"bootstrapped_collection_time_series[{node}]: \
                                        {bootstrapped_collection_time_series[node]}")
#         if verbose: print("\n>>>> len(bootstrapped_collection_time_series): ", len(bootstrapped_collection_time_series))
        if verbose: print("\n>>>> Last bootstrapped series of node 0:\nlen(bootstrapped_collection_time_series[0][-1]): ", len(bootstrapped_collection_time_series[0][-1]))
        if verbose: print("\n>>>> bootstrapped_collection_time_series[0][-1][-1].shape: ", bootstrapped_collection_time_series[0][-1][-1].shape)
    return bootstrapped_collection_time_series


def build_bootstrapped_df(bootstrapped_series, nodes, residuals, block_size=432, verbose=False):
    if verbose: print("\n\n\n<< build_bootstrapped_df >>")
    set_seed(5) # Fix seed
    # Initialize temp_series
    temp_series = {}
    for node in nodes:
        temp_series[node] = []
    # Iterate over bootsrapped series and fix their size if they are longer than the residuals length
    for i in range(len(bootstrapped_series[0])):
        if verbose: print(f"\n\n\nBootstrapped series **{i}**")
        # discard random number of values between 0 and block_size - 1 at the beginning
        number_of_elements_to_remove = np.random.randint(0, block_size-1)
        if verbose: print("number_of_elements_to_remove from the start: ", number_of_elements_to_remove)
#         print("number_of_elements_to_remove: ", number_of_elements_to_remove)

        for node in nodes:
            if verbose: print(f"\tnode: {node}")
            temp_series[node].append(np.concatenate(bootstrapped_series[node][i]))
            if verbose: print(f"\tBefore >> temp_series[{node}][{i}]: ", temp_series[node][i])
#             print(f"temp_series[{node}]: ", temp_series[node])
    
            temp_series[node][i] = temp_series[node][i][number_of_elements_to_remove:]
            if verbose: print(f"\tAfter start truncation>> temp_series[{node}][{i}]: ", temp_series[node][i])

            # Discard all elements after the last element of the original training series
            if len(temp_series[node][i]) > len(residuals[node]):
                if verbose: print("\tnumber_of_elements_to_remove from the end: ", 
                      len(temp_series[node][i])-len(residuals[node]))
                temp_series[node][i] = temp_series[node][i][:len(residuals[node])]  
                if verbose: print(f"\tAfter end truncation>> temp_series[{node}][{i}]: ", temp_series[node][i]) 
#             temp_series[node][i] = temp_series[node][i].reshape(-1,1)
    #    temp_series[i] = temp_series[i] + df_training['trend'].values + df_training['seasonality'].values
    if verbose: 
        for node in nodes:
            print(f"\n>> temp_series[{node}]: {temp_series[node]}")
    return temp_series


def add_trend_seasonality(bootstrapped_series, node_data_decomposed, n_boot_series, \
                          nodes, verbose=False):
    bootstrapped_series_final = {}
    for node in nodes:
        bootstrapped_series_final[node] = {}
        if verbose: print("\nnode: ", node)
        for bootstrap_idx in range(n_boot_series
                                  ):  
            bootstrapped_series_final[node][bootstrap_idx] = bootstrapped_series[node][bootstrap_idx] 
            # Add trend & seasonality to bootstrapped series
            if verbose: print(f"BEFORE bootstrapped_series[node={node}][bootstrap_idx={bootstrap_idx}]: {bootstrapped_series[node][bootstrap_idx]}")
            if verbose: print(f"node_data_decomposed['trend'][node={node}]: {node_data_decomposed['trend'][node]}")
            if verbose: print(f"node_data_decomposed['seasonality'][node={node}]: {node_data_decomposed['seasonality'][node]}")
#             bootstrapped_series[node][bootstrap_idx] += np.array(node_data_decomposed['trend'][node]) + \
#                                                         np.array(node_data_decomposed['seasonality'][node])
            bootstrapped_series_final[node][bootstrap_idx] = bootstrapped_series[node][bootstrap_idx]+ \
                                                            node_data_decomposed['trend'][node] + \
                                                            node_data_decomposed['seasonality'][node]
            
            if verbose: print(f"AFTER bootstrapped_series[node={node}][bootstrap_idx={bootstrap_idx}]: {bootstrapped_series_final[node][bootstrap_idx]}")
    return bootstrapped_series_final

#########################################################################################################
############################################ Loss Evaluation ############################################
#########################################################################################################
def get_eval_model(config, 
                   data_params_list,
                   node_features_list, 
                   n_nodes_list, 
                   static_edge_index_list,
                   device_eval,
                   device_trianed_with,
                   ):
    print("\ndevice_eval:", device_eval)
    print("\ndevice_trianed_with:", device_trianed_with)
# ######################################### !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!    
#     # Get model name based on the original device that was used to train the model
#     config.device = 'cpu'
# ######################################### !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  
    # Prepare the parameters
    pretrained_models_names = []
    hidden_warmup = []
    for dataset_name, datapath, params_iter, config_iter in data_params_list:
        pretrained_models_names.append(get_model_name(params_iter))
        hidden_warmup.append(params_iter.hidden)
    print(pretrained_models_names, hidden_warmup)
    # Create the model
    model = build_network(model_name = config.model_name,
                            n_input_periods = config.n_input_periods,
                            n_output_periods = config.n_output_periods,
                            batch_size = config.batch_size,
                            ensemble = config.ensemble,
                            hidden = config.hidden, 
                            hidden_warmup = hidden_warmup,
                            out_channels = config.out_channels,
                            dp = config.dp,
                            graph_version = config.graph_version,
                            nb_block = config.nb_block,
                            K = config.K,
                            nb_chev_filter = config.nb_chev_filter,
                            nb_time_filter = config.nb_time_filter,
                            time_strides = config.time_strides,
                            start_month = config.start_month,
                            end_month = config.end_month,
                            year = config.year,
                            company = config.company,
                            device = device_eval,
                            node_features = node_features_list, # Number of features
                            n_nodes = n_nodes_list, 
                            model_path = pretrained_models_names,
                            edge_index_list = static_edge_index_list,
                            dp_loaded = config.dp_loaded,
                          )
#     Get mode address
    # Get model name based on the original device that was used to train the model
    config.device = device_trianed_with
    model_address = "../models/" + get_model_name(config) + ".pth"
    # Bring the config back to evaluation device
    config.device = 'cpu' if device_eval.type == 'cpu' else 'gpu'
    print("\nModel under evaluation: ", model_address)
    model.load_state_dict(torch.load(model_address))
    # Set the device to evaluation device
    model.to(device_eval)
    model.eval()
    return model
    
# ######################################### !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!    
#     # Get model name based on the original device that was used to train the model
#     config.device = device_trianed_with
#     model_address = "../models/" + get_model_name(config) + ".pth"
#     # Bring the config back to evaluation device
#     config.device = 'cpu'
#     print("\nModel under evaluation: ", model_address)
#     model.load_state_dict(torch.load(model_address, map_location='cpu'))
#     model.to(torch.device('cpu'))
#     model.eval()
#     return model
# ######################################### !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
def get_bootstrap_loss(config, 
                       boot_DataLoader_list, 
                       model,
                       boot_mean_std_dict_list,
                       loss_function = 'mse',
                       test = True,
                      ):
    # Create the loss function
#     criterion = build_criterion(config.loss_function)
    criterion = build_criterion(loss_function)
    
    # Compute loss for each bootstrapped data
    losses = []
    separate_losses_list = []
    unnormalized_losses = []
    for boot_idx in range(len(boot_DataLoader_list)):
        loss, normalized_loss, loss_total, n_samples, separate_losses, _, unnormalized_loss = compute_val_loss(
                                                                 model=model, 
                                                                 model_name=config.model_name, 
                                                                 val_loader=boot_DataLoader_list[boot_idx], 
                                                                 criterion=criterion, 
                                                                 ensemble=config.ensemble,
                                                                 normalization=config.normalize,
                                                                 normalization_param_dict = boot_mean_std_dict_list[boot_idx],
                                                                 test = test,
                                                                        )
        losses.append(loss)
        unnormalized_losses.append(unnormalized_loss)
        print(f"Loss {boot_idx+1}: {loss:.4f}")
        print(f"Unnormalized loss {boot_idx+1}: {unnormalized_loss:.4f}")
        separate_losses_list.append(separate_losses)
#         print("loss: ", loss)
#         print("separate_losses: ", separate_losses)
#         print("normalized_loss: ", normalized_loss)
#         print("loss_total: ", loss_total)
    return losses, separate_losses_list, unnormalized_losses



#########################################################################################################
############################################ Stat Analysis ##############################################
#########################################################################################################
def get_boot_confidence_interval(losses, alpha):
    # https://www.geeksforgeeks.org/how-to-calculate-confidence-intervals-in-python/
    # create 95% confidence interval, using t-distribution
    t_lower_ci, t_upper_ci = stats.t.interval(alpha=alpha, 
                                          df=len(losses)-1, 
                                          loc=np.mean(losses), 
                                          scale=stats.sem(losses),
                                          ) 
    print(f"Confidence interval (t-test): ({t_lower_ci}, {t_upper_ci})")
    # https://www.geeksforgeeks.org/how-to-calculate-confidence-intervals-in-python/
    # create 95% confidence interval 
    # for population mean weight, using Normal distribution
    z_lower_ci, z_upper_ci = stats.norm.interval(alpha=alpha, 
                                             loc=np.mean(losses), 
                                             scale=stats.sem(losses),
                                             ) 
    print(f"Confidence interval (z-test): ({z_lower_ci}, {z_upper_ci})")
    return (t_lower_ci, t_upper_ci), (z_lower_ci, z_upper_ci)

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html
# t-test
def get_t_test(array_1, array_2):
    t_stat, p_value = stats.ttest_ind(array_1, array_2)
    print(f"t-stat: {t_stat}, p-value: {p_value}")
    
# https://www.geeksforgeeks.org/z-test/
def get_z_test(array_1, 
               array_2, 
               alpha, # Significance level
               hypothesis_diff = 0, # Null Hypothesis = mu_1-mu_2 = 0, Hypothesized difference (under the null hypothesis)
              ):
    # Group A (Offline Classes)
    n1 = len(array_1) # Number of observations
    x1 = np.mean(array_1) # Sample mean
    s1 = stats.sem(array_1)  # Sample std

    # Group B (Online Classes)
    n2 = len(array_2) # Number of observations
    x2 = np.mean(array_2) # Sample mean
    s2 = stats.sem(array_2)  # Sample std

    # Calculate the test statistic (z-score)
    z_score = ((x1 - x2) - hypothesis_diff) / np.sqrt((s1**2 / n1) + (s2**2 / n2))
    print('Z-Score:', np.abs(z_score))

    # Calculate the critical value
    z_critical = stats.norm.ppf(1 - alpha/2)
    print('Critical Z-Score:',z_critical)


    # Compare the test statistic with the critical value
    if np.abs(z_score) > z_critical:
        print("""Reject the null hypothesis.
    There is a significant difference.""")
    else:
        print("""Fail to reject the null hypothesis.
    There is not enough evidence to suggest a significant difference.""")

    # Approach 2: Using P-value
    # P-Value : Probability of getting less than a Z-score
    p_value = 2 * (1 - stats.norm.cdf(np.abs(z_score)))
    print('P-Value :',p_value)

    # Compare the p-value with the significance level
    if p_value < alpha:
        print("""Reject the null hypothesis.
    There is a significant difference.""")
    else:
        print("""Fail to reject the null hypothesis.
    There is not enough evidence to suggest significant difference.""")
    return p_value
        
        

#########################################################################################################
########################################## Data Stationarity ############################################
#########################################################################################################
# https://www.statsmodels.org/dev/generated/statsmodels.tsa.stattools.adfuller.html

# Null hypothesis: Non Stationarity exists in the series.
# Alternative Hypothesis: Stationarity exists in the series
def check_stationarity(data_array, regression, alpha=0.05, verbose=False):
    n_nodes = data_array.shape[1]
    results = {}
    for node in range(n_nodes):
        result = adfuller(data_array[:, node], regression=regression)
        results[node] = result
        if np.isnan(result[1]):
            pass
#             print(f"\nNode {node}: nan p-value!")
        elif result[0] <= alpha:
            if verbose:
                print(f"\nNode {node} REJECT: p-value={result[1]}") # Stationary exists
                print('ADF Statistic:', result[0])
                print('Critical Values:')
                for key, value in result[4].items():
                    print('\t%s: %.3f' % (key, value))
        else:
            print(f"\nNode {node}: ***CANNOT REJECT*** p-value={result[1]}") # Nonstationary exists
    return results
    
def check_feature_mtxes_stationarity(alpha, feature_mtxes_dict, verbose):
    alpha = 0.05
    adfuller_results = {}
    nonstationarity_flag = False
    for dataset_name, data_array in feature_mtxes_dict.items():
        print("\nDataset name: ", dataset_name)
        adfuller_results[dataset_name] = {}
        for regression in ['c', 'ct', 'ctt', 'nc']:
            print("regression: ", regression)
            adfuller_results[dataset_name][regression] = check_stationarity(data_array, 
                                                                            regression, 
                                                                            alpha, 
                                                                            verbose)
            for node, result in adfuller_results[dataset_name][regression].items():
                if not np.isnan(result[1]) and result[0] > alpha:
                    print(f"\nNode {node}: ***CANNOT REJECT*** p-value={result[1]}") # Nonstationary exists
                    nonstationarity_flag = True
                    break
    return nonstationarity_flag