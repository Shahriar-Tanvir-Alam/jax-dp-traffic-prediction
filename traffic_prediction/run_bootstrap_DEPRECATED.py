import argparse
import yaml
import os
import sys
import numpy as np
import pandas as pd
import torch
from traffic_prediction.utils import read_yaml, yaml_to_config, get_device, write_obj, read_file, file_exists,\
                                        create_folder
import traffic_prediction.dotdict
from train_private import get_data_params
from traffic_prediction.block_bootstrap import get_boot_input_data, \
                                    get_eval_model, get_bootstrap_loss, get_feature_mtxes_dict, \
                                    get_n_train_val_test_dict, get_test_feature_mtxes, check_stationarity
from traffic_prediction.block_bootstrap import get_boot_confidence_interval, get_t_test, get_z_test, \
                                            check_feature_mtxes_stationarity


def get_boot_configs(company_name):
    config_addresses_yaml = '../data/bootstrap/config_addresses.yaml'
    with open(config_addresses_yaml, 'r') as file:
        config_addresses = yaml.safe_load(file)
    for key, value in config_addresses.items():
        value = value.replace('COMPANY_NAME', company_name)
        config_addresses[key] = value

    paired_model_yaml = '../data/bootstrap/paired_model.yaml'
    with open(paired_model_yaml, 'r') as file:
        paired_model = yaml.safe_load(file)

    model_types_yaml = '../data/bootstrap/model_types.yaml'
    with open(model_types_yaml, 'r') as file:
        model_types = yaml.safe_load(file)
    return config_addresses, paired_model, model_types

def get_nbb_output_folder_name(config_addresses, model_type, n_boot_series):
    # Get the parameters of the first model between set of models
    config_0 = list(config_addresses.values())[0]
    configuration = read_yaml(config_0)
    yaml_config = yaml_to_config(**configuration['parameters']) 
    saved_boot_parent_folder_name = get_boot_parent_folder_name(yaml_config)
    saved_boot_parent_folder_address = os.path.join("../data", "bootstrap", saved_boot_parent_folder_name)
    saved_boot_folder_name = get_boot_file_name(model_type, n_boot_series)
    saved_boot_folder_address = os.path.join(saved_boot_parent_folder_address, saved_boot_folder_name)
    return saved_boot_folder_address


def get_boot_parent_folder_name(config):
    print("\nget_boot_parent_folder_name:")
    saved_boot_folder_name =  "_".join(['t', str(config.train_ratio), \
                                        'v', str(config.val_ratio), \
                                        'ip', str(config.n_input_periods), \
                                        'op', str(config.n_output_periods), \
                                        'y', str(config.year), \
                                        'sm', str(config.start_month), \
                                        'em', str(config.end_month), \
                                        'br', config.boroughs, \
                                        'bs', config.bucket_size, \
                                       ])

    # Company of data
    if config.company != "" and config.company != None:
        saved_boot_folder_name += '_' + 'cmp' + "_" + config.company.lower()   
        
    print("Bootstrap data folder name: ", saved_boot_folder_name)
    return saved_boot_folder_name


def get_boot_file_name(model_type, n_boot_series):
    print("\nget_boot_folder_name:")
    saved_boot_file_name =  "_".join(['m', model_type,\
                                        'n', str(n_boot_series),\
                                       ])
    
    print("Bootstrap data folder name: ", saved_boot_file_name)
    return saved_boot_file_name

def get_boot_losses(config,
                    data_params_list,
                    node_features_list,
                    n_nodes_list,
                    static_edge_index_list,
                    device,
                    boot_DataLoader_list,
                    model,
                    alpha_CI = 0.95,
                   ):
    # Get the evaluation model
    model = get_eval_model(config = config, 
                           data_params_list = data_params_list,
                           node_features_list = node_features_list, 
                           n_nodes_list = n_nodes_list, 
                           static_edge_index_list = static_edge_index_list,
                           device = device,
                          )
    
    # Get the losses for bootstrapped test datasets
    print("Compute test losses based on bootstrapped data")
    losses, separate_losses = get_bootstrap_loss(config = config, 
                              boot_DataLoader_list = boot_DataLoader_list, 
                              model = model,
                              )
    # Get confidence interval for loss values
    get_boot_confidence_interval(losses, alpha = alpha_CI)
    
    
def run_bootstrap_process(config_addresses, paired_model, model_types, device_trained_with, n_boot_series, model_type, period, block_size, overlapping_blocks, robust, nonstationarity_alpha):
    """
    Process the given model configuration addresses, paired models, model types, and additional parameters.

    Parameters:
    config_addresses (dict): Dictionary of model names and their configuration file addresses.
    paired_model (list): List of lists containing pairs of model names.
    model_types (dict): Dictionary of model types with model names.
    device_trained_with (str): Device used for training.
    n_boot_series (int): Number of bootstrap series.
    model_type (str): Type of model.
    period (int): Period for the model.
    block_size (int): Block size for bootstrap.
    overlapping_blocks (bool): Whether to use overlapping blocks.
    robust (bool): Whether to use robust STL for decomposition.
    nonstationarity_alpha (float): Alpha level for nonstationarity hypothesis test.

    Returns:
    None
    """
    # Example processing: Print the information
    print("Config Addresses:")
    for model, address in config_addresses.items():
        print(f"{model}: {address}")
    
    print("\nPaired Models:")
    for pair in paired_model:
        print(pair)
    
    print("\nModel Types:")
    for model_type_key, models in model_types.items():
        print(f"{model_type_key}: {models}")
    
    print("\nAdditional Parameters:")
    print(f"Device Trained With: {device_trained_with}")
    print(f"Number of Bootstrap Series: {n_boot_series}")
    print(f"Model Type: {model_type}")
    print(f"Period: {period}")
    print(f"Block Size: {block_size}")
    print(f"Overlapping Blocks: {overlapping_blocks}")
    print(f"Robust: {robust}")
    print(f"Nonstationarity Alpha: {nonstationarity_alpha}")
    
    # Get the parameters of the bootstrapping model
    config_0 = list(config_addresses.values())[0]
    configuration = read_yaml(config_0)
    yaml_config = yaml_to_config(**configuration['parameters']) 

    device = torch.device('cpu') if yaml_config.device.lower()=='cpu' else get_device()
    device_trianed_with = yaml_config.device
    print(f"Device: {device}")   
    
    data_params_list = get_data_params(yaml_config, device)
    
    # Prepare Bootstrap Data Based on Setting 1
    # Data address
    saved_boot_parent_folder_name = get_boot_parent_folder_name(yaml_config)
    saved_boot_parent_folder_address = os.path.join("../data", "bootstrap", saved_boot_parent_folder_name)
    saved_boot_folder_name = get_boot_file_name(model_type, n_boot_series)
    saved_boot_folder_address = os.path.join(saved_boot_parent_folder_address, saved_boot_folder_name)
    if file_exists(saved_boot_folder_address+"/data.pck"):
        boot_data = read_file(saved_boot_folder_address+"/data.pck")
        boot_DataLoader_list = boot_data["boot_DataLoader_list"]
        n_nodes_list = boot_data["n_nodes_list"]
        node_features_list = boot_data["node_features_list"]
        static_edge_index_list = boot_data["static_edge_index_list"]
        boot_mean_std_dict_list = boot_data["boot_mean_std_dict_list"]
        del boot_data
        
    else:
        # Get input data
        boot_DataLoader_list, n_nodes_list, node_features_list, static_edge_index_list, boot_mean_std_dict_list = \
                                                        get_boot_input_data(config=yaml_config, 
                                                                            data_params_list= data_params_list,
                                                                            n_boot_series= n_boot_series, 
                                                                            period = period, 
                                                                            block_size = block_size, 
                                                                            robust = robust,
                                                                            device = device,
                                                                            overlapping = overlapping_blocks,
                                                                            nonstationarity_alpha = nonstationarity_alpha,
                                                                            verbose=False,
                                                                           )
        # Create the folder if it does not exist
        create_folder(saved_boot_parent_folder_address)
        create_folder(saved_boot_folder_address)
        # Combine the data into a single container (dictionary or tuple)
        data_to_pickle = {
            'boot_DataLoader_list': boot_DataLoader_list,
            'n_nodes_list': n_nodes_list,
            'node_features_list': node_features_list,
            'static_edge_index_list': static_edge_index_list,
            'boot_mean_std_dict_list': boot_mean_std_dict_list,
        }
        # Serialize (pickle) the data
        write_obj(data_to_pickle, saved_boot_folder_address+"/data.pck")
        
    
    # Get the evaluation model
    model_dict = {}
    losses_dict = {}
    separate_losses_dict = {}
    unnormalized_losses_dict = {}
    df_losses = {}
    df_separate_losses = {}
    for model_name, config_address_iter in config_addresses.items():
        print("\n\n*----------------------------------------------------------------*")
        print("*----------------------------------------------------------------*")
        print("Model name and config:", model_name, config_address_iter)
        result_file_address = os.path.join(saved_boot_folder_address, model_name)
        create_folder(result_file_address)
        print("result_file_address: ", result_file_address)
        df_losses_address = os.path.join(result_file_address, 'df_losses.csv')
        df_separate_losses_address = os.path.join(result_file_address, 'df_separate_losses.csv')
        if not file_exists(df_losses_address) and not file_exists(df_separate_losses_address):
            print("*----------------------------------------------------------------*")
            # Get the parameters of the bootstrapping model
            print("Loading model")
            configuration_iter = read_yaml(config_address_iter)
            yaml_config_iter = yaml_to_config(**configuration_iter['parameters']) 
            model_dict[model_name] = get_eval_model(config = yaml_config_iter, 
                                                        data_params_list = data_params_list,
                                                        node_features_list = node_features_list, 
                                                        n_nodes_list = n_nodes_list, 
                                                        static_edge_index_list = static_edge_index_list,
                                                        device_eval = device,
                                                        device_trianed_with = device_trianed_with,
                                                     )
            print("*----------------------------------------------------------------*")
            print("Compute test losses based on bootstrapped data")
            # Get the parameters of the bootstrapping model
            configuration_iter = read_yaml(config_address_iter)
            yaml_config_iter = yaml_to_config(**configuration_iter['parameters']) 
            # Get the losses for bootstrapped test datasets
            losses_dict[model_name], separate_losses_dict[model_name], unnormalized_losses_dict[model_name] = \
                                            get_bootstrap_loss(config = yaml_config_iter,
                                                                 boot_DataLoader_list = boot_DataLoader_list,
                                                                 model = model_dict[model_name],
                                                                 boot_mean_std_dict_list = boot_mean_std_dict_list,
                                                                 test = True,
                                                                )
            # Save losses
            print("*----------------------------------------------------------------*")
            print("Save files")
            df_losses[model_name] = pd.DataFrame({"losses": losses_dict[model_name], \
                                                  "unnormalized_losses": unnormalized_losses_dict[model_name]})
            df_separate_losses[model_name] = pd.DataFrame(np.squeeze(np.array(separate_losses_dict[model_name]), axis = 2))
            # Save the DataFrame to a CSV file with a header but without the index
            df_losses[model_name].to_csv(df_losses_address, index=False)
            df_separate_losses[model_name].to_csv(df_separate_losses_address, index=False)
        else:
            print("Losses are already computed and saved.")
    

def main():
    parser = argparse.ArgumentParser(description="Process model configurations")
    parser.add_argument('--company_name', type=str, required=True, help='Name of the company')
#     parser.add_argument('--config_addresses', type=str, required=True, help='Path to the YAML file with config addresses')
#     parser.add_argument('--paired_model', type=str, required=True, help='Path to the YAML file with paired models')
#     parser.add_argument('--model_types', type=str, required=True, help='Path to the YAML file with model types')
    parser.add_argument('--device_trained_with', type=str, required=True, help='Device used for training')
    parser.add_argument('--n_boot_series', type=int, required=True, help='Number of bootstrap series')
    parser.add_argument('--model_type', type=str, required=True, help='Type of model')
    parser.add_argument('--period', type=int, required=True, help='Period for the model')
    parser.add_argument('--block_size', type=int, required=True, help='Block size for bootstrap')
    parser.add_argument('--overlapping_blocks', type=bool, required=True, help='Whether to use overlapping blocks')
    parser.add_argument('--robust', type=bool, required=True, help='Whether to use robust STL for decomposition')
    parser.add_argument('--nonstationarity_alpha', type=float, required=True, help='Alpha level for nonstationarity hypothesis test')

    args = parser.parse_args()

       
#     with open(args.config_addresses, 'r') as file:
#         config_addresses = yaml.safe_load(file)
#     for key, value in config_addresses.items():
#         value = value.replace('COMPANY_NAME', args.company_name)
#         config_addresses[key] = value

#     with open(args.paired_model, 'r') as file:
#         paired_model = yaml.safe_load(file)

#     with open(args.model_types, 'r') as file:
#         model_types = yaml.safe_load(file)
    config_addresses, paired_model, model_types = get_boot_configs(args.company_name)
    
    run_bootstrap_process(config_addresses, paired_model, model_types, args.device_trained_with, args.n_boot_series, args.model_type, args.period, args.block_size, args.overlapping_blocks, args.robust, args.nonstationarity_alpha)

    
    
# python run_bootstrap.py \
#     --config_addresses '../data/bootstrap/config_addresses.yaml' \
#     --paired_model '../data/bootstrap/paired_model.yaml' \
#     --model_types '../data/bootstrap/model_types.yaml' \
#     --device_trained_with 'gpu' \
#     --n_boot_series 1000 \
#     --model_type 'MA' \
#     --period 144 \
#     --block_size 144 \
#     --overlapping_blocks False \
#     --robust True \
#     --nonstationarity_alpha 0.05

# python run_bootstrap.py --config_addresses '../data/bootstrap/config_addresses.yaml' --paired_model '../data/bootstrap/paired_model.yaml' --model_types '../data/bootstrap/model_types.yaml' --device_trained_with 'gpu' --n_boot_series 1000 --model_type 'MA' --period 144 --block_size 144 --overlapping_blocks False --robust True --nonstationarity_alpha 0.05

if __name__ == "__main__":
    main()
