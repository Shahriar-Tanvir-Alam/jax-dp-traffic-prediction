import argparse

import sys
sys.path.insert(0, '../')

import torch
import torch.nn as nn
from torch_geometric_temporal.signal import temporal_signal_split
from tqdm import tqdm
import numpy as np
from traffic_prediction.datasets import RealtimeNYCDatasetLoader, TLCNYCDatasetLoader
try:
    from traffic_prediction.models import *
except Exception as e:
    print("Skipping original PyTorch model imports for JAX-only CPU run:", e)

import pandas as pd
import wandb
from traffic_prediction.utils import *
from traffic_prediction.datasets import borough_mapping, normalize_data, minmax_scale
from copy import deepcopy
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
import random
from torch_geometric_temporal.nn.attention import GMAN, MSTGCN
try:
    from traffic_prediction.astgcn_customized import Customized_ASTGCN
except Exception as e:
    print("Skipping Customized_ASTGCN PyTorch import for JAX-only CPU run:", e)
    Customized_ASTGCN = None


other_company_str_dict = {'uber': '_lyft', 'lyft': '_uber', 'c95': '_c5', 'c5': '_c95', 'c90': '_c10', 'c10': '_c90', 'c80': '_c20', 'c20': '_c80', 'c70': '_c30', 'c30': '_c70', 'c60': '_c40', 'c40': '_c60', 'c1_50': '_c2_50', 'c2_50': '_c1_50', 'gy': 'gy', 'gy_avg1': 'gy_avg1',\
                      'c99': '_c1', 'c1': '_c99', 'c99.9': '_c0.1', 'c0.1': '_c99.9', 'c99.99': '_c0.01', 'c0.01': '_c99.99', \
                         'uber_dp60': '_lyft', 'lyft_dp60': '_uber', 'c95_dp60': '_c5', 'c5_dp60': '_c95', 'c90_dp60': '_c10', 'c10_dp60': '_c90', 'c80_dp60': '_c20', 'c20_dp60': '_c80', 'c70_dp60': '_c30', 'c30_dp60': '_c70', 'c60_dp60': '_c40', 'c40_dp60': '_c60', 'c1_50_dp60': '_c2_50', 'c2_50_dp60': '_c1_50', \
                      'c99_dp60': '_c1', 'c1_dp60': '_c99', 'c99.9_dp60': '_c0.1', 'c0.1_dp60': '_c99.9', 'c99.99_dp60': '_c0.01', 'c0.01_dp60': '_c99.99', \
                         'uber_dp1440_eps10.0': '_lyft', 'lyft_dp1440_eps10.0': '_uber', 'c95_dp1440_eps10.0': '_c5', 'c5_dp1440_eps10.0': '_c95', 'c90_dp1440_eps10.0': '_c10', 'c10_dp1440_eps10.0': '_c90', 'c80_dp1440_eps10.0': '_c20', 'c20_dp1440_eps10.0': '_c80', 'c70_dp1440_eps10.0': '_c30', 'c30_dp1440_eps10.0': '_c70', 'c60_dp1440_eps10.0': '_c40', 'c40_dp1440_eps10.0': '_c60', 'c1_50_dp1440_eps10.0': '_c2_50', 'c2_50_dp1440_eps10.0': '_c1_50', \
                      'c99_dp1440_eps10.0': '_c1', 'c1_dp1440_eps10.0': '_c99', 'c99.9_dp1440_eps10.0': '_c0.1', 'c0.1_dp1440_eps10.0': '_c99.9', 'c99.99_dp1440_eps10.0': '_c0.01', 'c0.01_dp1440_eps10.0': '_c99.99', \
                         'uber_dp60_eps20.0': '_lyft', 'lyft_dp60_eps20.0': '_uber', 'c95_dp60_eps20.0': '_c5', 'c5_dp60_eps20.0': '_c95', 'c90_dp60_eps20.0': '_c10', 'c10_dp60_eps20.0': '_c90', 'c80_dp60_eps20.0': '_c20', 'c20_dp60_eps20.0': '_c80', 'c70_dp60_eps20.0': '_c30', 'c30_dp60_eps20.0': '_c70', 'c60_dp60_eps20.0': '_c40', 'c40_dp60_eps20.0': '_c60', 'c1_50_dp60_eps20.0': '_c2_50', 'c2_50_dp60_eps20.0': '_c1_50', \
                         'uber_dp60_eps5.0': '_lyft', 'lyft_dp60_eps5.0': '_uber', 'c95_dp60_eps5.0': '_c5', 'c5_dp60_eps5.0': '_c95', 'c90_dp60_eps5.0': '_c10', 'c10_dp60_eps5.0': '_c90', 'c80_dp60_eps5.0': '_c20', 'c20_dp60_eps5.0': '_c80', 'c70_dp60_eps5.0': '_c30', 'c30_dp60_eps5.0': '_c70', 'c60_dp60_eps5.0': '_c40', 'c40_dp60_eps5.0': '_c60', 'c1_50_dp60_eps5.0': '_c2_50', 'c2_50_dp60_eps5.0': '_c1_50', \
                      'c99_dp60_eps5.0': '_c1', 'c1_dp60_eps5.0': '_c99', 'c99.9_dp60_eps5.0': '_c0.1', 'c0.1_dp60_eps5.0': '_c99.9', 'c99.99_dp60_eps5.0': '_c0.01', 'c0.01_dp60_eps5.0': '_c99.99', \
                         'uber_dp60_eps2.5': '_lyft', 'lyft_dp60_eps2.5': '_uber', 'c95_dp60_eps2.5': '_c5', 'c5_dp60_eps2.5': '_c95', 'c90_dp60_eps2.5': '_c10', 'c10_dp60_eps2.5': '_c90', 'c80_dp60_eps2.5': '_c20', 'c20_dp60_eps2.5': '_c80', 'c70_dp60_eps2.5': '_c30', 'c30_dp60_eps2.5': '_c70', 'c60_dp60_eps2.5': '_c40', 'c40_dp60_eps2.5': '_c60', 'c1_50_dp60_eps2.5': '_c2_50', 'c2_50_dp60_eps2.5': '_c1_50', \
                      'c99_dp60_eps2.5': '_c1', 'c1_dp60_eps2.5': '_c99', 'c99.9_dp60_eps2.5': '_c0.1', 'c0.1_dp60_eps2.5': '_c99.9', 'c99.99_dp60_eps2.5': '_c0.01', 'c0.01_dp60_eps2.5': '_c99.99', \
                         'uber_dp60_eps0.5': '_lyft', 'lyft_dp60_eps0.5': '_uber', 'c95_dp60_eps0.5': '_c5', 'c5_dp60_eps0.5': '_c95', 'c90_dp60_eps0.5': '_c10', 'c10_dp60_eps0.5': '_c90', 'c80_dp60_eps0.5': '_c20', 'c20_dp60_eps0.5': '_c80', 'c70_dp60_eps0.5': '_c30', 'c30_dp60_eps0.5': '_c70', 'c60_dp60_eps0.5': '_c40', 'c40_dp60_eps0.5': '_c60', 'c1_50_dp60_eps0.5': '_c2_50', 'c2_50_dp60_eps0.5': '_c1_50', \
                      'c99_dp60_eps0.5': '_c1', 'c1_dp60_eps0.5': '_c99', 'c99.9_dp60_eps0.5': '_c0.1', 'c0.1_dp60_eps0.5': '_c99.9', 'c99.99_dp60_eps0.5': '_c0.01', 'c0.01_dp60_eps0.5': '_c99.99', \
                         }
surge_mode_dict = {"1": 'shortage', "2": 'surplus', "3": 'o_minus_i'}


def train(config=None):
    print("\nStart training the model:")
    # Fix randomization
    set_seed(5)
    
    # Fixed offline run: bypass W&B sweep/config overwrite
    if isinstance(config, dict):
        from types import SimpleNamespace
        config = SimpleNamespace(**config)

    class _DummyRun:
        def __enter__(self): return self
        def __exit__(self, exc_type, exc, tb): return False

    with _DummyRun():
        device = torch.device('cpu') if config.device.lower()=='cpu' else get_device() # Select CPU/GPU device
        print(f"Device being used: {device}")
        
        # ###########################################################
        # Data Loading
        # Load and build data
        data_params_list = get_data_params(config, device) # List of lists [Data name, Data address, warmed-up model(s) config, currently training model config]
        train_loader, train_data_list, val_loader_list, test_loader_list, static_edge_index_list, \
            node_features_list, n_nodes_list, pretrained_models_names, hidden_warmup, normalization_param_dict = \
                                                                get_datasets(data_params_list, device)
        
        # ###########################################################
        # Model initialization
        # Create the network
        print(f"\n\n\npretrained_models_names: {pretrained_models_names}\n\n\n")
        model = build_network(device = device,
                                node_features = node_features_list,  # Number of features
                                n_nodes = n_nodes_list, 
                                model_path = pretrained_models_names,
                                hidden_warmup=hidden_warmup,
                                edge_index_list=static_edge_index_list,
                                model_name = config.model_name,
                                n_input_periods = config.n_input_periods,
                                n_output_periods = config.n_output_periods,
                                batch_size = config.batch_size,
                                ensemble=config.ensemble,
                                hidden=config.hidden, 
                                out_channels=config.out_channels,
                                dp=config.dp,
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
                                dp_loaded = config.dp_loaded,
                             )
        # Fix the model for Opacus DP if DP != 0
        if config.dp:
            assert ModuleValidator.is_valid(model), print("Not a valid model")
            model = ModuleValidator.fix_and_validate(model)
            assert ModuleValidator.is_valid(model), print("Not a valid model")
            
        # Create the optimizer
        optimizer = build_optimizer(model, config.optimizer, config.learning_rate)
        # Create the loss function
        criterion = build_criterion(config.loss_function)
        loss_function_val = 'mae' if config.loss_function=='mse' else 'mse'
        criterion_val_2 = build_criterion(loss_function_val)
        
        # Create the Opacus Differential Privacy Engine 
        privacy_engine = None
        if config.dp:
            print(f"Running differentially private training")
            secure_rng = True if config.secure_rng else False
            # https://pytorch.org/blog/torchcsprng-release-blog/
            privacy_engine = PrivacyEngine(secure_mode=secure_rng, accountant=config.accountant) 
            # https://opacus.ai/api/privacy_engine.html
            if config.dp == 1:
                model, optimizer, train_loader = privacy_engine.make_private(
                                                                module=model,
                                                                optimizer=optimizer,
                                                                data_loader=train_loader,
                                                                noise_multiplier=config.sigma,
                                                                max_grad_norm=config.max_per_sample_grad_norm,
                                                                poisson_sampling=True,
                                                                batch_first=True,
                                                                grad_sample_mode="functorch", # ["hooks", "functorch", "ew", "no_op"]
                                                            )
            elif config.dp == 2:
                model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
                                                                    module=model,
                                                                    optimizer=optimizer,
                                                                    data_loader=train_loader,
                                                                    epochs=config.epochs,
                                                                    target_epsilon=config.epsilon,
                                                                    target_delta=config.delta,
                                                                    max_grad_norm=config.max_per_sample_grad_norm,
                                                                    poisson_sampling=True,
                                                                )
#             model = ModuleValidator.fix(model)
            assert ModuleValidator.is_valid(model), print("Not valid model")
    
        # Keep the record of the best model and performance
#         model.train()
        saved_model_name = get_model_name(config)
        best_epoch = 0
        train_loss_at_best_val = 0
        epsilon_at_best_val = 0
        best_val_loss = np.inf
        best_val_loss_unnormalized = np.inf
        best_model_state_dict = None
        assert config.val_step <= config.epochs, "Number of epochs > Validation steps!"
        
        # ###########################################################
        # Start Epochs
        for epoch in tqdm(range(config.epochs), position=0):
# #             Print all the layer names and their parameter values
#             for name, param in model.named_parameters():
#                 print(name, param.data)
            # ###########################################################
            # Validation
            best_val_loss_flag = False # Reset flag
            val_loss, separate_losses_avg, val_loss_unnormalized = validation_epoch(val_step=config.val_step, 
                                                                         epoch=epoch, 
                                                                         model=model, 
                                                                         model_name=config.model_name,
                                                                         val_loader=val_loader_list, 
                                                                         criterion=criterion, 
                                                                         ensemble=config.ensemble,
                                                                         normalization=config.normalize,
                                                                         normalization_param_dict=normalization_param_dict,
                                                                                    )
            # Analyze validation result
            if val_loss: # If validation loss check was run
                if val_loss < best_val_loss: # If a lower validation error is found
                    best_val_loss_flag = True # A better model observed
                    # Update the best validation error so far. Also, update the epoch at which it happened.
                    best_val_loss = val_loss
                    best_val_loss_unnormalized = val_loss_unnormalized
                    best_epoch = epoch
                    # Save the best model and keep it for testing phase
                    if config.dp:
                        torch.save(model._module.state_dict(), '../models/' + saved_model_name + '.pth')
                        best_model_state_dict = deepcopy(model._module.state_dict())
                    else:
                        torch.save(model.state_dict(), '../models/' + saved_model_name + '.pth')
    #                     torch.save(model, '../models/' + saved_model_name + '.pth')
                        best_model_state_dict = deepcopy(model.state_dict())
                    # Record the validation results
                    for i in range(len(separate_losses_avg)):
                        wandb.log({"separate_losses_" + str(i) + "_at_best_val": separate_losses_avg[i]}, commit=False) 
                # Record the validation results
                wandb.log({"val_loss": val_loss, "best_val_loss": best_val_loss, "best_epoch": best_epoch}, commit=False) 
                for i in range(len(separate_losses_avg)):
                    wandb.log({"separate_losses_avg_" + str(i): separate_losses_avg[i]}, commit=False) 
                if val_loss_unnormalized:
                    wandb.log({"val_loss_unnormalized": val_loss_unnormalized, 
                               "best_val_loss_unnormalized": best_val_loss_unnormalized}, commit=False) 
#                 # Computing the alternative loss function for analysis purposes
#                 val_loss_2, separate_losses_avg_2, val_loss_2_unnormalized = validation_epoch(val_step=config.val_step, 
#                                                                      epoch=epoch, 
#                                                                      model=model, 
#                                                                      model_name=config.model_name,
#                                                                      val_loader=val_loader_list, 
#                                                                      criterion=criterion_val_2, 
#                                                                      ensemble=config.ensemble,
#                                                                      normalization_param_dict=normalization_param_dict,
#                                                                     )
#                 # Record the validation results of the alternative loss function that is not used for training
#                 wandb.log({"alternative_" + loss_function_val + "_val_loss": val_loss_2, }, commit=False) 
#                 for i in range(len(separate_losses_avg_2)):
#                     wandb.log({"alternative_" + loss_function_val + "_separate_losses_avg_" + str(i): separate_losses_avg_2[i]}, commit=False)  
#                 if val_loss_2_unnormalized:
#                     wandb.log({"alternative_val_loss_unnormalized": val_loss_2_unnormalized}, commit=False)
            
            # ###########################################################
            # Training
            model, train_loss, epsilon = train_epoch(model=model, 
                                                     model_name=config.model_name,
                                                     epoch=epoch, 
                                                     train_loader=train_loader,
                                                     optimizer=optimizer, 
                                                     criterion=criterion, 
                                                     val_step=config.val_step, 
                                                     privacy_engine=privacy_engine, 
                                                     ensemble=config.ensemble, 
                                                     train_data=train_data_list,
                                                     dp=config.dp,
                                                     delta=config.delta,
                                                     )
#             wandb.log({"train_loss": train_loss, "epoch": epoch, "epsilon": epsilon}) 
            if best_val_loss_flag:
                train_loss_at_best_val = train_loss
                if config.dp:
                    epsilon_at_best_val = epsilon
            wandb.log({"train_loss_at_best_val": train_loss_at_best_val}, commit=False) 
            if config.dp:
                wandb.log({"epsilon_at_best_val": epsilon_at_best_val}, commit=False)
            wandb.log({"train_loss": train_loss, "epoch": epoch, "epsilon": epsilon})

            # Early stopping
            if epoch - best_epoch >= config.patience:
                print(f"Early stopping with patience {config.patience} at epoch {epoch}.")
                break
        # Training Finished
                
                
        # ###########################################################
        # Test
        if best_model_state_dict is None:
            print("best_model_state_dict is None")
            if config.dp:
                best_model_state_dict = deepcopy(model._module.state_dict())
            else:
                best_model_state_dict = deepcopy(model.state_dict())
        print(f"\n\n\npretrained_models_names: {pretrained_models_names}\n\n\n")
        best_model_to_test = build_network(device = device,
                                            node_features = node_features_list, # Number of features
                                            n_nodes = n_nodes_list, 
                                            model_path = pretrained_models_names,
                                            hidden_warmup=hidden_warmup,
                                            edge_index_list=static_edge_index_list,
                                            model_name = config.model_name,
                                            n_input_periods = config.n_input_periods,
                                            n_output_periods = config.n_output_periods,
                                            batch_size = config.batch_size,
                                            ensemble=config.ensemble,
                                            hidden=config.hidden, 
                                            out_channels=config.out_channels,
                                            dp=config.dp,
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
                                            dp_loaded = config.dp_loaded,
                                          )
        best_model_to_test.load_state_dict(best_model_state_dict)
        best_model_to_test.eval()
        test_loss, test_separate_losses_avg, test_loss_unnormalized = validation_epoch(val_step=config.val_step, 
                                                                   epoch=epoch, 
                                                                   model=best_model_to_test, 
                                                                   model_name=config.model_name,
                                                                   val_loader=test_loader_list, 
                                                                   criterion=criterion, 
                                                                   ensemble=config.ensemble,
                                                                   normalization=config.normalize,
                                                                   normalization_param_dict=normalization_param_dict,
                                                                   test=True,
                                                                                      )
        wandb.log({"test_loss": test_loss}, commit=False) 
        if test_loss_unnormalized:
            wandb.log({"test_loss_unnormalized": test_loss_unnormalized}, commit=False)
        for i in range(len(test_separate_losses_avg)):
            wandb.log({"test_separate_losses_avg_"+str(i): test_separate_losses_avg[i]}, commit=False)  
        # Computing the alternative loss function for analysis purposes
        test_loss_2, test_separate_losses_avg_2, test_loss_2_unnormalized = validation_epoch(val_step=config.val_step, 
                                                             epoch=epoch, 
                                                             model=best_model_to_test, 
                                                             model_name=config.model_name,
                                                             val_loader=test_loader_list, 
                                                             criterion=criterion_val_2, 
                                                             ensemble=config.ensemble,
                                                             normalization=config.normalize,
                                                             normalization_param_dict=normalization_param_dict,
                                                             test=True,
                                                            )
        
        # Record the test results of the alternative loss function that is not used for training
        for i in range(len(test_separate_losses_avg_2)):
            wandb.log({"test_" + loss_function_val + "_separate_losses_avg_" + str(i): test_separate_losses_avg_2[i]}, commit=False)  
        if test_loss_2_unnormalized:
            wandb.log({"test_" + loss_function_val + "_loss_unnormalized": test_loss_2_unnormalized}, commit=False)
        wandb.log({"test_" + loss_function_val + "_loss": test_loss_2, }, commit=True) 
            
        print("\n\n** Best validation {} observed at epoch {:.0f}: {:.2f}".format(config.loss_function, best_epoch, best_val_loss)) 
        print("\n** Train loss at best validation {} observed at epoch {:.0f}: {:.2f}".format(config.loss_function, best_epoch, train_loss_at_best_val)) 
        print("\n** Test loss of model observed at best validation {} at epoch {:.0f}: {:.2f}".format(config.loss_function, best_epoch, test_loss)) 
        
        
        



# Get the configuration of data (and warmed-up models) that should be loaded based on the name of the model
def get_data_params(config, device):
    print("\nget_data_params:")
    data_params_list = [] 
    
    # Name of the company that model is being trained for
    company_str = "_"+config.company.lower() if config.company!="" else ""
    # Epsilon of the DP warmed-up model, empty string if non-DP
    dp_loaded_str = "_"+config.dp_loaded if config.dp_loaded!="" else ""
    
    if config.model_name in ["multiprediction_1", "multiprediction_2", "multiprediction_2_2", "multiprediction_3", "multiprediction_4"]:
        dataname_path_list = config.data_name_path.split("*")
        for i, data_path_iter in enumerate(dataname_path_list):
            data_name, data_address = data_path_iter.split("=")
            print(f"Loading parameters of dataset {i}: {data_name}")
            data_params_list.append((data_name, data_address, config, config)) # pass in your keyword args

    elif config.model_name[:2] in ['MA', 'MM'] or config.model_name[:3] == 'avg':
        print("Model is in MA, MM, or avg category.")
        # Name of the cooperating/other company to load its data and/or DP/non-DP model, Ex.: Lyft and Uber
        other_company_str = other_company_str_dict[config.company.lower()]
        print("config.data_name_path: ", config.data_name_path)
        # List of data name and their configuration files address
        dataname_path_list = config.data_name_path.split("*")
        # Extend the data address of the other companies' data
        # tlc_nyc_inflow2 and tlc_nyc_outflow2 are inflow and outflow of the cooperating company, gy is the aggregated data of green and yellow taxi data
        dataname_path_list.extend(["tlc_nyc_inflow2=../data/tlc_nyc/fhvhv/", "tlc_nyc_outflow2=../data/tlc_nyc/fhvhv/", "gy_inflow=../data/tlc_nyc/gy/", "gy_outflow=../data/tlc_nyc/gy/", ])
        # List of configs of optimal individual models
        address_opt_configs = "../data/m_"+str(config.start_month)+"_"+str(config.end_month)+"_y_"+str(config.year)+"_optimals/"
        configuration_address_list = [address_opt_configs + "config_realtime_optimal_g", \
                                        address_opt_configs + "config_tlc_nyc_inflow_optimal" + company_str, \
                                        address_opt_configs + "config_tlc_nyc_outflow_optimal" + company_str, \
                                        address_opt_configs + "config_tlc_nyc_inflow_optimal" + other_company_str, \
                                        address_opt_configs + "config_tlc_nyc_outflow_optimal" + other_company_str, \
                                        address_opt_configs + "config_tlc_nyc_inflow_optimal_gy", \
                                        address_opt_configs + "config_tlc_nyc_outflow_optimal_gy"]
        
        for i, data_path_iter in enumerate(dataname_path_list):
            # Get the name and address of the data files
            data_name, data_address = data_path_iter.split("=")
            print(f"\nLoading parameters of dataset {i+1} out of {len(dataname_path_list)} data: {data_name}")
            # Define the graph version of the speed data. Graph version is not defined for inflow and outflow data because there is only one graph version for them.
            # Get the graph version for realtime data
            graph_version = str(config.graph_version) if data_name=='realtime' else ""
            if data_name=='realtime': 
                print("graph_version: ", graph_version)
            # Configuration file address
            configuration_address = configuration_address_list[i] + graph_version + ".yaml"
            print(f"Configuration address of {data_name} data: {configuration_address}")
            # Read YAML configuation file
            configuration_iter = read_yaml(configuration_address)
            yaml_config = yaml_to_config(**configuration_iter['parameters'])
            print(yaml_config)
            # Append configuation to other data configurations
            # Data name, Folder address of the data, warmed-up model config, currently training model config 
            data_params_list.append((data_name, data_address, yaml_config, config)) # pass in your keyword args
       
        
    elif config.model_name[:3] in ['MSA']:
        other_company_str = other_company_str_dict[config.company.lower()]
        print(config.data_name_path)
        dataname_path_list = config.data_name_path.split("*")
        # surge mode defines the surge data
        # 1: shortage, 2: surplus, 3: difference
        surge_mode = surge_mode_dict[config.model_name[5]]
        # Data folder addresses
        dataname_path_list.extend(["tlc_nyc_inflow2=../data/tlc_nyc/fhvhv/", "tlc_nyc_outflow2=../data/tlc_nyc/fhvhv/", \
                                   "gy_inflow=../data/tlc_nyc/gy/", "gy_outflow=../data/tlc_nyc/gy/", \
                                   surge_mode+"=../data/tlc_nyc/fhvhv/", surge_mode+"2=../data/tlc_nyc/fhvhv/", ])
        # List of configs of optimal individual models
        address_opt_configs = "../data/m_"+str(config.start_month)+"_"+str(config.end_month)+"_y_"+str(config.year)+"_optimals/"
        configuration_address_list = [address_opt_configs + "config_realtime_optimal_g", \
                                        address_opt_configs + "config_tlc_nyc_inflow_optimal" + company_str, \
                                        address_opt_configs + "config_tlc_nyc_outflow_optimal" + company_str, \
                                        address_opt_configs + "config_tlc_nyc_inflow_optimal" + other_company_str, \
                                        address_opt_configs + "config_tlc_nyc_outflow_optimal" + other_company_str, \
                                        address_opt_configs + "config_tlc_nyc_inflow_optimal_gy", \
                                        address_opt_configs + "config_tlc_nyc_outflow_optimal_gy",\
                                        address_opt_configs + "config_" + surge_mode + "_optimal" + company_str, \
                                        address_opt_configs + "config_" + surge_mode + "_optimal" + other_company_str]
        for i, data_path_iter in enumerate(dataname_path_list):
            data_name, data_address = data_path_iter.split("=")
            print(f"\n\nLoading parameters of dataset {i}: {data_name}")
            # Define the graph version of the speed data. Graph version is not defined for inflow and outflow data because there is only one graph version for them.
            graph_version = str(config.graph_version) if data_name=='realtime' else ""
            configuration_address = configuration_address_list[i] + graph_version + ".yaml"
            print(f"Configuration address of {data_name}: {configuration_address}")
            # Read YAML configuation file
            configuration_iter = read_yaml(configuration_address)
            yaml_config = yaml_to_config(**configuration_iter['parameters'])
            print(yaml_config)
            # Append configuation to other data configurations
            data_params_list.append((data_name, data_address, yaml_config, config)) # pass in your keyword args

    elif config.model_name in [ 'MP0_i', 'MP0_o', 'MP0_no_relu_i', 'MP0_no_relu_o', 'MP24', 'MP24PP', 'MP25_i', 'MP25_o', 'MP25X_i', 'MP25X_o', 'MP25P_i', 'MP25P_o', 'MP25P_2_i', 'MP25P_2_o', 'MP25PPSmall_i', 'MP25PPSmall_o', 'MP25PP_i', 'MP25PP_o', \
                              'M60A_i', 'M60A_o', 'M60B_i', 'M60B_o', 'M60C_i', 'M60C_o', 'M60D_i', 'M60D_o', 'M60E_i', 'M60E_o', 'M61A', 'M61B', 'M61C', \
                              'M64A_i', 'M64A_o', 'M64B_i', 'M64B_o', 'M64C_i', 'M64C_o', 'M65A', 'M65B', 'M65C', \
                              'M68A_i', 'M68A_o', 'M68B_i', 'M68B_o', 'M68C_i', 'M68C_o', 'M68D_i', 'M68D_o', \
                              ]:
        dataname_path_list = config.data_name_path.split("*")
        dataname_path_list.extend(["gy_inflow=../data/tlc_nyc/gy/", "gy_outflow=../data/tlc_nyc/gy/", ])
        # List of configs of optimal individual models
        address_opt_configs = "../data/m_"+str(config.start_month)+"_"+str(config.end_month)+"_y_"+str(config.year)+"_optimals/"
        configuration_address_list = [address_opt_configs + "config_realtime_optimal_g", \
                                        address_opt_configs + "config_tlc_nyc_inflow_optimal" + company_str, \
                                        address_opt_configs + "config_tlc_nyc_outflow_optimal" + company_str, \
                                        address_opt_configs + "config_tlc_nyc_inflow_optimal_gy", \
                                        address_opt_configs + "config_tlc_nyc_outflow_optimal_gy"]
        
        for i, data_path_iter in enumerate(dataname_path_list):
            data_name, data_address = data_path_iter.split("=")
            print(f"Loading parameters of dataset {i}: {data_name}")
            # Define the graph version of the speed data. Graph version is not defined for inflow and outflow data because there is only one graph version for them.
            graph_version = str(config.graph_version) if data_name=='realtime' else ""
            configuration_address = configuration_address_list[i] + graph_version + ".yaml"
            print(f"Configuration address of {data_name}: {configuration_address}")
            # Read YAML configuation file
            configuration_iter = read_yaml(configuration_address)
            yaml_config = yaml_to_config(**configuration_iter['parameters'])
            # Append configuation to other data configurations
            data_params_list.append((data_name, data_address, yaml_config, config)) # pass in your keyword args

    elif config.model_name in [ 'MP0_gy_avg1_i', 'MP0_gy_avg1_o', 'MP0_no_relu_gy_avg1_i', 'MP0_no_relu_gy_avg1_o', 'MP24_gy_avg1', 'MP24PP_gy_avg1', 'MP25_gy_avg1_i', 'MP25_gy_avg1_o', 'MP25X_gy_avg1_i', 'MP25X_gy_avg1_o', 'MP25P_gy_avg1_i', 'MP25P_gy_avg1_o', 'MP25P_2_gy_avg1_i', 'MP25P_2_gy_avg1_o', 'MP25PPSmall_gy_avg1_i', 'MP25PPSmall_gy_avg1_o', 'MP25PP_gy_avg1_i', 'MP25PP_gy_avg1_o', \
                              'M60A_gy_avg1_i', 'M60A_gy_avg1_o', 'M60B_gy_avg1_i', 'M60B_gy_avg1_o', 'M60C_gy_avg1_i', 'M60C_gy_avg1_o', 'M60D_gy_avg1_i', 'M60D_gy_avg1_o', 'M60E_gy_avg1_i', 'M60E_gy_avg1_o', 'M61A_gy_avg1', 'M61B_gy_avg1', 'M61C_gy_avg1', \
                              'M64A_gy_avg1_i', 'M64A_gy_avg1_o', 'M64B_gy_avg1_i', 'M64B_gy_avg1_o', 'M64C_gy_avg1_i', 'M64C_gy_avg1_o', 'M65A_gy_avg1', 'M65B_gy_avg1', 'M65C_gy_avg1', \
                              'M68A_gy_avg1_i', 'M68A_gy_avg1_o', 'M68B_gy_avg1_i', 'M68B_gy_avg1_o', 'M68C_gy_avg1_i', 'M68C_gy_avg1_o', 'M68D_gy_avg1_i', 'M68D_gy_avg1_o', \
                              ]:
        dataname_path_list = config.data_name_path.split("*")
        dataname_path_list.extend(["gy_inflow=../data/tlc_nyc/gy/", "gy_outflow=../data/tlc_nyc/gy/", ])
        # List of configs of optimal individual models
        address_opt_configs = "../data/m_"+str(config.start_month)+"_"+str(config.end_month)+"_y_"+str(config.year)+"_optimals/"
        configuration_address_list = [address_opt_configs + "config_realtime_optimal_g", \
                                        address_opt_configs + "config_tlc_nyc_inflow_optimal" + company_str, \
                                        address_opt_configs + "config_tlc_nyc_outflow_optimal" + company_str, \
                                        address_opt_configs + "config_tlc_nyc_inflow_optimal_gy_avg1", \
                                        address_opt_configs + "config_tlc_nyc_outflow_optimal_gy_avg1"]
        
        for i, data_path_iter in enumerate(dataname_path_list):
            data_name, data_address = data_path_iter.split("=")
            print(f"Loading parameters of dataset {i}: {data_name}")
            # Define the graph version of the speed data. Graph version is not defined for inflow and outflow data because there is only one graph version for them.
            graph_version = str(config.graph_version) if data_name=='realtime' else ""
            configuration_address = configuration_address_list[i] + graph_version + ".yaml"
            print(f"Configuration address of {data_name}: {configuration_address}")
            # Read YAML configuation file
            configuration_iter = read_yaml(configuration_address)
            yaml_config = yaml_to_config(**configuration_iter['parameters'])
            # Append configuation to other data configurations
            data_params_list.append((data_name, data_address, yaml_config, config)) # pass in your keyword args
    elif config.model_name in ["multiprediction_5", "multiprediction_6", "multiprediction_7", "multiprediction_8", "multiprediction_9", "multiprediction_10", "multiprediction_11", "multiprediction_12", "multiprediction_13", "multiprediction_14", "multiprediction_15", "multiprediction_16", "multiprediction_17", "multiprediction_18", "multiprediction_19", "multiprediction_21", "multiprediction_23", "multiprediction_24", "multiprediction_25", "multiprediction_25pp", "multiprediction_26", "multiprediction_26_dp", "multiprediction_26_o", "multiprediction_26_o_dp","multiprediction_27", "multiprediction_27_dp", "multiprediction_37_dp", "multiprediction_38_dp", "multiprediction_39_dp", "multiprediction_40_dp", "multiprediction_31_dp", "multiprediction_32_dp", "multiprediction_33_dp", "multiprediction_34_dp", "multiprediction_35_dp", "multiprediction_36_dp", "multiprediction_47", "multiprediction_48", "multiprediction_51", "multiprediction_52", "multiprediction_53", "multiprediction_54", "multiprediction_47_finetune", "multiprediction_48_finetune", "multiprediction_51_finetune", "multiprediction_52_finetune", "multiprediction_53_finetune", "multiprediction_54_finetune", "multiprediction_55", "multiprediction_56", "surge_1_1", "surge_1_2", "surge_1_3", "surge_1_4", "surge_1_dp_1", "surge_1_dp_2", "surge_1_dp_3", "surge_1_dp_4", "surge_2_1", "surge_2_2", "surge_2_3", "surge_2_4", "surge_2_dp_1", "surge_2_dp_2", "surge_2_dp_3", "surge_2_dp_4",  ]:
        dataname_path_list = config.data_name_path.split("*")
#         configuration_address_list = []
#         for dataname_path in dataname_path_list:
#             data_name, data_address = data_path_iter.split("=")
#             data_path_config = get_data_config_address(data_name, start_month, end_month, year, company_str="", graph_version="", dp_loaded_str="")
#             configuration_address_list.append()
        address_opt_configs = "../data/m_"+str(config.start_month)+"_"+str(config.end_month)+"_y_"+str(config.year)+"_optimals/"
        configuration_address_list = [address_opt_configs + "config_realtime_optimal_g", \
                                      address_opt_configs + "config_tlc_nyc_inflow_optimal" + company_str, \
                                      address_opt_configs + "config_tlc_nyc_outflow_optimal" + company_str]
        
        for i, data_path_iter in enumerate(dataname_path_list):
            # Get data name and address pair
            data_name, data_address = data_path_iter.split("=")
            print(f"Loading parameters of dataset {i}: {data_name}")
            # Define the graph version of the speed data. Graph version is not defined for inflow and outflow data because there is only one graph version for them.
            graph_version = str(config.graph_version) if data_name=='realtime' else ""
            configuration_address = configuration_address_list[i] + graph_version + ".yaml"
            print(f"Configuration address of {data_name}: {configuration_address}")
            # Read YAML configuation file
            configuration_iter = read_yaml(configuration_address)
            yaml_config = yaml_to_config(**configuration_iter['parameters'])
            # Append configuation to other data configurations
            data_params_list.append((data_name, data_address, yaml_config, config)) # pass in your keyword args

    elif config.model_name in ["dual_company_1_i_1", "dual_company_1_o_1", "dual_company_1_i_2", "dual_company_1_o_2", "dual_company_2_i_1", "dual_company_2_o_1", "dual_company_2_i_2", "dual_company_2_o_2", "dual_company_2_i_3", "dual_company_2_o_3", "dual_company_2_i_4", "dual_company_2_o_4", ]:
        print(f"get_data_params is called, config.model_name is {config.model_name}")
        other_company_str = other_company_str_dict[config.company.lower()]
#         if config.company.lower() == 'uber':
#             other_company_str = '_lyft'
#         else:
#             other_company_str = '_uber'
        dataname_path_list = config.data_name_path.split("*")
        configuration_address_list = []
        configuration_address_list.append("../data/m_"+str(config.start_month)+"_"+str(config.end_month)+"_y_"+str(config.year)+"_optimals/config_realtime_optimal_g")
        configuration_address_list.append("../data/m_"+str(config.start_month)+"_"+str(config.end_month)+"_y_"+str(config.year)+"_optimals/config_tlc_nyc_inflow_optimal" + company_str)
        configuration_address_list.append("../data/m_"+str(config.start_month)+"_"+str(config.end_month)+"_y_"+str(config.year)+"_optimals/config_tlc_nyc_outflow_optimal" + company_str)
        configuration_address_list.append("../data/m_"+str(config.start_month)+"_"+str(config.end_month)+"_y_"+str(config.year)+"_optimals/config_tlc_nyc_inflow_optimal" + other_company_str + dp_loaded_str)
        configuration_address_list.append("../data/m_"+str(config.start_month)+"_"+str(config.end_month)+"_y_"+str(config.year)+"_optimals/config_tlc_nyc_outflow_optimal" + other_company_str + dp_loaded_str)
        for i, data_path_iter in enumerate(dataname_path_list): # Speed AND Inflow/Outflow of the company
            data_name, data_address = data_path_iter.split("=")
            print(f"Loading parameters of dataset {i}: {data_name}, company: {company_str[1:]}")
            graph_version = str(config.graph_version) if data_name=='realtime' else ""
            configuration_address = configuration_address_list[i] + graph_version + ".yaml"
            print(f"Configuration address of {data_name}: {configuration_address}")
            configuration_iter = read_yaml(configuration_address)
            yaml_config = yaml_to_config(**configuration_iter['parameters'])
            data_params_list.append((data_name, data_address, yaml_config, config)) # pass in your keyword args
        
        for i, data_path_iter in enumerate(dataname_path_list[1:]): # Inflow/Outflow of the other company
            data_name, data_address = data_path_iter.split("=")
            print(f"Loading parameters of dataset {i}: {data_name}, company: {other_company_str[1:]}")
            graph_version = str(config.graph_version) if data_name=='realtime' else ""
            configuration_address = configuration_address_list[3+i] + graph_version + ".yaml"
            print(f"Configuration address of {data_name}: {configuration_address}")
            configuration_iter = read_yaml(configuration_address)
            yaml_config = yaml_to_config(**configuration_iter['parameters'])
            data_params_list.append((data_name, data_address, yaml_config, config)) # pass in your keyword args

    elif config.ensemble == 0:
        print(f"Loading parameters of dataset: {config.dataset}")
        data_params_list.append((config.dataset, config.datapath, config, config))
    
    else:
        config_paths_list = config.config_paths.split("*")
        for i, config_path_iter in enumerate(config_paths_list):
            dataset_name, configuration_address = config_path_iter.split("=")
            print(f"Loading parameters of dataset {i}: {dataset_name}")
            configuration_iter = read_yaml(configuration_address)
            yaml_config = yaml_to_config(**configuration_iter['parameters'])
            data_params_list.append((dataset_name, yaml_config.datapath, yaml_config, config)) # pass in your keyword args
            
    return data_params_list
    
    

def get_datasets(params_list, device):
    print("\nget_datasets,  Based on the list of data configs:")
    train_data_list, val_loader_list, test_loader_list, \
        static_edge_index_list, node_features_list, n_nodes_list, \
            pretrained_models_names, hidden_warmup, normalization_param_dict = [], [], [], [], [], [], [], [], {}
    for dataset_name, datapath, params_iter, config in params_list:
        print("dataset_name: ", dataset_name)
        # Check if the config of the warmed-up model and its training data is aligned with running config
        # Checks include, train/validation/test ratio, shuffle, normalize, date (year, start-end months), borough, bucket size, graph version (just for speed data), # of input time buckets, and # of output time buckets
        assert config.val_ratio == params_iter.val_ratio, print("Not similar val_ratio attribute between configs")
        assert config.train_ratio == params_iter.train_ratio, print("Not similar train_ratio attribute between configs")
        assert config.shuffle == params_iter.shuffle, print("Not similar shuffle attribute between configs")
        assert config.year == params_iter.year, print("Not similar year attribute between configs")
        assert config.start_month == params_iter.start_month, print("Not similar start_month attribute between configs")
        assert config.end_month == params_iter.end_month, print("Not similar end_month attribute between configs")
        assert config.boroughs == params_iter.boroughs, print("Not similar boroughs attribute between configs")
        assert config.bucket_size == params_iter.bucket_size, print("Not similar bucket_size attribute between configs")
        assert config.graph_version == params_iter.graph_version or dataset_name != 'realtime', print("Not similar graph_version attribute between configs")
        assert config.n_input_periods == params_iter.n_input_periods, print("Not similar n_input_periods attribute between configs")
        assert config.n_output_periods == params_iter.n_output_periods, print("Not similar n_output_periods attribute between configs")
        # No assert for company name as model might load data from other companies too
        if dataset_name != "realtime":
            company = (params_iter.company.lower() if params_iter.company is not None else '') # company name of the warmed-up model
        else:
            company = ""
        if config.normalize == 1: # Z-score normalization for all data
            assert config.normalize == params_iter.normalize, print("Not similar normalize attribute between configs")
            normalize = config.normalize
            print("Z-score normalization for all data")
        elif config.normalize == 2: # Hybrid normalization: z-score for speed and min-max scaling for others
            if dataset_name != "realtime":
                normalize = config.normalize
                print(f"Minmax scaling for {dataset_name} data")
            else:
                normalize = params_iter.normalize
                print("Z-score normalization for speed data")
        # mean_std_list: list of mean and std of features and the target for train, test, and validation data
        # mean_std_list size: 3 lists for train/test/validation, each list has 4 values=[mean of features, std of features, mean of target, std of target]
        _, train_data, val_loader, test_loader, static_edge_index, node_features, n_nodes, _, normalization_param_list = \
                                                            build_dataset(dataset_name = dataset_name, 
                                                                        datapath = datapath,
                                                                        device = device,
                                                                        batch_size = config.batch_size, 
                                                                        val_ratio = config.val_ratio,
                                                                        train_ratio = config.train_ratio, 
                                                                        shuffle = config.shuffle, 
#                                                                         normalize = config.normalize, 
                                                                        normalize = normalize, 
                                                                        year = config.year, 
                                                                        start_month = config.start_month, 
                                                                        end_month = config.end_month,
                                                                        boroughs = config.boroughs, 
                                                                        bucket_size = config.bucket_size,
                                                                        graph_version = config.graph_version,
                                                                        n_input_periods = config.n_input_periods, 
                                                                        n_output_periods = config.n_output_periods,
                                                                        company = company,
                                                                         )
        train_data_list.append(train_data)
        val_loader_list.append(val_loader)
        test_loader_list.append(test_loader)
        static_edge_index_list.append(static_edge_index)
        node_features_list.append(node_features) # Number of features
        n_nodes_list.append(n_nodes)
        pretrained_models_names.append(get_model_name(params_iter)) # Pretrained models have ensemble=0.
        hidden_warmup.append(params_iter.hidden)
        normalization_param_dict[dataset_name] = normalization_param_list
        
    # Create train loader based on data size
    n_sample = train_data_list[0][0].shape[0]
    data_index = torch.tensor(range(n_sample))
    tensor_data_index = torch.utils.data.TensorDataset(data_index, data_index)
    # Return index of data instead of data because Opacus cannot work with spatio temporal dataloader
    train_loader = torch.utils.data.DataLoader(tensor_data_index, 
                                               batch_size=config.batch_size, 
                                               shuffle=config.shuffle)
    
    return train_loader, train_data_list, val_loader_list, test_loader_list, static_edge_index_list, node_features_list, n_nodes_list, pretrained_models_names, hidden_warmup, normalization_param_dict


def build_dataset(dataset_name, 
                  datapath, 
                  year, 
                  start_month, 
                  end_month, 
                  boroughs, 
                  bucket_size, 
                  graph_version, 
                  n_input_periods, 
                  n_output_periods, 
                  batch_size, 
                  val_ratio, 
                  train_ratio, 
                  shuffle, 
                  normalize, 
                  device, 
                  company):
    print("\nbuild_dataset: load data object and split data into train/val/test")
    # Get the folder address of data and its graph
    datapath_data, datapath_graph = get_data_loader_path(year, start_month, end_month, boroughs, bucket_size, dataset_name, datapath, graph_version, company)
    # Get the data loader object
    loader = get_data_object(dataset_name, datapath_data, datapath_graph)
    # Get data from data object
    dataset = loader.get_dataset(num_timesteps_in=n_input_periods, num_timesteps_out=n_output_periods)
    # Number of features of each node
    node_features = loader.n_features
    # Number of nodes
    n_nodes = len(loader.nodelist)
    # Check batch size be smaller than number of samples
    if batch_size > len(dataset.features):
        print(f"batch_size is larger than data size. batch_size changed to size of data = {len(dataset.features)}")
        batch_size = len(dataset.features) # batch_size cannot be larger than datasize
    # Train/validation/test split
    if val_ratio > 0:
        train_dataset, test_val_dataset = temporal_signal_split(dataset, train_ratio=train_ratio)
        val_dataset, test_dataset = temporal_signal_split(test_val_dataset, train_ratio=val_ratio/(1-train_ratio))
    else:
        train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=train_ratio)
        val_dataset = None
    # Get DataLoader for test and validation data and train data itself
    train_loader, train_data, val_loader, test_loader, static_edge_index, normalization_param_list = get_dataloader(train_dataset=train_dataset, 
                                                                              val_dataset=val_dataset, 
                                                                              test_dataset=test_dataset, 
                                                                              batch_size=batch_size, 
                                                                              shuffle=shuffle, 
                                                                              device=device, 
                                                                              normalize=normalize)
    return train_loader, train_data, val_loader, test_loader, static_edge_index, node_features, n_nodes, loader, normalization_param_list
            

def get_data_object(dataset_name, datapath_data, datapath_graph):
    print("\nget_data_object:")
    if dataset_name == 'realtime':
        loader = RealtimeNYCDatasetLoader(data_path=datapath_data, edges_data_path=datapath_graph)
    elif dataset_name in ['tlc_nyc_inflow', 'tlc_nyc_inflow2', 'gy_inflow', 'tlc_nyc_outflow', 'tlc_nyc_outflow2', 'gy_outflow', 'o_minus_i', 'shortage', 'surplus', 'o_minus_i2', 'shortage2', 'surplus2']:
        loader = TLCNYCDatasetLoader(data_path=datapath_data, zones_metadata_path='../data/tlc_nyc/taxi_zones/taxi_zones_WGS84.shp')
    elif dataset_name == 'realtime_test1':
        loader = RealtimeNYCDatasetLoader(data_path=datapath_data, edges_data_path=datapath_graph)
    elif dataset_name in ['inflow_test1', 'outflow_test1']:
        loader = TLCNYCDatasetLoader(data_path=datapath_data, edges_data_path='../data/test_1/graph_taxi_test.csv')
    print(f'Loaded instance of data set {datapath_data} object ...')
    return loader

def get_data_loader_path(year, start_month, end_month, boroughs, bucket_size, dataset_name, datapath, graph_version, company):
    if company != "": 
        company_str = "_" + company    
    else:
        company_str = ""
    date_borough_bucket = f'_{str(year)}-{str(start_month).zfill(2)}_{str(year)}-{str(end_month).zfill(2)}' + '_B_' + boroughs + '_' + bucket_size 
    datapath_graph = None
    if dataset_name == 'realtime':
        datapath_data = datapath + "clean_aggregate_realtime" + date_borough_bucket + '.parquet'
        graph_file_name = "realtime_edges_final_v"+str(graph_version)+'_B_'+boroughs+".csv"
        datapath_graph = datapath + 'graph/' + graph_file_name
    elif dataset_name[:14] == 'tlc_nyc_inflow':
        datapath_data = datapath + "clean_aggregate_tlc_inflow" + date_borough_bucket + company_str + '.parquet'
    elif dataset_name[:15] == 'tlc_nyc_outflow':
        datapath_data = datapath + "clean_aggregate_tlc_outflow" + date_borough_bucket + company_str + '.parquet'
    elif dataset_name == 'gy_inflow':
        datapath_data = datapath + "clean_aggregate_tlc_inflow" + date_borough_bucket + '_gy.parquet'
    elif dataset_name == 'gy_outflow':
        datapath_data = datapath + "clean_aggregate_tlc_outflow" + date_borough_bucket + '_gy.parquet'
    elif dataset_name in ['o_minus_i', 'shortage', 'surplus']:
        datapath_data = datapath + "clean_aggregate_" + dataset_name + date_borough_bucket + company_str + '.parquet'
    elif dataset_name in ['o_minus_i2', 'shortage2', 'surplus2']:
        datapath_data = datapath + "clean_aggregate_" + dataset_name[:-1] + date_borough_bucket + company_str + '.parquet'
    elif dataset_name == 'realtime_test1':
        datapath_data = datapath + "test_clean_aggregate_realtime" + date_borough_bucket + '.parquet'
        graph_file_name = "realtime_edges_final_v"+str(graph_version)+'_B_'+boroughs+".csv"
        datapath_graph = '../data/test_1/graph_realtime_test.csv'
    elif dataset_name == 'inflow_test1':
        datapath_data = datapath + "clean_aggregate_tlc_inflow" + date_borough_bucket + '.parquet'
    elif dataset_name == 'outflow_test1':
        datapath_data = datapath + "clean_aggregate_tlc_outflow" + date_borough_bucket + '.parquet'
    return datapath_data, datapath_graph
    
def get_dataloader(train_dataset, val_dataset, test_dataset, batch_size, shuffle, device, normalize):
    print("\nget_dataloader:")
    # Fix randomization
    set_seed(5)
    shuffle = False if shuffle==0 else True
    normalization_param_list = []
    
    # Train: NumPy to Torch 
    train_x_tensor = torch.from_numpy(train_dataset.features).type(torch.FloatTensor).to(device)  # (B, N, F, T)
    train_target_tensor = torch.from_numpy(train_dataset.targets).type(torch.FloatTensor).to(device)  # (B, N, F, T)
    # Train: normalize features and target separately
    if normalize == 1:    
        train_x_tensor, train_mean_x, train_std_x = normalize_data(train_x_tensor)
        train_target_tensor, train_mean_target, train_std_target = normalize_data(train_target_tensor)
        normalization_param_list.append([train_mean_x.item(), train_std_x.item(), train_mean_target.item(), train_std_target.item()])
    elif normalize == 2:
        train_x_tensor, train_min_x, train_max_x = minmax_scale(train_x_tensor)
        train_target_tensor, train_min_target, train_max_target = minmax_scale(train_target_tensor)
        # Assumption: data has only 1 feature, otherwise, there is a min and a max for each feature
        normalization_param_list.append([train_min_x[0].item(), train_max_x[0].item(), train_min_target[0].item(), train_max_target[0].item()])
    # Train: Reshape target tensor
    train_target_tensor = torch.squeeze(train_target_tensor, 2)  # (B, N, T)
    print(f"train_x_tensor.shape: {train_x_tensor.shape}, train_target_tensor.shape: {train_target_tensor.shape}")
    # Train: Get DataLoader >> https://pytorch.org/docs/stable/data.html#torch.utils.data.TensorDataset
    train_dataset_new = torch.utils.data.TensorDataset(train_x_tensor, train_target_tensor)
#     train_loader = torch.utils.data.DataLoader(train_dataset_new, batch_size=batch_size, shuffle=shuffle, drop_last=True)
    # A different train_loader will be created in get_datasets and will be used.
    train_loader = torch.utils.data.DataLoader(train_dataset_new, batch_size=batch_size, shuffle=shuffle) # USELESS
    
    # Test: NumPy to Torch 
    test_x_tensor = torch.from_numpy(test_dataset.features).type(torch.FloatTensor).to(device)  # (B, N, F, T)
    test_target_tensor = torch.from_numpy(test_dataset.targets).type(torch.FloatTensor).to(device) # (B, N, F, T)
    # Test: normalize features and target separately
    if normalize == 1:    
        test_x_tensor, test_mean_x, test_std_x = normalize_data(test_x_tensor)
        test_target_tensor, test_mean_target, test_std_target = normalize_data(test_target_tensor)
        normalization_param_list.append([test_mean_x.item(), test_std_x.item(), test_mean_target.item(), test_std_target.item()])
    elif normalize == 2:
        test_x_tensor, test_min_x, test_max_x = minmax_scale(test_x_tensor)
        test_target_tensor, test_min_target, test_max_target = minmax_scale(test_target_tensor)
        # Assumption: data has only 1 feature
        normalization_param_list.append([test_min_x[0].item(), test_max_x[0].item(), test_min_target[0].item(), test_max_target[0].item()])
        
    # Test: Reshape target tensor
    test_target_tensor = torch.squeeze(test_target_tensor, 2)  # (B, N, T)
    print(f"test_x_tensor.shape: {test_x_tensor.shape}, test_target_tensor.shape: {test_target_tensor.shape}")
    # Test: Get DataLoader
    test_dataset_new = torch.utils.data.TensorDataset(test_x_tensor, test_target_tensor)
    test_loader = torch.utils.data.DataLoader(test_dataset_new, batch_size=1, shuffle=False) # default batch_size also 1

    if val_dataset is not None:
        # Validation data: NumPy to Torch 
        val_x_tensor = torch.from_numpy(val_dataset.features).type(torch.FloatTensor).to(device)  # (B, N, F, T)
        val_target_tensor = torch.from_numpy(val_dataset.targets).type(torch.FloatTensor).to(device) # (B, N, F, T)
        # Validation: normalize features and target separately
        if normalize == 1:    
            val_x_tensor, val_mean_x, val_std_x = normalize_data(val_x_tensor)
            val_target_tensor, val_mean_target, val_std_target = normalize_data(val_target_tensor)
            normalization_param_list.append([val_mean_x.item(), val_std_x.item(), val_mean_target.item(), val_std_target.item()])
        elif normalize == 2:
            val_x_tensor, val_min_x, val_max_x = minmax_scale(val_x_tensor)
            val_target_tensor, val_min_target, val_max_target = minmax_scale(val_target_tensor)
            # Assumption: data has only 1 feature
            normalization_param_list.append([val_min_x[0].item(), val_max_x[0].item(), val_min_target[0].item(), val_max_target[0].item()])
        # Validation: Reshape target tensor
        val_target_tensor = torch.squeeze(val_target_tensor, 2)  # (B, N, T)
        print(f"val_x_tensor.shape: {val_x_tensor.shape}, val_target_tensor.shape: {val_target_tensor.shape}")
        # Validation: Get DataLoader
        val_dataset_new = torch.utils.data.TensorDataset(val_x_tensor, val_target_tensor)
        val_loader = torch.utils.data.DataLoader(val_dataset_new, batch_size=1, shuffle=False) # default batch_size also 1
    else:
        val_loader = None
        
    # Loading the graph only once because it's a static graph
    for snapshot in train_dataset:
        static_edge_index = snapshot.edge_index.to(device)
        break 
        
        
    return train_loader, [train_x_tensor, train_target_tensor], val_loader, test_loader, static_edge_index, normalization_param_list
    

def get_model_name(config):
    print("\nget_model_name:")
#     bool2str = {True: "T", False: "F"}
    int2str = {0: "F", 1: "T"}
    ensemble_version = "" if config.ensemble<=1 else str(config.ensemble) # {0: "", 1: "", 2: "2", 3: "3", 4: "4", 5: "5"}
    model_str = config.model_name 
    if config.dp_loaded != "":
        model_str += "_" + (config.dp_loaded if config.dp_loaded is not None else "None")
    if config.normalize == 0:
        normalize = "F"
    elif config.normalize == 1:
        normalize = "T"
    elif config.normalize == 2:
        normalize = "T2"
        
    print("Model: ", model_str)    
    saved_model_name =  "_".join(['model', model_str + ensemble_version, \
                                    'd', str(config.dataset), 'lr', str(config.learning_rate), \
                                    'e', str(config.epochs), 't', str(config.train_ratio), \
                                    'v', str(config.val_ratio), 'vs', str(config.val_step), \
                                    'lf', config.loss_function, 'ip', str(config.n_input_periods), \
                                    'op', str(config.n_output_periods), 'b', str(config.batch_size), \
                                    'n', normalize, 'de', str(config.device), \
                                    'p', str(config.patience), \
                                    'sh', int2str[config.shuffle],\
                                    'y', str(config.year), 'sm', str(config.start_month), 'em', str(config.end_month), \
                                    'g', str(config.graph_version), 'br', config.boroughs, 'bs', config.bucket_size])
    # Number of nodes in the hidden layer
    if config.hidden>0:
        saved_model_name += '_' + 'h' + "_" + str(config.hidden)  
    # A3TGCN model
    if config.model_name.lower() == 'a3tgcn':
        saved_model_name += '_' + 'och' + "_" + str(config.out_channels)  
    # Company of data
    if config.company != "" and config.company != None:
        saved_model_name += '_' + 'cmp' + "_" + config.company.lower()        
    # DP setting
    if config.dp != 0:
        if config.dp == 1:
            saved_model_name += '_' + 'dp'  
            saved_model_name += '_' + 'sig' + "_" + str(config.sigma)  
        if config.dp == 2:
            saved_model_name += '_' + 'dp2'  
            saved_model_name += '_' + 'eps' + "_" + str(config.epsilon)
        saved_model_name += '_' + 'c' + "_" + str(config.max_per_sample_grad_norm)  
        saved_model_name += '_' + 'delta' + "_" + str(config.delta)  
        saved_model_name += '_' + 'secR' + "_" + int2str[config.secure_rng]
        if config.accountant != 'prv' and config.accountant != None:
            saved_model_name += '_' + 'acc' + "_" + config.accountant  
    # ASTGCN setting
    if config.nb_block != 2 and config.nb_block != None:
        saved_model_name += '_' + 'nbb' + "_" + str(config.nb_block)  
    if config.K != 3 and config.K != None:
        saved_model_name += '_' + 'K' + "_" + str(config.K)  
    if config.nb_chev_filter != 64 and config.nb_chev_filter != None:
        saved_model_name += '_' + 'nbcf' + "_" + str(config.nb_chev_filter)  
    if config.nb_time_filter != 64 and config.nb_time_filter != None:
        saved_model_name += '_' + 'nbtf' + "_" + str(config.nb_time_filter)  
    if config.time_strides != 1 and config.time_strides != None:
        saved_model_name += '_' + 'ts' + "_" + str(config.time_strides)  
        
    print("Saved model name: ", saved_model_name)
    return saved_model_name

    
# Define MAPE Criterion
class MAPE(nn.Module):
    def forward(self, y_pred, y_true):
        epsilon = 1e-3  # To avoid division by zero
        return torch.mean(torch.abs((y_true - y_pred) / (y_true + epsilon))) * 100
    
# class RMSE(nn.Module):
#     def __init__(self):
#         super(RMSE, self).__init__()
#         self.mse_loss = nn.MSELoss()
    
#     def forward(self, y_pred, y_true):
#         mse = self.mse_loss(y_pred, y_true)
#         return torch.sqrt(mse)

    
class log_abs_criterion(nn.Module):
    def __init__(self):
        super(log_abs_criterion, self).__init__()

    def forward(self, predictions, targets):
        loss = torch.log(1 + torch.abs(predictions - targets))
        return loss.mean()  # Return the mean loss over the batch

def build_criterion(loss_function):
    if loss_function.lower()=='mse':
        criterion = torch.nn.MSELoss()
    elif loss_function.lower()=='mae':
        criterion = torch.nn.L1Loss()
    elif loss_function.lower()=='huberloss':
        criterion = torch.nn.HuberLoss()
    elif loss_function.lower()=='logabsloss':
        criterion = log_abs_criterion()
    elif loss_function.lower()=='mape':
        criterion = MAPE()
    elif loss_function.lower()=='rmse':
        criterion = RMSE()
    return criterion


def build_optimizer(model, optimizer, learning_rate):
    # Fix randomization
    set_seed(5)
    if optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    elif optimizer == "adam":
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        
    print('Optimizer\'s state_dict:')  
    for var_name in optimizer.state_dict():
        print(var_name, '\t', optimizer.state_dict()[var_name])
    print('\n\n')
    return optimizer



# def compute_val_loss(model, val_loader, criterion, edge_index_data, ensemble=0, baseline_error=False):
def compute_val_loss(model, model_name, val_loader, criterion, normalization_param_dict, normalization, ensemble=0, baseline_error=False, test=False):
#     print("\ncompute_val_loss:")
    '''
    :param model: model
    :param val_loader: torch.utils.data.utils.DataLoader
    :param criterion: torch.nn.MSELoss or other defined loss function (e.g., MAE and HuberLoss)
    :return: val_loss
    '''
    # Fix randomization
    set_seed(5)
    
#     def compute_val_loss_step(encoder_inputs, labels, model, criterion, edge_index_data, ensemble):
    def compute_val_loss_step(encoder_inputs, labels, model, criterion):
#         outputs = model(encoder_inputs, edge_index_data)
        outputs = model(encoder_inputs)
        loss = criterion(outputs, labels)
        separate_losses = []
        return loss, separate_losses
    
    
    def compute_val_loss_array_output_step(encoder_inputs, labels, model, criterion, feature_sizes):
        outputs = model(encoder_inputs)
#         print(f"outputs.shape: {outputs.shape}")
        loss = criterion(outputs, labels)
        separate_losses = []
        for i in range(len(feature_sizes)):
            separate_losses.append(criterion(outputs[..., sum(feature_sizes[:i]):sum(feature_sizes[:i+1])], \
                                             labels[..., sum(feature_sizes[:i]):sum(feature_sizes[:i+1])]).item())
        return loss, separate_losses
    
    def compute_val_loss_array_output_step_with_unnormalized(encoder_inputs, labels, model, criterion, feature_sizes, normalization_param_1, normalization_param_2, normalization):
        outputs = model(encoder_inputs)
#         print(f"outputs.shape: {outputs.shape}")
        loss = criterion(outputs, labels)
        if normalization == 1:
            mu = normalization_param_1
            std = normalization_param_2
            loss_unnormalized = criterion(outputs*std+mu, labels*std+mu)
        elif normalization == 2:
            min_val = normalization_param_1
            max_val = normalization_param_2
            loss_unnormalized = criterion(outputs * (max_val - min_val) + min_val, 
                                             labels * (max_val - min_val) + min_val)
        separate_losses = []
        for i in range(len(feature_sizes)):
            separate_losses.append(criterion(outputs[..., sum(feature_sizes[:i]):sum(feature_sizes[:i+1])], \
                                             labels[..., sum(feature_sizes[:i]):sum(feature_sizes[:i+1])]).item())
        return loss, separate_losses, loss_unnormalized

    l_p_norm_labels_list = []
    if not baseline_error:
        # Set the model to evaluation mode
        model.train(False)  # ensure dropout layers are in evaluation mode, equivalent to model.eval() https://discuss.pytorch.org/t/what-does-nn-modules-train-true-train-false-do/4004/7
    with torch.no_grad():
        batch_loss = []  # batch loss
        batch_separate_losses = []  # batch loss of each input data
        batch_loss_unnormalized = []  # batch loss unnormalized input

        if model_name in ["multiprediction_5", "multiprediction_6", "multiprediction_7", "multiprediction_8"]:
            for (encoder_inputs1, labels1), (encoder_inputs2, labels2), (encoder_inputs3, labels3) \
                    in zip(*val_loader):
                if model_name == "multiprediction_5":
                    labels = labels2.reshape(labels2.shape[0], 1, 1, -1)
                    encoder_inputs = encoder_inputs1
                elif model_name == "multiprediction_6":
                    labels = labels3.reshape(labels3.shape[0], 1, 1, -1)
                    encoder_inputs = encoder_inputs1
                elif model_name == "multiprediction_7":
                    labels = labels2.reshape(labels2.shape[0], 1, 1, -1)
                    encoder_inputs = [encoder_inputs1, encoder_inputs2]
                elif model_name == "multiprediction_8":
                    labels = labels3.reshape(labels3.shape[0], 1, 1, -1)
                    encoder_inputs = [encoder_inputs1, encoder_inputs3]

                loss, _ = compute_val_loss_step(encoder_inputs=encoder_inputs, 
                                                labels=labels, 
                                                model=model, 
                                                criterion=criterion, 
                                                )
                batch_loss.append(loss.item())
                l_p_norm_labels = criterion(torch.zeros_like(labels.clone().detach().flatten()), \
                                                labels.clone().detach().flatten())
                l_p_norm_labels_list.append(l_p_norm_labels.item())


        elif model_name[:3] in ['MSA']:
            surge_mode = surge_mode_dict[model_name[5]]
            if test:
                normalization_param_1, normalization_param_2 = normalization_param_dict[surge_mode][-2][-2:]
            else:
                normalization_param_1, normalization_param_2 = normalization_param_dict[surge_mode][-1][-2:]
                
            for (s_input, s_labels), \
                (i_input_c1, i_labels_c1), (o_input_c1, o_labels_c1), \
                (i_input_c2, i_labels_c2), (o_input_c2, o_labels_c2), \
                (i_input_gy, i_labels_gy), (o_input_gy, o_labels_gy), \
                (surge_input_c1, surge_labels_c1), (surge_input_c2, surge_labels_c2) in zip(*val_loader):

                # Labels
                if model_name[-1] == "L":
                    labels = surge_labels_c1.reshape(surge_labels_c1.shape[0], 1, 1, -1)
                    feature_sizes = [labels.shape[-1]]
                else:
                    labels = surge_labels_c1
                    feature_sizes = [1]
                        
                # Inputs
                if model_name[:4] in ["MSA2"]:
                    encoder_inputs = [i_input_c1, o_input_c1]
                elif model_name[:4] in ["MSA3"]:
                    encoder_inputs = [s_input, i_input_c1, o_input_c1]
                elif model_name[:4] in ["MSA4"]:
                    encoder_inputs = [i_input_c1, o_input_c1, i_input_c2, o_input_c2]
                elif model_name[:4] in ["MSA5"]:
                    encoder_inputs = [s_input, i_input_c1, o_input_c1, i_input_c2, o_input_c2]
                    
                loss, separate_losses, loss_unnormalized = compute_val_loss_array_output_step_with_unnormalized(encoder_inputs=encoder_inputs, 
                                                        labels=labels, 
                                                        model=model, 
                                                        criterion=criterion, 
                                                        feature_sizes=feature_sizes,
                                                        normalization_param_1 = normalization_param_1,
                                                        normalization_param_2 = normalization_param_2,
                                                        normalization = normalization,
                                                     )
                batch_loss.append(loss.item())
                batch_separate_losses.append(separate_losses)
                batch_loss_unnormalized.append(loss_unnormalized.item())
                l_p_norm_labels = criterion(torch.zeros_like(labels.clone().detach().flatten()), \
                                                labels.clone().detach().flatten())
                l_p_norm_labels_list.append(l_p_norm_labels.item())
                    

        elif model_name[:2] in ['MA', 'MM'] or model_name[:3] == 'avg':
            for (s_input, s_labels), \
                (i_input_c1, i_labels_c1), (o_input_c1, o_labels_c1), \
                (i_input_c2, i_labels_c2), (o_input_c2, o_labels_c2), \
                (i_input_gy, i_labels_gy), (o_input_gy, o_labels_gy) in zip(*val_loader):
                
                # Labels
                if model_name[:5] in ["MA1_I", "MA2_I", "MA3_I", "MA4_I", "MA6_I", "MM1_I", "MM2_I", "MM3_I", "MM4_I"] or \
                    model_name[:7] in ["MA3_S_I", "MA5_S_I", "MM3_S_I"] or \
                    model_name[:8] in ["MA4_GY_I", "MM4_GY_I"] or \
                     model_name[:8] in ["MA3_S2_I", "MA5_S2_I"] or \
                        model_name=='avg_i':
                    if test:
                        normalization_param_1, normalization_param_2 = normalization_param_dict['tlc_nyc_inflow'][-2][-2:]
                    else:
                        normalization_param_1, normalization_param_2 = normalization_param_dict['tlc_nyc_inflow'][-1][-2:]
                    if model_name[-1] == "L":
                        labels = i_labels_c1.reshape(i_labels_c1.shape[0], 1, 1, -1)
                        feature_sizes = [labels.shape[-1]]
                    else:
                        labels = i_labels_c1
                        feature_sizes = [1]
                elif model_name[:5] in ["MA1_O", "MA2_O", "MA3_O", "MA4_O", "MA6_O", "MM1_O", "MM2_O", "MM3_O", "MM4_O"] or \
                    model_name[:7] in ["MA3_S_O", "MA5_S_O", "MM3_S_O"] or \
                    model_name[:8] in ["MA4_GY_O", "MM4_GY_O"] or \
                     model_name[:8] in ["MA3_S2_O", "MA5_S2_O"] or \
                        model_name=='avg_o':
                    if test:
                        normalization_param_1, normalization_param_2 = normalization_param_dict['tlc_nyc_outflow'][-2][-2:]
                    else:
                        normalization_param_1, normalization_param_2 = normalization_param_dict['tlc_nyc_outflow'][-1][-2:]
                    if model_name[-1] == "L":
                        labels = o_labels_c1.reshape(o_labels_c1.shape[0], 1, 1, -1)
                        feature_sizes = [labels.shape[-1]]
                    else:
                        labels = o_labels_c1
                        feature_sizes = [1]
                elif model_name[:8] in ["MA4_DP_I"] or \
                    model_name[:10] in ["MA5_S_DP_I"]:
                    if test:
                        normalization_param_1, normalization_param_2 = normalization_param_dict['tlc_nyc_inflow2'][-2][-2:]
                    else:
                        normalization_param_1, normalization_param_2 = normalization_param_dict['tlc_nyc_inflow2'][-1][-2:]
                    if model_name[-1] == "L":
                        labels = i_labels_c2.reshape(i_labels_c2.shape[0], 1, 1, -1)
                        feature_sizes = [labels.shape[-1]]
                    else:
                        labels = i_labels_c2
                        feature_sizes = [1]
                        
                elif model_name[:8] in ["MA4_DP_O"] or \
                    model_name[:10] in ["MA5_S_DP_O"]:
                    if test:
                        normalization_param_1, normalization_param_2 = normalization_param_dict['tlc_nyc_outflow2'][-2][-2:]
                    else:
                        normalization_param_1, normalization_param_2 = normalization_param_dict['tlc_nyc_outflow2'][-1][-2:]
                    if model_name[-1] == "L":
                        labels = o_labels_c2.reshape(o_labels_c2.shape[0], 1, 1, -1)
                        feature_sizes = [labels.shape[-1]]
                    else:
                        labels = o_labels_c2
                        feature_sizes = [1]
                        
                # Inputs
                if model_name[:5] in ["MA1_I", "MM1_I"] or model_name=='avg_i':
                    encoder_inputs = [i_input_c1]
                elif model_name[:5] in ["MA1_O", "MM1_O"] or model_name=='avg_o':
                    encoder_inputs = [o_input_c1]
                elif model_name=='avg_s':
                    encoder_inputs = [s_input]
                    
                elif model_name[:6] in ["MA2_I1", "MM2_I1"]:
                    encoder_inputs = [i_input_c1, i_input_c2]
                elif model_name[:6] in ["MA2_I2", "MM2_I2"]:
                    encoder_inputs = [i_input_c1, i_input_gy]
                elif model_name[:6] in ["MA2_I3", "MM2_I3"]:
                    encoder_inputs = [i_input_c1, o_input_c1]
                elif model_name[:6] in ["MA2_I4", "MM2_I4"]:
                    encoder_inputs = [i_input_c1, o_input_c2]
                elif model_name[:6] in ["MA2_I5", "MM2_I5"]:
                    encoder_inputs = [i_input_c1, o_input_gy]
                    
                elif model_name[:6] in ["MA2_O1", "MM2_O1"]:
                    encoder_inputs = [o_input_c1, o_input_c2]
                elif model_name[:6] in ["MA2_O2", "MM2_O2"]:
                    encoder_inputs = [o_input_c1, o_input_gy]
                elif model_name[:6] in ["MA2_O3", "MM2_O3"]:
                    encoder_inputs = [o_input_c1, i_input_c1]
                elif model_name[:6] in ["MA2_O4", "MM2_O4"]:
                    encoder_inputs = [o_input_c1, i_input_c2]
                elif model_name[:6] in ["MA2_O5", "MM2_O5"]:
                    encoder_inputs = [o_input_c1, i_input_gy]
                    
                elif model_name[:6] in ["MA3_I1", "MM3_I1"]:
                    encoder_inputs = [i_input_c1, o_input_c1, i_input_c2]
                elif model_name[:6] in ["MA3_I2", "MM3_I2"]:
                    encoder_inputs = [i_input_c1, o_input_c1, o_input_c2]
                elif model_name[:6] in ["MA3_I3", "MM3_I3"]:
                    encoder_inputs = [i_input_c1, o_input_c1, i_input_gy]
                elif model_name[:6] in ["MA3_I4", "MM3_I4"]:
                    encoder_inputs = [i_input_c1, o_input_c1, o_input_gy]
                    
                elif model_name[:6] in ["MA3_O1", "MM3_O1"]:
                    encoder_inputs = [i_input_c1, o_input_c1, i_input_c2]
                elif model_name[:6] in ["MA3_O2", "MM3_O2"]:
                    encoder_inputs = [i_input_c1, o_input_c1, o_input_c2]
                elif model_name[:6] in ["MA3_O3", "MM3_O3"]:
                    encoder_inputs = [i_input_c1, o_input_c1, i_input_gy]
                elif model_name[:6] in ["MA3_O4", "MM3_O4"]:
                    encoder_inputs = [i_input_c1, o_input_c1, o_input_gy]
                    
                elif model_name[:6] in ["MA4_I1", "MA4_O1", "MM4_I1", "MM4_O1"] or model_name[:8] in ["MA4_DP_I", "MA4_DP_O"]:
                    encoder_inputs = [i_input_c1, o_input_c1, i_input_c2, o_input_c2]
                    
                elif model_name[:6] in ["MA4_I2", "MA4_O2", "MM4_I2", "MM4_O2"] or \
                     model_name[:8] in ["MA4_GY_I", "MM4_GY_I", "MA4_GY_O", "MM4_GY_O"]:
                    if model_name[:9] in ["MA4_GY_O2", "MM4_GY_O2"]:
                        encoder_inputs = [o_input_c1, i_input_c1, i_input_gy, o_input_gy]
                    else:
                        encoder_inputs = [i_input_c1, o_input_c1, i_input_gy, o_input_gy]
                    
                elif model_name[:7] in ["MA3_S_I", "MA3_S_O", "MM3_S_I", "MM3_S_O"] or model_name[:8] in ["MA3_S2_I", "MA3_S2_O"]:
                    if model_name[:8] in ["MA3_S_O2", "MM3_S_O2"]:
                        encoder_inputs = [s_input, o_input_c1, i_input_c1,]
                    else:
                        encoder_inputs = [s_input, i_input_c1, o_input_c1]
                        
                elif model_name[:7] in ["MA5_S_I", "MA5_S_O"] or model_name[:8] in ["MA5_S2_I", "MA5_S2_O"] or \
                    model_name[:10] in ["MA5_S_DP_I", "MA5_S_DP_O"]:
                    encoder_inputs = [s_input, i_input_c1, o_input_c1, i_input_c2, o_input_c2]
                        
                elif model_name[:6] in ["MA6_I1", "MA6_O1"]:
                    encoder_inputs = [i_input_c1, o_input_c1, i_input_c2, o_input_c2, i_input_gy, o_input_gy]
                    
                    
                loss, separate_losses, loss_unnormalized = compute_val_loss_array_output_step_with_unnormalized(encoder_inputs=encoder_inputs, 
                                                labels=labels, 
                                                model=model, 
                                                criterion=criterion, 
                                                feature_sizes=feature_sizes,
                                                normalization_param_1 = normalization_param_1,
                                                normalization_param_2 = normalization_param_2,
                                                normalization = normalization,
                                            )
                batch_loss.append(loss.item())
                batch_separate_losses.append(separate_losses)
                batch_loss_unnormalized.append(loss_unnormalized.item())
                l_p_norm_labels = criterion(torch.zeros_like(labels.clone().detach().flatten()), \
                                                labels.clone().detach().flatten())
                l_p_norm_labels_list.append(l_p_norm_labels.item())
                
            
        elif model_name in ['MP0_i', 'MP0_o', 'MP0_no_relu_i', 'MP0_no_relu_o', 'MP24', 'MP24PP', 'MP25_i', 'MP25_o', 'MP25X_i', 'MP25X_o', 'MP25P_i', 'MP25P_o', 'MP25P_2_i', 'MP25P_2_o', 'MP25PPSmall_i', 'MP25PPSmall_o', 'MP25PP_i', 'MP25PP_o', \
                           'M60A_i', 'M60A_o', 'M60B_i', 'M60B_o', 'M60C_i', 'M60C_o', 'M60D_i', 'M60D_o', 'M60E_i', 'M60E_o', 'M61A', 'M61B', 'M61C', \
                           'M64A_i', 'M64A_o', 'M64B_i', 'M64B_o', 'M64C_i', 'M64C_o', 'M65A', 'M65B', 'M65C', \
                           'M68A_i', 'M68A_o', 'M68B_i', 'M68B_o', 'M68C_i', 'M68C_o', 'M68D_i', 'M68D_o', \
                           ]:
            for (s_input, s_labels), (i_input, i_labels), (o_input, o_labels), \
                (i_gy_input, i_gy_labels), (o_gy_input, o_gy_labels) in zip(*val_loader):
                
                # Labels
                if model_name in ["MP0_i", 'MP0_no_relu_i', "MP25P_i", 'MP25P_2_i', "MP25PPSmall_i", 'M60B_i', 'M60C_i']:
                    labels = i_labels
                    feature_sizes = [1]
                elif model_name in ["MP0_o", 'MP0_no_relu_o', "MP25P_o", 'MP25P_2_o', "MP25PPSmall_o", 'M60B_o', 'M60C_o']:
                    labels = o_labels
                    feature_sizes = [1]
                elif model_name in ["MP24", "MP24PP", 'M61A', 'M61B', 'M61C', 'M65A', 'M65B', 'M65C']:
                    i_labels_reshaped = i_labels.reshape(i_labels.shape[0], 1, 1, -1)
                    o_labels_reshaped = o_labels.reshape(o_labels.shape[0], 1, 1, -1)
                    labels_list = [i_labels_reshaped, o_labels_reshaped]
                    labels = torch.concat(labels_list, dim=-1)
                    feature_sizes = [i_labels_reshaped.shape[-1], o_labels_reshaped.shape[-1]]
                elif model_name in ["MP25_i", "MP25X_i", "MP25PP_i", 'M60A_i', 'M60D_i', 'M60E_i', 'M64A_i', 'M64B_i', 'M64C_i', 'M68A_i', 'M68B_i', 'M68C_i', 'M68D_i']:
                    i_labels_reshaped = i_labels.reshape(i_labels.shape[0], 1, 1, -1)
                    labels_list = [i_labels_reshaped]
                    labels = torch.concat(labels_list, dim=-1)
                    feature_sizes = [i_labels_reshaped.shape[-1]]
                elif model_name in ["MP25_o", "MP25X_o", "MP25PP_o", 'M60A_o', 'M60D_o', 'M60E_o', 'M64A_o', 'M64B_o', 'M64C_o', 'M68A_o', 'M68B_o', 'M68C_o', 'M68D_o']:
                    o_labels_reshaped = o_labels.reshape(o_labels.shape[0], 1, 1, -1)
                    labels_list = [o_labels_reshaped]
                    labels = torch.concat(labels_list, dim=-1)
                    feature_sizes = [o_labels_reshaped.shape[-1]]
                    
                # Inputs
                if model_name in ["MP0_i", 'MP0_no_relu_i']:
                    encoder_inputs = [i_gy_input]
                elif model_name in ["MP0_o", 'MP0_no_relu_o']:
                    encoder_inputs = [o_gy_input]
                elif model_name in ["MP25P_i", "MP25P_o", 'MP25P_2_i', 'MP25P_2_o', "MP25PPSmall_i", "MP25PPSmall_o", "MP24", "MP24PP", "MP25_i", "MP25X_i", "MP25PP_i", "MP25_o", "MP25X_o", "MP25PP_o"]:
                    encoder_inputs = [i_gy_input, o_gy_input]
                elif model_name in ['M60A_i', 'M60A_o', 'M60B_i', 'M60B_o', 'M60C_i', 'M60C_o', 'M60D_i', 'M60D_o', 'M60E_i', 'M60E_o', 'M61A', 'M61B', 'M61C']:
                    encoder_inputs = [i_input, o_input, i_gy_input, o_gy_input]
                elif model_name in ['M64A_i', 'M64A_o', 'M64B_i', 'M64B_o', 'M64C_i', 'M64C_o', 'M65A', 'M65B', 'M65C', 'M68A_i', 'M68A_o', 'M68B_i', 'M68B_o', 'M68C_i', 'M68C_o', 'M68D_i', 'M68D_o']:
                    encoder_inputs = [s_input, i_input, o_input, i_gy_input, o_gy_input]
                    
#                 print(f"encoder_inputs[0].shape: {encoder_inputs[0].shape}")
#                 print(f"labels.shape: {labels.shape}")
                loss, separate_losses = compute_val_loss_array_output_step(encoder_inputs=encoder_inputs, 
                                                labels=labels, 
                                                model=model, 
                                                criterion=criterion, 
                                                feature_sizes=feature_sizes,
                                            )
                batch_loss.append(loss.item())
                batch_separate_losses.append(separate_losses)
                l_p_norm_labels = criterion(torch.zeros_like(labels.clone().detach().flatten()), \
                                                labels.clone().detach().flatten())
                l_p_norm_labels_list.append(l_p_norm_labels.item())
                
                

        elif model_name in ['MP0_gy_avg1_i', 'MP0_gy_avg1_o', 'MP0_no_relu_gy_avg1_i', 'MP0_no_relu_gy_avg1_o', 'MP24_gy_avg1', 'MP24PP_gy_avg1', 'MP25_gy_avg1_i', 'MP25_gy_avg1_o', 'MP25X_gy_avg1_i', 'MP25X_gy_avg1_o', 'MP25P_gy_avg1_i', 'MP25P_gy_avg1_o', 'MP25P_2_gy_avg1_i', 'MP25P_2_gy_avg1_o', 'MP25PPSmall_gy_avg1_i', 'MP25PPSmall_gy_avg1_o', 'MP25PP_gy_avg1_i', 'MP25PP_gy_avg1_o', \
                           'M60A_gy_avg1_i', 'M60A_gy_avg1_o', 'M60B_gy_avg1_i', 'M60B_gy_avg1_o', 'M60C_gy_avg1_i', 'M60C_gy_avg1_o', 'M60D_gy_avg1_i', 'M60D_gy_avg1_o', 'M60E_gy_avg1_i', 'M60E_gy_avg1_o', 'M61A_gy_avg1', 'M61B_gy_avg1', 'M61C_gy_avg1', \
                           'M64A_gy_avg1_i', 'M64A_gy_avg1_o', 'M64B_gy_avg1_i', 'M64B_gy_avg1_o', 'M64C_gy_avg1_i', 'M64C_gy_avg1_o', 'M65A_gy_avg1', 'M65B_gy_avg1', 'M65C_gy_avg1', \
                           'M68A_gy_avg1_i', 'M68A_gy_avg1_o', 'M68B_gy_avg1_i', 'M68B_gy_avg1_o', 'M68C_gy_avg1_i', 'M68C_gy_avg1_o', 'M68D_gy_avg1_i', 'M68D_gy_avg1_o', \
                           ]:
            for (s_input, s_labels), (i_input, i_labels), (o_input, o_labels), \
                (i_gy_input, i_gy_labels), (o_gy_input, o_gy_labels) in zip(*val_loader):
                
                # Labels
                if model_name in ["MP0_gy_avg1_i", 'MP0_no_relu_gy_avg1_i', "MP25P_gy_avg1_i", 'MP25P_2_gy_avg1_i', "MP25PPSmall_gy_avg1_i", 'M60B_gy_avg1_i', 'M60C_gy_avg1_i']:
                    labels = i_labels
                    feature_sizes = [1]
                elif model_name in ["MP0_gy_avg1_o", 'MP0_no_relu_gy_avg1_o', "MP25P_gy_avg1_o", 'MP25P_2_gy_avg1_o', "MP25PPSmall_gy_avg1_o", 'M60B_gy_avg1_o', 'M60C_gy_avg1_o']:
                    labels = o_labels
                    feature_sizes = [1]
                elif model_name in ["MP24_gy_avg1", "MP24PP_gy_avg1", 'M61A_gy_avg1', 'M61B_gy_avg1', 'M61C_gy_avg1', 'M65A_gy_avg1', 'M65B_gy_avg1', 'M65C_gy_avg1']:
                    i_labels_reshaped = i_labels.reshape(i_labels.shape[0], 1, 1, -1)
                    o_labels_reshaped = o_labels.reshape(o_labels.shape[0], 1, 1, -1)
                    labels_list = [i_labels_reshaped, o_labels_reshaped]
                    labels = torch.concat(labels_list, dim=-1)
                    feature_sizes = [i_labels_reshaped.shape[-1], o_labels_reshaped.shape[-1]]
                elif model_name in ["MP25_gy_avg1_i", "MP25X_gy_avg1_i", "MP25PP_gy_avg1_i", 'M60A_gy_avg1_i', 'M60D_gy_avg1_i', 'M60E_gy_avg1_i', 'M64A_gy_avg1_i', 'M64B_gy_avg1_i', 'M64C_gy_avg1_i', 'M68A_gy_avg1_i', 'M68B_gy_avg1_i', 'M68C_gy_avg1_i', 'M68D_gy_avg1_i']:
                    i_labels_reshaped = i_labels.reshape(i_labels.shape[0], 1, 1, -1)
                    labels_list = [i_labels_reshaped]
                    labels = torch.concat(labels_list, dim=-1)
                    feature_sizes = [i_labels_reshaped.shape[-1]]
                elif model_name in ["MP25_gy_avg1_o", "MP25X_gy_avg1_o", "MP25PP_gy_avg1_o", 'M60A_gy_avg1_o', 'M60D_gy_avg1_o', 'M60E_gy_avg1_o', 'M64A_gy_avg1_o', 'M64B_gy_avg1_o', 'M64C_gy_avg1_o', 'M68A_gy_avg1_o', 'M68B_gy_avg1_o', 'M68C_gy_avg1_o', 'M68D_gy_avg1_o']:
                    o_labels_reshaped = o_labels.reshape(o_labels.shape[0], 1, 1, -1)
                    labels_list = [o_labels_reshaped]
                    labels = torch.concat(labels_list, dim=-1)
                    feature_sizes = [o_labels_reshaped.shape[-1]]
                    
                # Inputs
                if model_name in ["MP0_gy_avg1_i", 'MP0_no_relu_gy_avg1_i']:
                    encoder_inputs = [i_gy_input]
                elif model_name in ["MP0_gy_avg1_o", 'MP0_no_relu_gy_avg1_o']:
                    encoder_inputs = [o_gy_input]
                elif model_name in ["MP25P_gy_avg1_i", "MP25P_gy_avg1_o", 'MP25P_2_gy_avg1_i', 'MP25P_2_gy_avg1_o', "MP25PPSmall_gy_avg1_i", "MP25PPSmall_gy_avg1_o", "MP24_gy_avg1", "MP24PP_gy_avg1", "MP25_gy_avg1_i", "MP25X_gy_avg1_i", "MP25PP_gy_avg1_i", "MP25_gy_avg1_o", "MP25X_gy_avg1_o", "MP25PP_gy_avg1_o"]:
                    encoder_inputs = [i_gy_input, o_gy_input]
                elif model_name in ['M60A_gy_avg1_i', 'M60A_gy_avg1_o', 'M60B_gy_avg1_i', 'M60B_gy_avg1_o', 'M60C_gy_avg1_i', 'M60C_gy_avg1_o', 'M60D_gy_avg1_i', 'M60D_gy_avg1_o', 'M60E_gy_avg1_i', 'M60E_gy_avg1_o', 'M61A_gy_avg1', 'M61B_gy_avg1', 'M61C_gy_avg1']:
                    encoder_inputs = [i_input, o_input, i_gy_input, o_gy_input]
                elif model_name in ['M64A_gy_avg1_i', 'M64A_gy_avg1_o', 'M64B_gy_avg1_i', 'M64B_gy_avg1_o', 'M64C_gy_avg1_i', 'M64C_gy_avg1_o', 'M65A_gy_avg1', 'M65B_gy_avg1', 'M65C_gy_avg1', 'M68A_gy_avg1_i', 'M68A_gy_avg1_o', 'M68B_gy_avg1_i', 'M68B_gy_avg1_o', 'M68C_gy_avg1_i', 'M68C_gy_avg1_o', 'M68D_gy_avg1_i', 'M68D_gy_avg1_o']:
                    encoder_inputs = [s_input, i_input, o_input, i_gy_input, o_gy_input]
                    
#                 print(f"encoder_inputs[0].shape: {encoder_inputs[0].shape}")
#                 print(f"labels.shape: {labels.shape}")
                loss, separate_losses = compute_val_loss_array_output_step(encoder_inputs=encoder_inputs, 
                                                labels=labels, 
                                                model=model, 
                                                criterion=criterion, 
                                                feature_sizes=feature_sizes,
                                            )
                batch_loss.append(loss.item())
                batch_separate_losses.append(separate_losses)
                l_p_norm_labels = criterion(torch.zeros_like(labels.clone().detach().flatten()), \
                                                labels.clone().detach().flatten())
                l_p_norm_labels_list.append(l_p_norm_labels.item())
                
        elif model_name in ["multiprediction_9", "multiprediction_10", "multiprediction_11", "multiprediction_12", "multiprediction_13", "multiprediction_14", "multiprediction_15", "multiprediction_16", "multiprediction_17", "multiprediction_18", "multiprediction_19", "multiprediction_21", "multiprediction_23", "multiprediction_24", "multiprediction_25", "multiprediction_25pp", "multiprediction_26", "multiprediction_26_dp", "multiprediction_26_o", "multiprediction_26_o_dp", "multiprediction_27", "multiprediction_27_dp", "multiprediction_39_dp", "multiprediction_40_dp", "multiprediction_31_dp", "multiprediction_32_dp", "multiprediction_35_dp", "multiprediction_36_dp", "multiprediction_47", "multiprediction_48", "multiprediction_51", "multiprediction_52", "multiprediction_53", "multiprediction_54", "multiprediction_47_finetune", "multiprediction_48_finetune", "multiprediction_51_finetune", "multiprediction_52_finetune", "multiprediction_53_finetune", "multiprediction_54_finetune", "multiprediction_55", "multiprediction_56", "surge_1_1", "surge_1_2", "surge_1_3", "surge_1_4", "surge_1_dp_1", "surge_1_dp_2", "surge_1_dp_3", "surge_1_dp_4", "surge_2_1", "surge_2_2", "surge_2_3", "surge_2_4", "surge_2_dp_1", "surge_2_dp_2", "surge_2_dp_3", "surge_2_dp_4", ]:
            for (encoder_inputs1, labels1), (encoder_inputs2, labels2), (encoder_inputs3, labels3) \
                    in zip(*val_loader):
                if model_name in ["multiprediction_9", "multiprediction_13", "multiprediction_15"]:
                    labels1_reshaped = labels1.reshape(labels1.shape[0], 1, 1, -1)
                    labels2_reshaped = labels2.reshape(labels2.shape[0], 1, 1, -1)
                    labels_list = [labels1_reshaped, labels2_reshaped]
                    labels = torch.concat(labels_list, dim=-1)
                    encoder_inputs = [encoder_inputs1, encoder_inputs2]
                    feature_sizes = [labels1_reshaped.shape[-1], labels2_reshaped.shape[-1]]
                elif model_name in ["multiprediction_10", "multiprediction_14", "multiprediction_16"]:
                    labels1_reshaped = labels1.reshape(labels1.shape[0], 1, 1, -1)
                    labels3_reshaped = labels3.reshape(labels3.shape[0], 1, 1, -1)
                    labels_list = [labels1_reshaped, labels3_reshaped]
                    labels = torch.concat(labels_list, dim=-1)
                    encoder_inputs = [encoder_inputs1, encoder_inputs3]
                    feature_sizes = [labels1_reshaped.shape[-1], labels3_reshaped.shape[-1]]
                elif model_name in ["multiprediction_11"]:
                    labels1_reshaped = labels1.reshape(labels1.shape[0], 1, 1, -1)
                    labels2_reshaped = labels2.reshape(labels2.shape[0], 1, 1, -1)
                    labels_list = [labels1_reshaped, labels2_reshaped]
                    labels = torch.concat(labels_list, dim=-1)
                    encoder_inputs = [encoder_inputs1, encoder_inputs2]
                    feature_sizes = [labels1_reshaped.shape[-1], labels2_reshaped.shape[-1]]
                elif model_name in ["multiprediction_12"]:
                    labels1_reshaped = labels1.reshape(labels1.shape[0], 1, 1, -1)
                    labels3_reshaped = labels3.reshape(labels3.shape[0], 1, 1, -1)
                    labels_list = [labels1_reshaped, labels3_reshaped]
                    labels = torch.concat(labels_list, dim=-1)
                    encoder_inputs = [encoder_inputs1, encoder_inputs3]
                    feature_sizes = [labels1_reshaped.shape[-1], labels3_reshaped.shape[-1]]
                elif model_name in ["multiprediction_17", "multiprediction_31_dp", "multiprediction_35_dp", "multiprediction_39_dp", "multiprediction_47", "multiprediction_49", "multiprediction_51", "multiprediction_53", "multiprediction_47_finetune", "multiprediction_51_finetune", "multiprediction_53_finetune"]:
                    labels2_reshaped = labels2.reshape(labels2.shape[0], 1, 1, -1)
                    labels_list = [labels2_reshaped]
                    labels = torch.concat(labels_list, dim=-1)
                    encoder_inputs = [encoder_inputs1, encoder_inputs2, encoder_inputs3]
                    feature_sizes = [labels2_reshaped.shape[-1]]
                elif model_name in ["multiprediction_18", "multiprediction_26_o", "multiprediction_26_o_dp", "multiprediction_32_dp", "multiprediction_36_dp", "multiprediction_40_dp", "multiprediction_48", "multiprediction_52", "multiprediction_54", "multiprediction_48_finetune", "multiprediction_50_finetune", "multiprediction_52_finetune", "multiprediction_54_finetune", "multiprediction_56"]:
                    labels3_reshaped = labels3.reshape(labels3.shape[0], 1, 1, -1)
                    labels_list = [labels3_reshaped]
                    labels = torch.concat(labels_list, dim=-1)
                    encoder_inputs = [encoder_inputs1, encoder_inputs2, encoder_inputs3]
                    feature_sizes = [labels3_reshaped.shape[-1]]
                elif model_name in ["multiprediction_26", "multiprediction_26_dp", "multiprediction_55"]:
                    labels2_reshaped = labels2.reshape(labels2.shape[0], 1, 1, -1)
                    labels_list = [labels2_reshaped]
                    labels = torch.concat(labels_list, dim=-1)
                    encoder_inputs = [encoder_inputs1, encoder_inputs2]
                    feature_sizes = [labels2_reshaped.shape[-1]]
                elif model_name in ["multiprediction_19"]:
                    labels1_reshaped = labels1.reshape(labels1.shape[0], 1, 1, -1)
                    labels2_reshaped = labels2.reshape(labels2.shape[0], 1, 1, -1)
                    labels_list = [labels1_reshaped, labels2_reshaped]
                    labels = torch.concat(labels_list, dim=-1)
                    encoder_inputs = [encoder_inputs1, encoder_inputs2, encoder_inputs3]
                    feature_sizes = [labels1_reshaped.shape[-1], labels2_reshaped.shape[-1]]
                elif model_name in ["multiprediction_21"]:
                    labels1_reshaped = labels1.reshape(labels1.shape[0], 1, 1, -1)
                    labels2_reshaped = labels2.reshape(labels2.shape[0], 1, 1, -1)
                    labels3_reshaped = labels3.reshape(labels3.shape[0], 1, 1, -1)
                    labels_list = [labels1_reshaped, labels2_reshaped, labels3_reshaped]
                    labels = torch.concat(labels_list, dim=-1)
                    encoder_inputs = [encoder_inputs1, encoder_inputs2, encoder_inputs3]
                    feature_sizes = [labels1_reshaped.shape[-1], labels2_reshaped.shape[-1], labels3_reshaped.shape[-1]]
                elif model_name in ["multiprediction_23"]:
                    labels2_reshaped = labels2.reshape(labels2.shape[0], 1, 1, -1)
                    labels3_reshaped = labels3.reshape(labels3.shape[0], 1, 1, -1)
                    labels_list = [labels2_reshaped, labels3_reshaped]
                    labels = torch.concat(labels_list, dim=-1)
                    encoder_inputs = [encoder_inputs1, encoder_inputs2, encoder_inputs3]
                    feature_sizes = [labels2_reshaped.shape[-1], labels3_reshaped.shape[-1]]
                elif model_name in ["multiprediction_24"]:
                    labels2_reshaped = labels2.reshape(labels2.shape[0], 1, 1, -1)
                    labels3_reshaped = labels3.reshape(labels3.shape[0], 1, 1, -1)
                    labels_list = [labels2_reshaped, labels3_reshaped]
                    labels = torch.concat(labels_list, dim=-1)
                    encoder_inputs = [encoder_inputs2, encoder_inputs3]
                    feature_sizes = [labels2_reshaped.shape[-1], labels3_reshaped.shape[-1]]
                elif model_name in ["multiprediction_25", "multiprediction_25pp"]:
                    labels2_reshaped = labels2.reshape(labels2.shape[0], 1, 1, -1)
                    labels_list = [labels2_reshaped]
                    labels = torch.concat(labels_list, dim=-1)
                    encoder_inputs = [encoder_inputs2, encoder_inputs3]
                    feature_sizes = [labels2_reshaped.shape[-1]]
                elif model_name in ["multiprediction_27", "multiprediction_27_dp"]:
                    labels2_reshaped = labels2.reshape(labels2.shape[0], 1, 1, -1)
                    labels_list = [labels2_reshaped]
                    labels = torch.concat(labels_list, dim=-1)
                    encoder_inputs = [encoder_inputs1, encoder_inputs2]
                    feature_sizes = [labels2_reshaped.shape[-1]]
                elif model_name in ["surge_1_1", "surge_1_2", "surge_1_3", "surge_1_4", "surge_1_dp_1", "surge_1_dp_2", "surge_1_dp_3", "surge_1_dp_4", "surge_2_1", "surge_2_2", "surge_2_3", "surge_2_4", "surge_2_dp_1", "surge_2_dp_2", "surge_2_dp_3", "surge_2_dp_4", ]:
                    labels2_reshaped = labels2.reshape(labels2.shape[0], 1, 1, -1) # Inflow
                    labels3_reshaped = labels3.reshape(labels3.shape[0], 1, 1, -1) # Outflow
                    diff_outflow_inflow_labels = labels3_reshaped - labels2_reshaped # Outflow (Demand) - Inflow (Supply)
                    labels_list = [diff_outflow_inflow_labels]
                    labels = torch.concat(labels_list, dim=-1)
                    encoder_inputs = [encoder_inputs1, encoder_inputs2, encoder_inputs3] # Speed, Inflow, and Outflow
                    feature_sizes = [diff_outflow_inflow_labels.shape[-1]]
                    
                loss, separate_losses = compute_val_loss_array_output_step(encoder_inputs=encoder_inputs, 
                                                labels=labels, 
                                                model=model, 
                                                criterion=criterion, 
                                                feature_sizes=feature_sizes,
                                            )
                batch_loss.append(loss.item())
                batch_separate_losses.append(separate_losses)
                l_p_norm_labels = criterion(torch.zeros_like(labels.clone().detach().flatten()), \
                                                labels.clone().detach().flatten())
                l_p_norm_labels_list.append(l_p_norm_labels.item())
                
                
        
        elif model_name in ["dual_company_1_i_1", "dual_company_1_o_1", "dual_company_1_i_2", "dual_company_1_o_2", "dual_company_2_i_1", "dual_company_2_o_1", "dual_company_2_i_2", "dual_company_2_o_2", "dual_company_2_i_3", "dual_company_2_o_3", "dual_company_2_i_4", "dual_company_2_o_4", ]:
            for (speed_inputs, speed_labels), (inflow_inputs_1, inflow_labels_1), (outflow_inputs_1, outflow_labels_1), \
                    (inflow_inputs_2, inflow_labels_2), (outflow_inputs_2, outflow_labels_2) in zip(*val_loader):
                if model_name in ["dual_company_1_i_1", "dual_company_1_i_2", ]:
                    labels = inflow_labels_1
                    encoder_inputs = [inflow_inputs_1, inflow_inputs_2]
                elif model_name in ["dual_company_1_o_1", "dual_company_1_o_2", ]:
                    labels = outflow_labels_1
                    encoder_inputs = [outflow_inputs_1, outflow_inputs_2]
                elif model_name in ["dual_company_2_i_1", "dual_company_2_i_2", "dual_company_2_i_3", "dual_company_2_i_4", ]:
                    labels = inflow_labels_1.reshape(inflow_labels_1.shape[0], 1, 1, -1)
                    encoder_inputs = [inflow_inputs_1, inflow_inputs_2]
                elif model_name in ["dual_company_2_o_1", "dual_company_2_o_2", "dual_company_2_o_3", "dual_company_2_o_4", ]:
                    labels = outflow_labels_1.reshape(outflow_labels_1.shape[0], 1, 1, -1)
                    encoder_inputs = [outflow_inputs_1, outflow_inputs_2]
                loss, _, loss_unnormalized = compute_val_loss_step(encoder_inputs=encoder_inputs, 
                                                                    labels=labels, 
                                                                    model=model, 
                                                                    criterion=criterion, 
                                                                    mu=mu, 
                                                                    std=std,
                                                                  )
                batch_loss.append(loss.item())
                batch_loss_unnormalized.append(loss_unnormalized.item())
                l_p_norm_labels = criterion(torch.zeros_like(labels.clone().detach().flatten()), \
                                                labels.clone().detach().flatten())
                l_p_norm_labels_list.append(l_p_norm_labels.item())
                
                
                    
                    
        elif model_name in ["multiprediction_37_dp", "multiprediction_38_dp", "multiprediction_33_dp", "multiprediction_34_dp"]:
            for (encoder_inputs1, labels1), (encoder_inputs2, labels2), (encoder_inputs3, labels3) \
                    in zip(*val_loader):
                if model_name in ["multiprediction_33_dp", "multiprediction_37_dp"]: 
                    labels = labels2
                elif model_name in ["multiprediction_34_dp", "multiprediction_38_dp"]: 
                    labels = labels3
                encoder_inputs = [encoder_inputs1, encoder_inputs2, encoder_inputs3]
                loss, _, loss_unnormalized = compute_val_loss_step(encoder_inputs, 
                                                labels, 
                                                model, 
                                                criterion, 
                                                mu=mu, 
                                                std=std,
                                               )
                batch_loss.append(loss.item())
                batch_loss_unnormalized.append(loss_unnormalized.item())
                l_p_norm_labels = criterion(torch.zeros_like(labels.clone().detach().flatten()), \
                                                labels.clone().detach().flatten())
                l_p_norm_labels_list.append(l_p_norm_labels.item())
            
        elif ensemble==0: # No ensemble
            print("************** ensemble==0 **************")
            for encoder_inputs, labels in val_loader[0]:
#                 loss, _ = compute_val_loss_step(encoder_inputs, labels, model, criterion, edge_index_data[0], ensemble)
                loss, _ = compute_val_loss_step(encoder_inputs, labels, model, criterion)
                batch_loss.append(loss.item())
                l_p_norm_labels = criterion(torch.zeros_like(labels.clone().detach().flatten()), \
                                                labels.clone().detach().flatten())
                l_p_norm_labels_list.append(l_p_norm_labels.item())
                
                
        validation_loss = sum(batch_loss) / len(batch_loss)
        if batch_loss_unnormalized:
            val_loss_unnormalized = sum(batch_loss_unnormalized) / len(batch_loss_unnormalized)      
        else:
            val_loss_unnormalized = None
        l_p_norm_labels_avg = sum(l_p_norm_labels_list) / len(l_p_norm_labels_list)        

        return validation_loss, sum(batch_loss)/sum(l_p_norm_labels_list), sum(batch_loss), len(batch_loss), batch_separate_losses, l_p_norm_labels_avg, val_loss_unnormalized


    
def validation_epoch(val_step, epoch, model, model_name, val_loader, criterion, normalization_param_dict, normalization, ensemble=0, test=False):
    if (epoch+1) % val_step == 0:
        val_loss, normalized_val_loss, val_loss_total, n_samples, separate_losses, _, val_loss_unnormalized = \
                                        compute_val_loss(model=model, 
                                                         model_name=model_name, 
                                                         val_loader=val_loader, 
                                                         criterion=criterion, 
                                                         ensemble=ensemble,
                                                         normalization=normalization,
                                                         normalization_param_dict=normalization_param_dict,
                                                         test=test,
                                                        )
        err_type = "Test" if test else "Validation"
        print('\nEpoch %s, %s Error %s: %.4f, Normalized Validation Error: %.4f' % (epoch, err_type, criterion, val_loss, normalized_val_loss))
        if model_name in ["multiprediction_1", "multiprediction_2", "multiprediction_2_2", "multiprediction_3", "multiprediction_4", "multiprediction_9", "multiprediction_10", "multiprediction_11", "multiprediction_12", "multiprediction_13", "multiprediction_14", "multiprediction_15", "multiprediction_16", "multiprediction_17", "multiprediction_18", "multiprediction_19", "multiprediction_21", "multiprediction_23", "multiprediction_24", "multiprediction_25", "multiprediction_25pp", "multiprediction_26", "multiprediction_26_dp", "multiprediction_26_o", "multiprediction_26_o_dp", "multiprediction_27", "multiprediction_27_dp", "multiprediction_55", "multiprediction_56", "MP24", "MP24PP"] or ensemble!=0:
            separate_losses_avg = np.mean(np.array(separate_losses), axis=0)
            print(f"Validation separate_losses_avg: {separate_losses_avg}")
        else:
            separate_losses_avg = [val_loss]

#         print('Validation SE Total: %.4f, # of Samples: %.4f' % (val_loss_total, n_samples))
        return val_loss, separate_losses_avg, val_loss_unnormalized
    else:
        return None, [], None
    
    
# def train_epoch(model, epoch, train_loader, optimizer, criterion, val_step, edge_index_data, privacy_engine, train_data, ensemble=0, dp=False, delta=1e-5):
def train_epoch(model, 
                model_name, 
                epoch, 
                train_loader, 
                optimizer, 
                criterion, 
                val_step, 
                privacy_engine, 
                train_data, 
                ensemble = 0, 
                dp = False, 
                delta = 1e-5,
               ):
    # Fix randomization
    set_seed(5)
    
    def train_epoch_step(criterion, y_hat, labels, optimizer):
        loss = criterion(y_hat, labels) 
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return loss
        
    # Training
    model.train()
    loss_list = []
    if model_name=="multiprediction_1": #https://discuss.pytorch.org/t/a-model-with-multiple-outputs/10440
        for batch_idx, _ in train_loader:
            # Get model predictions
            print(f"train_data[0][0][batch_idx].shape: {train_data[0][0][batch_idx].shape}")
            y_hat_list = model(train_data[0][0][batch_idx])
            print(f"len(train_data): {len(train_data)}")
            # Loss computation
            loss = 0
            for output_idx in range(len(train_data)):
#                 label = train_data[output_idx][1][batch_idx].reshape(batch_idx.shape[0], 1, 1, -1)
                label = train_data[output_idx][1][batch_idx]
                loss += criterion(y_hat_list[output_idx], label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_list.append(loss.item())
        
    elif model_name=="multiprediction_2" or model_name=="multiprediction_2_2" or model_name=="multiprediction_3" or model_name=="multiprediction_4": #https://discuss.pytorch.org/t/a-model-with-multiple-outputs/10440
        for batch_idx, _ in train_loader:
            # Get model predictions
#             y_hat = model([train_data[0][0][batch_idx], \
#                                    train_data[1][0][batch_idx], \
#                                    train_data[2][0][batch_idx]])
            y_hat = model(train_data[0][0][batch_idx])
            # Labels
            label1 = train_data[0][1][batch_idx].reshape(batch_idx.shape[0], 1, 1, -1)
            label2 = train_data[1][1][batch_idx].reshape(batch_idx.shape[0], 1, 1, -1)
            label3 = train_data[2][1][batch_idx].reshape(batch_idx.shape[0], 1, 1, -1)
            labels = torch.concat((label1, 
                                   label2, 
                                   label3), 
                                   dim=-1
                                  )
            # Loss computation
#             print(f"y_hat.shape: {y_hat.shape}")
#             print(f"labels.shape: {labels.shape}")
            loss = train_epoch_step(criterion, y_hat, labels, optimizer)
            loss_list.append(loss.item())
                    
    elif model_name[:3] in ["MSA"]:
        for batch_idx, _ in train_loader:
            # Indicies: 0: speed, 1: c1 inflow, 2: c1 outflow, 3: c2 inflow, 4: c2 outflow, 5: gy inflow, 6: gy outflow
            # 7: c1 surge, 8: c2 surge
            # Get model predictions given inputs
            if model_name[:4] in ["MSA3"]: 
                y_hat = model([train_data[0][0][batch_idx],\
                               train_data[1][0][batch_idx],\
                               train_data[2][0][batch_idx],\
                              ])
            elif model_name[:4] in ["MSA5"]: 
                y_hat = model([train_data[0][0][batch_idx],\
                               train_data[1][0][batch_idx],\
                               train_data[2][0][batch_idx],\
                               train_data[3][0][batch_idx],\
                               train_data[4][0][batch_idx],\
                              ])
                
            # Labels
            if model_name[-1] == "L":
                labels = train_data[7][1][batch_idx].reshape(batch_idx.shape[0], 1, 1, -1)
            else:
                labels = train_data[7][1][batch_idx]
                        
            loss = train_epoch_step(criterion, y_hat, labels, optimizer)
            loss_list.append(loss.item())
                        
            
                    
    elif model_name[:2] in ["MA", "MM"] or model_name[:3] == 'avg':
        for batch_idx, _ in train_loader:
            # Indicies: 0: speed, 1: c1 inflow, 2: c1 outflow, 3: c2 inflow, 4: c2 outflow, 5: gy inflow, 6: gy outflow
            # Get model predictions given inputs
            if model_name[:5] in ["MA1_I", "MM1_I"] or model_name=='avg_i': 
                y_hat = model([train_data[1][0][batch_idx]])
            elif model_name[:5] in ["MA1_O", "MM1_O"] or model_name=='avg_o': 
                y_hat = model([train_data[2][0][batch_idx]])
            elif model_name[:5] in ["MA1_O", "MM1_O"] or model_name=='avg_s': 
                y_hat = model([train_data[0][0][batch_idx]])
                
            elif model_name[:6] in ["MA2_I1", "MM2_I1"]: 
                y_hat = model([train_data[1][0][batch_idx],
                               train_data[3][0][batch_idx],])
            elif model_name[:6] in ["MA2_I2", "MM2_I2"]: 
                y_hat = model([train_data[1][0][batch_idx],
                               train_data[5][0][batch_idx],])
            elif model_name[:6] in ["MA2_I3", "MM2_I3"]: 
                y_hat = model([train_data[1][0][batch_idx],
                               train_data[2][0][batch_idx],])
            elif model_name[:6] in ["MA2_I4", "MM2_I4"]: 
                y_hat = model([train_data[1][0][batch_idx],
                               train_data[4][0][batch_idx],])
            elif model_name[:6] in ["MA2_I5", "MM2_I5"]: 
                y_hat = model([train_data[1][0][batch_idx],
                               train_data[6][0][batch_idx],])
                
            elif model_name[:6] in ["MA2_O1", "MM2_O1"]: 
                y_hat = model([train_data[2][0][batch_idx],
                               train_data[4][0][batch_idx],])
            elif model_name[:6] in ["MA2_O2", "MM2_O2"]: 
                y_hat = model([train_data[2][0][batch_idx],
                               train_data[6][0][batch_idx],])
            elif model_name[:6] in ["MA2_O3", "MM2_O3"]: 
                y_hat = model([train_data[2][0][batch_idx],
                               train_data[1][0][batch_idx],])
            elif model_name[:6] in ["MA2_O4", "MM2_O4"]: 
                y_hat = model([train_data[2][0][batch_idx],
                               train_data[3][0][batch_idx],])
            elif model_name[:6] in ["MA2_O5", "MM2_O5"]: 
                y_hat = model([train_data[2][0][batch_idx],
                               train_data[5][0][batch_idx],])
                
            elif model_name[:6] in ["MA3_I1", "MA3_O1", "MM3_I1", "MM3_O1"]: 
                y_hat = model([train_data[1][0][batch_idx],
                               train_data[2][0][batch_idx],
                               train_data[3][0][batch_idx],])
            elif model_name[:6] in ["MA3_I2", "MA3_O2", "MM3_I2", "MM3_O2"]: 
                y_hat = model([train_data[1][0][batch_idx],
                               train_data[2][0][batch_idx],
                               train_data[4][0][batch_idx],])
            elif model_name[:6] in ["MA3_I3", "MA3_O3", "MM3_I3", "MM3_O3"]: 
                y_hat = model([train_data[1][0][batch_idx],
                               train_data[2][0][batch_idx],
                               train_data[5][0][batch_idx],])
            elif model_name[:6] in ["MA3_I4", "MA3_O4", "MM3_I4", "MM3_O4"]: 
                y_hat = model([train_data[1][0][batch_idx],
                               train_data[2][0][batch_idx],
                               train_data[6][0][batch_idx],])
                
            elif model_name[:6] in ["MA4_I1", "MA4_O1", "MM4_I1", "MM4_O1"] or model_name[:8] in ["MA4_DP_I", "MA4_DP_O"]: 
                y_hat = model([train_data[1][0][batch_idx],
                               train_data[2][0][batch_idx],
                               train_data[3][0][batch_idx],
                               train_data[4][0][batch_idx],])
                
            elif model_name[:6] in ["MA4_I2", "MA4_O2", "MM4_I2", "MM4_O2"] or \
                     model_name[:8] in ["MA4_GY_I", "MM4_GY_I", "MA4_GY_O", "MM4_GY_O"]: 
                if model_name[:9] in ["MA4_GY_O2", "MM4_GY_O2"]:
                    y_hat = model([train_data[2][0][batch_idx],
                                   train_data[1][0][batch_idx],
                                   train_data[5][0][batch_idx],
                                   train_data[6][0][batch_idx],])
                else:
                    y_hat = model([train_data[1][0][batch_idx],
                                   train_data[2][0][batch_idx],
                                   train_data[5][0][batch_idx],
                                   train_data[6][0][batch_idx],])
                
            elif model_name[:7] in ["MA3_S_I", "MA3_S_O", "MM3_S_I", "MM3_S_O"] or model_name[:8] in ["MA3_S2_I", "MA3_S2_O"]: 
                if model_name[:8] in ["MA3_S_O2", "MM3_S_O2"]:
                    y_hat = model([train_data[0][0][batch_idx],
                                   train_data[2][0][batch_idx],
                                   train_data[1][0][batch_idx],])
                else:
                    y_hat = model([train_data[0][0][batch_idx],
                                   train_data[1][0][batch_idx],
                                   train_data[2][0][batch_idx],])
                    
            elif model_name[:7] in ["MA5_S_I", "MA5_S_O"] or model_name[:8] in ["MA5_S2_I", "MA5_S2_O"] or \
                    model_name[:10] in ["MA5_S_DP_I", "MA5_S_DP_O"]:
                y_hat = model([train_data[0][0][batch_idx],
                               train_data[1][0][batch_idx],
                               train_data[2][0][batch_idx],
                               train_data[3][0][batch_idx],
                               train_data[4][0][batch_idx],])
                    
            elif model_name[:6] in ["MA6_I1", "MA6_O1"]: 
                y_hat = model([train_data[1][0][batch_idx],
                               train_data[2][0][batch_idx],
                               train_data[3][0][batch_idx],
                               train_data[4][0][batch_idx],
                               train_data[5][0][batch_idx],
                               train_data[6][0][batch_idx],])
                
                
            # Labels
            if model_name[:5] in ["MA1_I", "MA2_I", "MA3_I", "MA4_I", "MA6_I", "MM1_I", "MM2_I", "MM3_I", "MM4_I"] or \
                    model_name[:7] in ["MA3_S_I", "MA5_S_I", "MM3_S_I"] or model_name=='avg_i' or \
                    model_name[:8] in ["MA3_S2_I", "MA5_S2_I"] or \
                     model_name[:8] in ["MA4_GY_I", "MM4_GY_I"]:
                if model_name[-1] == "L":
                    labels = train_data[1][1][batch_idx].reshape(batch_idx.shape[0], 1, 1, -1)
                else:
                    labels = train_data[1][1][batch_idx]
            elif model_name[:5] in ["MA1_O", "MA2_O", "MA3_O", "MA4_O", "MA6_O", "MM1_O", "MM2_O", "MM3_O", "MM4_O"] or \
                    model_name[:7] in ["MA3_S_O", "MA5_S_O", "MM3_S_O"] or model_name=='avg_o' or \
                    model_name[:8] in ["MA3_S2_O", "MA5_S2_O"] or \
                     model_name[:8] in ["MA4_GY_O", "MM4_GY_O"]:
                if model_name[-1] == "L":
                    labels = train_data[2][1][batch_idx].reshape(batch_idx.shape[0], 1, 1, -1)
                else:
                    labels = train_data[2][1][batch_idx]
            elif model_name[:8] in ["MA4_DP_I"] or \
                    model_name[:10] in ["MA5_S_DP_I"]:
                if model_name[-1] == "L":
                    labels = train_data[3][1][batch_idx].reshape(batch_idx.shape[0], 1, 1, -1)
                else:
                    labels = train_data[3][1][batch_idx]
            elif model_name[:8] in ["MA4_DP_O"] or \
                    model_name[:10] in ["MA5_S_DP_O"]:
                if model_name[-1] == "L":
                    labels = train_data[4][1][batch_idx].reshape(batch_idx.shape[0], 1, 1, -1)
                else:
                    labels = train_data[4][1][batch_idx]
            elif model_name=='avg_s':
                if model_name[-1] == "L":
                    labels = train_data[0][1][batch_idx].reshape(batch_idx.shape[0], 1, 1, -1)
                else:
                    labels = train_data[0][1][batch_idx]
                        
            loss = train_epoch_step(criterion, y_hat, labels, optimizer)
            loss_list.append(loss.item())
                        
                        
    elif model_name in ['MP0_i', 'MP0_o', 'MP0_no_relu_i', 'MP0_no_relu_o', 'MP24', 'MP24PP', 'MP25_i', 'MP25_o', 'MP25X_i', 'MP25X_o', 'MP25P_i', 'MP25P_o', 'MP25P_2_i', 'MP25P_2_o', "MP25PPSmall_i", "MP25PPSmall_o", 'MP25PP_i', 'MP25PP_o', \
                       'M60A_i', 'M60A_o', 'M60B_i', 'M60B_o', 'M60C_i', 'M60C_o', 'M60D_i', 'M60D_o', 'M60E_i', 'M60E_o', 'M61A', 'M61B', 'M61C', \
                       'M64A_i', 'M64A_o', 'M64B_i', 'M64B_o', 'M64C_i', 'M64C_o', 'M65A', 'M65B', 'M65C', \
                       'M68A_i', 'M68A_o', 'M68B_i', 'M68B_o', 'M68C_i', 'M68C_o', 'M68D_i', 'M68D_o', \
                       ]:
        for batch_idx, _ in train_loader:
            # Get model predictions given inputs
            if model_name in ["MP0_i", 'MP0_no_relu_i']: # Input: inflow of gy
                y_hat = model([train_data[3][0][batch_idx]])
            
            elif model_name in ["MP0_o", 'MP0_no_relu_o']: # Input: outflow of gy
                y_hat = model([train_data[4][0][batch_idx]])
                
            elif model_name in ["MP24", 'MP24PP', 'MP25_i', 'MP25_o', 'MP25X_i', 'MP25X_o', 'MP25P_i', 'MP25P_o', 'MP25P_2_i', 'MP25P_2_o', "MP25PPSmall_i", "MP25PPSmall_o", 'MP25PP_i', 'MP25PP_o']: # Inputs: inflow of gy + outflow of gy 
                y_hat = model([train_data[3][0][batch_idx],
                               train_data[4][0][batch_idx],])
                
            elif model_name in ['M60A_i', 'M60A_o', 'M60B_i', 'M60B_o', 'M60C_i', 'M60C_o', 'M60D_i', 'M60D_o', 'M60E_i', 'M60E_o', 'M61A', 'M61B', 'M61C']: # Inputs: inflow + outflow + inflow of gy + outflow of gy 
                y_hat = model([train_data[1][0][batch_idx],
                               train_data[2][0][batch_idx],
                               train_data[3][0][batch_idx],
                               train_data[4][0][batch_idx],])
                
            elif model_name in ['M64A_i', 'M64A_o', 'M64B_i', 'M64B_o', 'M64C_i', 'M64C_o', 'M65A', 'M65B', 'M65C', 'M68A_i', 'M68A_o', 'M68B_i', 'M68B_o', 'M68C_i', 'M68C_o', 'M68D_i', 'M68D_o']: # Inputs: inflow + outflow + inflow of gy + outflow of gy 
                y_hat = model([train_data[0][0][batch_idx],
                               train_data[1][0][batch_idx],
                               train_data[2][0][batch_idx],
                               train_data[3][0][batch_idx],
                               train_data[4][0][batch_idx],])
                
            # Labels
            if model_name in ["MP0_i", 'MP0_no_relu_i', "MP25P_i", 'MP25P_2_i', "MP25PPSmall_i", 'M60B_i', 'M60C_i']: # Label(s): inflow
#                 labels = train_data[1][1][batch_idx].reshape(batch_idx.shape[0], 1, 1, -1)
                labels = train_data[1][1][batch_idx]
            
            elif model_name in ["MP0_o", 'MP0_no_relu_o', "MP25P_o", 'MP25P_2_o', "MP25PPSmall_o", 'M60B_o', 'M60C_o']: # Label(s): outflow
#                 labels = train_data[2][1][batch_idx].reshape(batch_idx.shape[0], 1, 1, -1)
                labels = train_data[2][1][batch_idx]
    
            if model_name in ["MP25_i", "MP25X_i", "MP25PP_i", 'M60A_i', 'M60D_i', 'M60E_i', 'M64A_i', 'M64B_i', 'M64C_i', 'M68A_i', 'M68B_i', 'M68C_i', 'M68D_i']: # Label(s): inflow
                labels = train_data[1][1][batch_idx].reshape(batch_idx.shape[0], 1, 1, -1)
                
            elif model_name in ["MP25_o", "MP25X_o", "MP25PP_o", 'M60A_o', 'M60D_o', 'M60E_o', 'M64A_o', 'M64B_o', 'M64C_o', 'M68A_o', 'M68B_o', 'M68C_o', 'M68D_o']: # Label(s): outflow
                labels = train_data[2][1][batch_idx].reshape(batch_idx.shape[0], 1, 1, -1)
                
            elif model_name in ["MP24", 'MP24PP', 'M61A', 'M61B', 'M61C', 'M65A', 'M65B', 'M65C']: # Label(s): inflow + outflow
                label_i = train_data[1][1][batch_idx].reshape(batch_idx.shape[0], 1, 1, -1)
                label_o = train_data[2][1][batch_idx].reshape(batch_idx.shape[0], 1, 1, -1)
                labels = torch.concat((label_i, 
                                       label_o), 
                                       dim=-1
                                      )
                
            loss = train_epoch_step(criterion, y_hat, labels, optimizer)
            loss_list.append(loss.item())
            
    
    elif model_name in ['MP0_gy_avg1_i', 'MP0_gy_avg1_o', 'MP0_no_relu_gy_avg1_i', 'MP0_no_relu_gy_avg1_o', 'MP24_gy_avg1', 'MP24PP_gy_avg1', 'MP25_gy_avg1_i', 'MP25_gy_avg1_o', 'MP25X_gy_avg1_i', 'MP25X_gy_avg1_o', 'MP25P_gy_avg1_i', 'MP25P_gy_avg1_o', 'MP25P_2_gy_avg1_i', 'MP25P_2_gy_avg1_o', "MP25PPSmall_gy_avg1_i", "MP25PPSmall_gy_avg1_o", 'MP25PP_gy_avg1_i', 'MP25PP_gy_avg1_o', \
                       'M60A_gy_avg1_i', 'M60A_gy_avg1_o', 'M60B_gy_avg1_i', 'M60B_gy_avg1_o', 'M60C_gy_avg1_i', 'M60C_gy_avg1_o', 'M60D_gy_avg1_i', 'M60D_gy_avg1_o', 'M60E_gy_avg1_i', 'M60E_gy_avg1_o', 'M61A_gy_avg1', 'M61B_gy_avg1', 'M61C_gy_avg1', \
                       'M64A_gy_avg1_i', 'M64A_gy_avg1_o', 'M64B_gy_avg1_i', 'M64B_gy_avg1_o', 'M64C_gy_avg1_i', 'M64C_gy_avg1_o', 'M65A_gy_avg1', 'M65B_gy_avg1', 'M65C_gy_avg1', \
                       'M68A_gy_avg1_i', 'M68A_gy_avg1_o', 'M68B_gy_avg1_i', 'M68B_gy_avg1_o', 'M68C_gy_avg1_i', 'M68C_gy_avg1_o', 'M68D_gy_avg1_i', 'M68D_gy_avg1_o', \
                       ]:
        for batch_idx, _ in train_loader:
            # Get model predictions given inputs
            if model_name in ["MP0_gy_avg1_i", 'MP0_no_relu_gy_avg1_i']: # Input: inflow of gy
                y_hat = model([train_data[3][0][batch_idx]])
            
            elif model_name in ["MP0_gy_avg1_o", 'MP0_no_relu_gy_avg1_o']: # Input: outflow of gy
                y_hat = model([train_data[4][0][batch_idx]])
                
            elif model_name in ["MP24_gy_avg1", 'MP24PP_gy_avg1', 'MP25_gy_avg1_i', 'MP25_gy_avg1_o', 'MP25X_gy_avg1_i', 'MP25X_gy_avg1_o', 'MP25P_gy_avg1_i', 'MP25P_gy_avg1_o', 'MP25P_2_gy_avg1_i', 'MP25P_2_gy_avg1_o', "MP25PPSmall_gy_avg1_i", "MP25PPSmall_gy_avg1_o", 'MP25PP_gy_avg1_i', 'MP25PP_gy_avg1_o']: # Inputs: inflow of gy + outflow of gy 
                y_hat = model([train_data[3][0][batch_idx],
                               train_data[4][0][batch_idx],])
                
            elif model_name in ['M60A_gy_avg1_i', 'M60A_gy_avg1_o', 'M60B_gy_avg1_i', 'M60B_gy_avg1_o', 'M60C_gy_avg1_i', 'M60C_gy_avg1_o', 'M60D_gy_avg1_i', 'M60D_gy_avg1_o', 'M60E_gy_avg1_i', 'M60E_gy_avg1_o', 'M61A_gy_avg1', 'M61B_gy_avg1', 'M61C_gy_avg1']: # Inputs: inflow + outflow + inflow of gy + outflow of gy 
                y_hat = model([train_data[1][0][batch_idx],
                               train_data[2][0][batch_idx],
                               train_data[3][0][batch_idx],
                               train_data[4][0][batch_idx],])
                
            elif model_name in ['M64A_gy_avg1_i', 'M64A_gy_avg1_o', 'M64B_gy_avg1_i', 'M64B_gy_avg1_o', 'M64C_gy_avg1_i', 'M64C_gy_avg1_o', 'M65A_gy_avg1', 'M65B_gy_avg1', 'M65C_gy_avg1', 'M68A_gy_avg1_i', 'M68A_gy_avg1_o', 'M68B_gy_avg1_i', 'M68B_gy_avg1_o', 'M68C_gy_avg1_i', 'M68C_gy_avg1_o', 'M68D_gy_avg1_i', 'M68D_gy_avg1_o']: # Inputs: inflow + outflow + inflow of gy + outflow of gy 
                y_hat = model([train_data[0][0][batch_idx],
                               train_data[1][0][batch_idx],
                               train_data[2][0][batch_idx],
                               train_data[3][0][batch_idx],
                               train_data[4][0][batch_idx],])
                
            # Labels
            if model_name in ["MP0_gy_avg1_i", 'MP0_no_relu_gy_avg1_i', "MP25P_gy_avg1_i", 'MP25P_2_gy_avg1_i', "MP25PPSmall_gy_avg1_i", 'M60B_gy_avg1_i', 'M60C_gy_avg1_i']: # Label(s): inflow
#                 labels = train_data[1][1][batch_idx].reshape(batch_idx.shape[0], 1, 1, -1)
                labels = train_data[1][1][batch_idx]
            
            elif model_name in ["MP0_gy_avg1_o", 'MP0_no_relu_gy_avg1_o', "MP25P_gy_avg1_o", 'MP25P_2_gy_avg1_o', "MP25PPSmall_gy_avg1_o", 'M60B_gy_avg1_o', 'M60C_gy_avg1_o']: # Label(s): outflow
#                 labels = train_data[2][1][batch_idx].reshape(batch_idx.shape[0], 1, 1, -1)
                labels = train_data[2][1][batch_idx]
    
            elif model_name in ["MP25_gy_avg1_i", "MP25X_gy_avg1_i", "MP25PP_gy_avg1_i", 'M60A_gy_avg1_i', 'M60D_gy_avg1_i', 'M60E_gy_avg1_i', 'M64A_gy_avg1_i', 'M64B_gy_avg1_i', 'M64C_gy_avg1_i', 'M68A_gy_avg1_i', 'M68B_gy_avg1_i', 'M68C_gy_avg1_i', 'M68D_gy_avg1_i']: # Label(s): inflow
                labels = train_data[1][1][batch_idx].reshape(batch_idx.shape[0], 1, 1, -1)
                
            elif model_name in ["MP25_gy_avg1_o", "MP25X_gy_avg1_o", "MP25PP_gy_avg1_o", 'M60A_gy_avg1_o', 'M60D_gy_avg1_o', 'M60E_gy_avg1_o', 'M64A_gy_avg1_o', 'M64B_gy_avg1_o', 'M64C_gy_avg1_o', 'M68A_gy_avg1_o', 'M68B_gy_avg1_o', 'M68C_gy_avg1_o', 'M68D_gy_avg1_o']: # Label(s): outflow
                labels = train_data[2][1][batch_idx].reshape(batch_idx.shape[0], 1, 1, -1)
                
            elif model_name in ["MP24_gy_avg1", 'MP24PP_gy_avg1', 'M61A_gy_avg1', 'M61B_gy_avg1', 'M61C_gy_avg1', 'M65A_gy_avg1', 'M65B_gy_avg1', 'M65C_gy_avg1']: # Label(s): inflow + outflow
                label_i = train_data[1][1][batch_idx].reshape(batch_idx.shape[0], 1, 1, -1)
                label_o = train_data[2][1][batch_idx].reshape(batch_idx.shape[0], 1, 1, -1)
                labels = torch.concat((label_i, 
                                       label_o), 
                                       dim=-1
                                      )
                
            loss = train_epoch_step(criterion, y_hat, labels, optimizer)
            loss_list.append(loss.item())
    elif model_name in ["multiprediction_5", "multiprediction_6", "multiprediction_7", "multiprediction_8", "multiprediction_9", "multiprediction_10", "multiprediction_11", "multiprediction_12", "multiprediction_13", "multiprediction_14", "multiprediction_15", "multiprediction_16", "multiprediction_17", "multiprediction_18", "multiprediction_19", "multiprediction_21", "multiprediction_23", "multiprediction_24", "multiprediction_25", "multiprediction_25pp", "multiprediction_26", "multiprediction_26_dp", "multiprediction_26_o", "multiprediction_26_o_dp", "multiprediction_27", "multiprediction_27_dp", "multiprediction_37_dp", "multiprediction_38_dp", "multiprediction_39_dp", "multiprediction_40_dp", "multiprediction_31_dp", "multiprediction_32_dp", "multiprediction_33_dp", "multiprediction_34_dp", "multiprediction_35_dp", "multiprediction_36_dp", "multiprediction_47", "multiprediction_48", "multiprediction_51", "multiprediction_52", "multiprediction_53", "multiprediction_54", "multiprediction_47_finetune", "multiprediction_48_finetune", "multiprediction_51_finetune", "multiprediction_52_finetune", "multiprediction_53_finetune", "multiprediction_54_finetune", "multiprediction_55", "multiprediction_56", "dual_company_1_i_1", "dual_company_1_o_1", "dual_company_1_i_2", "dual_company_1_o_2", "dual_company_2_i_1", "dual_company_2_o_1", "dual_company_2_i_2", "dual_company_2_o_2", "dual_company_2_i_3", "dual_company_2_o_3", "dual_company_2_i_4", "dual_company_2_o_4", "surge_1_1", "surge_1_2", "surge_1_3", "surge_1_4", "surge_1_dp_1", "surge_1_dp_2", "surge_1_dp_3", "surge_1_dp_4", "surge_2_1", "surge_2_2", "surge_2_3", "surge_2_4", "surge_2_dp_1", "surge_2_dp_2", "surge_2_dp_3", "surge_2_dp_4",]:
        for batch_idx, _ in train_loader:
            # Get model predictions
            if model_name in ["multiprediction_5", "multiprediction_6"]: # Inputs: Speed
                y_hat = model(train_data[0][0][batch_idx])
            elif model_name in ["multiprediction_7", "multiprediction_9", "multiprediction_11", "multiprediction_13", "multiprediction_15", "multiprediction_26", "multiprediction_26_dp", "multiprediction_27", "multiprediction_27_dp", "multiprediction_55"]: # Inputs: Speed and inflow 
                y_hat = model([train_data[0][0][batch_idx],
                               train_data[1][0][batch_idx],])
            elif model_name in ["multiprediction_8", "multiprediction_10", "multiprediction_12", "multiprediction_14", "multiprediction_16", "multiprediction_26_o", "multiprediction_26_o_dp", "multiprediction_56"]: # Inputs: Speed and outflow 
                y_hat = model([train_data[0][0][batch_idx],
                               train_data[2][0][batch_idx],])
            elif model_name in ["multiprediction_17", "multiprediction_18", "multiprediction_19", "multiprediction_21", "multiprediction_23", "multiprediction_37_dp", "multiprediction_38_dp", "multiprediction_39_dp", "multiprediction_40_dp", "multiprediction_31_dp", "multiprediction_32_dp", "multiprediction_33_dp", "multiprediction_34_dp", "multiprediction_35_dp", "multiprediction_36_dp", "multiprediction_47", "multiprediction_48", "multiprediction_51", "multiprediction_52", "multiprediction_53", "multiprediction_54", "multiprediction_47_finetune", "multiprediction_48_finetune", "multiprediction_51_finetune", "multiprediction_52_finetune", "multiprediction_53_finetune", "multiprediction_54_finetune", "surge_1_1", "surge_1_2", "surge_1_3", "surge_1_4", "surge_1_dp_1", "surge_1_dp_2", "surge_1_dp_3", "surge_1_dp_4", "surge_2_1", "surge_2_2", "surge_2_3", "surge_2_4", "surge_2_dp_1", "surge_2_dp_2", "surge_2_dp_3", "surge_2_dp_4",]: # Inputs: Speed + inflow + outflow 
                y_hat = model([train_data[0][0][batch_idx],
                               train_data[1][0][batch_idx],
                               train_data[2][0][batch_idx],])
            elif model_name in ["multiprediction_24", "multiprediction_25", "multiprediction_25pp"]: # Inputs: inflow + outflow 
                y_hat = model([train_data[1][0][batch_idx],
                               train_data[2][0][batch_idx],])
            elif model_name in ["dual_company_1_i_1", "dual_company_1_i_2", "dual_company_2_i_1", "dual_company_2_i_2", "dual_company_2_i_3", "dual_company_2_i_4", ]: # Inputs: inflow 
                y_hat = model([train_data[1][0][batch_idx],  # Inflow of the company
                               train_data[3][0][batch_idx],])  # Inflow of the other company
            elif model_name in ["dual_company_1_o_1", "dual_company_1_o_2", "dual_company_2_o_1", "dual_company_2_o_2", "dual_company_2_o_3", "dual_company_2_o_4", ]: # Inputs: outflow 
                y_hat = model([train_data[2][0][batch_idx], # Outflow of the company
                               train_data[4][0][batch_idx],])  # Outflow of the other company
                
                
                
            # Labels
            if model_name in ["multiprediction_5", "multiprediction_7", "multiprediction_17", "multiprediction_25", "multiprediction_25pp", "multiprediction_26", "multiprediction_26_dp", "multiprediction_27", "multiprediction_27_dp", "multiprediction_39_dp", "multiprediction_35_dp", "multiprediction_31_dp", "multiprediction_47", "multiprediction_51", "multiprediction_53", "multiprediction_47_finetune", "multiprediction_51_finetune", "multiprediction_53_finetune", "multiprediction_55", "dual_company_2_i_1", "dual_company_2_i_2", "dual_company_2_i_3", "dual_company_2_i_4",]: # Label(s): Inflow
                labels = train_data[1][1][batch_idx].reshape(batch_idx.shape[0], 1, 1, -1)
                
            elif model_name in ["multiprediction_6", "multiprediction_8", "multiprediction_18", "multiprediction_26_o", "multiprediction_26_o_dp", "multiprediction_40_dp", "multiprediction_36_dp", "multiprediction_32_dp", "multiprediction_48", "multiprediction_52", "multiprediction_54", "multiprediction_48_finetune", "multiprediction_52_finetune", "multiprediction_54_finetune", "multiprediction_56", "dual_company_2_o_1", "dual_company_2_o_2", "dual_company_2_o_3", "dual_company_2_o_4", ]: # Outflow as labels
                labels = train_data[2][1][batch_idx].reshape(batch_idx.shape[0], 1, 1, -1)
                
            elif model_name in ["multiprediction_9", "multiprediction_11", "multiprediction_13", "multiprediction_15", "multiprediction_19"]: # Label(s): Speed + inflow
                label1 = train_data[0][1][batch_idx].reshape(batch_idx.shape[0], 1, 1, -1)
                label2 = train_data[1][1][batch_idx].reshape(batch_idx.shape[0], 1, 1, -1)
                labels = torch.concat((label1, 
                                       label2), 
                                       dim=-1
                                      )
            elif model_name in ["multiprediction_10", "multiprediction_12", "multiprediction_14", "multiprediction_16"]: # Label(s): Speed + outflow
                label1 = train_data[0][1][batch_idx].reshape(batch_idx.shape[0], 1, 1, -1)
                label3 = train_data[2][1][batch_idx].reshape(batch_idx.shape[0], 1, 1, -1)
                labels = torch.concat((label1, 
                                       label3), 
                                       dim=-1
                                      )
            elif model_name in ["multiprediction_21"]: # Label(s): Speed + inflow + outflow
                label1 = train_data[0][1][batch_idx].reshape(batch_idx.shape[0], 1, 1, -1)
                label2 = train_data[1][1][batch_idx].reshape(batch_idx.shape[0], 1, 1, -1)
                label3 = train_data[2][1][batch_idx].reshape(batch_idx.shape[0], 1, 1, -1)
                labels = torch.concat((label1, 
                                       label2, 
                                       label3), 
                                       dim=-1
                                      )
            elif model_name in ["multiprediction_23", "multiprediction_24"]: # Label(s): Speed + inflow + outflow
                label2 = train_data[1][1][batch_idx].reshape(batch_idx.shape[0], 1, 1, -1)
                label3 = train_data[2][1][batch_idx].reshape(batch_idx.shape[0], 1, 1, -1)
                labels = torch.concat((label2, 
                                       label3), 
                                       dim=-1
                                      )
            elif model_name in ["dual_company_1_i_1", "dual_company_1_i_2", "multiprediction_33_dp", "multiprediction_37_dp"]: # Label(s): inflow
                labels = train_data[1][1][batch_idx]
            elif model_name in ["dual_company_1_o_1", "dual_company_1_o_2", "multiprediction_34_dp", "multiprediction_38_dp"]: # Label(s): outflow
                labels = train_data[2][1][batch_idx]
            elif model_name in ["surge_1_1", "surge_1_2", "surge_1_3", "surge_1_4", "surge_1_dp_1", "surge_1_dp_2", "surge_1_dp_3", "surge_1_dp_4", "surge_2_1", "surge_2_2", "surge_2_3", "surge_2_4", "surge_2_dp_1", "surge_2_dp_2", "surge_2_dp_3", "surge_2_dp_4",]: # Label: outflow - inflow
                label2 = train_data[1][1][batch_idx].reshape(batch_idx.shape[0], 1, 1, -1) # Inflow
                label3 = train_data[2][1][batch_idx].reshape(batch_idx.shape[0], 1, 1, -1) # Outflow
                labels = label3 - label2
                
            loss = train_epoch_step(criterion, y_hat, labels, optimizer)
            loss_list.append(loss.item())
    elif model_name in ['astgcn_customized_no_relu']:
        for batch_idx, _ in train_loader:
#             encoder_input = train_data[0][0]
            encoder_input_batch = train_data[0][0][batch_idx]
            # Get model predictions
            y_hat = model(encoder_input_batch) # Get model predictions
            # Labeles
#             labels = train_data[0][1]
            labels_batch = train_data[0][1][batch_idx]
            # Loss computation
            loss = train_epoch_step(criterion, y_hat, labels_batch, optimizer)
            loss_list.append(loss.item())
        
    elif ensemble==0: # No ensemble
        for batch_idx, _ in train_loader:
#             encoder_input = train_data[0][0]
            encoder_input_batch = train_data[0][0][batch_idx]
            # Get model predictions
            y_hat = model(x=encoder_input_batch) # Get model predictions
            # Labeles
#             labels = train_data[0][1]
            labels_batch = train_data[0][1][batch_idx]
            # Loss computation
            loss = train_epoch_step(criterion, y_hat, labels_batch, optimizer)
            loss_list.append(loss.item())
        
    train_loss = sum(loss_list)/len(loss_list)
    if (epoch+1) % val_step == 0:
        print("Epoch {}, Train Error: {:.2f}".format(epoch, train_loss))
    
    epsilon = None
    if dp:
        epsilon = privacy_engine.accountant.get_epsilon(delta=delta)
        print(f"(ε = {epsilon:.2f}, δ = {delta})")
        

    return model, train_loss, epsilon
    

    
def get_model_path(config_address):
    configuration_temp = read_yaml(config_address)
    yaml_config = yaml_to_config(**configuration_temp['parameters'])
    return get_model_name(yaml_config)

          
def build_network(model_name, n_input_periods, n_output_periods, batch_size, device, edge_index_list, \
                  node_features, n_nodes, ensemble, out_channels, dp, graph_version, \
                  start_month, end_month, year, company, dp_loaded,\
                  hidden=0, model_path = None, hidden_warmup=None, nb_block=2, K=3, \
                  nb_chev_filter=64, nb_time_filter=64, time_strides=1):
    # Fix randomization
    set_seed(5)
    
    company_str, dp_loaded_str = "", ""
    if company != "": 
        company_str = "_" + company
    if dp_loaded != "": 
        dp_loaded_str = "_" + dp_loaded
    
    model_func_dict_1_4 = {'multiprediction_1': MultiPredictionNet1,
                            'multiprediction_2': MultiPredictionNet2,
                            'multiprediction_2_2': MultiPredictionNet2_2,
                            'multiprediction_3': MultiPredictionNet3,
                            'multiprediction_4': MultiPredictionNet4,
                          }
    
    model_func_dict = {'multiprediction_5': MultiPredictionNet5,
                        'multiprediction_6': MultiPredictionNet6,
                        'multiprediction_7': MultiPredictionNet7,
                        'multiprediction_8': MultiPredictionNet8,
                        'multiprediction_11': MultiPredictionNet11,
                        'multiprediction_12': MultiPredictionNet12,
                        'multiprediction_17': MultiPredictionNet17,
                        'multiprediction_18': MultiPredictionNet17,
                        'multiprediction_19': MultiPredictionNet19,
                        'multiprediction_21': MultiPredictionNet21,
                        'multiprediction_23': MultiPredictionNet23,
                        'multiprediction_24': MultiPredictionNet24,
                        'mp0_i': MP0_i,
                        'mp0_o': MP0_o,
                        'mp0_no_relu_i': MP0_no_relu,
                        'mp0_no_relu_o': MP0_no_relu,
                        'mp24': MP24,
                        'mp24pp': MP24PP,
                        'multiprediction_25': MultiPredictionNet25,
                        'mp25_i': MP25,
                        'mp25_o': MP25,
                        'mp25x_i': MP25X,
                        'mp25x_o': MP25X,
                        'multiprediction_25pp': MultiPredictionNet25pp,
                        'mp25p_i': MP25P,
                        'mp25p_o': MP25P,
                        'mp25p_2_i': MP25P_2,
                        'mp25p_2_o': MP25P_2,
                        'mp25pp_i': MP25PP,
                        'mp25pp_o': MP25PP,
                        'mp25ppsmall_i': MP25PPSmall,
                        'mp25ppsmall_o': MP25PPSmall,
                        'm60a_i': M60A,
                        'm60a_o': M60A,
                        'm60b_i': M60B,
                        'm60b_o': M60B,
                        'm60c_i': M60C,
                        'm60c_o': M60C,
                        'm60d_i': M60D,
                        'm60d_o': M60D,
                        'm60e_i': M60E,
                        'm60e_o': M60E,
                        'm61a': M61A,
                        'm61b': M61B,
                        'm61c': M61C,
                        'mp0_gy_avg1_i': MP0_i,
                        'mp0_gy_avg1_o': MP0_o,
                        'mp0_no_relu_gy_avg1_i': MP0_no_relu,
                        'mp0_no_relu_gy_avg1_o': MP0_no_relu,
                        'mp24_gy_avg1': MP24,
                        'mp24pp_gy_avg1': MP24PP,
                        'multiprediction_25': MultiPredictionNet25,
                        'mp25_gy_avg1_i': MP25,
                        'mp25_gy_avg1_o': MP25,
                        'mp25x_gy_avg1_i': MP25X,
                        'mp25x_gy_avg1_o': MP25X,
                        'multiprediction_25pp': MultiPredictionNet25pp,
                        'mp25p_gy_avg1_i': MP25P,
                        'mp25p_gy_avg1_o': MP25P,
                        'mp25p_2_gy_avg1_i': MP25P_2,
                        'mp25p_2_gy_avg1_o': MP25P_2,
                        'mp25pp_gy_avg1_i': MP25PP,
                        'mp25pp_gy_avg1_o': MP25PP,
                        'mp25ppsmall_gy_avg1_i': MP25PPSmall,
                        'mp25ppsmall_gy_avg1_o': MP25PPSmall,
                        'm60a_gy_avg1_i': M60A,
                        'm60a_gy_avg1_o': M60A,
                        'm60b_gy_avg1_i': M60B,
                        'm60b_gy_avg1_o': M60B,
                        'm60c_gy_avg1_i': M60C,
                        'm60c_gy_avg1_o': M60C,
                        'm60d_gy_avg1_i': M60D,
                        'm60d_gy_avg1_o': M60D,
                        'm60e_gy_avg1_i': M60E,
                        'm60e_gy_avg1_o': M60E,
                        'm61a_gy_avg1': M61A,
                        'm61b_gy_avg1': M61B,
                        'm61c_gy_avg1': M61C,
#                         'm62A_i': M62A,
#                         'm62A_o': M62A,
#                         'm62B_i': M62B,
#                         'm62B_o': M62B,
#                         'm62C_i': M62C,
#                         'm62C_o': M62C,
#                         'm62D_i': M62D,
#                         'm62D_o': M62D,
#                         'm62E_i': M62E,
#                         'm62E_o': M62E,
#                         'm63A': M63A,
#                         'm63B': M63B,
#                         'm63C': M63C,
                        'm64a_gy_avg1_i': M64A,
                        'm64a_gy_avg1_o': M64A,
                        'm64b_gy_avg1_i': M64B,
                        'm64b_gy_avg1_o': M64B,
                        'm64c_gy_avg1_i': M64C,
                        'm64c_gy_avg1_o': M64C,
                        'm65a_gy_avg1': M65A,
                        'm65a_gy_avg1': M65B,
                        'm65a_gy_avg1': M65C,
#                         'm66A_i': M66A,
#                         'm66A_o': M66A,
#                         'm66B_i': M66B,
#                         'm66B_o': M66B,
#                         'm66C_i': M66C,
#                         'm66C_o': M66C,
#                         'm67A': M67A,
#                         'm67B': M67B,
#                         'm67C': M67C,
                        'ma1_i': MA_1in,
                        'ma1_o': MA_1in,
                        'ma2_i1': MA_2in,
                        'ma2_i2': MA_2in,
                        'ma2_i3': MA_2in,
                        'ma2_i4': MA_2in,
                        'ma2_i5': MA_2in,
                        'ma2_o1': MA_2in,
                        'ma2_o2': MA_2in,
                        'ma2_o3': MA_2in,
                        'ma2_o4': MA_2in,
                        'ma2_o5': MA_2in,
                        'ma2_i1_l': MA_2in_linear,
                        'ma2_i2_l': MA_2in_linear,
                        'ma2_i3_l': MA_2in_linear,
                        'ma2_i4_l': MA_2in_linear,
                        'ma2_i5_l': MA_2in_linear,
                        'ma2_o1_l': MA_2in_linear,
                        'ma2_o2_l': MA_2in_linear,
                        'ma2_o3_l': MA_2in_linear,
                        'ma2_o4_l': MA_2in_linear,
                        'ma2_o5_l': MA_2in_linear,
                        'ma3_i1_l': MA_3in_linear,
                        'ma3_i2_l': MA_3in_linear,
                        'ma3_o1_l': MA_3in_linear,
                        'ma3_o2_l': MA_3in_linear,
                        'ma4_i1_l': MA_4in_linear,
                        'ma4_i2_l': MA_4in_linear,
                        'ma4_o1_l': MA_4in_linear,
                        'ma4_o2_l': MA_4in_linear,
                        'ma4_i1': MA_4in,
                        'ma4_i2': MA_4in,
                        'ma4_o1': MA_4in,
                        'ma4_o2': MA_4in,
                        'ma4_dp_i1': MA_4in,
                        'ma4_dp_o1': MA_4in,
                        'ma6_i1': MA_6in,
                        'ma6_o1': MA_6in,
#                         'mm1_i': MM_1in,
#                         'mm1_o': MM_1in,
#                         'mm2_i1': MM_2in,
#                         'mm2_i2': MM_2in,
#                         'mm2_i3': MM_2in,
#                         'mm2_i4': MM_2in,
#                         'mm2_i5': MM_2in,
#                         'mm2_o1': MM_2in,
#                         'mm2_o2': MM_2in,
#                         'mm2_o3': MM_2in,
#                         'mm2_o4': MM_2in,
#                         'mm2_o5': MM_2in,
#                         'mm2_i1_l': MM_2in_linear,
#                         'mm2_i2_l': MM_2in_linear,
#                         'mm2_i3_l': MM_2in_linear,
#                         'mm2_i4_l': MM_2in_linear,
#                         'mm2_i5_l': MM_2in_linear,
#                         'mm2_o1_l': MM_2in_linear,
#                         'mm2_o2_l': MM_2in_linear,
#                         'mm2_o3_l': MM_2in_linear,
#                         'mm2_o4_l': MM_2in_linear,
#                         'mm2_o5_l': MM_2in_linear,
#                         'mm3_i1_l': MM_3in_linear,
#                         'mm3_i2_l': MM_3in_linear,
#                         'mm3_o1_l': MM_3in_linear,
#                         'mm3_o2_l': MM_3in_linear,
#                         'mm4_i1_l': MM_4in_linear,
#                         'mm4_i2_l': MM_4in_linear,
#                         'mm4_o1_l': MM_4in_linear,
#                         'mm4_o2_l': MM_4in_linear,
#                         'mm4_i1': MM_4in,
#                         'mm4_i2': MM_4in,
#                         'mm4_o1': MM_4in,
#                         'mm4_o2': MM_4in,
                      } 

    

    print('\nbuild_network >> Building model ', model_name.lower())
    
    
    if model_name.lower() in model_func_dict.keys():
        model = model_func_dict[model_name.lower()](n_input_periods=n_input_periods,
                                                     n_output_periods=n_output_periods,
                                                     node_features=node_features,
                                                     n_nodes=n_nodes, 
                                                     hidden_warmup=hidden_warmup,
                                                     edge_index_list=edge_index_list,
                                                     model_path=model_path,
                                                   )
        
    # Surge models
    elif model_name.lower()[:3] in ["msa"]:
        # Check which the model is being trained for which company
        other_company_str = other_company_str_dict[company.lower()]
        
        no_backprop_models_dict = {"1": False, "2": True}
        s_no_backprop_dict = {"1": False, "2": False, "3": True, "4": True}
        dp_no_backprop_dict = {"1": False, "2": True, "3": False, "4": True}
        if model_name.lower()[:4] in ["msa5"]:
            no_backprop_models = no_backprop_models_dict[model_name[8]] # 1, 2
            no_backprop_s_models = no_backprop_models_dict[model_name[9]] # 1, 2
        elif model_name.lower()[:4] in ["msa3"]:
            no_backprop_models = no_backprop_models_dict[model_name[8]] # 1, 2
            no_backprop_s_models = s_no_backprop_dict[model_name[9]] # 1, 2, 3, 4
            no_backprop_dp_models = dp_no_backprop_dict[model_name[9]] # 1, 2, 3, 4
            
        configuration_addresses = []   
        if model_name.lower()[:4] in ["msa5"]:
            # Load MA5_S_I1X as a warmed up model
            configuration_addresses.append(("../data/m_"+str(start_month)+"_"+str(end_month)+"_y_"+str(year)+"_optimals/config_MA5_S_I1" + model_name[9] + "_optimal" + company_str + ".yaml"))
            # Load MA5_S_I1X as a warmed up model
            configuration_addresses.append(("../data/m_"+str(start_month)+"_"+str(end_month)+"_y_"+str(year)+"_optimals/config_MA5_S_O1" + model_name[9] + "_optimal" + company_str + ".yaml"))
            
        elif model_name.lower()[:4] in ["msa3"]:
            assert dp_loaded_str != "", print("Define the epsilon of the model!")
            # Model 5
            configuration_addresses.append("../data/m_"+str(start_month)+"_"+str(end_month)+"_y_"+str(year)+"_optimals/config_dp_multi5_g" + str(graph_version) + "_optimal" + other_company_str + dp_loaded_str + ".yaml")
            # Model 6
            configuration_addresses.append("../data/m_"+str(start_month)+"_"+str(end_month)+"_y_"+str(year)+"_optimals/config_dp_multi6_g" + str(graph_version) + "_optimal" + other_company_str + dp_loaded_str + ".yaml")
            # Load MA3_S_I5X as a warmed up model
            configuration_addresses.append(("../data/m_"+str(start_month)+"_"+str(end_month)+"_y_"+str(year)+"_optimals/config_MA3_S_I5" + model_name[9] + "_optimal" + company_str + dp_loaded_str + ".yaml"))
            # Load MA3_S_O5X as a warmed up model
            configuration_addresses.append(("../data/m_"+str(start_month)+"_"+str(end_month)+"_y_"+str(year)+"_optimals/config_MA3_S_O5" + model_name[9] + "_optimal" + company_str + dp_loaded_str + ".yaml"))
            
        # Convert YAML file addresses to model address
        for configuration_address in configuration_addresses:
            model_path.append(get_model_path(configuration_address))
        print(f"model_path of loaded models:")
        print(*model_path, sep='\n')
        
        
        if model_name.lower()[:4] in ["msa3"]:
            model_func = MSA_3in_1_L if model_name.lower()[-1] == 'l' else MSA_3in_1
            model = model_func(n_input_periods=n_input_periods,
                                 n_output_periods=n_output_periods,
                                 node_features=node_features,
                                 n_nodes=n_nodes, 
                                 hidden_warmup=hidden_warmup,
                                 edge_index_list=edge_index_list,
                                 model_path=model_path,
                                 no_backprop_models=no_backprop_models,
                                 no_backprop_s_models=no_backprop_s_models,
                                 no_backprop_dp_models=no_backprop_dp_models,
                               )
        elif model_name.lower()[:4] in ["msa5"]:
            model_func = MSA_5in_1_L if model_name.lower()[-1] == 'l' else MSA_5in_1
            model = model_func(n_input_periods=n_input_periods,
                                 n_output_periods=n_output_periods,
                                 node_features=node_features,
                                 n_nodes=n_nodes, 
                                 hidden_warmup=hidden_warmup,
                                 edge_index_list=edge_index_list,
                                 model_path=model_path,
                                 no_backprop_models=no_backprop_models,
                                 no_backprop_s_models=no_backprop_s_models,
                               )
        
        
      
        
    elif model_name.lower()[:7] in ["ma3_s_i", "ma3_s_o"] or model_name.lower()[:8] in ["ma3_s2_i", "ma3_s2_o"]:
        # Check which the model is being trained for which company
        other_company_str = other_company_str_dict[company.lower()]
        configuration_addresses = []       
        if model_name.lower()[:8] in ["ma3_s_i1", "ma3_s_o1", "ma3_s_i2", "ma3_s_o2", "ma3_s_i3", "ma3_s_o3", "ma3_s_i5", "ma3_s_o5"] or model_name.lower()[:8] in ["ma3_s2_i", "ma3_s2_o"]:
            # Model 5 other company
            configuration_addresses.append("../data/m_"+str(start_month)+"_"+str(end_month)+"_y_"+str(year)+"_optimals/config_dp_multi5_g" + str(graph_version) + "_optimal" + other_company_str + dp_loaded_str + ".yaml")
            # Model 6 other company
            configuration_addresses.append("../data/m_"+str(start_month)+"_"+str(end_month)+"_y_"+str(year)+"_optimals/config_dp_multi6_g" + str(graph_version) + "_optimal" + other_company_str + dp_loaded_str + ".yaml")
        elif model_name.lower()[:8] in ["ma3_s_i6", "ma3_s_o6", "ma3_s_i7", "ma3_s_o7"]:
            # Model 5 of company itself
            configuration_addresses.append("../data/m_"+str(start_month)+"_"+str(end_month)+"_y_"+str(year)+"_optimals/config_dp_multi5_g" + str(graph_version) + "_optimal" + company_str + dp_loaded_str + ".yaml")
            # Model 6 of company itself
            configuration_addresses.append("../data/m_"+str(start_month)+"_"+str(end_month)+"_y_"+str(year)+"_optimals/config_dp_multi6_g" + str(graph_version) + "_optimal" + company_str + dp_loaded_str + ".yaml")
                  
        elif model_name.lower()[:8] in ["ma3_s_i8", "ma3_s_o8"]:
            # Model 5 other company
            configuration_addresses.append("../data/m_"+str(start_month)+"_"+str(end_month)+"_y_"+str(year)+"_optimals/config_dp_multi5_g" + str(graph_version) + "_optimal" + other_company_str + dp_loaded_str + ".yaml")
            # Model 6 other company
            configuration_addresses.append("../data/m_"+str(start_month)+"_"+str(end_month)+"_y_"+str(year)+"_optimals/config_dp_multi6_g" + str(graph_version) + "_optimal" + other_company_str + dp_loaded_str + ".yaml")
            # Model 5 of company itself
            configuration_addresses.append("../data/m_"+str(start_month)+"_"+str(end_month)+"_y_"+str(year)+"_optimals/config_dp_multi5_g" + str(graph_version) + "_optimal" + company_str + dp_loaded_str + ".yaml")
            # Model 6 of company itself
            configuration_addresses.append("../data/m_"+str(start_month)+"_"+str(end_month)+"_y_"+str(year)+"_optimals/config_dp_multi6_g" + str(graph_version) + "_optimal" + company_str + dp_loaded_str + ".yaml")
            
        if model_name.lower()[:8] in ["ma3_s_i2"]:
            # Load MA2_I3 as a warmed up model
            configuration_addresses.append(("../data/m_"+str(start_month)+"_"+str(end_month)+"_y_"+str(year)+"_optimals/config_MA2_I3_optimal" + company_str + ".yaml"))
        elif model_name.lower()[:8] in ["ma3_s_o2"]:
            # Load MA2_O3 as a warmed up model
            configuration_addresses.append(("../data/m_"+str(start_month)+"_"+str(end_month)+"_y_"+str(year)+"_optimals/config_MA2_O3_optimal" + company_str + ".yaml"))
        # Convert YAML file addresses to model address
        for configuration_address in configuration_addresses:
            model_path.append(get_model_path(configuration_address))
        print(f"model_path of loaded models:")
        print(*model_path, sep='\n')
        
        if model_name.lower() in ["ma3_s_i1_l", "ma3_s_o1_l"]:
            model = MA_3in_speed_linear(n_input_periods=n_input_periods,
                                         n_output_periods=n_output_periods,
                                         node_features=node_features,
                                         n_nodes=n_nodes, 
                                         hidden_warmup=hidden_warmup,
                                         edge_index_list=edge_index_list,
                                         model_path=model_path,
                                       )
        elif model_name.lower()[:8] in ["ma3_s_i2", "ma3_s_o2"]:
            nondp_no_backprop_dict = {"ma3_s_i21": False, "ma3_s_i22": False, "ma3_s_i23": True, "ma3_s_i24": True, \
                                       "ma3_s_o21": False, "ma3_s_o22": False, "ma3_s_o23": True, "ma3_s_o24": True}
            dp_no_backprop_dict = {"ma3_s_i21": False, "ma3_s_i22": True, "ma3_s_i23": False, "ma3_s_i24": True, \
                                    "ma3_s_o21": False, "ma3_s_o22": True, "ma3_s_o23": False, "ma3_s_o24": True}
            model_selected = MA_3in_warm_speed_linear if model_name[-1]=="L" else MA_3in_warm_speed
            model = model_selected(n_input_periods=n_input_periods,
                                     n_output_periods=n_output_periods,
                                     node_features=node_features,
                                     n_nodes=n_nodes, 
                                     hidden_warmup=hidden_warmup,
                                     edge_index_list=edge_index_list,
                                     model_path=model_path,
                                     no_backprop_nondp_models = dp_no_backprop_dict[model_name.lower()[:9]],
                                     no_backprop_dp_models = nondp_no_backprop_dict[model_name.lower()[:9]],
                                  )
            
        elif model_name.lower()[:8] in ["ma3_s_i3", "ma3_s_o3", "ma3_s_i7", "ma3_s_o7"]:
            dp_no_backprop_dict = {"ma3_s_i31": False, "ma3_s_i32": True, \
                                      "ma3_s_o31": False, "ma3_s_o32": True, \
                                  "ma3_s_i71": False, "ma3_s_i72": True, \
                                      "ma3_s_o71": False, "ma3_s_o72": True}
            model = MA_3in_speed(n_input_periods=n_input_periods,
                                 n_output_periods=n_output_periods,
                                 node_features=node_features,
                                 n_nodes=n_nodes, 
                                 hidden_warmup=hidden_warmup,
                                 edge_index_list=edge_index_list,
                                 model_path=model_path,
                                 no_backprop_dp_models = dp_no_backprop_dict[model_name.lower()],
                                )
            
        elif model_name.lower()[:8] in ["ma3_s_i8", "ma3_s_o8"]:
            dp_no_backprop_dict = {"ma3_s_i81": False, "ma3_s_i82": True, \
                                      "ma3_s_o81": False, "ma3_s_o82": True}
            model = MA_3in_speed_extra_plus(n_input_periods=n_input_periods,
                                 n_output_periods=n_output_periods,
                                 node_features=node_features,
                                 n_nodes=n_nodes, 
                                 hidden_warmup=hidden_warmup,
                                 edge_index_list=edge_index_list,
                                 model_path=model_path,
                                 no_backprop_dp_models = dp_no_backprop_dict[model_name.lower()],
                                )
            
        elif model_name.lower()[:8] in ["ma3_s_i4", "ma3_s_o4", "ma3_s_i9", "ma3_s_o9"]:
            s_no_backprop_dict = {"ma3_s_i41": False, "ma3_s_i42": True, \
                                  "ma3_s_o41": False, "ma3_s_o42": True, \
                                  "ma3_s_i91_l": False, "ma3_s_i92_l": True, \
                                  "ma3_s_o91_l": False, "ma3_s_o92_l": True, \
                                 }
            model_func_dict = {"ma3_s_i4": MA_3in_speed_baseline, "ma3_s_o4": MA_3in_speed_baseline, 
                               "ma3_s_i9": MA_3in_speed_linear2_baseline, "ma3_s_o9": MA_3in_speed_linear2_baseline, 
                              }
            model = model_func_dict[model_name.lower()[:8]](n_input_periods=n_input_periods,
                                                             n_output_periods=n_output_periods,
                                                             node_features=node_features,
                                                             n_nodes=n_nodes, 
                                                             hidden_warmup=hidden_warmup,
                                                             edge_index_list=edge_index_list,
                                                             model_path=model_path,
                                                             no_backprop_s_models = s_no_backprop_dict[model_name.lower()],
                                                            )

        elif model_name.lower()[:8] in ["ma3_s_i5", "ma3_s_o5", "ma3_s_i6", "ma3_s_o6"]:
            s_no_backprop_dict = {"ma3_s_i51": False, "ma3_s_i52": False, "ma3_s_i53": True, "ma3_s_i54": True, \
                                       "ma3_s_o51": False, "ma3_s_o52": False, "ma3_s_o53": True, "ma3_s_o54": True, \
                                 "ma3_s_i61": False, "ma3_s_i62": False, "ma3_s_i63": True, "ma3_s_i64": True, \
                                       "ma3_s_o61": False, "ma3_s_o62": False, "ma3_s_o63": True, "ma3_s_o64": True}
            dp_no_backprop_dict = {"ma3_s_i51": False, "ma3_s_i52": True, "ma3_s_i53": False, "ma3_s_i54": True, \
                                    "ma3_s_o51": False, "ma3_s_o52": True, "ma3_s_o53": False, "ma3_s_o54": True, 
                                  "ma3_s_i61": False, "ma3_s_i62": True, "ma3_s_i63": False, "ma3_s_i64": True, \
                                    "ma3_s_o61": False, "ma3_s_o62": True, "ma3_s_o63": False, "ma3_s_o64": True}
            model = MA_3in_speed_extra(n_input_periods=n_input_periods,
                                     n_output_periods=n_output_periods,
                                     node_features=node_features,
                                     n_nodes=n_nodes, 
                                     hidden_warmup=hidden_warmup,
                                     edge_index_list=edge_index_list,
                                     model_path=model_path,
                                     no_backprop_s_models = s_no_backprop_dict[model_name.lower()],
                                     no_backprop_dp_models = dp_no_backprop_dict[model_name.lower()],
                                    )
            
                        
        elif model_name.lower()[:8] in ["ma3_s2_i", "ma3_s2_o"]:
            no_backprop_dict = {"ma3_s2_i11": False, "ma3_s2_i12": True, \
                                  "ma3_s2_o11": False, "ma3_s2_o12": True, \
                                  "ma3_s2_i21": False, "ma3_s2_i22": True, \
                                  "ma3_s2_o21": False, "ma3_s2_o22": True, \
                                 }
            model_func_dict = {"ma3_s2_i1": MA_3in_speed_extra_pool_sum, "ma3_s2_o1": MA_3in_speed_extra_pool_sum, 
                               "ma3_s2_i2": MA_3in_pool_sum, "ma3_s2_o2": MA_3in_pool_sum, 
                              }
            model = model_func_dict[model_name.lower()[:9]](n_input_periods=n_input_periods,
                                                             n_output_periods=n_output_periods,
                                                             node_features=node_features,
                                                             n_nodes=n_nodes, 
                                                             hidden_warmup=hidden_warmup,
                                                             edge_index_list=edge_index_list,
                                                             model_path=model_path,
                                                             no_backprop_models = no_backprop_dict[model_name.lower()],
                                                            )

    elif model_name.lower()[:8] in ["ma4_gy_i", "ma4_gy_o"]:
        # Check which the model is being trained for which company
        other_company_str = other_company_str_dict[company.lower()]
        configuration_addresses = []            
        # Model MP0_i
        configuration_addresses.append("../data/m_"+str(start_month)+"_"+str(end_month)+"_y_"+str(year)+"_optimals/config_MP0_i_optimal" + other_company_str + dp_loaded_str + ".yaml")
        # Model MP0_o
        configuration_addresses.append("../data/m_"+str(start_month)+"_"+str(end_month)+"_y_"+str(year)+"_optimals/config_MP0_o_optimal" + other_company_str + dp_loaded_str + ".yaml")
        if model_name.lower()[:9] in ["ma4_gy_i2"]:
            # Load MA2_I as a warmed up model
            configuration_addresses.append(("../data/m_"+str(start_month)+"_"+str(end_month)+"_y_"+str(year)+"_optimals/config_MA2_I3_optimal" + company_str + ".yaml"))
        elif model_name.lower()[:9] in ["ma4_gy_o2"]:
            # Load MA2_O as a warmed up model
            configuration_addresses.append(("../data/m_"+str(start_month)+"_"+str(end_month)+"_y_"+str(year)+"_optimals/config_MA2_O3_optimal" + company_str + ".yaml"))
        # Convert YAML file addresses to model address
        for configuration_address in configuration_addresses:
            model_path.append(get_model_path(configuration_address))
        print(f"model_path of loaded models:")
        print(*model_path, sep='\n')
        
        if model_name.lower() in ["ma4_gy_i1_l", "ma4_gy_o1_l"]:
            model = MA_4in_gy_linear(n_input_periods=n_input_periods,
                                         n_output_periods=n_output_periods,
                                         node_features=node_features,
                                         n_nodes=n_nodes, 
                                         hidden_warmup=hidden_warmup,
                                         edge_index_list=edge_index_list,
                                         model_path=model_path,
                                       )
        elif model_name.lower()[:9] in ["ma4_gy_i2", "ma4_gy_o2"]:
            nondp_no_backprop_dict = {"ma4_gy_i21": False, "ma4_gy_i22": False, "ma4_gy_i23": True, "ma4_gy_i24": True,\
                                      "ma4_gy_o21": False, "ma4_gy_o22": False, "ma4_gy_o23": True, "ma4_gy_o24": True}
            dp_no_backprop_dict = {"ma4_gy_i21": False, "ma4_gy_i22": True, "ma4_gy_i23": False, "ma4_gy_i24": True, \
                                   "ma4_gy_o21": False, "ma4_gy_o22": True, "ma4_gy_o23": False, "ma4_gy_o24": True}
            model_selected = MA_4in_warm_gy_linear if model_name[-1]=="L" else MA_4in_warm_gy
            model = model_selected(n_input_periods=n_input_periods,
                                         n_output_periods=n_output_periods,
                                         node_features=node_features,
                                         n_nodes=n_nodes, 
                                         hidden_warmup=hidden_warmup,
                                         edge_index_list=edge_index_list,
                                         model_path=model_path,
                                         no_backprop_nondp_models = dp_no_backprop_dict[model_name.lower()[:10]],
                                         no_backprop_dp_models = nondp_no_backprop_dict[model_name.lower()[:10]],
                                         )
            
        elif model_name.lower()[:9] in ["ma4_gy_i3", "ma4_gy_o3"]:
            dp_no_backprop_dict = {"ma4_gy_i31": False, "ma4_gy_i32": True, \
                                      "ma4_gy_o31": False, "ma4_gy_o32": True}
            model = MA_4in_gy(n_input_periods=n_input_periods,
                             n_output_periods=n_output_periods,
                             node_features=node_features,
                             n_nodes=n_nodes, 
                             hidden_warmup=hidden_warmup,
                             edge_index_list=edge_index_list,
                             model_path=model_path,
                             no_backprop_dp_models = dp_no_backprop_dict[model_name.lower()],
                             )
            
        elif model_name.lower()[:9] in ["ma4_gy_i4", "ma4_gy_o4"]:
            dp_no_backprop_dict = {"ma4_gy_i41": False, "ma4_gy_i42": True, \
                                      "ma4_gy_o41": False, "ma4_gy_o42": True}
            model = MA_4in_gy_extra(n_input_periods=n_input_periods,
                             n_output_periods=n_output_periods,
                             node_features=node_features,
                             n_nodes=n_nodes, 
                             hidden_warmup=hidden_warmup,
                             edge_index_list=edge_index_list,
                             model_path=model_path,
                             no_backprop_dp_models = dp_no_backprop_dict[model_name.lower()],
                             )
            
            
    elif model_name.lower()[:7] in ["ma5_s_i", "ma5_s_o"] or model_name.lower()[:8] in ["ma5_s2_i", "ma5_s2_o"] or \
             model_name.lower()[:10] in ["ma5_s_dp_i", "ma5_s_dp_o"]:
        # Check which the model is being trained for which company
        other_company_str = other_company_str_dict[company.lower()]
        no_backprop_models_dict = {"1": False, "2": True}
        if model_name.lower()[:8] in ["ma5_s_i1", "ma5_s_o1"]:
            no_backprop_s_models = model_name[8]
        elif model_name.lower()[:8] in ["ma5_s_i4", "ma5_s_o4", "ma5_s_i5", "ma5_s_o5"]:
            no_backprop_s_models = model_name[10]
        elif model_name.lower()[:9] in ["ma5_s2_i1", "ma5_s2_o1", "ma5_s2_i2", "ma5_s2_o2", ]:
            no_backprop_s_models = model_name[9]
        configuration_addresses = []   
        if model_name.lower()[:8] in ["ma5_s_i2", "ma5_s_o2", "ma5_s_i3", "ma5_s_o3", "ma5_s_i4", "ma5_s_o4", "ma5_s_i5", "ma5_s_o5"]:
            no_backprop_s_baseline_models = model_name[8]
            no_backprop_baseline_models = model_name[9]
            # Load MA3_S_I4X as a warmed up model
            configuration_addresses.append(("../data/m_"+str(start_month)+"_"+str(end_month)+"_y_"+str(year)+"_optimals/config_MA3_S_I4" + no_backprop_s_baseline_models + "_optimal" + company_str + ".yaml"))
            # Load MA3_S_O4X as a warmed up model
            configuration_addresses.append(("../data/m_"+str(start_month)+"_"+str(end_month)+"_y_"+str(year)+"_optimals/config_MA3_S_O4" + no_backprop_s_baseline_models + "_optimal" + company_str + ".yaml"))
            
        elif model_name.lower()[:8] in ["ma5_s_i6", "ma5_s_o6", "ma5_s_i7", "ma5_s_o7", "ma5_s_i8", "ma5_s_o8"]:
            no_backprop_s_baseline_models = model_name[8]
            no_backprop_baseline_models = model_name[9]
            if model_name.lower()[:8] in ["ma5_s_i6", "ma5_s_o6", "ma5_s_i8", "ma5_s_o8"]:
                no_backprop_dp_i_models = model_name[10]
                no_backprop_dp_o_models = model_name[11]
            elif model_name.lower()[:8] in ["ma5_s_i7", "ma5_s_o7"]:
                no_backprop_s_models = model_name[10]
                no_backprop_dp_i_models = model_name[11]
                no_backprop_dp_o_models = model_name[12]
            # Load MA3_S_I4X as a warmed up model
            configuration_addresses.append(("../data/m_"+str(start_month)+"_"+str(end_month)+"_y_"+str(year)+"_optimals/config_MA3_S_I4" + no_backprop_s_baseline_models + "_optimal" + company_str + ".yaml"))
            # Load MA3_S_O4X as a warmed up model
            configuration_addresses.append(("../data/m_"+str(start_month)+"_"+str(end_month)+"_y_"+str(year)+"_optimals/config_MA3_S_O4" + no_backprop_s_baseline_models + "_optimal" + company_str + ".yaml"))
            # Model 5
            configuration_addresses.append("../data/m_"+str(start_month)+"_"+str(end_month)+"_y_"+str(year)+"_optimals/config_dp_multi5_g" + str(graph_version) + "_optimal" + other_company_str + dp_loaded_str + ".yaml")
            # Model 6
            configuration_addresses.append("../data/m_"+str(start_month)+"_"+str(end_month)+"_y_"+str(year)+"_optimals/config_dp_multi6_g" + str(graph_version) + "_optimal" + other_company_str + dp_loaded_str + ".yaml")
        # Convert YAML file addresses to model address
        for configuration_address in configuration_addresses:
            model_path.append(get_model_path(configuration_address))
        print(f"model_path of loaded models:")
        print(*model_path, sep='\n')
        
        if model_name.lower()[:8] in ["ma5_s_i1", "ma5_s_o1"]:
            model = MA_5in_speed(n_input_periods=n_input_periods,
                                 n_output_periods=n_output_periods,
                                 node_features=node_features,
                                 n_nodes=n_nodes, 
                                 hidden_warmup=hidden_warmup,
                                 edge_index_list=edge_index_list,
                                 model_path=model_path,
                                 no_backprop_s_models = no_backprop_models_dict[no_backprop_s_models],
                                )
        
        elif model_name.lower()[:9] in ["ma5_s2_i1", "ma5_s2_o1", "ma5_s2_i2", "ma5_s2_o2", ]:
            model_func_dict = {'ma5_s2_i1': MA_5in_speed_pool_sum_baseline,
                               'ma5_s2_o1': MA_5in_speed_pool_sum_baseline,
                               'ma5_s2_i2': MA_5in_pool_sum_baseline,
                               'ma5_s2_o2': MA_5in_pool_sum_baseline,
                              }
            model = model_func_dict[model_name.lower()[:9]](n_input_periods=n_input_periods,
                                                     n_output_periods=n_output_periods,
                                                     node_features=node_features,
                                                     n_nodes=n_nodes, 
                                                     hidden_warmup=hidden_warmup,
                                                     edge_index_list=edge_index_list,
                                                     model_path=model_path,
                                                     no_backprop_s_models = no_backprop_models_dict[no_backprop_s_models],
                                                    )
            
        elif model_name.lower()[:8] in ["ma5_s_i2", "ma5_s_o2", "ma5_s_i3", "ma5_s_o3"]:
            model_func_dict = {'ma5_s_i2': MA_5in_speed2,
                               'ma5_s_o2': MA_5in_speed2,
                               'ma5_s_i3': MA_5in_speed3,
                               'ma5_s_o3': MA_5in_speed3,
                              }
            model = model_func_dict[model_name.lower()[:8]](n_input_periods=n_input_periods,
                                     n_output_periods=n_output_periods,
                                     node_features=node_features,
                                     n_nodes=n_nodes, 
                                     hidden_warmup=hidden_warmup,
                                     edge_index_list=edge_index_list,
                                     model_path=model_path,
                                     no_backprop_s_baseline_models = no_backprop_models_dict[no_backprop_s_baseline_models],
                                     no_backprop_baseline_models = no_backprop_models_dict[no_backprop_baseline_models],
                                    )
            
        elif model_name.lower()[:8] in ["ma5_s_i4", "ma5_s_o4", "ma5_s_i5", "ma5_s_o5"]:
            model_func_dict = {'ma5_s_i4': MA_5in_speed4,
                               'ma5_s_o4': MA_5in_speed4,
                               'ma5_s_i5': MA_5in_speed5,
                               'ma5_s_o5': MA_5in_speed5,
                              }
            model = model_func_dict[model_name.lower()[:8]](n_input_periods=n_input_periods,
                                     n_output_periods=n_output_periods,
                                     node_features=node_features,
                                     n_nodes=n_nodes, 
                                     hidden_warmup=hidden_warmup,
                                     edge_index_list=edge_index_list,
                                     model_path=model_path,
                                     no_backprop_s_baseline_models = no_backprop_models_dict[no_backprop_s_baseline_models],
                                     no_backprop_baseline_models = no_backprop_models_dict[no_backprop_baseline_models],
                                     no_backprop_s_models = no_backprop_models_dict[no_backprop_s_models],
                                 )
            
        elif model_name.lower()[:8] in ["ma5_s_i6", "ma5_s_o6", "ma5_s_i8", "ma5_s_o8"]:
            model_func_dict = {'ma5_s_i6': MA_5in_speed6,
                               'ma5_s_o6': MA_5in_speed6,
                               'ma5_s_i8': MA_5in_speed8,
                               'ma5_s_o8': MA_5in_speed8,
                              }
            model = model_func_dict[model_name.lower()[:8]](n_input_periods=n_input_periods,
                                     n_output_periods=n_output_periods,
                                     node_features=node_features,
                                     n_nodes=n_nodes, 
                                     hidden_warmup=hidden_warmup,
                                     edge_index_list=edge_index_list,
                                     model_path=model_path,
                                     no_backprop_s_baseline_models = no_backprop_models_dict[no_backprop_s_baseline_models],
                                     no_backprop_baseline_models = no_backprop_models_dict[no_backprop_baseline_models],
                                     no_backprop_dp_i_models = no_backprop_models_dict[no_backprop_dp_i_models],
                                     no_backprop_dp_o_models = no_backprop_models_dict[no_backprop_dp_o_models],
                                 )
            
        elif model_name.lower()[:8] in ["ma5_s_i7", "ma5_s_o7"]:
            model_func_dict = {'ma5_s_i7': MA_5in_speed7,
                               'ma5_s_o7': MA_5in_speed7,
                              }
            model = model_func_dict[model_name.lower()[:8]](n_input_periods=n_input_periods,
                                     n_output_periods=n_output_periods,
                                     node_features=node_features,
                                     n_nodes=n_nodes, 
                                     hidden_warmup=hidden_warmup,
                                     edge_index_list=edge_index_list,
                                     model_path=model_path,
                                     no_backprop_s_baseline_models = no_backprop_models_dict[no_backprop_s_baseline_models],
                                     no_backprop_baseline_models = no_backprop_models_dict[no_backprop_baseline_models],
                                     no_backprop_s_models = no_backprop_models_dict[no_backprop_s_models],
                                     no_backprop_dp_i_models = no_backprop_models_dict[no_backprop_dp_i_models],
                                     no_backprop_dp_o_models = no_backprop_models_dict[no_backprop_dp_o_models],
                                 )
            
        elif model_name.lower()[:11] in ["ma5_s_dp_i1", "ma5_s_dp_o1"]:
            s_no_backprop_dict = {"ma5_s_dp_i11": False, "ma5_s_dp_i12": True, \
                                  "ma5_s_dp_o11": False, "ma5_s_dp_o12": True, }
            model = MA_3in_speed_extra_dp(n_input_periods=n_input_periods,
                                          n_output_periods=n_output_periods,
                                          node_features=node_features,
                                          n_nodes=n_nodes, 
                                          hidden_warmup=hidden_warmup,
                                          edge_index_list=edge_index_list,
                                          model_path=model_path,
                                          no_backprop_s_models = s_no_backprop_dict[model_name.lower()],
                                         )
        
        
# https://pytorch-geometric-temporal.readthedocs.io/en/latest/modules/root.html#torch_geometric_temporal.nn.attention.mstgcn.MSTGCN
# https://pytorch-geometric-temporal.readthedocs.io/en/latest/_modules/torch_geometric_temporal/nn/attention/mstgcn.html#MSTGCN
    elif model_name.lower() == "mstgcn":
        model = MSTGCN_temp(nb_block = 2, 
                           in_channels = 1, 
                           K = 3, 
                           nb_chev_filter = 64, 
                           nb_time_filter = 64, 
                           time_strides = 1, 
                           num_for_predict = n_output_periods, 
                           len_input = n_input_periods,
                           edge_index=edge_index_list[0],
                          )
        
    elif model_name.lower()[:3] == "avg":
        model = AverageModel()
        
    elif model_name.lower() == 'a3tgcn':
        assert n_input_periods==n_output_periods, "Number of input and output times should be same in a3tgcn"
        model = A3TGCN_RecurrentGCN(node_features=node_features[0], periods=n_input_periods, batch_size=batch_size, out_channels=out_channels, edge_index=edge_index_list[0])
        
    elif model_name.lower() == 'astgcn_customized_no_relu':
        model = Customized_ASTGCN(len_input = n_input_periods,
                                    num_for_predict = n_output_periods,
                                    in_channels = node_features[0],
                                    num_of_vertices = n_nodes[0],
                                    nb_block = nb_block,
                                    K = K,
                                    nb_chev_filter = nb_chev_filter,
                                    nb_time_filter = nb_time_filter,
                                    time_strides = time_strides,
                                    edge_index=edge_index_list[0]) 
        
    elif model_name.lower() == 'astgcn':
        model = ASTGCN_AttentionGCN(len_input = n_input_periods,
                                    num_for_predict = n_output_periods,
                                    in_channels = node_features[0],
                                    num_of_vertices = n_nodes[0],
                                    nb_block = nb_block,
                                    K = K,
                                    nb_chev_filter = nb_chev_filter,
                                    nb_time_filter = nb_time_filter,
                    #                         time_strides = num_of_hours,
                                    time_strides = time_strides,
                                    hidden = hidden,
                                    edge_index=edge_index_list[0])  
        
    elif model_name.lower() == 'fnntest':
        model = FNNTest(num_for_predict = n_output_periods,)
        
    elif model_name.lower() == 'fnntestlarge':
        model = FNNTestLarge(num_for_predict = n_output_periods,)
#                         len_input = n_input_periods,
#                         num_of_vertices = n_nodes[0],) 

    elif model_name.lower() == 'astgcn_customized':
        model = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                num_for_predict = n_output_periods,
                                                in_channels = node_features[0],
                                                num_of_vertices = n_nodes[0],
                                                nb_block = nb_block,
                                                K = K,
                                                nb_chev_filter = nb_chev_filter,
                                                nb_time_filter = nb_time_filter,
                                #                         time_strides = num_of_hours,
                                                time_strides = time_strides,
                                                hidden = hidden,
                                                edge_index=edge_index_list[0])  
# https://pytorch-geometric-temporal.readthedocs.io/en/latest/_modules/torch_geometric_temporal/nn/attention/gman.html#GMAN
# https://fanxlxmu.github.io/publication/aaai2020/
# https://github.com/zhengchuanpan/GMAN/blob/master/METR/model.py#L227
# https://github.com/zhengchuanpan/GMAN/blob/master/METR/train.py
# https://github.com/viz27/GMAN

#     elif model_name.lower() == 'gman':
#         model = GMAN(L = 3, 
#                      K = 8, 
#                      d = 8, 
#                      num_his = in_channels, 
#                      bn_decay = , 
#                      steps_per_day = , 
#                      use_bias = , 
#                      mask = )
#         num_nodes = 10
#         in_channels = 3
#         out_channels = 1
#         hidden_dim = 32
#         num_heads = 4
#         num_layers = 4

#         model = GMAN(num_nodes=num_nodes, 
#                      in_channels=in_channels, 
#                      out_channels=out_channels, 
#                      hidden_dim=hidden_dim, 
#                      num_heads=num_heads, 
#                      num_layers=num_layers)

    elif model_name.lower() == 'ensemble':
        # Create models and load state_dicts    
        models = []
        for model_idx in range(len(node_features)):
#             model_temp = ASTGCN_AttentionGCN(len_input = n_input_periods, 
#                                                 num_for_predict = n_output_periods,
#                                                 in_channels = node_features[model_idx], 
#                                                 num_of_vertices = n_nodes[model_idx],
#                                                 nb_block = 2, 
#                                                 K = 3, 
#                                                 nb_chev_filter = 64, 
#                                                 nb_time_filter = 64,
#                                                 # time_strides = num_of_hours,
#                                                 time_strides = 1,
#                                                 hidden = hidden_warmup[model_idx],
#                                                 edge_index=edge_index_list[0],
#                                             )
            model_temp = ASTGCN_AttentionGCN_customized(len_input = n_input_periods,
                                                    num_for_predict = n_output_periods,
                                                    in_channels = node_features[model_idx],
                                                    num_of_vertices = n_nodes[model_idx],
                                                    nb_block = 2,
                                                    K = 3,
                                                    nb_chev_filter = 64,
                                                    nb_time_filter = 64,
                                    #                         time_strides = num_of_hours,
                                                    time_strides = 1,
                                                    hidden = hidden_warmup[model_idx],
                                                    edge_index=edge_index_list[model_idx],
                                                       )  

            # Load state dicts
            model_temp.load_state_dict(torch.load("../models/" + model_path[model_idx] + ".pth").state_dict())
            model_temp.train()
#             model_temp.load_state_dict(torch.load("../models/" + model_path[model_idx] + ".pth"))
            models.append(model_temp)
            
        if ensemble==1:
            model = EnsembleNet(*models, periods=n_output_periods, n_nodes=n_nodes)
        elif ensemble==2:
            model = EnsembleNetSpeed(*models, hidden=hidden, periods=n_output_periods, n_nodes=n_nodes)
        elif ensemble==3:
            model = EnsembleNetSpeedResidual(*models, hidden=hidden, periods=n_output_periods, n_nodes=n_nodes)
        elif ensemble==4:
            model = EnsembleNetSpeedGraphEnd(*models, hidden=hidden, periods=n_output_periods, n_nodes=n_nodes)
        elif ensemble==5:
            model = EnsembleNetSpeedResidualGraphEnd(*models, hidden=hidden, periods=n_output_periods, n_nodes=n_nodes)
        elif ensemble==6:
            model = EnsembleNetResidual(*models, hidden=hidden, periods=n_output_periods, n_nodes=n_nodes)
    
    
    elif model_name.lower() in model_func_dict_1_4.keys():
        model = model_func_dict_1_4[model_name.lower()](n_input_periods=n_input_periods, 
                                                        n_output_periods=n_output_periods, 
                                                        node_features=node_features, 
                                                        n_nodes=n_nodes, 
                                                        edge_index_list=edge_index_list, 
                                                        device = device,
                                                       )
                                                    

    elif model_name.lower() == 'multiprediction_9':
        configuration_address = "../data/m_"+str(start_month)+"_"+str(end_month)+"_y_"+str(year)+"_optimals/config_dp_multi6_g" + str(graph_version) + "_optimal" + company_str + ".yaml"
        model_path.append(get_model_path(configuration_address))
        
        model = MultiPredictionNet9(n_input_periods=n_input_periods,
                                     n_output_periods=n_output_periods,
                                     node_features=node_features,
                                     n_nodes=n_nodes, 
                                     hidden_warmup=hidden_warmup,
                                     edge_index_list=edge_index_list,
                                     model_path=model_path,
                                   )
        

    elif model_name.lower() == 'multiprediction_10':
        configuration_address = "../data/m_"+str(start_month)+"_"+str(end_month)+"_y_"+str(year)+"_optimals/config_dp_multi5_g" + str(graph_version) + "_optimal" + company_str + ".yaml"
        model_path.append(get_model_path(configuration_address))
        
        model = MultiPredictionNet10(n_input_periods=n_input_periods,
                                     n_output_periods=n_output_periods,
                                     node_features=node_features,
                                     n_nodes=n_nodes, 
                                     hidden_warmup=hidden_warmup,
                                     edge_index_list=edge_index_list,
                                     model_path=model_path,
                                   )

    elif model_name.lower() == 'multiprediction_13':
        configuration_address = "../data/m_"+str(start_month)+"_"+str(end_month)+"_y_"+str(year)+"_optimals/config_dp_multi6_g" + str(graph_version) + "_optimal" + company_str + ".yaml"
        model_path.append(get_model_path(configuration_address))
        model = MultiPredictionNet13(n_input_periods=n_input_periods,
                                     n_output_periods=n_output_periods,
                                     node_features=node_features,
                                     n_nodes=n_nodes, 
                                     hidden_warmup=hidden_warmup,
                                     edge_index_list=edge_index_list,
                                     model_path=model_path,
                                   )
        

    elif model_name.lower() == 'multiprediction_14':
        configuration_address = "../data/m_"+str(start_month)+"_"+str(end_month)+"_y_"+str(year)+"_optimals/config_dp_multi5_g" + str(graph_version) + "_optimal" + company_str + ".yaml"
        model_path.append(get_model_path(configuration_address))
        model = MultiPredictionNet14(n_input_periods=n_input_periods,
                                     n_output_periods=n_output_periods,
                                     node_features=node_features,
                                     n_nodes=n_nodes, 
                                     hidden_warmup=hidden_warmup,
                                     edge_index_list=edge_index_list,
                                     model_path=model_path,
                                   )
        
    elif model_name.lower() == 'multiprediction_15':
        configuration_address = "../data/m_"+str(start_month)+"_"+str(end_month)+"_y_"+str(year)+"_optimals/config_multi6_g" + str(graph_version) + "_optimal" + company_str + ".yaml"
        model_path.append(get_model_path(configuration_address))
        model = MultiPredictionNet9(n_input_periods=n_input_periods,
                                     n_output_periods=n_output_periods,
                                     node_features=node_features,
                                     n_nodes=n_nodes, 
                                     hidden_warmup=hidden_warmup,
                                     edge_index_list=edge_index_list,
                                     model_path=model_path,
                                   )
        

    elif model_name.lower() == 'multiprediction_16':
        configuration_address = "../data/m_"+str(start_month)+"_"+str(end_month)+"_y_"+str(year)+"_optimals/config_multi5_g" + str(graph_version) + "_optimal" + company_str + ".yaml"
        model_path.append(get_model_path(configuration_address))
        model = MultiPredictionNet10(n_input_periods=n_input_periods,
                                     n_output_periods=n_output_periods,
                                     node_features=node_features,
                                     n_nodes=n_nodes, 
                                     hidden_warmup=hidden_warmup,
                                     edge_index_list=edge_index_list,
                                     model_path=model_path,
                                   )

       
   
        
    elif model_name.lower() == 'multiprediction_26':
        configuration_address = "../data/m_"+str(start_month)+"_"+str(end_month)+"_y_"+str(year)+"_optimals/config_multi6_g" + str(graph_version) + "_optimal" + company_str + ".yaml"
        model_path.append(get_model_path(configuration_address))
        model = MultiPredictionNet26(n_input_periods=n_input_periods,
                                     n_output_periods=n_output_periods,
                                     node_features=node_features,
                                     n_nodes=n_nodes, 
                                     hidden_warmup=hidden_warmup,
                                     edge_index_list=edge_index_list,
                                     model_path=model_path,
                                   )
        
    elif model_name.lower() == "multiprediction_26_dp":
        print("\n\n\n", model_name.lower() + dp_loaded_str)
        configuration_address = "../data/m_"+str(start_month)+"_"+str(end_month)+"_y_"+str(year)+"_optimals/config_dp_multi6_g" + str(graph_version) + "_optimal" + company_str + dp_loaded_str + ".yaml"
        print(f"configuration_address: {configuration_address}")
        model_path.append(get_model_path(configuration_address))
        print(f"model_path of loaded models:")
        print(*model_path, sep='\n')
        model = MultiPredictionNet26(n_input_periods=n_input_periods,
                                     n_output_periods=n_output_periods,
                                     node_features=node_features,
                                     n_nodes=n_nodes, 
                                     hidden_warmup=hidden_warmup,
                                     edge_index_list=edge_index_list,
                                     model_path=model_path,
                                   )
        
    elif model_name.lower() == 'multiprediction_26_o':
        configuration_address = "../data/m_"+str(start_month)+"_"+str(end_month)+"_y_"+str(year)+"_optimals/config_multi5_g" + str(graph_version) + "_optimal" + company_str + ".yaml"
        model_path.append(get_model_path(configuration_address))
        model = MultiPredictionNet26_o(n_input_periods=n_input_periods,
                                     n_output_periods=n_output_periods,
                                     node_features=node_features,
                                     n_nodes=n_nodes, 
                                     hidden_warmup=hidden_warmup,
                                     edge_index_list=edge_index_list,
                                     model_path=model_path,
                                   )
        
    elif model_name.lower() == 'multiprediction_26_o_dp':
        configuration_address = "../data/m_"+str(start_month)+"_"+str(end_month)+"_y_"+str(year)+"_optimals/config_dp_multi5_g" + str(graph_version) + "_optimal" + company_str + dp_loaded_str + ".yaml"
        print(f"configuration_address: {configuration_address}")
        model_path.append(get_model_path(configuration_address))
        print(f"model_path of loaded models:")
        print(*model_path, sep='\n')
        model = MultiPredictionNet26_o(n_input_periods=n_input_periods,
                                     n_output_periods=n_output_periods,
                                     node_features=node_features,
                                     n_nodes=n_nodes, 
                                     hidden_warmup=hidden_warmup,
                                     edge_index_list=edge_index_list,
                                     model_path=model_path,
                                   )
        
    elif model_name.lower() == 'multiprediction_27':
        configuration_address = "../data/m_"+str(start_month)+"_"+str(end_month)+"_y_"+str(year)+"_optimals/config_multi6_g" + str(graph_version) + "_optimal" + company_str + ".yaml"
        model_path.append(get_model_path(configuration_address))
        model = MultiPredictionNet27(n_input_periods=n_input_periods,
                                     n_output_periods=n_output_periods,
                                     node_features=node_features,
                                     n_nodes=n_nodes, 
                                     hidden_warmup=hidden_warmup,
                                     edge_index_list=edge_index_list,
                                     model_path=model_path,
                                   )
        
    elif model_name.lower() == 'multiprediction_27_dp':
        configuration_address = "../data/m_"+str(start_month)+"_"+str(end_month)+"_y_"+str(year)+"_optimals/config_dp_multi6_g" + str(graph_version) + "_optimal" + company_str + dp_loaded_str + ".yaml"
        model_path.append(get_model_path(configuration_address))
        print(f"model_path of loaded models:")
        print(*model_path, sep='\n')
        model = MultiPredictionNet27(n_input_periods=n_input_periods,
                                     n_output_periods=n_output_periods,
                                     node_features=node_features,
                                     n_nodes=n_nodes, 
                                     hidden_warmup=hidden_warmup,
                                     edge_index_list=edge_index_list,
                                     model_path=model_path,
                                   )
        
    elif model_name.lower() in ['multiprediction_37_dp', 'multiprediction_38_dp', 'multiprediction_39_dp', 'multiprediction_40_dp', 'multiprediction_31_dp', 'multiprediction_32_dp', 'multiprediction_33_dp', 'multiprediction_34_dp', 'multiprediction_35_dp', 'multiprediction_36_dp', "multiprediction_47", "multiprediction_48", "multiprediction_51", "multiprediction_52", "multiprediction_53", "multiprediction_54"]:
        configuration_addresses = []
        other_company_str = other_company_str_dict[company.lower()]
#         if company.lower() == 'uber':
#             other_company_str = '_lyft'
#         else:
#             other_company_str = '_uber'
        configuration_addresses.append("../data/m_"+str(start_month)+"_"+str(end_month)+"_y_"+str(year)+"_optimals/config_dp_multi5_g" + str(graph_version) + "_optimal" + other_company_str + dp_loaded_str + ".yaml")
        configuration_addresses.append("../data/m_"+str(start_month)+"_"+str(end_month)+"_y_"+str(year)+"_optimals/config_dp_multi6_g" + str(graph_version) + "_optimal" + other_company_str + dp_loaded_str + ".yaml")
            
        if model_name.lower() in ["multiprediction_47", "multiprediction_47_eps20", "multiprediction_47_wo_eps", "multiprediction_51", "multiprediction_51_eps20", "multiprediction_51_wo_eps", "multiprediction_53"]:
            configuration_addresses.append("../data/m_"+str(start_month)+"_"+str(end_month)+"_y_"+str(year)+"_optimals/config_multi17_g" + str(graph_version) + "_optimal" + company_str + ".yaml")
        elif model_name.lower() in ["multiprediction_48", "multiprediction_48_eps20", "multiprediction_48_wo_eps", "multiprediction_52", "multiprediction_52_eps20", "multiprediction_52_wo_eps", "multiprediction_54"]:
            configuration_addresses.append("../data/m_"+str(start_month)+"_"+str(end_month)+"_y_"+str(year)+"_optimals/config_multi18_g" + str(graph_version) + "_optimal" + company_str + ".yaml")
            
        for configuration_address in configuration_addresses:
            model_path.append(get_model_path(configuration_address))
        print(f"model_path of loaded models:")
        print(*model_path, sep='\n')
        
        model_func_dict_31_40 = {'multiprediction_31_dp': MultiPredictionNet31,
                                 'multiprediction_32_dp': MultiPredictionNet32,
                                 'multiprediction_33_dp': MultiPredictionNet33,
                                 'multiprediction_34_dp': MultiPredictionNet34,
                                 'multiprediction_35_dp': MultiPredictionNet35,
                                 'multiprediction_36_dp': MultiPredictionNet36,
                                 'multiprediction_37_dp': MultiPredictionNet37,
                                 'multiprediction_38_dp': MultiPredictionNet38,
                                 'multiprediction_39_dp': MultiPredictionNet39,
                                 'multiprediction_40_dp': MultiPredictionNet40,
                                } 
        
        if model_name.lower() in model_func_dict_31_40.keys():
            model = model_func_dict_31_40[model_name.lower()](n_input_periods=n_input_periods,
                                                                 n_output_periods=n_output_periods,
                                                                 node_features=node_features,
                                                                 n_nodes=n_nodes, 
                                                                 hidden_warmup=hidden_warmup,
                                                                 edge_index_list=edge_index_list,
                                                                 model_path=model_path,
                                                               )

        elif model_name.lower() in ["multiprediction_47", "multiprediction_48"]:
            model = MultiPredictionNet47(n_input_periods=n_input_periods,
                                         n_output_periods=n_output_periods,
                                         node_features=node_features,
                                         n_nodes=n_nodes, 
                                         hidden_warmup=hidden_warmup,
                                         edge_index_list=edge_index_list,
                                         model_path=model_path,
                                         no_backprop_nondp_models = False,
                                         no_backprop_dp_models = True
                                       )
        elif model_name.lower() in ['multiprediction_51', 'multiprediction_52']:
            model = MultiPredictionNet47(n_input_periods=n_input_periods,
                                         n_output_periods=n_output_periods,
                                         node_features=node_features,
                                         n_nodes=n_nodes, 
                                         hidden_warmup=hidden_warmup,
                                         edge_index_list=edge_index_list,
                                         model_path=model_path,
                                         no_backprop_nondp_models = True,
                                         no_backprop_dp_models = True
                                       )
        elif model_name.lower() in ['multiprediction_53', 'multiprediction_54']:
            model = MultiPredictionNet47(n_input_periods=n_input_periods,
                                         n_output_periods=n_output_periods,
                                         node_features=node_features,
                                         n_nodes=n_nodes, 
                                         hidden_warmup=hidden_warmup,
                                         edge_index_list=edge_index_list,
                                         model_path=model_path,
                                         no_backprop_nondp_models = False,
                                         no_backprop_dp_models = False
                                       )
    elif model_name.lower() in ["multiprediction_47_finetune", "multiprediction_48_finetune", "multiprediction_51_finetune", "multiprediction_52_finetune", "multiprediction_53_finetune", "multiprediction_54_finetune"]:
        configuration_addresses = []
        other_company_str = other_company_str_dict[company.lower()]
#         if company.lower() == 'uber':
# #             other_company_str = '_lyft'
#             configuration_addresses.append("../data/m_"+str(start_month)+"_"+str(end_month)+"_y_"+str(year)+"_optimals/config_dp_multi5_g" + str(graph_version) + "_optimal" + other_company_str + dp_loaded_str + ".yaml")
#             configuration_addresses.append("../data/m_"+str(start_month)+"_"+str(end_month)+"_y_"+str(year)+"_optimals/config_dp_multi6_g" + str(graph_version) + "_optimal" + other_company_str + dp_loaded_str + ".yaml")
#         else:
# #             other_company_str = '_uber'
        configuration_addresses.append("../data/m_"+str(start_month)+"_"+str(end_month)+"_y_"+str(year)+"_optimals/config_dp_multi5_g" + str(graph_version) + "_optimal" + other_company_str + dp_loaded_str + ".yaml")
        configuration_addresses.append("../data/m_"+str(start_month)+"_"+str(end_month)+"_y_"+str(year)+"_optimals/config_dp_multi6_g" + str(graph_version) + "_optimal" + other_company_str + dp_loaded_str + ".yaml")
        if model_name.lower() in ["multiprediction_47_finetune", "multiprediction_51_finetune", "multiprediction_53_finetune"]:
            configuration_addresses.append("../data/m_"+str(start_month)+"_"+str(end_month)+"_y_"+str(year)+"_optimals/config_multi17_g" + str(graph_version) + "_optimal" + company_str + ".yaml")
        if model_name.lower() in ["multiprediction_48_finetune", "multiprediction_52_finetune", "multiprediction_54_finetune"]:
            configuration_addresses.append("../data/m_"+str(start_month)+"_"+str(end_month)+"_y_"+str(year)+"_optimals/config_multi18_g" + str(graph_version) + "_optimal" + company_str + ".yaml")
        for configuration_address in configuration_addresses:
            model_path.append(get_model_path(configuration_address))
    
        print(f"model_path of loaded models:")
        print(*model_path, sep='\n')
        model_number = model_name.lower()[16:18]
        if model_name.lower() in ['multiprediction_47_finetune', 'multiprediction_48_finetune']:
            configuration_address = "../data/m_"+str(start_month)+"_"+str(end_month)+"_y_"+str(year)+"_optimals/config_multi" + model_number + "_g" + str(graph_version) + "_optimal" + company_str + dp_loaded_str + ".yaml"
            model_path.append(get_model_path(configuration_address))
            model = MultiPredictionNet47_finetune(n_input_periods=n_input_periods,
                                         n_output_periods=n_output_periods,
                                         node_features=node_features,
                                         n_nodes=n_nodes, 
                                         hidden_warmup=hidden_warmup,
                                         edge_index_list=edge_index_list,
                                         model_path=model_path,
                                         no_backprop_nondp_models = False,
                                         no_backprop_dp_models = True
                                       )
            
        elif model_name.lower() in ['multiprediction_51_finetune', 'multiprediction_52_finetune']:
            configuration_address = "../data/m_"+str(start_month)+"_"+str(end_month)+"_y_"+str(year)+"_optimals/config_multi" + model_number  + "_g" + str(graph_version) + "_optimal" + company_str + dp_loaded_str + ".yaml"
            model_path.append(get_model_path(configuration_address))
            model = MultiPredictionNet47_finetune(n_input_periods=n_input_periods,
                                         n_output_periods=n_output_periods,
                                         node_features=node_features,
                                         n_nodes=n_nodes, 
                                         hidden_warmup=hidden_warmup,
                                         edge_index_list=edge_index_list,
                                         model_path=model_path,
                                         no_backprop_nondp_models = True,
                                         no_backprop_dp_models = True
                                       )
            
        elif model_name.lower() in ['multiprediction_53_finetune', 'multiprediction_54_finetune']:
            configuration_address = "../data/m_"+str(start_month)+"_"+str(end_month)+"_y_"+str(year)+"_optimals/config_multi" + model_number + "_g" + str(graph_version) + "_optimal" + company_str + dp_loaded_str + ".yaml"
            model_path.append(get_model_path(configuration_address))
            model = MultiPredictionNet47_finetune(n_input_periods=n_input_periods,
                                         n_output_periods=n_output_periods,
                                         node_features=node_features,
                                         n_nodes=n_nodes, 
                                         hidden_warmup=hidden_warmup,
                                         edge_index_list=edge_index_list,
                                         model_path=model_path,
                                         no_backprop_nondp_models = False,
                                         no_backprop_dp_models = False
                                       )
    elif model_name.lower() == 'multiprediction_55':
        print("\n\n\n", model_name.lower() + dp_loaded_str)
        configuration_address = "../data/m_"+str(start_month)+"_"+str(end_month)+"_y_"+str(year)+"_optimals/config_dp_multi6_g" + str(graph_version) + "_optimal" + company_str + dp_loaded_str + ".yaml"
#         print(f"configuration_address: {configuration_address}")
        model_path.append(get_model_path(configuration_address))
        configuration_address2 = "../data/m_"+str(start_month)+"_"+str(end_month)+"_y_"+str(year)+"_optimals/config_multi7_g" + str(graph_version) + "_optimal" + company_str + ".yaml"
#         print(f"configuration_address2: {configuration_address2}")
        model_path.append(get_model_path(configuration_address2))
        print(f"model_path of loaded models:")
        print(*model_path, sep='\n')
        model = MultiPredictionNet55(n_input_periods=n_input_periods,
                                     n_output_periods=n_output_periods,
                                     node_features=node_features,
                                     n_nodes=n_nodes, 
                                     hidden_warmup=hidden_warmup,
                                     edge_index_list=edge_index_list,
                                     model_path=model_path,
                                     no_backprop_nondp_models = False,
                                     no_backprop_dp_models = False
                                   )
    elif model_name.lower() == 'multiprediction_56':
        configuration_address = "../data/m_"+str(start_month)+"_"+str(end_month)+"_y_"+str(year)+"_optimals/config_dp_multi5_g" + str(graph_version) + "_optimal" + company_str + dp_loaded_str + ".yaml"
#         print(f"configuration_address: {configuration_address}")
        model_path.append(get_model_path(configuration_address))
        configuration_address2 = "../data/m_"+str(start_month)+"_"+str(end_month)+"_y_"+str(year)+"_optimals/config_multi8_g" + str(graph_version) + "_optimal" + company_str + ".yaml"
#         print(f"configuration_address2: {configuration_address2}")
        model_path.append(get_model_path(configuration_address2))
        print(f"model_path of loaded models:")
        print(*model_path, sep='\n')
        model = MultiPredictionNet56(n_input_periods=n_input_periods,
                                     n_output_periods=n_output_periods,
                                     node_features=node_features,
                                     n_nodes=n_nodes, 
                                     hidden_warmup=hidden_warmup,
                                     edge_index_list=edge_index_list,
                                     model_path=model_path,
                                     no_backprop_nondp_models = False,
                                     no_backprop_dp_models = False
                                       )
            
    elif model_name.lower() in ["dual_company_1_i_1", "dual_company_1_o_1", "dual_company_1_i_2", "dual_company_1_o_2", "dual_company_2_i_1", "dual_company_2_o_1", "dual_company_2_i_2", "dual_company_2_o_2", "dual_company_2_i_3", "dual_company_2_o_3", "dual_company_2_i_4", "dual_company_2_o_4",]:
        other_company_str = other_company_str_dict[company.lower()]
#         if company.lower() == 'uber':
#             other_company_str = '_lyft'
#         else:
#             other_company_str = '_uber'
        if model_name.lower() in ["dual_company_1_i_1", "dual_company_1_i_2", "dual_company_2_i_1", "dual_company_2_i_2", "dual_company_2_i_3", "dual_company_2_i_4", ]:
            configuration_address_inflow = "../data/m_"+str(start_month)+"_"+str(end_month)+"_y_"+str(year)+"_optimals/config_tlc_nyc_inflow_optimal" + other_company_str + dp_loaded_str + ".yaml"
            model_path.append(get_model_path(configuration_address_inflow))
        elif model_name.lower() in ["dual_company_1_o_1", "dual_company_1_o_2", "dual_company_2_o_1", "dual_company_2_o_2", "dual_company_2_o_3", "dual_company_2_o_4", ]:
            configuration_address_outflow = "../data/m_"+str(start_month)+"_"+str(end_month)+"_y_"+str(year)+"_optimals/config_tlc_nyc_outflow_optimal" + other_company_str + dp_loaded_str + ".yaml"
            model_path.append(get_model_path(configuration_address_outflow))
#         configuration_address_inflow = "../data/m_"+str(start_month)+"_"+str(end_month)+"_y_"+str(year)+"_optimals/config_tlc_nyc_inflow_optimal" + other_company_str + dp_loaded_str + ".yaml"
#         model_path.append(get_model_path(configuration_address_inflow))
#         configuration_address_outflow = "../data/m_"+str(start_month)+"_"+str(end_month)+"_y_"+str(year)+"_optimals/config_tlc_nyc_outflow_optimal" + other_company_str + dp_loaded_str + ".yaml"
#         model_path.append(get_model_path(configuration_address_outflow))
        print(f"model_path of loaded models:")
        print(*model_path, sep='\n')
        
        if model_name.lower() in ["dual_company_2_i_1", "dual_company_2_i_2", "dual_company_2_i_3", "dual_company_2_i_4", ]:
            inflow_pred = True
        else:
            inflow_pred = False
        if model_name.lower() in ["dual_company_1_i_1", "dual_company_1_o_1"]:
            model = DualCompany_1(n_input_periods=n_input_periods,
                                     n_output_periods=n_output_periods,
                                     node_features=node_features,
                                     n_nodes=n_nodes, 
                                     hidden_warmup=hidden_warmup,
                                     edge_index_list=edge_index_list,
                                     model_path=model_path,
                                     backprop_loaded_model = True,
                                   )
        elif model_name.lower() in ["dual_company_1_i_2", "dual_company_1_o_2",]:
            model = DualCompany_1(n_input_periods=n_input_periods,
                                     n_output_periods=n_output_periods,
                                     node_features=node_features,
                                     n_nodes=n_nodes, 
                                     hidden_warmup=hidden_warmup,
                                     edge_index_list=edge_index_list,
                                     model_path=model_path,
                                     backprop_loaded_model = False,
                                   )
        elif model_name.lower() in ["dual_company_2_i_1", "dual_company_2_o_1",]:
            model = DualCompany_2(n_input_periods=n_input_periods,
                                     n_output_periods=n_output_periods,
                                     node_features=node_features,
                                     n_nodes=n_nodes, 
                                     hidden_warmup=hidden_warmup,
                                     edge_index_list=edge_index_list,
                                     model_path=model_path,
                                     backprop_loaded_model_other_company = True,
                                     backprop_loaded_model_company = True,
                                     inflow_pred = inflow_pred,
                                   )
        elif model_name.lower() in ["dual_company_2_i_2", "dual_company_2_o_2",]:
            model = DualCompany_2(n_input_periods=n_input_periods,
                                     n_output_periods=n_output_periods,
                                     node_features=node_features,
                                     n_nodes=n_nodes, 
                                     hidden_warmup=hidden_warmup,
                                     edge_index_list=edge_index_list,
                                     model_path=model_path,
                                     backprop_loaded_model_other_company = True,
                                     backprop_loaded_model_company = False,
                                     inflow_pred = inflow_pred,
                                   )
        elif model_name.lower() in ["dual_company_2_i_3", "dual_company_2_o_3",]:
            model = DualCompany_2(n_input_periods=n_input_periods,
                                     n_output_periods=n_output_periods,
                                     node_features=node_features,
                                     n_nodes=n_nodes, 
                                     hidden_warmup=hidden_warmup,
                                     edge_index_list=edge_index_list,
                                     model_path=model_path,
                                     backprop_loaded_model_other_company = False,
                                     backprop_loaded_model_company = True,
                                     inflow_pred = inflow_pred,
                                   )
        elif model_name.lower() in ["dual_company_2_i_4", "dual_company_2_o_4",]:
            model = DualCompany_2(n_input_periods=n_input_periods,
                                     n_output_periods=n_output_periods,
                                     node_features=node_features,
                                     n_nodes=n_nodes, 
                                     hidden_warmup=hidden_warmup,
                                     edge_index_list=edge_index_list,
                                     model_path=model_path,
                                     backprop_loaded_model_other_company = False,
                                     backprop_loaded_model_company = False,
                                     inflow_pred = inflow_pred,
                                   )

            
            
    elif model_name.lower() in ["surge_1_1", "surge_1_2", "surge_1_3", "surge_1_4", "surge_1_dp_1", "surge_1_dp_2", "surge_1_dp_3", "surge_1_dp_4", "surge_2_1", "surge_2_2", "surge_2_3", "surge_2_4", "surge_2_dp_1", "surge_2_dp_2", "surge_2_dp_3", "surge_2_dp_4", ]:
        # Check which the model is being trained for which company
        other_company_str = other_company_str_dict[company.lower()]
#         if company.lower() == 'uber':
#             other_company_str = '_lyft'
#         else:
#             other_company_str = '_uber'
        # Create a list of addresses of models that are loaded and are part of the model
        configuration_addresses = []
        # Load models 17 and 18 as what the company would do alone
        if model_name.lower() in ["surge_1_1", "surge_1_2", "surge_1_3", "surge_1_4", "surge_2_1", "surge_2_2", "surge_2_3", "surge_2_4", ]:
            surge_dp = False
            # Model 17
            configuration_addresses.append("../data/m_"+str(start_month)+"_"+str(end_month)+"_y_"+str(year)+"_optimals/config_multi17_g" + str(graph_version) + "_optimal" + company_str + dp_loaded_str + ".yaml")
            # Model 18
            configuration_addresses.append("../data/m_"+str(start_month)+"_"+str(end_month)+"_y_"+str(year)+"_optimals/config_multi18_g" + str(graph_version) + "_optimal" + company_str + dp_loaded_str + ".yaml")
            
        # Load models 51 and 52 as what the company would do when cooperating with the other company
        elif model_name.lower() in ["surge_1_dp_1", "surge_1_dp_2", "surge_1_dp_3", "surge_1_dp_4", "surge_2_dp_1", "surge_2_dp_2", "surge_2_dp_3", "surge_2_dp_4", ]:
            surge_dp = True
            # Model 5
            configuration_addresses.append("../data/m_"+str(start_month)+"_"+str(end_month)+"_y_"+str(year)+"_optimals/config_dp_multi5_g" + str(graph_version) + "_optimal" + other_company_str + dp_loaded_str + ".yaml")
            # Model 6
            configuration_addresses.append("../data/m_"+str(start_month)+"_"+str(end_month)+"_y_"+str(year)+"_optimals/config_dp_multi6_g" + str(graph_version) + "_optimal" + other_company_str + dp_loaded_str + ".yaml")
            # Model 17
            configuration_addresses.append("../data/m_"+str(start_month)+"_"+str(end_month)+"_y_"+str(year)+"_optimals/config_multi17_g" + str(graph_version) + "_optimal" + company_str + ".yaml")
            # Model 18
            configuration_addresses.append("../data/m_"+str(start_month)+"_"+str(end_month)+"_y_"+str(year)+"_optimals/config_multi18_g" + str(graph_version) + "_optimal" + company_str + ".yaml")
            # Model 51
            configuration_addresses.append("../data/m_"+str(start_month)+"_"+str(end_month)+"_y_"+str(year)+"_optimals/config_multi51_g" + str(graph_version) + "_optimal" + company_str + dp_loaded_str + ".yaml")
            # Model 52
            configuration_addresses.append("../data/m_"+str(start_month)+"_"+str(end_month)+"_y_"+str(year)+"_optimals/config_multi52_g" + str(graph_version) + "_optimal" + company_str + dp_loaded_str + ".yaml")
        
        # Convert YAML file addresses to model address
        for configuration_address in configuration_addresses:
            model_path.append(get_model_path(configuration_address))
        print(f"model_path of loaded models:")
        print(*model_path, sep='\n')
        
        # Backprop on the models of the other company
        # Backprop on the models of the company itself
        if model_name.lower() in ["surge_1_1", "surge_1_dp_1"]:
            model = Surge_1(n_input_periods=n_input_periods,
                                     n_output_periods=n_output_periods,
                                     node_features=node_features,
                                     n_nodes=n_nodes, 
                                     hidden_warmup=hidden_warmup,
                                     edge_index_list=edge_index_list,
                                     model_path=model_path,
                                     backprop_loaded_model_inflow = True,
                                     backprop_loaded_model_outflow = True,
                                     surge_dp = surge_dp, 
                                   )
        # Backprop on the models of the other company
        # No backprop on the models of the company itself
        elif model_name.lower() in ["surge_1_2", "surge_1_dp_2",]:
            model = Surge_1(n_input_periods=n_input_periods,
                                     n_output_periods=n_output_periods,
                                     node_features=node_features,
                                     n_nodes=n_nodes, 
                                     hidden_warmup=hidden_warmup,
                                     edge_index_list=edge_index_list,
                                     model_path=model_path,
                                     backprop_loaded_model_inflow = True,
                                     backprop_loaded_model_outflow = False,
                                     surge_dp = surge_dp, 
                                   )
        # No backprop on the models of the other company
        # Backprop on the models of the company itself
        elif model_name.lower() in ["surge_1_3", "surge_1_dp_3",]:
            model = Surge_1(n_input_periods=n_input_periods,
                                     n_output_periods=n_output_periods,
                                     node_features=node_features,
                                     n_nodes=n_nodes, 
                                     hidden_warmup=hidden_warmup,
                                     edge_index_list=edge_index_list,
                                     model_path=model_path,
                                     backprop_loaded_model_inflow = False,
                                     backprop_loaded_model_outflow = True,
                                     surge_dp = surge_dp, 
                                   )
        # No backprop on the models of the other company
        # No backprop on the models of the company itself
        elif model_name.lower() in ["surge_1_4", "surge_1_dp_4",]:
            model = Surge_1(n_input_periods=n_input_periods,
                                     n_output_periods=n_output_periods,
                                     node_features=node_features,
                                     n_nodes=n_nodes, 
                                     hidden_warmup=hidden_warmup,
                                     edge_index_list=edge_index_list,
                                     model_path=model_path,
                                     backprop_loaded_model_inflow = False,
                                     backprop_loaded_model_outflow = False,
                                     surge_dp = surge_dp, 
                                   )
         
        # Backprop on the models of the other company
        # Backprop on the models of the company itself
        elif model_name.lower() in ["surge_2_1", "surge_2_dp_1"]:
            model = Surge_2(n_input_periods=n_input_periods,
                                     n_output_periods=n_output_periods,
                                     node_features=node_features,
                                     n_nodes=n_nodes, 
                                     hidden_warmup=hidden_warmup,
                                     edge_index_list=edge_index_list,
                                     model_path=model_path,
                                     backprop_loaded_model_inflow = True,
                                     backprop_loaded_model_outflow = True,
                                     surge_dp = surge_dp, 
                                   )
        # Backprop on the models of the other company
        # No backprop on the models of the company itself
        elif model_name.lower() in ["surge_2_2", "surge_2_dp_2",]:
            model = Surge_2(n_input_periods=n_input_periods,
                                     n_output_periods=n_output_periods,
                                     node_features=node_features,
                                     n_nodes=n_nodes, 
                                     hidden_warmup=hidden_warmup,
                                     edge_index_list=edge_index_list,
                                     model_path=model_path,
                                     backprop_loaded_model_inflow = True,
                                     backprop_loaded_model_outflow = False,
                                     surge_dp = surge_dp, 
                                   )
        # No backprop on the models of the other company
        # Backprop on the models of the company itself
        elif model_name.lower() in ["surge_2_3", "surge_2_dp_3",]:
            model = Surge_2(n_input_periods=n_input_periods,
                                     n_output_periods=n_output_periods,
                                     node_features=node_features,
                                     n_nodes=n_nodes, 
                                     hidden_warmup=hidden_warmup,
                                     edge_index_list=edge_index_list,
                                     model_path=model_path,
                                     backprop_loaded_model_inflow = False,
                                     backprop_loaded_model_outflow = True,
                                     surge_dp = surge_dp, 
                                   )
        # No backprop on the models of the other company
        # No backprop on the models of the company itself
        elif model_name.lower() in ["surge_2_4", "surge_2_dp_4",]:
            model = Surge_2(n_input_periods=n_input_periods,
                                     n_output_periods=n_output_periods,
                                     node_features=node_features,
                                     n_nodes=n_nodes, 
                                     hidden_warmup=hidden_warmup,
                                     edge_index_list=edge_index_list,
                                     model_path=model_path,
                                     backprop_loaded_model_inflow = False,
                                     backprop_loaded_model_outflow = False,
                                     surge_dp = surge_dp, 
                                   )
    elif model_name.lower() in ['m68a_i', 'm68a_o', 'm68b_i', 'm68b_o', 'm68c_i', 'm68c_o', 'm68d_i', 'm68d_o', ]:
        # Name of the other company that we should load its DP model
        other_company_str = other_company_str_dict[company.lower()]
        # DP models trained by the other company
        if model_name.lower() in ['m68a_i', 'm68a_o', 'm68b_i', 'm68b_o', 'm68c_i', 'm68c_o', 'm68d_i', 'm68d_o', ]:
            dp_model_i = 'MP25PP_i'
            dp_model_o = 'MP25PP_o'
        # Best model trained by the company itself
        if model_name.lower() in ['m68a_i', 'm68b_i', 'm68c_i', 'm68d_i']:
            model_company = 'multi17'
        elif model_name.lower() in ['m68a_o', 'm68b_o', 'm68c_o', 'm68d_o']:
            model_company = 'multi18'
        # Map model name to model function
        model_func_dict_68 = {'m68a_i': M68,
                              'm68a_o': M68,
                              'm68b_i': M68,
                              'm68b_o': M68,
                              'm68c_i': M68,
                              'm68c_o': M68,
                              'm68d_i': M68,
                              'm68d_o': M68,
                             } 
        # Map model name to whether it backpropogates through the non-DP model's layers
        no_backprop_nondp_models_dict = {'m68a_i': False,
                                         'm68a_o': False,
                                         'm68b_i': False,
                                         'm68b_o': False,
                                         'm68c_i': True,
                                         'm68c_o': True,
                                         'm68d_i': True,
                                         'm68d_o': True,
                                        } 
        # Map model name to whether it backpropogates through the DP model's layers
        no_backprop_dp_models_dict = {'m68a_i': False,
                                      'm68a_o': False,
                                      'm68b_i': True,
                                      'm68b_o': True,
                                      'm68c_i': False,
                                      'm68c_o': False,
                                      'm68d_i': True,
                                      'm68d_o': True,
                                     } 
        
            
        # Address of configutation (YAML) files of other models that should be loaded
        configuration_addresses = []
        # DP models of the other company
        if model_name.lower() in ["m68a_i", "m68a_o", "m68b_i", "m68b_o", "m68c_i", "m68c_o", "m68d_i", "m68d_o", ]:
            configuration_addresses.append("../data/m_"+str(start_month)+"_"+str(end_month)+"_y_"+str(year)+"_optimals/config_" + dp_model_i + "_optimal" + other_company_str + dp_loaded_str + ".yaml")
            configuration_addresses.append("../data/m_"+str(start_month)+"_"+str(end_month)+"_y_"+str(year)+"_optimals/config_" + dp_model_o + "_optimal" + other_company_str + dp_loaded_str + ".yaml")
        # Model of the company itself
        if model_name.lower() in ["m68a_i", "m68b_i", "m68c_i", "m68d_i"]:
            configuration_addresses.append("../data/m_"+str(start_month)+"_"+str(end_month)+"_y_"+str(year)+"_optimals/config_" + model_company + "_g" + str(graph_version) + "_optimal" + company_str + ".yaml")
        elif model_name.lower() in ["m68a_o", "m68b_o", "m68c_o", "m68d_o"]:
            configuration_addresses.append("../data/m_"+str(start_month)+"_"+str(end_month)+"_y_"+str(year)+"_optimals/config_" + model_company + "_g" + str(graph_version) + "_optimal" + company_str + ".yaml")
            
        for configuration_address in configuration_addresses:
            model_path.append(get_model_path(configuration_address))
        print(f"\n\nmodel_path of loaded models:")
        # Loop through the list with indexes
        for index, element in enumerate(model_path):
            print(f"{index}: {element}")
        if model_name.lower() in model_func_dict_68.keys():
            model = model_func_dict_68[model_name.lower()](n_input_periods=n_input_periods,
                                                 n_output_periods=n_output_periods,
                                                 node_features=node_features,
                                                 n_nodes=n_nodes, 
                                                 hidden_warmup=hidden_warmup,
                                                 edge_index_list=edge_index_list,
                                                 model_path=model_path,
                                                 dp_model_i = dp_model_i,
                                                 dp_model_o = dp_model_o,
                                                 model_company = model_company,
                                                 no_backprop_nondp_models = no_backprop_nondp_models_dict[model_name.lower()],
                                                 no_backprop_dp_models = no_backprop_dp_models_dict[model_name.lower()],
                                               )
    model = model.to(device)
    model.train()
    
    print("\n\n\nNet's layers, backpropagation status, and size:")
    total_param = 0
    # Print the layers and their backpropagation status
    for name, param in model.named_parameters():
        print(f"Layer: {name}, Backpropagate: {param.requires_grad}, Size: {model.state_dict()[name].size()}")
        total_param += np.prod(model.state_dict()[name].size())
    print('Net\'s total params:', total_param)
    print("\n\n")

    return model



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-y', '--yaml_address', type=str, default='../data/config_multiprediction.yaml')
    args = parser.parse_args()
    
    wandb.login()
    sweep_configuration = read_yaml(args.yaml_address)
    sweep_id = wandb.sweep(sweep=sweep_configuration, entity='ghafelebashi', project='ptp-sweep-3')
    wandb.agent(sweep_id, function=train)
