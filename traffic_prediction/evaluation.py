import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
# import datetime
# import networkx as nx

def mean_baseline(input_sample, _=None):
    n_priods = input_sample.shape[-1]
    mean_prediction = torch.mean(input_sample, dim=(2, 3)).unsqueeze(-1).repeat_interleave(n_priods, dim=2)
    return mean_prediction


def get_labels_and_predictions(model, loader, static_edge_index):
    if model != mean_baseline:
        model.eval()
    test_labels = []
    test_predictions = []
    with torch.no_grad():
        for encoder_inputs, labels in loader:
            # Get model predictions
            y_hat = model(encoder_inputs, static_edge_index)
            # Store for analysis below
            test_labels.append(labels)
            test_predictions.append(y_hat)
    test_labels = torch.stack(test_labels, dim=0)
    test_predictions = torch.stack(test_predictions, dim=0)
    return test_labels, test_predictions

def get_one_day_comparison(df, sensor, test_labels, predictions, baseline_predictions, periods, plot_times: [0]):
    datetimes_temp = df.index.get_level_values(0)
    interval_per_day = datetimes_temp[datetimes_temp.date==datetimes_temp[0].date()].nunique()
    
    
    labs  = np.array([label[0][sensor][0].detach().cpu().numpy() for label in test_labels[-(interval_per_day-periods):]])
    for timestep in range(periods):
        if timestep in plot_times:
            preds = np.array([pred[0][sensor][timestep].detach().cpu().numpy() for pred in predictions[-(interval_per_day-periods):]])
            plt.plot(preds, label='Prediction, '+str(timestep))

    base_preds = np.array([label[0][sensor][0].detach().cpu().numpy() for label in baseline_predictions[-(interval_per_day-periods):]])
    plt.plot(base_preds, label='Baseline Prediction')
    plt.plot(labs, label='Ground Truth')
    plt.legend()
    plt.show()
    
    
    
# def plot_err(df, cols, col_labels, title, x_label, y_label, skip_n_row: int = 0, val: bool = False):
# #     plt.rcParams['figure.figsize'] = [5, 3]
# #     plt.rcParams['figure.dpi'] = 75
    
#     ax = df.loc[skip_n_row:].plot(y=cols, label=col_labels, title=title)
#     ax.set_xlabel(x_label)
#     ax.set_ylabel(y_label)
#     if val:
#         ax.set_xticklabels(df.index)
#     plt.legend()
# #     plt.figure(figsize=(7,4))
#     plt.show()

# def plot_hist(df, cols, title, x_label, y_label, bins: int = 25):
#     ax = df.hist(column=['test_err'], bins = bins)
#     ax[0, 0].set_title(title)
#     ax[0, 0].set_xlabel(x_label)
#     ax[0, 0].set_ylabel(y_label)
# #     ax.set_xticklabels(ax.get_xticklabels(), rotation=30)    
#     plt.show()


# n_sample = 100
# def get_flattened_samples(label, prediction, n_sample: int = 100):
#     labels_flattened = torch.cat(label, axis=0).flatten().cpu()
#     predictions_flattened = torch.cat(prediction, axis=0).flatten().cpu()
#     torch.manual_seed(1)
#     indices = torch.randperm(n=len(labels_flattened))[:n_sample]
#     selected_test_labels = labels_flattened[indices].detach().numpy()
#     selected_predictions = predictions_flattened[indices].detach().numpy()
#     return selected_test_labels, selected_predictions

# def plot_label_vs_prediction(selected_test_labels, selected_predictions):
#     n_sample = len(selected_test_labels)
#     plt.plot(selected_test_labels, label='Ground Truth')
#     mae_ground_truth = np.mean(selected_test_labels)
#     plt.plot([mae_ground_truth]*n_sample, label='Mean Ground Truth, MAE: ' + "{:0.2f}".format(mae_ground_truth))
#     std_ground_truth = np.std(selected_test_labels)
#     plt.plot([mae_ground_truth+std_ground_truth]*n_sample, label='Mean + STD of Ground Truth, STD: ' + "{:0.2f}".format(std_ground_truth) + ', Mean + STD: ' + "{:0.2f}".format(mae_ground_truth+std_ground_truth))
#     plt.plot([mae_ground_truth-std_ground_truth]*n_sample, label='Mean - STD of Ground Truth, STD: ' + "{:0.2f}".format(std_ground_truth) + ', Mean - STD: ' + "{:0.2f}".format(mae_ground_truth-std_ground_truth))
#     plt.plot(selected_predictions, label='Prediction')
#     mae_predictions = np.mean(selected_predictions)
#     plt.plot([mae_predictions]*n_sample, label='Mean Predictions, MAE: ' + "{:0.2f}".format(mae_predictions))
#     std_predictions = np.std(selected_predictions)
#     plt.plot([mae_predictions+std_predictions]*n_sample, label='Mean + STD of Predictions, STD: ' + "{:0.2f}".format(std_predictions) + ', Mean + STD: ' + "{:0.2f}".format(mae_predictions+std_predictions))
#     plt.plot([mae_predictions-std_predictions]*n_sample, label='Mean - STD of Predictions, STD: ' + "{:0.2f}".format(std_predictions) + ', Mean - STD: ' + "{:0.2f}".format(mae_predictions-std_predictions))
#     plt.xlabel('Sample #')
#     plt.ylabel('Value')
#     plt.legend(bbox_to_anchor=(1.1, 1.05))
#     plt.show()

# def plot_mae(selected_test_labels, selected_predictions):
#     n_sample = len(selected_test_labels)
    
#     ae = np.abs(selected_test_labels-selected_predictions)
#     plt.plot(ae, label='Absolute Error (AE)')
    
#     mae = np.mean(np.abs(selected_test_labels-selected_predictions))
#     plt.plot([mae]*n_sample, label='MAE, MAE: ' + "{:0.2f}".format(mae))
    
#     std_ae = np.std(ae)
#     plt.plot([mae+std_ae]*n_sample, label='MAE + STD of AE, STD: ' + "{:0.2f}".format(std_ae) + ', MAE + STD: ' + "{:0.2f}".format(mae+std_ae))
#     plt.plot([mae-std_ae]*n_sample, label='MAE - STD of AE, STD: ' + "{:0.2f}".format(std_ae) + ', MAE - STD: ' + "{:0.2f}".format(mae-std_ae))
    
#     plt.xlabel('Sample #')
#     plt.ylabel('Error')
#     plt.legend(bbox_to_anchor=(1.1, 1.05))
#     plt.show()
