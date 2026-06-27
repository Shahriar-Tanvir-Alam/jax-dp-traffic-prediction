
import matplotlib.pyplot as plt
import numpy as np 
import math
import random 
from tqdm import tqdm
from traffic_prediction.utils import set_seed

class LinearRegression:
    def __init__(self, learning_rate, n_epochs, X_train, y_train, X_val, y_val, X_test, y_test, verbose = False):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.X_train, self.y_train = X_train, y_train
        self.X_val, self.y_val = X_val, y_val
        self.X_test, self.y_test = X_test, y_test
#         print(X_train.shape)
        self.verbose = verbose
        self.mse_training = []
        self.mse_val = []
        self.mse_test = []
        self.divergence_counter = 1
        
    def initialization(self):
        n_feature = self.X_train.shape[1]
        self.W = np.zeros((1, n_feature))
        self.bias = 0
        
    def fit(self):
        self.initialization()
#         for epoch in tqdm(range(self.n_epochs), disable=(not self.verbose)):
        for epoch in range(self.n_epochs):
            # Update Weights
            self.update_weight(self.X_train, self.y_train)
            # Record MSE
            mse_train = self.get_mse(self.X_train, self.y_train)
            mse_test = self.get_mse(self.X_test, self.y_test)
            self.mse_training.append(mse_train)
            self.mse_test.append(mse_test)
            if self.X_val.shape[0] > 0:
                mse_val = self.get_mse(self.X_val, self.y_val)
                self.mse_val.append(mse_val)
            # Check Divergence
            if epoch > 0 and self.check_divergence():
                if self.verbose:
                    print(f"Validation MSE diverged at epoch {epoch} for more than 1 time!")
                return 
            
    def update_weight(self, X, y):
        # Prediction
        y_hat = self.get_pred(X)
        # Gradients
        n_sample = X.shape[0]
        dW = -2 * X.T @ (y - y_hat) / n_sample
        db = 2 * np.sum(y - y_hat) / n_sample
        # Update
        self.W -= self.learning_rate * dW
        self.bias -= self.learning_rate * db
        
    def get_pred(self, X):
        return X @ self.W + self.bias
    
    def get_mse(self, X, y):
        # Prediction
        n_sample = X.shape[0]
        y_hat = self.get_pred(X)
        assert n_sample > 0, print("No sample here!")
        return (((y - y_hat).T @ (y - y_hat)) / n_sample).item()
            
    def check_divergence(self):
        if len(self.mse_val) > 0 and self.mse_val[-1] > self.mse_val[-2]:
#         if self.mse_training[-1] > self.mse_training[-2]:
            self.divergence_counter -= 1
            if self.divergence_counter == 0:
                return True
        return False
        
# class LinearRegressionDualGradient:
#     def __init__(self, learning_rate, n_epochs, X_train1, y_train1, X_train2, y_train2, X_val, y_val, X_test, y_test, verbose = False):
#         self.learning_rate = learning_rate
#         self.n_epochs = n_epochs
#         self.X_train1, self.y_train1 = X_train1, y_train1
#         self.X_train2, self.y_train2 = X_train2, y_train2
#         self.n_sample1 = self.X_train1.shape[0]
#         self.n_sample2 = self.X_train2.shape[0]
#         self.n_sample_total = self.n_sample1 + self.n_sample2
#         self.X_val, self.y_val = X_val, y_val
#         self.X_test, self.y_test = X_test, y_test
#         self.verbose = verbose
#         self.mse_training = []
#         self.mse_val = []
#         self.mse_test = []
#         self.divergence_counter = 10
        
#     def initialization(self):
#         n_feature = self.X_train1.shape[1]
#         self.W = np.zeros((1, n_feature))
#         self.bias = 0
        
#     def fit(self):
#         self.initialization()
#         for epoch in tqdm(range(self.n_epochs), disable=(not self.verbose)):
#             # Update Weights
#             dW1, db1 = self.get_gradient(self.X_train1, self.y_train1)
#             dW2, db2 = self.get_gradient(self.X_train2, self.y_train2)
#             self.update_weight(dW1, db1, dW2, db2, )
#             # Record MSE
# #             mse_train = self.get_mse(self.X_train, self.y_train)
#             mse_test = self.get_mse(self.X_test, self.y_test)
# #             self.mse_training.append(mse_train)
#             self.mse_test.append(mse_test)
#             if self.X_val.shape[0] > 0:
#                 mse_val = self.get_mse(self.X_val, self.y_val)
#                 self.mse_val.append(mse_val)
#             # Check Divergence
#             if epoch > 0 and self.check_divergence():
#                 if self.verbose:
#                     print(f"Validation MSE diverged at epoch {epoch} for more than 1 time!")
#                 return 
            
            
#     def get_gradient(self, X, y):
#         # Prediction
#         y_hat = self.get_pred(X)
#         # Gradients
#         n_sample = X.shape[0]
#         dW = -2 * X.T @ (y - y_hat) / n_sample
#         db = 2 * np.sum(y - y_hat) / n_sample
#         return dW, db
    
#     def update_weight(self, dW1, db1, dW2, db2, ):
#         # Update
#         self.W -= self.learning_rate * (dW1 + dW2)/2
#         self.bias -= self.learning_rate * (db1 + db2)/2
        
# #         # Update
# #         self.W -= self.learning_rate * (dW1 * (self.n_sample1/self.n_sample_total) + \
# #                                         dW2 * (self.n_sample2/self.n_sample_total))
# #         self.bias -= self.learning_rate * (db1 * (self.n_sample1/self.n_sample_total) + \
# #                                            db2 * (self.n_sample2/self.n_sample_total))
        
#     def get_pred(self, X):
#         return X @ self.W + self.bias
    
#     def get_mse(self, X, y):
#         # Prediction
#         n_sample = X.shape[0]
#         y_hat = self.get_pred(X)
#         return (((y - y_hat).T @ (y - y_hat)) / n_sample).item()
            
#     def check_divergence(self):
#         if len(self.mse_val) > 0 and self.mse_val[-1] > self.mse_val[-2]:
# #         if self.mse_training[-1] > self.mse_training[-2]:
#             self.divergence_counter -= 1
#             if self.divergence_counter == 0:
#                 return True
#         return False
        
        
        
def get_n_train_val_test(n_sample, train_ratio, val_test_ratio, verbose=False):
    n_train = int(n_sample * train_ratio)
    n_val_test = n_sample - n_train
    n_val = int(n_val_test * val_test_ratio)
    n_test = n_val_test - n_val
#     print(f"Number of train samples: \t{n_train}/{n_sample}")
#     print(f"Number of validation samples: \t{n_val}/{n_sample}")
#     print(f"Number of test samples: \t{n_test}/{n_sample}")
    return n_train, n_val, n_test



def get_slope_intercept(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    slope = (y2 - y1)/(x2 - x1)
    intercept = 0.1 - slope
    print(f"slope: {slope:.3f}, intercept: {intercept:.3f}")
    plt.plot([x1, x2], [y1, y2])
    plt.plot([0, x2+1], [1, 1], linestyle='dashed')
    plt.plot([0, x2+1], [0, 0], linestyle='dashed')
    plt.show()
    return slope, intercept


def get_random_data(n_sample, x_min, x_max, seed = 42, verbose=False):
    set_seed(seed)
    r_uniform = np.random.uniform(size = n_sample) * (x_max - x_min + 1) + x_min
    x_values = [math.floor(x) for x in r_uniform]
    if verbose:
        print(f"First 5 generated initial random values: {r_uniform[:5]}")
        print(f"First 5 generated random integer values: {x_values[:5]}")
    return x_values


def sigmoid(x):
    return 1/(1 + math.exp(x))


def get_labels(x_values, slope, intercept, seed = 42, verbose = False):
    set_seed(seed)
    labels = []
    for x in x_values:
        threshold_p_x = slope * x + intercept
        rand_val = random.random()
        if rand_val > threshold_p_x:
            label = 1
        else:
            label = 0
        labels.append(label)
        if verbose:
            print(f"\nx: {x}")
            print(f"threshold_p_x (ax+b): {threshold_p_x:.3f} ({slope:.3f} * {x} + {intercept:.3f})")
            print(f"rand_val: {rand_val:.3f}")
            print(f"label: {label}")
            
#     for x in x_values:
#         threshold_p_x = slope * x + intercept
#         rand_val = (random.random() - 0.5) * 1
#         prob = sigmoid(rand_val)
#         if prob <= threshold_p_x:
#             label = 1
#         else:
#             label = 0
#         labels.append(label)
#         if verbose:
#             print(f"\nx: {x}")
#             print(f"threshold_p_x (ax+b): {threshold_p_x:.3f} ({slope:.3f} * {x} + {intercept:.3f})")
#             print(f"rand_val: {rand_val:.3f}, prob: {prob:.3f}")
#             print(f"label: {label}")
    return labels

def split_train_val_test(x_values, label_values, n_train, n_val, n_test, verbose=False):
    data = {}
    labels = {}
    # Divide
    data['train'] = np.array(x_values[:n_train])
    labels['train'] = np.array(label_values[:n_train])
    data['val'] = np.array(x_values[n_train:n_train+n_val])
    labels['val'] = np.array(label_values[n_train:n_train+n_val])
    data['test'] = np.array(x_values[n_train+n_val:])
    labels['test'] = np.array(label_values[n_train+n_val:])
    # Reshape
    data['train'] = np.reshape(data['train'], (-1, 1))
    data['val'] = np.reshape(data['val'], (-1, 1))
    data['test'] = np.reshape(data['test'], (-1, 1))
    labels['train'] = np.reshape(labels['train'], (-1, 1))
    labels['val'] = np.reshape(labels['val'], (-1, 1))
    labels['test'] = np.reshape(labels['test'], (-1, 1))
    # Print Size
    if verbose:
        print(f"Train data  size (x, labels): \t({len(data['train'])}, {len(labels['train'])})")
        print(f"Val. data size (x, labels): \t({len(data['val'])}, {len(labels['val'])})")
        print(f"Test data size (x, labels): \t({len(data['test'])}, {len(labels['test'])})")
    return data, labels


def split_company_train_val_test(data, labels, c_sizes, n_train, n_val, n_test, verbose=False):
    data_company = {'train': {}, 'val': {}, 'test': {}}
    labels_company = {'train': {}, 'val': {}, 'test': {}}
#     print(f"* ------------------------------------------------------------------ *")
    for c_idx, c_size in enumerate(c_sizes):
#         if verbose: print("\n* ------------------------------------------------------ *")
#         if verbose: print(f"Company {c_idx+1}:")
        
        n_train_prev_companies = int(n_train * sum(c_sizes[:c_idx]))
        n_train_so_far_companies = int(n_train * sum(c_sizes[:c_idx+1]))
        n_train_company = n_train_so_far_companies - n_train_prev_companies
        n_val_prev_companies = int(n_val * sum(c_sizes[:c_idx]))
        n_val_so_far_companies = int(n_val * sum(c_sizes[:c_idx+1]))
        n_val_company = n_val_so_far_companies - n_val_prev_companies
        n_test_prev_companies = int(n_test * sum(c_sizes[:c_idx]))
        n_test_so_far_companies = int(n_test * sum(c_sizes[:c_idx+1]))
        n_test_company = n_test_so_far_companies - n_test_prev_companies
        if verbose:
            print(f"Train data size: {n_train_company}")
            print(f"Val. data size: {n_val_company}")
            print(f"Test data size: {n_test_company}")
#         print(f"Company {c_idx+1} ({c_size}) train/val/test data size: {n_train_company}/{n_val_company}/{n_test_company}")


        data_company['train'][c_idx] = np.array(data['train'][n_train_prev_companies:n_train_so_far_companies])
        data_company['val'][c_idx] = np.array(data['val'][n_val_prev_companies:n_val_so_far_companies])
        data_company['test'][c_idx] = np.array(data['test'][n_test_prev_companies:n_test_so_far_companies])
        labels_company['train'][c_idx] = np.array(labels['train'][n_train_prev_companies:n_train_so_far_companies])
        labels_company['val'][c_idx] = np.array(labels['val'][n_val_prev_companies:n_val_so_far_companies])
        labels_company['test'][c_idx] = np.array(labels['test'][n_test_prev_companies:n_test_so_far_companies])
    return data_company, labels_company


# def agg_train_val_test(data, labels, x_min, x_max, verbose=False):
#     data_agg = {}
#     labels_agg = {}
#     # Divide
#     data_agg['train'], labels_agg['train'] = agg_data_labels(data['train'], labels['train'], x_min, x_max)
#     data_agg['val'], labels_agg['val'] = agg_data_labels(data['val'], labels['val'], x_min, x_max)
#     data_agg['test'], labels_agg['test'] = agg_data_labels(data['test'], labels['test'], x_min, x_max)
#     # Print Size
#     if verbose:
#         print(f"Train data  size (x, labels): \t({len(data_agg['train'])}, {len(labels_agg['train'])})")
#         print(f"Val. data size (x, labels): \t({len(data_agg['val'])}, {len(labels_agg['val'])})")
#         print(f"Test data size (x, labels): \t({len(data_agg['test'])}, {len(labels_agg['test'])})")
#     return data_agg, labels_agg

def agg_data_labels(data, labels, x_min, x_max):
    val_count_1 = [0 for i in range(x_min, x_max+1)]
    val_count_total = [0 for i in range(x_min, x_max+1)]
    for x, y in zip(data.flatten(), labels.flatten()):
        val_count_total[x - x_min] += 1
        if y == 1:
            val_count_1[x - x_min] += 1

    data_agg = []
    labels_agg = []
    for idx in range(x_max - x_min + 1):
        x_val = idx + x_min
        count_1 = val_count_1[idx]
        count_total = val_count_total[idx]
        if count_total > 0:
            data_agg.append(x_val)
            labels_agg.append(count_1/count_total)
#         else:
#             data_agg.append(x_val)
#             labels_agg.append(0)
            
    data_agg = np.array(data_agg)
    data_agg = np.reshape(data_agg, (-1, 1))
    labels_agg = np.array(labels_agg)
    labels_agg = np.reshape(labels_agg, (-1, 1))
    return data_agg, labels_agg


def get_aggregated_data(data, labels, data_company, labels_company, c_sizes, x_min, x_max):
    data_agg = {}
    labels_agg = {}
    # Divide
#     data, labels, x_min, x_max
    data_agg['train'], labels_agg['train'] = agg_data_labels(data=data['train'], labels=labels['train'], x_min=x_min, x_max=x_max)
    data_agg['val'], labels_agg['val'] = agg_data_labels(data=data['val'], labels=labels['val'], x_min=x_min, x_max=x_max)
    data_agg['test'], labels_agg['test'] = agg_data_labels(data=data['test'], labels=labels['test'], x_min=x_min, x_max=x_max)

    # [[0.25, 0.75], [0.5, 0.5]]
    data_company_agg = {}
    labels_company_agg = {}
    for company_setting in range(len(c_sizes)):
        data_company_agg[company_setting] = {'train': {}, 'val': {}, 'test': {}}
        labels_company_agg[company_setting] = {'train': {}, 'val': {}, 'test': {}}
        for c_idx, c_size in enumerate(c_sizes[company_setting]):
    #         if verbose: print("\n* ------------------------------------------------------ *")
    #         if verbose: print(f"Company {c_idx+1}:")
            data_company_agg[company_setting]['train'][c_idx], labels_company_agg[company_setting]['train'][c_idx] = agg_data_labels(
                data=data_company[company_setting]['train'][c_idx],
                labels=labels_company[company_setting]['train'][c_idx],
                x_min=x_min,
                x_max=x_max,
                 )
            data_company_agg[company_setting]['val'][c_idx], labels_company_agg[company_setting]['val'][c_idx] = agg_data_labels(
                data=data_company[company_setting]['val'][c_idx],
                labels=labels_company[company_setting]['val'][c_idx],
                x_min=x_min,
                x_max=x_max,
                 )
            data_company_agg[company_setting]['test'][c_idx], labels_company_agg[company_setting]['test'][c_idx] = agg_data_labels(
                data=data_company[company_setting]['test'][c_idx],
                labels=labels_company[company_setting]['test'][c_idx],
                x_min=x_min,
                x_max=x_max,
                 )
    return data_agg, labels_agg, data_company_agg, labels_company_agg


def get_lr_datasets(n_sample, slope, intercept, c_sizes, train_ratio, val_test_ratio, verbose, x_min=1, x_max=10, seed=42):
    # get x and p(x)
    x_values = get_random_data(n_sample=n_sample, x_min=x_min, x_max=x_max, seed=seed, verbose=verbose)
    label_values = get_labels(x_values=x_values, slope=slope, intercept=intercept, seed=seed, verbose=verbose)
    
    # Get number of samples per train/val/test
    n_train, n_val, n_test = get_n_train_val_test(n_sample=n_sample, 
                                                  train_ratio=train_ratio, 
                                                  val_test_ratio=val_test_ratio, 
                                                  verbose=verbose)
    
    # Get train/val/test data
    data, labels = split_train_val_test(x_values=x_values,
                                        label_values=label_values,
                                        n_train=n_train,
                                        n_val=n_val,
                                        n_test=n_test,
                                        verbose=verbose)
    
    # Get train/val/test data per company
    data_company = {}
    labels_company = {}
    for company_setting in range(len(c_sizes)):
        data_company[company_setting], labels_company[company_setting] = split_company_train_val_test(data=data,
                                                                                    labels=labels,
                                                                                    c_sizes=c_sizes[company_setting],
                                                                                    n_train=n_train,
                                                                                    n_val=n_val,
                                                                                    n_test=n_test,
                                                                                    verbose = verbose,  
                                                                                   )
    return data, labels, data_company, labels_company, n_train, n_val, n_test


def get_trained_LR(c_sizes,
                   learning_rate,
                   n_epochs,
                   data,
                   labels,
                   data_company,
                   labels_company,
                   coop,
                   verbose,
                  ):
    LRs = {}
#     for idx_setting in tqdm(range(len(c_sizes)), desc="Company setting processed: "):
    for idx_setting in range(len(c_sizes)):
        LRs[idx_setting] = {}
        if verbose: print(f"\n* ------------------- {c_sizes[idx_setting]} ------------------------- *")
        for c_idx in range(len(c_sizes[idx_setting])):

            # Linear Regression Object
            if coop:
                X_train = data['train']
                y_train = labels['train']
                X_val = data['val']
                y_val = labels['val']
            else:
                X_train = data_company[idx_setting]['train'][c_idx]
                y_train = labels_company[idx_setting]['train'][c_idx]
                X_val = data_company[idx_setting]['val'][c_idx]
                y_val = labels_company[idx_setting]['val'][c_idx]
#                 print(X_train.shape)
#             X_val = data_company[idx_setting]['val'][c_idx]
#             y_val = labels_company[idx_setting]['val'][c_idx]
            LRs[idx_setting][c_idx] = LinearRegression(learning_rate = learning_rate,
                                                       n_epochs = n_epochs,
                                                       X_train = X_train,
                                                       y_train = y_train,
                                                       X_val = X_val,
                                                       y_val = y_val,
                                                       X_test = data_company[idx_setting]['test'][c_idx],
                                                       y_test = labels_company[idx_setting]['test'][c_idx],
                                                       verbose = verbose,
                                                      )
            # Fit the model
            LRs[idx_setting][c_idx].fit()
            # Results
            if verbose:
                print(f">> Company {c_idx+1} with size {c_sizes[idx_setting][c_idx]}:")
                print(f"First vs. Last MSE: {LRs[idx_setting][c_idx].mse_training[0]:.3f} vs. {LRs[idx_setting][c_idx].mse_training[-1]:.3f}")
                print(f"(W, bias) = ({LRs[idx_setting][c_idx].W.item():.3f}, {LRs[idx_setting][c_idx].bias:.3f})")
                print(f"Test MSE = {LRs[idx_setting][c_idx].mse_test[-1]:.3f}")
    return LRs



# def get_trained_LR_dual_gradient(c_sizes,
#                    learning_rate,
#                    n_epochs,
#                    data,
#                    labels,
#                    data_company,
#                    labels_company,
#                    coop,
#                    verbose,
#                   ):
#     LRs = {}
#     for idx_setting in tqdm(range(len(c_sizes)), desc="Company setting processed: "):
#         LRs[idx_setting] = {}
#         if verbose: print(f"\n* ------------------- {c_sizes[idx_setting]} ------------------------- *")
#         for c_idx in range(len(c_sizes[idx_setting])):

#             # Linear Regression Object
#             if coop:
#                 X_train1 = data_company[idx_setting]['train'][0]
#                 y_train1 = labels_company[idx_setting]['train'][0]
#                 X_train2 = data_company[idx_setting]['train'][1]
#                 y_train2 = labels_company[idx_setting]['train'][1]
#                 X_val = data['val']
#                 y_val = labels['val']
#                 LRs[idx_setting][c_idx] = LinearRegressionDualGradient(learning_rate = learning_rate,
#                                                    n_epochs = n_epochs,
#                                                    X_train1 = X_train1,
#                                                    y_train1 = y_train1,
#                                                    X_train2 = X_train2,
#                                                    y_train2 = y_train2,
#                                                    X_val = X_val,
#                                                    y_val = y_val,
#                                                    X_test = data_company[idx_setting]['test'][c_idx],
#                                                    y_test = labels_company[idx_setting]['test'][c_idx],
#                                                    verbose = verbose,
#                                                   )
#                 # Fit the model
#                 LRs[idx_setting][c_idx].fit()
#             else:
#                 X_train = data_company[idx_setting]['train'][c_idx]
#                 y_train = labels_company[idx_setting]['train'][c_idx]
#                 X_val = data_company[idx_setting]['val'][c_idx]
#                 y_val = labels_company[idx_setting]['val'][c_idx]
#                 LRs[idx_setting][c_idx] = LinearRegression(learning_rate = learning_rate,
#                                                            n_epochs = n_epochs,
#                                                            X_train = X_train,
#                                                            y_train = y_train,
#                                                            X_val = X_val,
#                                                            y_val = y_val,
#                                                            X_test = data_company[idx_setting]['test'][c_idx],
#                                                            y_test = labels_company[idx_setting]['test'][c_idx],
#                                                            verbose = verbose,
#                                                           )
#                 # Fit the model
#                 LRs[idx_setting][c_idx].fit()
                
#             # Results
#             if verbose:
#                 print(f">> Company {c_idx+1} with size {c_sizes[idx_setting][c_idx]}:")
#                 print(f"First vs. Last MSE: {LRs[idx_setting][c_idx].mse_training[0]:.3f} vs. {LRs[idx_setting][c_idx].mse_training[-1]:.3f}")
#                 print(f"(W, bias) = ({LRs[idx_setting][c_idx].W.item():.3f}, {LRs[idx_setting][c_idx].bias:.3f})")
#                 print(f"Test MSE = {LRs[idx_setting][c_idx].mse_test[-1]:.3f}")
#     return LRs

def get_lr_improv(c_sizes,
                  LRs,
                  LRs_coop,
                  verbose,
                 ):
    improvements_c_size = []
    improvements_perc = []
    for idx_setting in range(len(c_sizes)):
        if verbose:
            print(f"\n* ------------------- {c_sizes[idx_setting]} ------------------------- *")
        for c_idx in range(len(c_sizes[idx_setting])):
            c_size = c_sizes[idx_setting][c_idx]*100
            test_mse_wo_coop = LRs[idx_setting][c_idx].mse_test[-1]
            test_mse_w_coop = LRs_coop[idx_setting][c_idx].mse_test[-1]
            if test_mse_wo_coop == 0:
                test_mse_wo_coop = 1e-5
                improv_perc = (test_mse_wo_coop - test_mse_w_coop)/(test_mse_wo_coop)*100
            else:
                improv_perc = (test_mse_wo_coop - test_mse_w_coop)/(test_mse_wo_coop)*100
            improvements_c_size.append(c_size)
            improvements_perc.append(improv_perc)
            if verbose:
                print(f">> Company {c_idx+1} with size {c_sizes[idx_setting][c_idx]*100}%: Test MSE Improvement After Collaboration = {improv_perc:.3f}")
    return improvements_c_size, improvements_perc
    