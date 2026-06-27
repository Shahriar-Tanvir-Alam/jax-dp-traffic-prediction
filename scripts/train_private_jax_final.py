import sys
import os
sys.path.insert(0, "../")

import jax
import jax.numpy as jnp
import optax
import numpy as np
from opacus.accountants import RDPAccountant

from traffic_prediction.utils import read_yaml, yaml_to_config
from train_private import get_data_params, get_datasets


# ---------------------------------------------------------
# JAX Customized_ASTGCN with DP-SGD
#
# This file replaces the PyTorch/Opacus training mechanism with:
#   1. per-sample gradients
#   2. per-sample gradient clipping
#   3. Gaussian noise addition
#   4. Optax optimizer update
#
# Exact privacy accountant is not included yet.
# This is the first working JAX DP-SGD implementation.
# ---------------------------------------------------------


def edge_index_to_adjacency(edge_index, n_nodes):
    if hasattr(edge_index, "detach"):
        edge_index = edge_index.detach().cpu().numpy()
    else:
        edge_index = np.array(edge_index)

    if edge_index.shape[0] != 2 and edge_index.shape[1] == 2:
        edge_index = edge_index.T

    adj = np.zeros((n_nodes, n_nodes), dtype=np.float32)

    for source, target in zip(edge_index[0], edge_index[1]):
        source = int(source)
        target = int(target)

        if 0 <= source < n_nodes and 0 <= target < n_nodes:
            adj[source, target] = 1.0
            adj[target, source] = 1.0

    adj = adj + np.eye(n_nodes, dtype=np.float32)

    row_sum = adj.sum(axis=1, keepdims=True)
    adj_norm = adj / np.maximum(row_sum, 1.0)

    return jnp.array(adj_norm)


def compute_chebyshev_polynomials(adj_norm, K):
    n_nodes = adj_norm.shape[0]

    polys = [jnp.eye(n_nodes, dtype=jnp.float32)]

    if K == 1:
        return jnp.stack(polys, axis=0)

    polys.append(adj_norm)

    for _ in range(2, K):
        next_poly = 2.0 * (adj_norm @ polys[-1]) - polys[-2]
        polys.append(next_poly)

    return jnp.stack(polys, axis=0)


def init_matrix(key, shape, scale=0.01):
    return jax.random.normal(key, shape, dtype=jnp.float32) * scale


def init_block_params(
    keys,
    in_channels,
    n_nodes,
    input_periods,
    K,
    nb_chev_filter,
    nb_time_filter,
):
    return {
        "T_U1": init_matrix(keys[0], (n_nodes,)),
        "T_U2": init_matrix(keys[1], (in_channels, n_nodes)),
        "T_U3": init_matrix(keys[2], (in_channels,)),
        "T_be": jnp.zeros((1, input_periods, input_periods), dtype=jnp.float32),
        "T_Ve": init_matrix(keys[3], (input_periods, input_periods)),

        "S_W1": init_matrix(keys[4], (input_periods,)),
        "S_W2": init_matrix(keys[5], (in_channels, input_periods)),
        "S_W3": init_matrix(keys[6], (in_channels,)),
        "S_bs": jnp.zeros((1, n_nodes, n_nodes), dtype=jnp.float32),
        "S_Vs": init_matrix(keys[7], (n_nodes, n_nodes)),

        "cheb_theta": init_matrix(keys[8], (K, in_channels, nb_chev_filter)),
        "cheb_bias": jnp.zeros((nb_chev_filter,), dtype=jnp.float32),

        "time_W": init_matrix(keys[9], (nb_time_filter, nb_chev_filter, 3)),
        "time_b": jnp.zeros((nb_time_filter,), dtype=jnp.float32),

        "res_W": init_matrix(keys[10], (nb_time_filter, in_channels)),
        "res_b": jnp.zeros((nb_time_filter,), dtype=jnp.float32),

        "ln_gamma": jnp.ones((nb_time_filter,), dtype=jnp.float32),
        "ln_beta": jnp.zeros((nb_time_filter,), dtype=jnp.float32),
    }


def init_model_params(
    key,
    nb_block,
    in_channels,
    n_nodes,
    input_periods,
    output_periods,
    K,
    nb_chev_filter,
    nb_time_filter,
):
    keys = jax.random.split(key, nb_block * 11 + 2)

    blocks = []

    for block_id in range(nb_block):
        block_in_channels = in_channels if block_id == 0 else nb_time_filter
        block_keys = keys[block_id * 11:(block_id + 1) * 11]

        block_params = init_block_params(
            keys=block_keys,
            in_channels=block_in_channels,
            n_nodes=n_nodes,
            input_periods=input_periods,
            K=K,
            nb_chev_filter=nb_chev_filter,
            nb_time_filter=nb_time_filter,
        )

        blocks.append(block_params)

    params = {
        "blocks": blocks,
        "final_W": init_matrix(keys[-2], (output_periods, input_periods, nb_time_filter)),
        "final_b": jnp.zeros((output_periods,), dtype=jnp.float32),
    }

    return params


def temporal_attention(block, X):
    X_perm = jnp.transpose(X, (0, 3, 2, 1))

    LHS = jnp.einsum("btfn,n->btf", X_perm, block["T_U1"])
    LHS = jnp.einsum("btf,fn->btn", LHS, block["T_U2"])

    RHS = jnp.einsum("f,bnft->bnt", block["T_U3"], X)

    E = jnp.einsum("btn,bns->bts", LHS, RHS)
    E = E + block["T_be"]
    E = jnp.einsum("ij,bjs->bis", block["T_Ve"], jax.nn.sigmoid(E))
    E = jax.nn.softmax(E, axis=1)

    return E


def apply_temporal_attention(X, E):
    batch_size, n_nodes, n_features, input_periods = X.shape

    X_flat = X.reshape((batch_size, n_nodes * n_features, input_periods))
    X_tilde = jnp.einsum("bqt,bts->bqs", X_flat, E)
    X_tilde = X_tilde.reshape((batch_size, n_nodes, n_features, input_periods))

    return X_tilde


def spatial_attention(block, X):
    LHS = jnp.einsum("bnft,t->bnf", X, block["S_W1"])
    LHS = jnp.einsum("bnf,ft->bnt", LHS, block["S_W2"])

    RHS = jnp.einsum("f,bnft->bnt", block["S_W3"], X)
    RHS = jnp.transpose(RHS, (0, 2, 1))

    S = jnp.einsum("bnt,btm->bnm", LHS, RHS)
    S = S + block["S_bs"]
    S = jnp.einsum("ij,bjm->bim", block["S_Vs"], jax.nn.sigmoid(S))
    S = jax.nn.softmax(S, axis=1)

    return S


def cheb_conv_attention(block, X, S, cheb_polynomials):
    weighted_polys = cheb_polynomials[None, :, :, :] * S[:, None, :, :]

    x_k = jnp.einsum("bkij,bjft->bikft", weighted_polys, X)

    out = jnp.einsum("bikft,kfo->biot", x_k, block["cheb_theta"])
    out = out + block["cheb_bias"][None, None, :, None]

    return out


def time_convolution(block, X):
    batch_size, n_nodes, channels, input_periods = X.shape

    X_pad = jnp.pad(X, ((0, 0), (0, 0), (0, 0), (1, 1)))

    windows = jnp.stack(
        [X_pad[:, :, :, i:i + input_periods] for i in range(3)],
        axis=-1,
    )

    out = jnp.einsum("bnctk,ock->bnot", windows, block["time_W"])
    out = out + block["time_b"][None, None, :, None]

    return out


def residual_convolution(block, X):
    out = jnp.einsum("bnft,of->bnot", X, block["res_W"])
    out = out + block["res_b"][None, None, :, None]
    return out


def layer_norm_feature_axis(block, X, eps=1e-5):
    mean = jnp.mean(X, axis=2, keepdims=True)
    var = jnp.mean((X - mean) ** 2, axis=2, keepdims=True)

    X_norm = (X - mean) / jnp.sqrt(var + eps)

    X_norm = X_norm * block["ln_gamma"][None, None, :, None]
    X_norm = X_norm + block["ln_beta"][None, None, :, None]

    return X_norm


def astgcn_block_forward(block, X, cheb_polynomials):
    E = temporal_attention(block, X)
    X_tilde = apply_temporal_attention(X, E)

    S = spatial_attention(block, X_tilde)

    X_hat = cheb_conv_attention(block, X, S, cheb_polynomials)
    X_hat = jax.nn.relu(X_hat)

    X_hat = time_convolution(block, X_hat)
    X_res = residual_convolution(block, X)

    X_out = jax.nn.relu(X_hat + X_res)
    X_out = layer_norm_feature_axis(block, X_out)

    return X_out


def model_forward(params, X, cheb_polynomials):
    for block in params["blocks"]:
        X = astgcn_block_forward(block, X, cheb_polynomials)

    y = jnp.einsum("bnft,otf->bno", X, params["final_W"])
    y = y + params["final_b"][None, None, :]

    return y


def mse_loss(params, X, y, cheb_polynomials):
    pred = model_forward(params, X, cheb_polynomials)
    return jnp.mean((pred - y) ** 2)


def single_sample_loss(params, x_single, y_single, cheb_polynomials):
    x_single = x_single[None, ...]
    y_single = y_single[None, ...]

    pred = model_forward(params, x_single, cheb_polynomials)

    return jnp.mean((pred - y_single) ** 2)


def compute_metrics(y_true, y_pred):
    eps = 1e-5

    mse = jnp.mean((y_pred - y_true) ** 2)
    rmse = jnp.sqrt(mse)
    mae = jnp.mean(jnp.abs(y_pred - y_true))

    mask = jnp.abs(y_true) > eps
    mape = jnp.mean(jnp.where(mask, jnp.abs((y_true - y_pred) / y_true), 0.0)) * 100.0

    return mse, rmse, mae, mape


def global_norm_per_sample(per_sample_grads):
    leaves = jax.tree_util.tree_leaves(per_sample_grads)

    squared_norms = None

    for leaf in leaves:
        reduce_axes = tuple(range(1, leaf.ndim))
        leaf_squared = jnp.sum(leaf ** 2, axis=reduce_axes)

        if squared_norms is None:
            squared_norms = leaf_squared
        else:
            squared_norms = squared_norms + leaf_squared

    return jnp.sqrt(squared_norms + 1e-12)


def clip_and_average_per_sample_grads(per_sample_grads, max_grad_norm):
    norms = global_norm_per_sample(per_sample_grads)

    clip_factors = jnp.minimum(1.0, max_grad_norm / (norms + 1e-6))

    def clip_leaf(leaf):
        reshape_shape = (leaf.shape[0],) + (1,) * (leaf.ndim - 1)
        return leaf * clip_factors.reshape(reshape_shape)

    clipped_grads = jax.tree_util.tree_map(clip_leaf, per_sample_grads)

    avg_clipped_grads = jax.tree_util.tree_map(
        lambda g: jnp.mean(g, axis=0),
        clipped_grads
    )

    avg_clip_fraction = jnp.mean((norms > max_grad_norm).astype(jnp.float32))
    avg_grad_norm = jnp.mean(norms)

    return avg_clipped_grads, avg_grad_norm, avg_clip_fraction


def add_gaussian_noise_to_grads(grads, key, noise_multiplier, max_grad_norm, batch_size):
    leaves, treedef = jax.tree_util.tree_flatten(grads)
    keys = jax.random.split(key, len(leaves))

    noise_std = noise_multiplier * max_grad_norm / batch_size

    noised_leaves = []

    for leaf, leaf_key in zip(leaves, keys):
        noise = jax.random.normal(leaf_key, leaf.shape, dtype=leaf.dtype) * noise_std
        noised_leaves.append(leaf + noise)

    noised_grads = jax.tree_util.tree_unflatten(treedef, noised_leaves)

    return noised_grads


optimizer = optax.adam(learning_rate=1e-3)


@jax.jit
def train_step_nonprivate(params, opt_state, X, y, cheb_polynomials):
    loss, grads = jax.value_and_grad(mse_loss)(params, X, y, cheb_polynomials)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)

    return params, opt_state, loss


@jax.jit
def train_step_private(
    params,
    opt_state,
    X,
    y,
    cheb_polynomials,
    rng_key,
    noise_multiplier,
    max_grad_norm,
):
    batch_size = X.shape[0]

    per_sample_grad_fn = jax.vmap(
        jax.grad(single_sample_loss),
        in_axes=(None, 0, 0, None)
    )

    per_sample_grads = per_sample_grad_fn(params, X, y, cheb_polynomials)

    clipped_avg_grads, avg_grad_norm, avg_clip_fraction = clip_and_average_per_sample_grads(
        per_sample_grads,
        max_grad_norm,
    )

    rng_key, noise_key = jax.random.split(rng_key)

    noised_grads = add_gaussian_noise_to_grads(
        clipped_avg_grads,
        noise_key,
        noise_multiplier,
        max_grad_norm,
        batch_size,
    )

    updates, opt_state = optimizer.update(noised_grads, opt_state, params)
    params = optax.apply_updates(params, updates)

    loss = mse_loss(params, X, y, cheb_polynomials)

    return params, opt_state, rng_key, loss, avg_grad_norm, avg_clip_fraction


@jax.jit
def eval_step(params, X, y, cheb_polynomials):
    pred = model_forward(params, X, cheb_polynomials)
    return compute_metrics(y, pred)


def dataloader_to_jax(loader):
    xs = []
    ys = []

    for batch_x, batch_y in loader:
        xs.append(batch_x.detach().cpu().numpy())
        ys.append(batch_y.detach().cpu().numpy())

    X = jnp.array(np.concatenate(xs, axis=0))
    y = jnp.array(np.concatenate(ys, axis=0))

    return X, y


def rough_epsilon_estimate(sample_rate, noise_multiplier, steps, delta):
    if noise_multiplier <= 0:
        return np.inf

    return (sample_rate * np.sqrt(2.0 * steps * np.log(1.0 / delta))) / noise_multiplier


def main():
    print("JAX devices:", jax.devices())
    print("JAX backend:", jax.default_backend())

    config_path = "data/m_2_3_y_2019_optimals_check/config_tlc_nyc_inflow_optimal_c90.yaml"
    config_dict = read_yaml(config_path)

    run_mode = os.environ.get("JAX_RUN_MODE", "debug").lower()
    if run_mode == "debug":
        # CPU debug mode on laptop
        # Keep this small so PyCharm/Mac CPU can test the code quickly.
        config_dict["epochs"] = 1
        config_dict["batch_size"] = 4
    else:
        print("JAX_RUN_MODE=full: using original YAML/paper experiment settings.")

    config = yaml_to_config(**config_dict["parameters"])

    if run_mode == "paper_test":
        config.epochs = int(os.environ.get("JAX_PAPER_TEST_EPOCHS", "2"))
        print(f"JAX_RUN_MODE=paper_test: using paper data/settings with epochs={config.epochs}.")


    # -----------------------------------------------------
    # Training settings
    # -----------------------------------------------------
    if run_mode == "debug":
        config.epochs = 1

    # DP-SGD is memory-heavy because it computes per-sample gradients.
    # Therefore, use smaller batch size for this first private run.
    if run_mode == "debug":
        config.batch_size = 4
    config.device = "gpu"
    config.shuffle = 1

    # -----------------------------------------------------
    # DP-SGD settings
    # -----------------------------------------------------
    use_private_training = True

    noise_multiplier = 1.0
    max_grad_norm = 1.0
    delta = 1e-5

    print("\nDP-SGD settings")
    print("---------------")
    print("use_private_training:", use_private_training)
    print("noise_multiplier sigma:", noise_multiplier)
    print("max_grad_norm:", max_grad_norm)
    print("delta:", delta)
    print("Exact epsilon accounting: not included yet")
    print("This run verifies the JAX DP-SGD mechanism.\n")

    device = "cpu"

    data_params_list = get_data_params(config, device)

    train_loader, train_data_list, val_loader_list, test_loader_list, static_edge_index_list, \
        node_features_list, n_nodes_list, pretrained_models_names, hidden_warmup, normalization_param_dict = \
        get_datasets(data_params_list, device)

    train_x_torch = train_data_list[0][0]
    train_y_torch = train_data_list[0][1]

    train_x = jnp.array(train_x_torch.detach().cpu().numpy())
    train_y = jnp.array(train_y_torch.detach().cpu().numpy())

    val_x, val_y = dataloader_to_jax(val_loader_list[0])
    test_x, test_y = dataloader_to_jax(test_loader_list[0])

    n_nodes = train_x.shape[1]
    in_channels = train_x.shape[2]
    input_periods = train_x.shape[3]
    output_periods = train_y.shape[2]

    print("JAX train_x shape:", train_x.shape)
    print("JAX train_y shape:", train_y.shape)
    print("JAX val_x shape:", val_x.shape)
    print("JAX val_y shape:", val_y.shape)
    print("JAX test_x shape:", test_x.shape)
    print("JAX test_y shape:", test_y.shape)

    print("n_nodes:", n_nodes)
    print("in_channels:", in_channels)
    print("input_periods:", input_periods)
    print("output_periods:", output_periods)

    edge_index = static_edge_index_list[0]
    adj_norm = edge_index_to_adjacency(edge_index, n_nodes)

    K = 3
    cheb_polynomials = compute_chebyshev_polynomials(adj_norm, K)

    print("Adjacency shape:", adj_norm.shape)
    print("Chebyshev polynomials shape:", cheb_polynomials.shape)

    nb_block = 2
    nb_chev_filter = 64
    nb_time_filter = 64

    print("nb_block:", nb_block)
    print("nb_chev_filter:", nb_chev_filter)
    print("nb_time_filter:", nb_time_filter)

    key = jax.random.PRNGKey(5)
    rng_key = jax.random.PRNGKey(123)

    params = init_model_params(
        key=key,
        nb_block=nb_block,
        in_channels=in_channels,
        n_nodes=n_nodes,
        input_periods=input_periods,
        output_periods=output_periods,
        K=K,
        nb_chev_filter=nb_chev_filter,
        nb_time_filter=nb_time_filter,
    )

    opt_state = optimizer.init(params)

    batch_size = config.batch_size
    n_samples = train_x.shape[0]
    steps_per_epoch = int(np.ceil(n_samples / batch_size))
    total_steps = steps_per_epoch * config.epochs
    sample_rate = batch_size / n_samples

    print("\nDP sampling information")
    print("-----------------------")
    print("n_samples:", n_samples)
    print("batch_size:", batch_size)
    print("steps_per_epoch:", steps_per_epoch)
    print("total_steps:", total_steps)
    print("sample_rate:", sample_rate)
    accountant = RDPAccountant()

    print("Privacy accountant: Opacus RDPAccountant")
    print("Initial epsilon:", accountant.get_epsilon(delta=delta))
    print()

    for epoch in range(config.epochs):
        losses = []
        grad_norms = []
        clip_fractions = []

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)

            batch_x = train_x[start:end]
            batch_y = train_y[start:end]

            if use_private_training:
                params, opt_state, rng_key, loss, avg_grad_norm, avg_clip_fraction = train_step_private(
                    params,
                    opt_state,
                    batch_x,
                    batch_y,
                    cheb_polynomials,
                    rng_key,
                    noise_multiplier,
                    max_grad_norm,
                )

                accountant.step(
                    noise_multiplier=float(noise_multiplier),
                    sample_rate=float(sample_rate),
                )

                grad_norms.append(float(avg_grad_norm))
                clip_fractions.append(float(avg_clip_fraction))

            else:
                params, opt_state, loss = train_step_nonprivate(
                    params,
                    opt_state,
                    batch_x,
                    batch_y,
                    cheb_polynomials,
                )

            losses.append(float(loss))

        val_mse, val_rmse, val_mae, val_mape = eval_step(
            params, val_x, val_y, cheb_polynomials
        )

        if use_private_training:
            epsilon = accountant.get_epsilon(delta=delta)

            print(
                f"Epoch {epoch+1:03d}/{config.epochs}, "
                f"train MSE = {np.mean(losses):.6f}, "
                f"val MSE = {float(val_mse):.6f}, "
                f"val RMSE = {float(val_rmse):.6f}, "
                f"val MAE = {float(val_mae):.6f}, "
                f"avg grad norm = {np.mean(grad_norms):.6f}, "
                f"clip fraction = {np.mean(clip_fractions):.4f}, "
                f"epsilon = {epsilon:.4f}, "
                f"delta = {delta}"
            )
        else:
            print(
                f"Epoch {epoch+1:03d}/{config.epochs}, "
                f"train MSE = {np.mean(losses):.6f}, "
                f"val MSE = {float(val_mse):.6f}, "
                f"val RMSE = {float(val_rmse):.6f}, "
                f"val MAE = {float(val_mae):.6f}"
            )

    test_mse, test_rmse, test_mae, test_mape = eval_step(
        params, test_x, test_y, cheb_polynomials
    )

    print("\nFinal test metrics")
    print("------------------")
    print("Final test MSE :", float(test_mse))
    print("Final test RMSE:", float(test_rmse))
    print("Final test MAE :", float(test_mae))
    print("Final test MAPE:", float(test_mape), "%")

    if use_private_training:
        final_epsilon = accountant.get_epsilon(delta=delta)
        print("\nFinal privacy accounting")
        print("------------------------")
        print("Final epsilon:", final_epsilon)
        print("Final delta:", delta)
        print("Noise multiplier sigma:", noise_multiplier)
        print("Max grad norm:", max_grad_norm)

    print("\nJAX Customized_ASTGCN private DP-SGD training with RDP accountant finished successfully.")


if __name__ == "__main__":
    main()
