# src/visualization/expert_analysis.py

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

from .routing import (
    _get_expert_names,
    _get_display_indices,
    _format_labels,
    DECODER_DISPLAY_ORDER,
    DECODER_LABELS,
)


# ---------------------------------------------------------------------
# Expert response maps
# ---------------------------------------------------------------------

@torch.no_grad()
def expert_response_maps(expert_features):
    """
    Compute the magnitude of each expert's feature response.

    Args:
        expert_features:
            Tensor of shape [B, K, C, D, H, W].

    Returns:
        response_maps:
            Tensor of shape [B, K, D, H, W].
    """

    if expert_features.ndim != 6:
        raise ValueError(
            "Expected expert_features with shape "
            "[B, K, C, D, H, W], "
            f"got {tuple(expert_features.shape)}"
        )

    response_maps = torch.sqrt(
        torch.mean(
            expert_features ** 2,
            dim=2,
        )
        + 1e-8
    )

    return response_maps


# ---------------------------------------------------------------------
# Pairwise expert correlations
# ---------------------------------------------------------------------

@torch.no_grad()
def pairwise_expert_correlations(response_maps):
    """
    Compute pairwise Pearson-style correlations between expert
    response maps.

    Args:
        response_maps:
            Tensor of shape [B, K, D, H, W].

    Returns:
        corr_matrix:
            Tensor of shape [K, K].
    """

    if response_maps.ndim != 5:
        raise ValueError(
            "Expected response_maps with shape "
            "[B, K, D, H, W], "
            f"got {tuple(response_maps.shape)}"
        )

    B, K, D, H, W = response_maps.shape

    # ---------------------------------------------------------------
    # Flatten samples and spatial locations
    # ---------------------------------------------------------------

    x = response_maps.permute(
        1, 0, 2, 3, 4
    )

    x = x.reshape(
        K,
        -1,
    )

    # ---------------------------------------------------------------
    # Center each expert
    # ---------------------------------------------------------------

    x = x - x.mean(
        dim=1,
        keepdim=True,
    )

    # ---------------------------------------------------------------
    # Normalize each expert
    # ---------------------------------------------------------------

    x = x / (
        torch.sqrt(
            torch.sum(
                x ** 2,
                dim=1,
                keepdim=True,
            )
        )
        + 1e-8
    )

    # ---------------------------------------------------------------
    # Correlation matrix
    # ---------------------------------------------------------------

    corr = x @ x.T

    return corr


# ---------------------------------------------------------------------
# Expert response analysis
# ---------------------------------------------------------------------

@torch.no_grad()
def analyze_expert_response(
    model,
    loader,
    device,
):
    """
    Analyze response similarity between decoder experts over a dataset.

    The decoder is expected to support:

        return_weights=True
        return_expert_features=True

    Args:
        model:
            Autoencoder model.

        loader:
            DataLoader containing the evaluation dataset.

        device:
            Torch device.

    Returns:
        Dictionary containing:
            response_maps
            correlation
            expert_names
            expert_labels
    """

    model.eval()

    all_response_maps = []

    # ---------------------------------------------------------------
    # Determine decoder expert ordering
    # ---------------------------------------------------------------

    decoder_names_model = _get_expert_names(
        model.dec2.conv_suite
    )

    decoder_indices, decoder_names = _get_display_indices(
        model_names=decoder_names_model,
        display_order=DECODER_DISPLAY_ORDER,
        module_name="Decoder",
    )

    decoder_labels = _format_labels(
        decoder_names,
        DECODER_LABELS,
    )

    # ---------------------------------------------------------------
    # Dataset loop
    # ---------------------------------------------------------------

    pbar = tqdm(
        loader,
        desc="Analyzing expert responses",
    )

    for batch in pbar:

        if isinstance(batch, (list, tuple)):
            x = batch[0]
        else:
            x = batch

        x = x.to(
            device,
            non_blocking=True,
        )

        # -----------------------------------------------------------
        # Encode
        # -----------------------------------------------------------

        z, _ = model.encode(x)

        # -----------------------------------------------------------
        # Decode and return expert features
        # -----------------------------------------------------------

        reconstruction, weights, expert_features = (
            model.decode(
                z,
                return_weights=True,
                return_expert_features=True,
            )
        )

        if expert_features is None:
            raise RuntimeError(
                "Decoder did not return expert features. "
                "Make sure `return_expert_features=True` is "
                "implemented in DecoderBlock."
            )

        # -----------------------------------------------------------
        # Reorder experts to match visualization convention
        # -----------------------------------------------------------

        expert_features = expert_features[
            :,
            decoder_indices,
        ]

        # -----------------------------------------------------------
        # Compute response magnitude
        # -----------------------------------------------------------

        response_maps = expert_response_maps(
            expert_features
        )

        all_response_maps.append(
            response_maps.cpu()
        )

    if not all_response_maps:
        raise RuntimeError(
            "No batches were processed by the DataLoader."
        )

    # ---------------------------------------------------------------
    # Combine dataset
    # ---------------------------------------------------------------

    response_maps = torch.cat(
        all_response_maps,
        dim=0,
    )

    # ---------------------------------------------------------------
    # Pairwise correlations
    # ---------------------------------------------------------------

    correlation = pairwise_expert_correlations(
        response_maps
    )

    return {
        "response_maps": response_maps,
        "correlation": correlation,
        "expert_names": decoder_names,
        "expert_labels": decoder_labels,
    }


# ---------------------------------------------------------------------
# Correlation visualization
# ---------------------------------------------------------------------

def plot_expert_correlation(
    corr,
    expert_labels=None,
):
    """
    Plot pairwise expert response correlations.

    Args:
        corr:
            Correlation matrix [K, K].

        expert_labels:
            Optional list of expert labels.
    """

    if isinstance(corr, torch.Tensor):
        corr = corr.detach().cpu().numpy()

    K = corr.shape[0]

    if expert_labels is None:
        expert_labels = [
            f"Expert {k}"
            for k in range(K)
        ]

    if len(expert_labels) != K:
        raise ValueError(
            "Number of labels must match the number "
            "of experts."
        )

    plt.figure(
        figsize=(6, 5)
    )

    sns.heatmap(
        corr,
        annot=True,
        fmt=".3f",
        xticklabels=expert_labels,
        yticklabels=expert_labels,
        vmin=-1,
        vmax=1,
        square=True,
        cmap="coolwarm",
    )

    plt.xlabel("Expert")
    plt.ylabel("Expert")
    plt.title(
        "Pairwise Expert Response Correlation"
    )

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------
# Response redundancy
# ---------------------------------------------------------------------

@torch.no_grad()
def response_redundancy(corr):
    """
    Compute mean absolute pairwise correlation between distinct
    experts.

    Diagonal self-correlations are excluded.

    Args:
        corr:
            Correlation matrix [K, K].

    Returns:
        Mean absolute inter-expert correlation.
    """

    if corr.ndim != 2:
        raise ValueError(
            "Expected a 2D correlation matrix."
        )

    K = corr.shape[0]

    mask = ~torch.eye(
        K,
        dtype=torch.bool,
        device=corr.device,
    )

    mean_abs_corr = torch.mean(
        torch.abs(
            corr[mask]
        )
    )

    return mean_abs_corr.item()


# ---------------------------------------------------------------------
# Routing statistics
# ---------------------------------------------------------------------

@torch.no_grad()
def routing_statistics(weights):
    """
    Compute global routing statistics.

    Args:
        weights:
            Tensor of shape [B, K, D, H, W].

    Returns:
        Dictionary containing:

            mean_usage:
                Mean routing weight for each expert.

            spatial_variance:
                Variance of routing weights for each expert.

            mean_entropy:
                Mean routing entropy.

            normalized_entropy:
                Entropy normalized by log(K).
    """

    if weights.ndim != 5:
        raise ValueError(
            "Expected weights with shape "
            "[B, K, D, H, W], "
            f"got {tuple(weights.shape)}"
        )

    B, K, D, H, W = weights.shape

    # ---------------------------------------------------------------
    # Mean expert usage
    # ---------------------------------------------------------------

    mean_usage = weights.mean(
        dim=(0, 2, 3, 4)
    )

    # ---------------------------------------------------------------
    # Spatial variance
    # ---------------------------------------------------------------

    spatial_variance = weights.var(
        dim=(0, 2, 3, 4),
        unbiased=False,
    )

    # ---------------------------------------------------------------
    # Routing entropy
    # ---------------------------------------------------------------

    entropy = -torch.sum(
        weights
        * torch.log(
            weights + 1e-8
        ),
        dim=1,
    )

    mean_entropy = entropy.mean()

    # ---------------------------------------------------------------
    # Normalize by maximum entropy
    # ---------------------------------------------------------------

    if K > 1:
        normalized_entropy = (
            mean_entropy
            / np.log(K)
        )
    else:
        normalized_entropy = torch.tensor(
            0.0,
            device=weights.device,
        )

    return {
        "mean_usage": mean_usage,
        "spatial_variance": spatial_variance,
        "mean_entropy": mean_entropy,
        "normalized_entropy": normalized_entropy,
    }


# ---------------------------------------------------------------------
# Dataset-level routing analysis
# ---------------------------------------------------------------------

@torch.no_grad()
def analyze_routing(
    model,
    loader,
    device,
):
    """
    Analyze decoder routing over an entire dataset.

    Args:
        model:
            Autoencoder model.

        loader:
            Evaluation DataLoader.

        device:
            Torch device.

    Returns:
        Dictionary containing routing weights and statistics.
    """

    model.eval()

    all_weights = []

    # ---------------------------------------------------------------
    # Decoder expert ordering
    # ---------------------------------------------------------------

    decoder_names_model = _get_expert_names(
        model.dec2.conv_suite
    )

    decoder_indices, decoder_names = _get_display_indices(
        model_names=decoder_names_model,
        display_order=DECODER_DISPLAY_ORDER,
        module_name="Decoder",
    )

    decoder_labels = _format_labels(
        decoder_names,
        DECODER_LABELS,
    )

    # ---------------------------------------------------------------
    # Dataset loop
    # ---------------------------------------------------------------

    pbar = tqdm(
        loader,
        desc="Analyzing routing",
    )

    for batch in pbar:

        if isinstance(batch, (list, tuple)):
            x = batch[0]
        else:
            x = batch

        x = x.to(
            device,
            non_blocking=True,
        )

        # -----------------------------------------------------------
        # Encode
        # -----------------------------------------------------------

        z, _ = model.encode(x)

        # -----------------------------------------------------------
        # Decode
        # -----------------------------------------------------------

        reconstruction, weights, expert_features = (
            model.decode(
                z,
                return_weights=True,
                return_expert_features=True,
            )
        )

        if weights is None:
            raise RuntimeError(
                "Decoder did not return routing weights."
            )

        # -----------------------------------------------------------
        # Reorder routing channels
        # -----------------------------------------------------------

        weights = weights[
            :,
            decoder_indices,
        ]

        all_weights.append(
            weights.cpu()
        )

    if not all_weights:
        raise RuntimeError(
            "No batches were processed by the DataLoader."
        )

    # ---------------------------------------------------------------
    # Combine dataset
    # ---------------------------------------------------------------

    weights = torch.cat(
        all_weights,
        dim=0,
    )

    # ---------------------------------------------------------------
    # Compute statistics
    # ---------------------------------------------------------------

    stats = routing_statistics(
        weights
    )

    return {
        "weights": weights,
        "mean_usage": stats["mean_usage"],
        "spatial_variance": stats["spatial_variance"],
        "mean_entropy": stats["mean_entropy"],
        "normalized_entropy": stats[
            "normalized_entropy"
        ],
        "expert_names": decoder_names,
        "expert_labels": decoder_labels,
    }


# ---------------------------------------------------------------------
# Pretty-print routing statistics
# ---------------------------------------------------------------------

def print_routing_statistics(stats):
    """
    Print routing statistics in a compact format.
    """

    labels = stats["expert_labels"]

    mean_usage = stats["mean_usage"]
    spatial_variance = stats["spatial_variance"]

    print("\nDecoder routing statistics:")

    for k, label in enumerate(labels):

        print(
            f"{label}: "
            f"usage={mean_usage[k].item():.4f} | "
            f"variance={spatial_variance[k].item():.6f}"
        )

    print(
        "\nMean routing entropy: "
        f"{stats['mean_entropy'].item():.4f}"
    )

    print(
        "Normalized routing entropy: "
        f"{stats['normalized_entropy'].item():.4f}"
    )

# ---------------------------------------------------------------------
# Structure-response correlations
# ---------------------------------------------------------------------

def plot_structure_correlations(
    structure_results,
    mode="response",
    title=None,
    expert_labels=None,
):
    """
    Plot correlations between expert responses/routed responses
    and structural properties of the input.

    Args:
        structure_results:
            Dictionary containing correlation results for each
            structural property.

            Expected structure:

                structure_results["gradient"]["response"]
                structure_results["gradient"]["routed"]

                structure_results["inplane_gradient"]["response"]
                structure_results["inplane_gradient"]["routed"]

                structure_results["depth_gradient"]["response"]
                structure_results["depth_gradient"]["routed"]

                structure_results["laplacian"]["response"]
                structure_results["laplacian"]["routed"]

            Each entry should contain correlations across test
            volumes with shape [N, K].

        mode:
            "response":
                Correlation between expert response A_k and the
                structural property S.

            "routed":
                Correlation between routed response w_k A_k and
                the structural property S.

        title:
            Optional plot title.

        expert_labels:
            Optional list of expert labels. If None, generic
            expert labels are used.

    Returns:
        matrix:
            NumPy array of shape [4, K] containing mean correlations.
    """

    # ---------------------------------------------------------------
    # Validate mode
    # ---------------------------------------------------------------

    valid_modes = {
        "response",
        "routed",
    }

    if mode not in valid_modes:
        raise ValueError(
            f"Invalid mode '{mode}'. "
            f"Expected one of {sorted(valid_modes)}."
        )

    # ---------------------------------------------------------------
    # Structural properties
    # ---------------------------------------------------------------

    structure_names = [
        "gradient",
        "inplane_gradient",
        "depth_gradient",
        "laplacian",
    ]

    structure_labels = [
        "Gradient",
        "In-plane gradient",
        "Depth gradient",
        "Laplacian",
    ]

    # ---------------------------------------------------------------
    # Build correlation matrix
    # ---------------------------------------------------------------

    matrix = []

    for name in structure_names:

        if name not in structure_results:
            raise KeyError(
                f"Missing '{name}' from structure_results."
            )

        if mode not in structure_results[name]:
            raise KeyError(
                f"Missing mode '{mode}' for "
                f"structure '{name}'."
            )

        correlations = np.asarray(
            structure_results[name][mode]
        )

        if correlations.ndim != 2:
            raise ValueError(
                f"Expected correlations for '{name}' "
                f"to have shape [N, K], "
                f"got {correlations.shape}."
            )

        # Mean across test volumes.
        mean_corr = correlations.mean(
            axis=0
        )

        matrix.append(
            mean_corr
        )

    matrix = np.stack(
        matrix,
        axis=0,
    )

    K = matrix.shape[1]

    # ---------------------------------------------------------------
    # Expert labels
    # ---------------------------------------------------------------

    if expert_labels is None:

        expert_labels = [
            f"E{k}"
            for k in range(K)
        ]

    if len(expert_labels) != K:
        raise ValueError(
            f"Expected {K} expert labels, "
            f"got {len(expert_labels)}."
        )

    # ---------------------------------------------------------------
    # Plot
    # ---------------------------------------------------------------

    plt.figure(
        figsize=(7, 5)
    )

    sns.heatmap(
        matrix,
        annot=True,
        fmt=".3f",
        vmin=-1,
        vmax=1,
        center=0,
        xticklabels=expert_labels,
        yticklabels=structure_labels,
        square=True,
        cmap="coolwarm",
    )

    plt.xlabel("Expert")
    plt.ylabel("Structural property")

    if title is not None:
        plt.title(title)
    else:
        if mode == "response":
            plt.title(
                "Expert Response–Structure Correlation"
            )
        else:
            plt.title(
                "Routed Response–Structure Correlation"
            )

    plt.tight_layout()
    plt.show()

    return matrix
    
# from src.visualization.expert_analysis import (
#     analyze_expert_response,
#     plot_expert_correlation,
#     response_redundancy,
# )

# response_stats = analyze_expert_response(
#     model,
#     test_loader,
#     device,
# )

# plot_expert_correlation(
#     response_stats["correlation"],
#     response_stats["expert_labels"],
# )

# redundancy = response_redundancy(
#     response_stats["correlation"]
# )

# print(
#     f"Mean absolute inter-expert correlation: "
#     f"{redundancy:.4f}"
# )

# from src.visualization.expert_analysis import (
#     analyze_routing,
#     print_routing_statistics,
# )

# stats = analyze_routing(
#     model,
#     test_loader,
#     device,
# )

# print_routing_statistics(stats)
