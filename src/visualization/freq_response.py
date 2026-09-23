import numpy as np
import torch
import matplotlib.pyplot as plt


def make_3d_sinusoid(
    frequency,
    size=(128, 128, 128),
    axis="x",
    amplitude=1.0,
    phase=0.0,
    device="cuda",
):
    """
    Create a normalized 3D sinusoid.

    Args:
        frequency: Spatial frequency in cycles per volume.
        size: Volume size as (D, H, W).
        axis: Direction of the sinusoid: "x", "y", or "z".
        amplitude: Signal amplitude.
        phase: Phase offset in radians.
        device: Torch device.

    Returns:
        Tensor of shape [1, 1, D, H, W].
    """

    D, H, W = size

    if axis == "x":
        coord = torch.arange(
            W,
            device=device,
            dtype=torch.float32,
        )

        signal = torch.sin(
            2 * np.pi * frequency * coord / W
            + phase
        )

        signal = signal.view(1, 1, 1, 1, W)

        signal = signal.expand(
            1, 1, D, H, W
        )

    elif axis == "y":
        coord = torch.arange(
            H,
            device=device,
            dtype=torch.float32,
        )

        signal = torch.sin(
            2 * np.pi * frequency * coord / H
            + phase
        )

        signal = signal.view(1, 1, 1, H, 1)

        signal = signal.expand(
            1, 1, D, H, W
        )

    elif axis == "z":
        coord = torch.arange(
            D,
            device=device,
            dtype=torch.float32,
        )

        signal = torch.sin(
            2 * np.pi * frequency * coord / D
            + phase
        )

        signal = signal.view(1, 1, D, 1, 1)

        signal = signal.expand(
            1, 1, D, H, W
        )

    else:
        raise ValueError(
            "axis must be 'x', 'y', or 'z'"
        )

    return amplitude * signal


@torch.no_grad()
def get_routing_weights(
    model,
    x,
    device,
):
    """
    Run a volume through the autoencoder and
    return decoder routing weights.

    Returns:
        Routing weights of shape [B, K, D, H, W].
    """

    model.eval()

    x = x.to(
        device,
        non_blocking=True,
    )

    z, _ = model.encode(x)

    (
        _,
        weights,
        _,
    ) = model.decode(
        z,
        return_weights=True,
        return_expert_features=True,
    )

    if weights is None:
        raise RuntimeError(
            "Decoder did not return routing weights."
        )

    return weights


@torch.no_grad()
def measure_frequency_routing(
    model,
    frequencies,
    size=(128, 128, 128),
    axis="x",
    amplitude=1.0,
    phase=0.0,
    device="cuda",
):
    """
    Measure the mean decoder routing weight for
    each expert as a function of spatial frequency.

    Returns:
        NumPy array of shape
        [num_frequencies, num_experts].
    """

    model.eval()

    results = []

    for frequency in frequencies:

        x = make_3d_sinusoid(
            frequency=frequency,
            size=size,
            axis=axis,
            amplitude=amplitude,
            phase=phase,
            device=device,
        )

        weights = get_routing_weights(
            model=model,
            x=x,
            device=device,
        )

        # [B, K, D, H, W] -> [K]
        mean_weights = weights.mean(
            dim=(0, 2, 3, 4)
        )

        results.append(
            mean_weights.cpu().numpy()
        )

    return np.stack(
        results,
        axis=0,
    )


@torch.no_grad()
def measure_frequency_response_and_routing(
    model,
    frequencies,
    size=(128, 128, 128),
    axis="x",
    amplitude=1.0,
    phase=0.0,
    device="cuda",
):
    """
    Measure both decoder routing weights and
    unrouted expert response magnitudes as a
    function of spatial frequency.

    Routing:
        Mean routing weight for each expert.

    Unrouted response:
        Mean magnitude of each expert feature
        before routing.

    Args:
        model: Trained autoencoder.
        frequencies: Iterable of spatial frequencies.
        size: Volume size as (D, H, W).
        axis: Direction of the sinusoid.
        amplitude: Signal amplitude.
        phase: Phase offset in radians.
        device: Torch device.

    Returns:
        routing:
            NumPy array of shape
            [num_frequencies, num_experts].

        response:
            NumPy array of shape
            [num_frequencies, num_experts].
    """

    model.eval()

    routing_results = []
    response_results = []

    for frequency in frequencies:

        x = make_3d_sinusoid(
            frequency=frequency,
            size=size,
            axis=axis,
            amplitude=amplitude,
            phase=phase,
            device=device,
        )

        x = x.to(
            device,
            non_blocking=True,
        )

        # --------------------------------------------------
        # Encode
        # --------------------------------------------------

        z, _ = model.encode(x)

        # --------------------------------------------------
        # Decode and retrieve expert features + routing
        # --------------------------------------------------

        (
            _,
            weights,
            expert_features,
        ) = model.decode(
            z,
            return_weights=True,
            return_expert_features=True,
        )

        if weights is None:
            raise RuntimeError(
                "Decoder did not return routing weights."
            )

        if expert_features is None:
            raise RuntimeError(
                "Decoder did not return expert features."
            )

        # --------------------------------------------------
        # Routing
        #
        # weights: [B, K, D, H, W]
        # --------------------------------------------------

        mean_weights = weights.mean(
            dim=(0, 2, 3, 4)
        )

        # --------------------------------------------------
        # Unrouted expert response
        #
        # expert_features: [B, K, C, D, H, W]
        #
        # Compute RMS magnitude over channels.
        # --------------------------------------------------

        response_maps = torch.sqrt(
            torch.mean(
                expert_features ** 2,
                dim=2,
            )
            + 1e-8
        )

        # [B, K, D, H, W] -> [K]
        mean_response = response_maps.mean(
            dim=(0, 2, 3, 4)
        )

        routing_results.append(
            mean_weights.cpu().numpy()
        )

        response_results.append(
            mean_response.cpu().numpy()
        )

    routing = np.stack(
        routing_results,
        axis=0,
    )

    response = np.stack(
        response_results,
        axis=0,
    )

    return routing, response


def plot_frequency_routing(
    frequencies,
    routing,
    title=None,
    expert_labels=None,
):
    """
    Plot mean routing weight against spatial frequency.
    """

    K = routing.shape[1]

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

    plt.figure(
        figsize=(8, 5)
    )

    for k in range(K):

        plt.plot(
            frequencies,
            routing[:, k],
            marker="o",
            label=expert_labels[k],
        )

    plt.xlabel(
        "Spatial frequency (cycles/volume)"
    )

    plt.ylabel(
        "Mean routing weight"
    )

    if title is not None:
        plt.title(title)

    plt.ylim(0.1, 0.4)

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()
    plt.show()


def plot_frequency_response(
    frequencies,
    response,
    title=None,
    expert_labels=None,
):
    """
    Plot unrouted expert response magnitude
    against spatial frequency.

    Args:
        frequencies: Spatial frequencies.
        response: Array of shape
            [num_frequencies, num_experts].
        title: Optional plot title.
        expert_labels: Optional expert labels.
    """

    K = response.shape[1]

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

    plt.figure(
        figsize=(8, 5)
    )

    for k in range(K):

        plt.plot(
            frequencies,
            response[:, k],
            marker="o",
            label=expert_labels[k],
        )

    plt.xlabel(
        "Spatial frequency (cycles/volume)"
    )

    plt.ylabel(
        "Mean expert response magnitude"
    )

    if title is not None:
        plt.title(title)

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()
    plt.show()


def routing_distribution_shift(
    routing,
):
    """
    Compute the L1 shift in routing distribution
    relative to the first frequency.

    Args:
        routing:
            Array of shape
            [num_frequencies, num_experts].

    Returns:
        Array of shape [num_frequencies].
    """

    reference = routing[0]

    shift = np.sum(
        np.abs(
            routing - reference[None, :]
        ),
        axis=1,
    )

    return shift


def plot_routing_distribution_shift(
    frequencies,
    shift,
    title=None,
):
    """
    Plot L1 routing distribution shift
    against spatial frequency.
    """

    plt.figure(
        figsize=(7, 4)
    )

    plt.plot(
        frequencies,
        shift,
        marker="o",
    )

    plt.xlabel(
        "Spatial frequency"
    )

    plt.ylabel(
        r"$L_1$ routing distribution shift"
    )

    if title is not None:
        plt.title(title)

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()
    plt.show()

# from src.visualization.freq_response import (
#     measure_frequency_response_and_routing,
#     plot_frequency_response,
#     plot_frequency_routing,
#     routing_distribution_shift,
#     plot_routing_distribution_shift,
# )


# frequencies = [
#     0.5,
#     1,
#     2,
#     4,
#     8,
#     12,
#     16,
#     24,
#     32,
# ]


# routing, response = measure_frequency_response_and_routing(
#     model=ae,
#     frequencies=frequencies,
#     size=(128, 128, 128),
#     axis="x",
#     amplitude=1.0,
#     phase=0.0,
#     device=device,
# )


# # Unrouted expert response
# plot_frequency_response(
#     frequencies,
#     response,
#     title="Unrouted Expert Response vs. Spatial Frequency",
# )


# # Routing weights
# plot_frequency_routing(
#     frequencies,
#     routing,
#     title="Routing Weight vs. Spatial Frequency",
# )


# # Routing distribution shift
# shift_x = routing_distribution_shift(
#     routing
# )

# plot_routing_distribution_shift(
#     frequencies,
#     shift_x,
# )
