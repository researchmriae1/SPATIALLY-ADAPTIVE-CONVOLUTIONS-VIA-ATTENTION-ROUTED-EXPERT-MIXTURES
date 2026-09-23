# src/visualization/routing.py

import torch
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Expert display configuration
# ---------------------------------------------------------------------
#
# Encoder expert semantics:
#
#   low           : low-frequency / local averaging transformation
#   spatial       : in-plane spatial convolution (1, 3, 3)
#   point         : pointwise 1x1 convolution
#   identity_like : pointwise 1x1 convolution
#
# NOTE:
# "point" and "identity_like" are both pointwise learned
# transformations. They are separate expert instances with separate
# learned parameters, but they have the same convolutional form.
#
# "identity_like" is retained as the actual model name so that the
# architecture and checkpoint naming remain unchanged.
#
# The ordering below is only for visualization.
#
# ---------------------------------------------------------------------

ENCODER_DISPLAY_ORDER = [
    "low",
    "spatial",
    "point",
    "identity_like",
    "depth",
]

DECODER_DISPLAY_ORDER = [
    "low",
    "high",
    "spatial",
    "point",
    "depth",
]


ENCODER_LABELS = {
    "low": "Low-frequency",
    "spatial": "Spatial",
    "point": "Pointwise",
    "identity_like": "Identity-like",
    "depth": "Through-plane",
}


DECODER_LABELS = {
    "low": "Low-frequency",
    "high": "High-frequency",
    "spatial": "Spatial",
    "point": "Pointwise",
    "depth": "Through-plane",
}


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def _get_expert_names(module):
    """
    Retrieve expert names from a routing module.

    The model is expected to expose:
        module.expert_names

    Raises:
        AttributeError: if expert names are not available.
    """
    if not hasattr(module, "expert_names"):
        raise AttributeError(
            f"{module.__class__.__name__} does not expose "
            "`expert_names`."
        )

    return list(module.expert_names)


def _get_display_indices(
    model_names,
    display_order,
    module_name="module",
):
    """
    Convert expert names into indices according to the desired
    visualization order.

    Only experts present in the model are included.

    Raises:
        ValueError: if duplicate expert names exist.
    """
    if len(model_names) != len(set(model_names)):
        raise ValueError(
            f"{module_name} contains duplicate expert names: "
            f"{model_names}"
        )

    indices = []
    names = []

    for name in display_order:
        if name in model_names:
            indices.append(model_names.index(name))
            names.append(name)

    # Include any model experts not explicitly specified in the
    # display order at the end. This keeps the function robust to
    # future expert configurations.
    for name in model_names:
        if name not in names:
            indices.append(model_names.index(name))
            names.append(name)

    return indices, names


def _format_labels(names, label_map):
    """
    Convert internal expert names into human-readable labels.
    """
    return [
        label_map.get(name, name.replace("_", " ").title())
        for name in names
    ]


def _plot_routing_maps(
    weights,
    expert_names,
    title,
    cmap="viridis",
):
    """
    Plot routing weights for the centre depth slice.

    Args:
        weights:
            Tensor of shape [B, K, D, H, W].

        expert_names:
            Names corresponding to the K experts.

        title:
            Figure title.

        cmap:
            Matplotlib colormap.
    """
    if weights is None:
        return

    if weights.ndim != 5:
        raise ValueError(
            f"Expected routing weights with shape "
            f"[B, K, D, H, W], got {tuple(weights.shape)}"
        )

    # Use the centre slice along the depth dimension.
    slice_idx = weights.shape[2] // 2

    num_experts = weights.shape[1]

    fig, axes = plt.subplots(
        1,
        num_experts,
        figsize=(4 * num_experts, 4),
        squeeze=False,
    )

    axes = axes[0]

    for k in range(num_experts):
        ax = axes[k]

        ax.imshow(
            weights[0, k, slice_idx].detach().cpu(),
            cmap=cmap,
        )

        ax.set_title(expert_names[k])
        ax.axis("off")

        fig.colorbar(
            ax.images[0],
            ax=ax,
            fraction=0.046,
            pad=0.04,
        )

    fig.suptitle(
        f"{title} — Centre Depth Slice",
        fontsize=14,
    )

    plt.tight_layout()
    plt.show()


def _calculate_entropy(weights):
    """
    Calculate per-voxel routing entropy.

    Args:
        weights:
            Tensor of shape [B, K, D, H, W].

    Returns:
        Tensor of shape [B, D, H, W].
    """
    if weights is None:
        return None

    weights = weights.clamp_min(1e-8)

    entropy = -(
        weights * torch.log(weights)
    ).sum(dim=1)

    return entropy


# ---------------------------------------------------------------------
# Main visualization function
# ---------------------------------------------------------------------

@torch.no_grad()
def visualize_routing(
    ae,
    x,
    plot_encoder=True,
    plot_decoder=True,
    print_shapes=True,
):
    """
    Visualize encoder and decoder routing weights.

    Args:
        ae:
            AutoEncoder model.

        x:
            Input MRI tensor. The first sample is visualized.

        plot_encoder:
            Whether to plot encoder routing maps.

        plot_decoder:
            Whether to plot decoder routing maps.

        print_shapes:
            Whether to print tensor shapes.

    Returns:
        Dictionary containing:
            z
            w_enc
            w_dec
            encoder_entropy
            decoder_entropy
            output
    """

    # ---------------------------------------------------------------
    # Model setup
    # ---------------------------------------------------------------

    ae.eval()

    device = next(ae.parameters()).device

    # Only visualize the first sample.
    x = x[:1].to(device)

    # ---------------------------------------------------------------
    # Forward pass
    # ---------------------------------------------------------------

    z, w_enc = ae.encode(x)

    out, w_dec = ae.decode(
        z,
        return_weights=True,
    )

    # ---------------------------------------------------------------
    # Move outputs to CPU
    # ---------------------------------------------------------------

    w_enc = (
        w_enc.detach().cpu()
        if w_enc is not None
        else None
    )

    w_dec = (
        w_dec.detach().cpu()
        if w_dec is not None
        else None
    )

    z_cpu = z.detach().cpu()
    out_cpu = out.detach().cpu()

    # ---------------------------------------------------------------
    # Print shapes
    # ---------------------------------------------------------------

    if print_shapes:
        print("Input shape:", tuple(x.shape))
        print("Latent shape:", tuple(z.shape))

        if w_enc is not None:
            print(
                "Encoder routing shape:",
                tuple(w_enc.shape),
            )

        else:
            print("Encoder routing shape: None")

        if w_dec is not None:
            print(
                "Decoder routing shape:",
                tuple(w_dec.shape),
            )

        else:
            print("Decoder routing shape: None")

    # ---------------------------------------------------------------
    # Retrieve model expert names
    # ---------------------------------------------------------------

    encoder_names_model = _get_expert_names(
        ae.enc2.conv_suite
    )

    decoder_names_model = _get_expert_names(
        ae.dec2.conv_suite
    )

    # ---------------------------------------------------------------
    # Determine visualization ordering
    # ---------------------------------------------------------------

    encoder_indices, encoder_names = _get_display_indices(
        model_names=encoder_names_model,
        display_order=ENCODER_DISPLAY_ORDER,
        module_name="Encoder",
    )

    decoder_indices, decoder_names = _get_display_indices(
        model_names=decoder_names_model,
        display_order=DECODER_DISPLAY_ORDER,
        module_name="Decoder",
    )

    # ---------------------------------------------------------------
    # Reorder routing channels
    # ---------------------------------------------------------------

    w_enc_plot = None

    if w_enc is not None:

        if w_enc.shape[1] != len(encoder_names_model):
            raise ValueError(
                "Number of encoder routing channels does not match "
                "the number of encoder experts.\n"
                f"Routing channels: {w_enc.shape[1]}\n"
                f"Experts: {encoder_names_model}"
            )

        w_enc_plot = w_enc[:, encoder_indices]

    w_dec_plot = None

    if w_dec is not None:

        if w_dec.shape[1] != len(decoder_names_model):
            raise ValueError(
                "Number of decoder routing channels does not match "
                "the number of decoder experts.\n"
                f"Routing channels: {w_dec.shape[1]}\n"
                f"Experts: {decoder_names_model}"
            )

        w_dec_plot = w_dec[:, decoder_indices]

    # ---------------------------------------------------------------
    # Human-readable labels
    # ---------------------------------------------------------------

    encoder_labels = _format_labels(
        encoder_names,
        ENCODER_LABELS,
    )

    decoder_labels = _format_labels(
        decoder_names,
        DECODER_LABELS,
    )

    # ---------------------------------------------------------------
    # Print expert ordering
    # ---------------------------------------------------------------

    if print_shapes:
        print("\nEncoder experts:")
        for i, (name, label) in enumerate(
            zip(encoder_names, encoder_labels)
        ):
            print(
                f"  {i}: {name} -> {label}"
            )

        print("\nDecoder experts:")
        for i, (name, label) in enumerate(
            zip(decoder_names, decoder_labels)
        ):
            print(
                f"  {i}: {name} -> {label}"
            )

    # ---------------------------------------------------------------
    # Plot encoder routing
    # ---------------------------------------------------------------

    if plot_encoder and w_enc_plot is not None:

        _plot_routing_maps(
            weights=w_enc_plot,
            expert_names=encoder_labels,
            title="Encoder Routing",
        )

    # ---------------------------------------------------------------
    # Plot decoder routing
    # ---------------------------------------------------------------

    if plot_decoder and w_dec_plot is not None:

        _plot_routing_maps(
            weights=w_dec_plot,
            expert_names=decoder_labels,
            title="Decoder Routing",
        )

    # ---------------------------------------------------------------
    # Entropy
    # ---------------------------------------------------------------

    encoder_entropy = _calculate_entropy(w_enc)

    decoder_entropy = _calculate_entropy(w_dec)

    # ---------------------------------------------------------------
    # Print entropy statistics
    # ---------------------------------------------------------------

    if print_shapes:

        if encoder_entropy is not None:
            print("\nEncoder routing entropy:")
            print(
                f"  Mean: {encoder_entropy.mean().item():.6f}"
            )
            print(
                f"  Std:  {encoder_entropy.std().item():.6f}"
            )
            print(
                f"  Min:  {encoder_entropy.min().item():.6f}"
            )
            print(
                f"  Max:  {encoder_entropy.max().item():.6f}"
            )

        if decoder_entropy is not None:
            print("\nDecoder routing entropy:")
            print(
                f"  Mean: {decoder_entropy.mean().item():.6f}"
            )
            print(
                f"  Std:  {decoder_entropy.std().item():.6f}"
            )
            print(
                f"  Min:  {decoder_entropy.min().item():.6f}"
            )
            print(
                f"  Max:  {decoder_entropy.max().item():.6f}"
            )

    # ---------------------------------------------------------------
    # Return results
    # ---------------------------------------------------------------

    return {
        "z": z_cpu,
        "w_enc": w_enc,
        "w_dec": w_dec,
        "w_enc_plot": w_enc_plot,
        "w_dec_plot": w_dec_plot,
        "encoder_entropy": encoder_entropy,
        "decoder_entropy": decoder_entropy,
        "output": out_cpu,
        "encoder_expert_names": encoder_names,
        "decoder_expert_names": decoder_names,
        "encoder_labels": encoder_labels,
        "decoder_labels": decoder_labels,
    }
