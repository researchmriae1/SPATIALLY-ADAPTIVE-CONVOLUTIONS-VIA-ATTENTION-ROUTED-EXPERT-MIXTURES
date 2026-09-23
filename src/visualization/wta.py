# ---------------------------------------------------------------------
# Winner-takes-all routing
# ---------------------------------------------------------------------

@torch.no_grad()
def _get_decoder_routing(ae, x):
    """
    Run the autoencoder and return decoder routing weights.

    Returns:
        z:
            Latent representation.

        w_enc:
            Encoder routing weights, if available.

        w_dec:
            Decoder routing weights.

        out:
            Reconstruction.
    """
    ae.eval()

    device = next(ae.parameters()).device
    x = x[:1].to(device)

    z, w_enc = ae.encode(x)

    out, w_dec = ae.decode(
        z,
        return_weights=True,
    )

    if w_dec is None:
        raise RuntimeError(
            "Decoder routing weights are None. "
            "Make sure the decoder routing block is configured "
            "to return weights."
        )

    return (
        z.detach().cpu(),
        w_enc.detach().cpu() if w_enc is not None else None,
        w_dec.detach().cpu(),
        out.detach().cpu(),
    )


def _get_decoder_display_info(ae):
    """
    Get decoder expert indices and human-readable labels using the
    same ordering as the routing visualization.

    Returns:
        indices:
            Indices mapping model expert order -> display order.

        names:
            Internal expert names in display order.

        labels:
            Human-readable labels in display order.
    """
    decoder_names_model = _get_expert_names(
        ae.dec2.conv_suite
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

    return (
        decoder_indices,
        decoder_names,
        decoder_labels,
    )


# ---------------------------------------------------------------------
# Winner-takes-all map
# ---------------------------------------------------------------------

@torch.no_grad()
def visualize_winner_takes_all(
    ae,
    x,
    slice_idx=None,
):
    """
    Visualize the winner-takes-all decoder routing map for one
    axial slice.

    Each voxel is assigned to the decoder expert with the highest
    routing weight.

    Args:
        ae:
            AutoEncoder model.

        x:
            Input MRI tensor.

        slice_idx:
            Axial/depth slice to visualize. If None, the centre
            slice is used.

    Returns:
        Dictionary containing:
            z
            w_enc
            w_dec
            winner_map
            output
    """

    # ---------------------------------------------------------------
    # Forward pass
    # ---------------------------------------------------------------

    z, w_enc, w_dec, out = _get_decoder_routing(
        ae,
        x,
    )

    print("z shape:", tuple(z.shape))
    print("w_dec shape:", tuple(w_dec.shape))

    # ---------------------------------------------------------------
    # Decoder expert information
    # ---------------------------------------------------------------

    decoder_indices, decoder_names, decoder_labels = (
        _get_decoder_display_info(ae)
    )

    # Reorder routing channels to match visualization order.
    w_dec_plot = w_dec[:, decoder_indices]

    # ---------------------------------------------------------------
    # Winner-takes-all
    # ---------------------------------------------------------------

    winner_map = w_dec_plot.argmax(dim=1)
    # Shape: [B, D, H, W]

    if slice_idx is None:
        slice_idx = winner_map.shape[1] // 2

    if not 0 <= slice_idx < winner_map.shape[1]:
        raise ValueError(
            f"slice_idx={slice_idx} is outside the valid range "
            f"[0, {winner_map.shape[1] - 1}]"
        )

    winner_slice = winner_map[0, slice_idx]

    # ---------------------------------------------------------------
    # Visualize
    # ---------------------------------------------------------------

    K = w_dec_plot.shape[1]

    plt.figure(figsize=(6, 6))

    plt.imshow(
        winner_slice,
        cmap="Set2",
        vmin=0,
        vmax=K - 1,
        interpolation="nearest",
    )

    cbar = plt.colorbar(
        ticks=np.arange(K)
    )

    cbar.ax.set_yticklabels(
        decoder_labels
    )

    plt.title(
        "Decoder Winner-Takes-All Routing "
        f"(Axial Slice {slice_idx})"
    )

    plt.xlabel("Width")
    plt.ylabel("Height")

    plt.tight_layout()
    plt.show()

    # ---------------------------------------------------------------
    # Winner statistics
    # ---------------------------------------------------------------

    total = winner_map.numel()

    print("\nWinner distribution:")

    for k, label in enumerate(decoder_labels):

        count = (
            winner_map == k
        ).sum().item()

        percentage = (
            100.0 * count / total
        )

        print(
            f"{label}: "
            f"{count:,} locations "
            f"({percentage:.2f}%)"
        )

    # ---------------------------------------------------------------
    # Return
    # ---------------------------------------------------------------

    return {
        "z": z,
        "w_enc": w_enc,
        "w_dec": w_dec,
        "w_dec_plot": w_dec_plot,
        "winner_map": winner_map,
        "decoder_expert_names": decoder_names,
        "decoder_labels": decoder_labels,
        "output": out,
    }


# ---------------------------------------------------------------------
# Winner distribution
# ---------------------------------------------------------------------

@torch.no_grad()
def plot_winner_distribution(
    ae,
    x,
):
    """
    Plot the global winner-takes-all distribution of decoder experts.

    For every voxel, the expert with the highest routing weight is
    selected as the winner. The plot shows the percentage of voxels
    assigned to each expert.
    """

    # ---------------------------------------------------------------
    # Forward pass
    # ---------------------------------------------------------------

    z, w_enc, w_dec, out = _get_decoder_routing(
        ae,
        x,
    )

    print("z shape:", tuple(z.shape))
    print("w_dec shape:", tuple(w_dec.shape))

    # ---------------------------------------------------------------
    # Decoder expert information
    # ---------------------------------------------------------------

    decoder_indices, decoder_names, decoder_labels = (
        _get_decoder_display_info(ae)
    )

    # Reorder routing channels to match visualization order.
    w_dec_plot = w_dec[:, decoder_indices]

    # ---------------------------------------------------------------
    # Winner-takes-all
    # ---------------------------------------------------------------

    winner_map = w_dec_plot.argmax(dim=1)
    # Shape: [B, D, H, W]

    K = w_dec_plot.shape[1]

    # ---------------------------------------------------------------
    # Count winners
    # ---------------------------------------------------------------

    counts = torch.zeros(
        K,
        dtype=torch.long,
    )

    for k in range(K):
        counts[k] = (
            winner_map == k
        ).sum()

    total = counts.sum()

    percentages = (
        100.0
        * counts.float()
        / total.float()
    )

    # ---------------------------------------------------------------
    # Print statistics
    # ---------------------------------------------------------------

    print("\nWinner distribution:")

    for k, label in enumerate(decoder_labels):

        print(
            f"{label}: "
            f"{counts[k].item():,} locations "
            f"({percentages[k].item():.2f}%)"
        )

    # ---------------------------------------------------------------
    # Bar plot
    # ---------------------------------------------------------------

    percentage_values = percentages.numpy()

    plt.figure(figsize=(8, 5))

    bars = plt.bar(
        np.arange(K),
        percentage_values,
    )

    plt.xticks(
        np.arange(K),
        decoder_labels,
        rotation=20,
        ha="right",
    )

    plt.ylabel("Winner Frequency (%)")
    plt.xlabel("Expert")
    plt.title(
        "Decoder Winner-Takes-All Expert Distribution"
    )

    # ---------------------------------------------------------------
    # Percentage labels
    # ---------------------------------------------------------------

    for bar, percentage in zip(
        bars,
        percentage_values,
    ):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{percentage:.2f}%",
            ha="center",
            va="bottom",
        )

    max_percentage = percentage_values.max()

    plt.ylim(
        0,
        max_percentage * 1.15
        if max_percentage > 0
        else 1,
    )

    plt.tight_layout()
    plt.show()

    # ---------------------------------------------------------------
    # Return
    # ---------------------------------------------------------------

    return {
        "z": z,
        "w_enc": w_enc,
        "w_dec": w_dec,
        "w_dec_plot": w_dec_plot,
        "winner_map": winner_map,
        "winner_counts": counts,
        "winner_percentages": percentages,
        "decoder_expert_names": decoder_names,
        "decoder_labels": decoder_labels,
        "output": out,
    }
