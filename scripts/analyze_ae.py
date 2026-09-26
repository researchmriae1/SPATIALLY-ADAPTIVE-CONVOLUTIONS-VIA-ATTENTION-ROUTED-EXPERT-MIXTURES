import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from configs import ae, data, frequency_response
from data import MRIDataset
from models import AutoEncoder
from training import load_checkpoint

from visualization import (
    visualize_routing,
    visualize_winner_takes_all,
    plot_winner_distribution,
    analyze_expert_response,
    plot_expert_correlation,
    response_redundancy,
    analyze_routing,
    print_routing_statistics,
    plot_structure_correlations,
    measure_frequency_response_and_routing,
    plot_frequency_response,
    plot_frequency_routing,
    routing_distribution_shift,
    plot_routing_distribution_shift,
)


# ================================================================
# Data
# ================================================================

def create_test_loader():
    test_dataset = MRIDataset(
        root_dir=data.TEST_DIR,
        crop_size=data.CROP_SIZE,
        normalize=data.NORMALIZE,
        augment=False,
        random_crop=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=data.BATCH_SIZE,
        shuffle=False,
        num_workers=data.NUM_WORKERS,
        pin_memory=data.PIN_MEMORY,
        persistent_workers=(
            data.PERSISTENT_WORKERS
            and data.NUM_WORKERS > 0
        ),
    )

    print(f"Test samples: {len(test_dataset)}")

    return test_loader


# ================================================================
# Model
# ================================================================

def create_model(device):
    model = AutoEncoder(
        encoder_experts=ae.ENCODER_EXPERTS,
        decoder_experts=ae.DECODER_EXPERTS,
    ).to(device)

    checkpoint_path = os.path.join(
        ae.CHECKPOINT_DIR,
        ae.BEST_CHECKPOINT_NAME,
    )

    load_checkpoint(
        checkpoint_path,
        model=model,
        device=device,
    )

    model.eval()

    print(f"Loaded checkpoint: {checkpoint_path}")

    return model


# ================================================================
# Single-volume routing analysis
# ================================================================

def run_routing_analysis(model, test_loader, device):
    print("\n" + "=" * 70)
    print("ROUTING ANALYSIS")
    print("=" * 70)

    x = next(iter(test_loader))

    if isinstance(x, (list, tuple)):
        x = x[0]

    x = x.to(
        device,
        non_blocking=True,
    )

    results = visualize_routing(
        model,
        x,
        plot_encoder=True,
        plot_decoder=True,
        print_shapes=True,
    )

    return x, results


# ================================================================
# Winner-Takes-All analysis
# ================================================================

def run_wta_analysis(model, x):
    print("\n" + "=" * 70)
    print("WINNER-TAKES-ALL ANALYSIS")
    print("=" * 70)

    wta_results = visualize_winner_takes_all(
        model,
        x,
    )

    plot_winner_distribution(
        model,
        x,
    )

    return wta_results


# ================================================================
# Expert-response analysis
# ================================================================

def run_expert_analysis(model, test_loader, device):
    print("\n" + "=" * 70)
    print("EXPERT RESPONSE ANALYSIS")
    print("=" * 70)

    results = analyze_expert_response(
        model,
        test_loader,
        device,
    )

    corr = results["correlation"]

    plot_expert_correlation(
        corr,
    )

    redundancy = response_redundancy(
        corr,
    )

    print("\nResponse redundancy:")
    print(redundancy)

    return results


# ================================================================
# Routing statistics
# ================================================================

def run_routing_statistics(model, test_loader, device):
    print("\n" + "=" * 70)
    print("ROUTING STATISTICS")
    print("=" * 70)

    results = analyze_routing(
        model,
        test_loader,
        device,
    )

    print_routing_statistics(
        results,
    )

    return results


# ================================================================
# Frequency-response analysis
# ================================================================

def run_frequency_analysis(model, device):
    print("\n" + "=" * 70)
    print("FREQUENCY RESPONSE ANALYSIS")
    print("=" * 70)

    frequencies = frequency_response.FREQUENCIES

    all_results = {}

    for axis in frequency_response.AXES:

        print(f"\nAnalyzing {axis}-axis frequency response...")

        routing, response = (
            measure_frequency_response_and_routing(
                model=model,
                frequencies=frequencies,
                size=frequency_response.VOLUME_SIZE,
                axis=axis,
                amplitude=frequency_response.AMPLITUDE,
                phase=frequency_response.PHASE,
                device=device,
            )
        )

        # --------------------------------------------------------
        # Unrouted expert response
        # --------------------------------------------------------

        plot_frequency_response(
            frequencies,
            response,
            title=(
                f"Unrouted Expert Response "
                f"vs. Spatial Frequency ({axis}-axis)"
            ),
        )

        # --------------------------------------------------------
        # Routing response
        # --------------------------------------------------------

        plot_frequency_routing(
            frequencies,
            routing,
            title=(
                f"Routing Weight vs. Spatial Frequency "
                f"({axis}-axis)"
            ),
        )

        # --------------------------------------------------------
        # Distribution shift
        # --------------------------------------------------------

        shift = routing_distribution_shift(
            routing,
        )

        plot_routing_distribution_shift(
            frequencies,
            shift,
            title=(
                f"Routing Distribution Shift "
                f"({axis}-axis)"
            ),
        )

        all_results[axis] = {
            "routing": routing,
            "response": response,
            "shift": shift,
        }

    return all_results


# ================================================================
# Main
# ================================================================

def main():

    # ------------------------------------------------------------
    # Device
    # ------------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")

    if torch.cuda.is_available():
        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    # ------------------------------------------------------------
    # Data
    # ------------------------------------------------------------

    test_loader = create_test_loader()

    # ------------------------------------------------------------
    # Model
    # ------------------------------------------------------------

    model = create_model(device)

    # ------------------------------------------------------------
    # 1. Routing
    # ------------------------------------------------------------

    x, routing_results = run_routing_analysis(
        model,
        test_loader,
        device,
    )

    # ------------------------------------------------------------
    # 2. Winner-Takes-All
    # ------------------------------------------------------------

    wta_results = run_wta_analysis(
        model,
        x,
    )

    # ------------------------------------------------------------
    # 3. Expert response analysis
    # ------------------------------------------------------------

    expert_results = run_expert_analysis(
        model,
        test_loader,
        device,
    )

    # ------------------------------------------------------------
    # 4. Routing statistics
    # ------------------------------------------------------------

    routing_statistics = run_routing_statistics(
        model,
        test_loader,
        device,
    )

    # ------------------------------------------------------------
    # 5. Frequency response
    # ------------------------------------------------------------

    frequency_results = run_frequency_analysis(
        model,
        device,
    )

    print("\n" + "=" * 70)
    print("ALL ANALYSES COMPLETE")
    print("=" * 70)

    return {
        "routing": routing_results,
        "wta": wta_results,
        "expert": expert_results,
        "routing_statistics": routing_statistics,
        "frequency": frequency_results,
    }


if __name__ == "__main__":
    main()
