from .routing import visualize_routing

from .wta import (
    visualize_winner_takes_all,
    plot_winner_distribution,
)

from .expert_analysis import (
    expert_response_maps,
    pairwise_expert_correlations,
    analyze_expert_response,
    plot_expert_correlation,
    response_redundancy,
    routing_statistics,
    analyze_routing,
    print_routing_statistics,
    plot_structure_correlations,
)

from .freq_response import (
    make_3d_sinusoid,
    measure_frequency_response_and_routing,
    plot_frequency_response,
    plot_frequency_routing,
    routing_distribution_shift,
    plot_routing_distribution_shift,
)

__all__ = [
    # Routing
    "visualize_routing",

    # Winner-takes-all
    "visualize_winner_takes_all",
    "plot_winner_distribution",

    # Expert analysis
    "expert_response_maps",
    "pairwise_expert_correlations",
    "analyze_expert_response",
    "plot_expert_correlation",
    "response_redundancy",
    "routing_statistics",
    "analyze_routing",
    "print_routing_statistics",
    "plot_structure_correlations",

    # Frequency response
    "make_3d_sinusoid",
    "measure_frequency_response_and_routing",
    "plot_frequency_response",
    "plot_frequency_routing",
    "routing_distribution_shift",
    "plot_routing_distribution_shift",
]
