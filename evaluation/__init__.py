from .evaluator import (
    EDPEvaluator,
    compare_models,
)

from .figures import (
    set_paper_style,
    plot_loss_vs_flops,
    plot_acceleration_vs_depth,
    plot_layer_utilization_heatmap,
    plot_entropy_vs_depth,
    plot_flops_distribution,
    plot_ablation_comparison,
    generate_all_figures,
)

__all__ = [
    # Evaluator
    'EDPEvaluator',
    'compare_models',
    # Figures
    'set_paper_style',
    'plot_loss_vs_flops',
    'plot_acceleration_vs_depth',
    'plot_layer_utilization_heatmap',
    'plot_entropy_vs_depth',
    'plot_flops_distribution',
    'plot_ablation_comparison',
    'generate_all_figures',
]
