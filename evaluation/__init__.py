"""
Evaluation module for portfolio management agents.

Provides comprehensive evaluation with:
- Multiple seed runs for statistical robustness
- Per-validation-window breakdown
- 95% confidence intervals
- Publication-ready output tables

Metrics module provides all performance metric computations:
- CAGR, Sharpe, Sortino, Calmar ratios
- Maximum drawdown, annualized volatility
- Hit rate, excess returns

Visualizer module provides publication-ready charts:
- plot_portfolio_values_with_ci: Portfolio values with confidence bands
- plot_drawdown: Drawdown evolution charts
- plot_cumulative_returns_comparison: Cumulative returns with drawdown subplot
- plot_rolling_sharpe: Rolling Sharpe ratio over time
- plot_daily_returns_distribution: Return distribution histograms
- plot_allocation_evolution: Stacked area allocation charts
- plot_bar_comparison_with_ci: Bar chart comparison with error bars
- plot_metric_heatmap: Performance heatmaps

Tables module provides publication-ready tables:
- generate_latex_table: LaTeX tables for papers
- generate_markdown_table: Markdown tables for docs
- generate_html_table: HTML tables for reports
"""

from evaluation.evaluator import (
    EvaluationConfig,
    AgentResult,
    AggregatedResult,
    DetailedAgentResult,
    Evaluator,
    compute_confidence_interval,
    run_single_evaluation,
    run_detailed_evaluation,
    aggregate_results,
)

from evaluation.metrics import (
    compute_cagr,
    compute_sharpe_ratio,
    compute_sortino_ratio,
    compute_calmar_ratio,
    compute_max_drawdown,
    compute_annualized_volatility,
    compute_hit_rate,
    compute_excess_return,
    compute_all_metrics,
)

from evaluation.visualizer import (
    StrategyTimeSeries,
    plot_portfolio_values,
    plot_portfolio_values_with_ci,
    plot_drawdown,
    plot_drawdown_with_ci,
    plot_learning_curves,
    plot_multi_agent_learning_curves,
    plot_bar_comparison_with_ci,
    plot_multi_metric_comparison,
    plot_allocation_evolution,
    plot_turnover_comparison,
    plot_metric_heatmap,
    plot_cumulative_returns_comparison,
    plot_rolling_sharpe,
    plot_daily_returns_distribution,
    plot_weight_evolution_selected,
    plot_multi_agent_allocation_comparison,
    save_all_figures,
)

from evaluation.tables import (
    TableConfig,
    generate_latex_table,
    generate_markdown_table,
    generate_text_table,
    generate_html_table,
    generate_comparison_table_with_ci,
    generate_per_window_table,
    save_all_tables,
)

__all__ = [
    # Evaluator classes
    "EvaluationConfig",
    "AgentResult", 
    "AggregatedResult",
    "DetailedAgentResult",
    "Evaluator",
    # Evaluator functions
    "compute_confidence_interval",
    "run_single_evaluation",
    "run_detailed_evaluation",
    "aggregate_results",
    # Metrics functions
    "compute_cagr",
    "compute_sharpe_ratio",
    "compute_sortino_ratio",
    "compute_calmar_ratio",
    "compute_max_drawdown",
    "compute_annualized_volatility",
    "compute_hit_rate",
    "compute_excess_return",
    "compute_all_metrics",
    # Visualizer classes
    "StrategyTimeSeries",
    # Visualizer functions
    "plot_portfolio_values",
    "plot_portfolio_values_with_ci",
    "plot_drawdown",
    "plot_drawdown_with_ci",
    "plot_learning_curves",
    "plot_multi_agent_learning_curves",
    "plot_bar_comparison_with_ci",
    "plot_multi_metric_comparison",
    "plot_allocation_evolution",
    "plot_turnover_comparison",
    "plot_metric_heatmap",
    "plot_cumulative_returns_comparison",
    "plot_rolling_sharpe",
    "plot_daily_returns_distribution",
    "plot_weight_evolution_selected",
    "plot_multi_agent_allocation_comparison",
    "save_all_figures",
    # Tables classes
    "TableConfig",
    # Tables functions
    "generate_latex_table",
    "generate_markdown_table",
    "generate_text_table",
    "generate_html_table",
    "generate_comparison_table_with_ci",
    "generate_per_window_table",
    "save_all_tables",
]
