"""
Visualization utilities for portfolio strategy evaluation.

Publication-ready visualizations following academic paper standards:
- Portfolio value evolution with confidence bands
- Log-scale portfolio value charts
- Drawdown evolution charts
- Learning curves (training loss/reward)
- Bar chart comparisons with confidence intervals
- Allocation evolution (stacked area charts)
- Turnover comparison charts

References:
- Jiang et al. (2017) - EIIE paper visualizations
- Lucarelli & Borrotti (2020) - DQN crypto paper
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import PercentFormatter
from matplotlib.patches import Patch
from scipy import stats

# Set publication-quality style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.figsize': (12, 6),
})


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class StrategyTimeSeries:
    """
    Time series data collected from running a strategy.
    
    Attributes
    ----------
    name : str
        Strategy/agent name
    dates : List[np.datetime64]
        Trading dates
    portfolio_values : List[float]
        Portfolio values at each step
    weights_history : List[Dict[str, float]]
        Portfolio weights at each step {asset_id: weight}
    asset_ids_history : List[List[str]]
        Available assets at each step
    turnovers : List[float]
        Turnover at each step
    rewards : List[float]
        Rewards at each step
    """
    name: str
    dates: List[np.datetime64]
    portfolio_values: List[float]
    weights_history: List[Dict[str, float]]
    asset_ids_history: List[List[str]]
    turnovers: List[float]
    rewards: List[float]


# =============================================================================
# Color Palettes
# =============================================================================

def get_agent_colors(n_agents: int) -> np.ndarray:
    """Get color palette for agents."""
    return plt.cm.tab10(np.linspace(0, 1, max(n_agents, 1)))


def get_type_colors() -> Dict[str, str]:
    """Get colors by agent type."""
    return {
        'baseline': 'steelblue',
        'rl': 'coral',
        'ensemble': 'forestgreen',
    }


# =============================================================================
# Portfolio Value Visualizations
# =============================================================================

def plot_portfolio_values(
    strategies: List[StrategyTimeSeries],
    validation_windows: Optional[List[Dict]] = None,
    title: str = "Portfolio Value Evolution",
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (14, 6),
    log_scale: bool = False,
) -> plt.Figure:
    """
    Plot portfolio value evolution for multiple strategies.
    
    Parameters
    ----------
    strategies : List[StrategyTimeSeries]
        Time series data from different strategies
    validation_windows : List[Dict], optional
        List of validation window definitions with 'name', 'start', 'end'
    title : str
        Chart title
    output_path : Path, optional
        If provided, save figure to this path
    log_scale : bool
        Whether to use log scale on y-axis (recommended for long horizons)
        
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    colors = get_agent_colors(len(strategies))
    
    for strat, color in zip(strategies, colors):
        steps = np.arange(len(strat.portfolio_values))
        ax.plot(steps, strat.portfolio_values, label=strat.name, 
                linewidth=2, color=color)
    
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Initial Value')
    
    # Add validation window boundaries
    if strategies and validation_windows:
        boundaries = _find_window_boundaries(strategies[0].dates, validation_windows)
        for step_idx, window_name in boundaries:
            ax.axvline(x=step_idx, color='red', linestyle=':', alpha=0.6, linewidth=1.5)
            y_pos = ax.get_ylim()[1] * 0.98
            short_name = window_name.replace('val_', '').replace('_', '\n')
            ax.text(step_idx + 2, y_pos, short_name, fontsize=8, 
                   verticalalignment='top', color='darkred', fontweight='bold')
    
    ax.set_xlabel('Trading Day (Step)')
    if log_scale:
        ax.set_yscale('log')
        ax.set_ylabel('Portfolio Value (log scale)')
    else:
        ax.set_ylabel('Portfolio Value')
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{(x-1)*100:.0f}%'))
    
    ax.set_title(title, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    return fig


def plot_portfolio_values_with_ci(
    multi_seed_data: Dict[str, List[List[float]]],
    validation_windows: Optional[List[Dict]] = None,
    title: str = "Portfolio Value Evolution with 95% CI",
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (14, 6),
    log_scale: bool = False,
) -> plt.Figure:
    """
    Plot portfolio value evolution with confidence bands from multi-seed runs.
    
    Parameters
    ----------
    multi_seed_data : Dict[str, List[List[float]]]
        Mapping agent_name -> list of portfolio_values lists (one per seed)
    validation_windows : List[Dict], optional
        List of validation window definitions
    title : str
        Chart title
    output_path : Path, optional
        If provided, save figure to this path
    log_scale : bool
        Whether to use log scale on y-axis
        
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    colors = get_agent_colors(len(multi_seed_data))
    
    for (agent_name, pv_runs), color in zip(multi_seed_data.items(), colors):
        if not pv_runs:
            continue
            
        # Pad to uniform length
        max_len = max(len(pv) for pv in pv_runs)
        padded_runs = []
        for pv in pv_runs:
            if len(pv) < max_len:
                pv = pv + [pv[-1]] * (max_len - len(pv))
            padded_runs.append(pv)
        
        pv_array = np.array(padded_runs)
        n_seeds = pv_array.shape[0]
        
        # Compute mean and CI
        mean_pv = pv_array.mean(axis=0)
        std_pv = pv_array.std(axis=0, ddof=1)
        
        # 95% CI using t-distribution
        t_value = stats.t.ppf(0.975, df=n_seeds - 1) if n_seeds > 1 else 0
        ci_margin = t_value * std_pv / np.sqrt(n_seeds)
        ci_lower = mean_pv - ci_margin
        ci_upper = mean_pv + ci_margin
        
        steps = np.arange(len(mean_pv))
        
        # Plot mean and CI band
        ax.plot(steps, mean_pv, label=f"{agent_name}", linewidth=2, color=color)
        ax.fill_between(steps, ci_lower, ci_upper, color=color, alpha=0.2)
    
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Initial')
    
    ax.set_xlabel('Trading Day (Step)')
    if log_scale:
        ax.set_yscale('log')
        ax.set_ylabel('Portfolio Value (log scale)')
    else:
        ax.set_ylabel('Portfolio Value')
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{(x-1)*100:.0f}%'))
    
    ax.set_title(title, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    return fig


# =============================================================================
# Drawdown Visualizations
# =============================================================================

def plot_drawdown(
    strategies: List[StrategyTimeSeries],
    title: str = "Drawdown Evolution",
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (14, 5),
) -> plt.Figure:
    """
    Plot drawdown evolution for multiple strategies.
    
    Drawdown = (Peak - Current) / Peak, shown as negative values.
    
    Parameters
    ----------
    strategies : List[StrategyTimeSeries]
        Time series data from different strategies
    title : str
        Chart title
    output_path : Path, optional
        If provided, save figure to this path
        
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    colors = get_agent_colors(len(strategies))
    
    for strat, color in zip(strategies, colors):
        pv = np.array(strat.portfolio_values)
        running_max = np.maximum.accumulate(pv)
        drawdown = (pv - running_max) / running_max
        
        steps = np.arange(len(drawdown))
        ax.fill_between(steps, drawdown, 0, alpha=0.3, color=color)
        ax.plot(steps, drawdown, label=strat.name, linewidth=1.5, color=color)
    
    ax.set_xlabel('Trading Day (Step)')
    ax.set_ylabel('Drawdown')
    ax.set_title(title, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax.set_ylim(None, 0.05)  # Small margin above 0
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    return fig


def plot_drawdown_with_ci(
    multi_seed_data: Dict[str, List[List[float]]],
    title: str = "Drawdown Evolution with 95% CI",
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (14, 5),
) -> plt.Figure:
    """
    Plot drawdown evolution with confidence bands from multi-seed runs.
    
    Parameters
    ----------
    multi_seed_data : Dict[str, List[List[float]]]
        Mapping agent_name -> list of portfolio_values lists (one per seed)
    title : str
        Chart title
    output_path : Path, optional
        If provided, save figure to this path
        
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    colors = get_agent_colors(len(multi_seed_data))
    
    for (agent_name, pv_runs), color in zip(multi_seed_data.items(), colors):
        if not pv_runs:
            continue
        
        # Compute drawdowns for each seed
        drawdown_runs = []
        max_len = max(len(pv) for pv in pv_runs)
        
        for pv in pv_runs:
            pv_arr = np.array(pv)
            if len(pv_arr) < max_len:
                pv_arr = np.concatenate([pv_arr, np.full(max_len - len(pv_arr), pv_arr[-1])])
            running_max = np.maximum.accumulate(pv_arr)
            dd = (pv_arr - running_max) / running_max
            drawdown_runs.append(dd)
        
        dd_array = np.array(drawdown_runs)
        n_seeds = dd_array.shape[0]
        
        mean_dd = dd_array.mean(axis=0)
        std_dd = dd_array.std(axis=0, ddof=1)
        
        t_value = stats.t.ppf(0.975, df=n_seeds - 1) if n_seeds > 1 else 0
        ci_margin = t_value * std_dd / np.sqrt(n_seeds)
        
        steps = np.arange(len(mean_dd))
        
        ax.fill_between(steps, mean_dd - ci_margin, mean_dd + ci_margin, 
                       color=color, alpha=0.2)
        ax.plot(steps, mean_dd, label=agent_name, linewidth=1.5, color=color)
    
    ax.set_xlabel('Trading Day (Step)')
    ax.set_ylabel('Drawdown')
    ax.set_title(title, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax.set_ylim(None, 0.05)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    return fig


# =============================================================================
# Learning Curve Visualizations
# =============================================================================

def plot_learning_curves(
    training_history: pd.DataFrame,
    metrics: List[str] = ['episode_return', 'loss'],
    smoothing_window: int = 10,
    title: str = "Training Learning Curves",
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (14, 5),
) -> plt.Figure:
    """
    Plot training learning curves with optional smoothing.
    
    Parameters
    ----------
    training_history : pd.DataFrame
        Training history with columns for episode metrics
    metrics : List[str]
        Metric columns to plot
    smoothing_window : int
        Rolling window for smoothing curves
    title : str
        Chart title
    output_path : Path, optional
        If provided, save figure to this path
        
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(figsize[0], figsize[1]))
    if n_metrics == 1:
        axes = [axes]
    
    for ax, metric in zip(axes, metrics):
        if metric not in training_history.columns:
            ax.set_title(f'{metric} (not found)')
            continue
        
        values = training_history[metric].dropna()
        episodes = np.arange(len(values))
        
        # Raw values (light)
        ax.plot(episodes, values, alpha=0.3, color='steelblue', linewidth=0.5)
        
        # Smoothed values (bold)
        if len(values) >= smoothing_window:
            smoothed = values.rolling(window=smoothing_window, min_periods=1).mean()
            ax.plot(episodes, smoothed, color='steelblue', linewidth=2, 
                   label=f'{smoothing_window}-ep moving avg')
        
        ax.set_xlabel('Episode')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'{metric.replace("_", " ").title()}')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=9)
    
    fig.suptitle(title, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    return fig


def plot_multi_agent_learning_curves(
    histories: Dict[str, pd.DataFrame],
    metric: str = 'episode_return',
    smoothing_window: int = 20,
    title: str = "Training Comparison",
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 6),
) -> plt.Figure:
    """
    Compare learning curves across multiple agents.
    
    Parameters
    ----------
    histories : Dict[str, pd.DataFrame]
        Mapping agent_name -> training_history DataFrame
    metric : str
        Metric column to compare
    smoothing_window : int
        Rolling window for smoothing
    title : str
        Chart title
    output_path : Path, optional
        If provided, save figure to this path
        
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    colors = get_agent_colors(len(histories))
    
    for (agent_name, history), color in zip(histories.items(), colors):
        if metric not in history.columns:
            continue
        
        values = history[metric].dropna()
        episodes = np.arange(len(values))
        
        # Raw (light)
        ax.plot(episodes, values, alpha=0.2, color=color, linewidth=0.5)
        
        # Smoothed
        if len(values) >= smoothing_window:
            smoothed = values.rolling(window=smoothing_window, min_periods=1).mean()
            ax.plot(episodes, smoothed, color=color, linewidth=2, label=agent_name)
    
    ax.set_xlabel('Episode')
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(title, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    return fig


# =============================================================================
# Bar Chart Comparisons
# =============================================================================

def plot_bar_comparison_with_ci(
    results_df: pd.DataFrame,
    metric: str = "Return (%)",
    ci_lower_col: Optional[str] = None,
    ci_upper_col: Optional[str] = None,
    title: str = "Agent Comparison",
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 6),
    color_by_type: bool = True,
) -> plt.Figure:
    """
    Create bar chart comparing agents with 95% CI error bars.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Results DataFrame with Agent, Type, and metric columns
    metric : str
        Column name for metric to compare
    ci_lower_col : str, optional
        Column name for CI lower bound
    ci_upper_col : str, optional  
        Column name for CI upper bound
    title : str
        Chart title
    output_path : Path, optional
        If provided, save figure to this path
    color_by_type : bool
        Whether to color bars by agent type
        
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    agents = results_df['Agent'].tolist()
    values = results_df[metric].tolist()
    
    # Get error bars if available
    errors = None
    if ci_lower_col and ci_upper_col:
        if ci_lower_col in results_df.columns and ci_upper_col in results_df.columns:
            ci_lower = results_df[ci_lower_col].tolist()
            ci_upper = results_df[ci_upper_col].tolist()
            errors = [[v - l for v, l in zip(values, ci_lower)],
                      [u - v for v, u in zip(values, ci_upper)]]
    
    # Colors
    type_colors = get_type_colors()
    if color_by_type and 'Type' in results_df.columns:
        colors = [type_colors.get(t, 'gray') for t in results_df['Type'].tolist()]
    else:
        colors = ['steelblue'] * len(agents)
    
    x = np.arange(len(agents))
    bars = ax.bar(x, values, color=colors, alpha=0.8, edgecolor='black')
    
    if errors:
        ax.errorbar(x, values, yerr=errors, fmt='none', color='black', 
                   capsize=5, capthick=2, linewidth=2)
    
    ax.set_xticks(x)
    ax.set_xticklabels(agents, rotation=45, ha='right')
    ax.set_ylabel(metric)
    ax.set_title(title, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Value labels
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.annotate(f'{val:.1f}' if abs(val) > 1 else f'{val:.2f}',
                   xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Legend
    if color_by_type:
        legend_elements = [Patch(facecolor=c, alpha=0.8, label=t.title()) 
                         for t, c in type_colors.items()]
        ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    return fig


def plot_multi_metric_comparison(
    results_df: pd.DataFrame,
    metrics: List[str] = ['Return (%)', 'Sharpe', 'Max DD (%)'],
    title: str = "Multi-Metric Agent Comparison",
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (16, 6),
) -> plt.Figure:
    """
    Create side-by-side bar charts for multiple metrics.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Results DataFrame with Agent and metric columns
    metrics : List[str]
        List of metric column names to plot
    title : str
        Chart title
    output_path : Path, optional
        If provided, save figure to this path
        
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    if n_metrics == 1:
        axes = [axes]
    
    type_colors = get_type_colors()
    agents = results_df['Agent'].tolist()
    
    for ax, metric in zip(axes, metrics):
        if metric not in results_df.columns:
            ax.set_title(f'{metric} (not found)')
            continue
        
        values = results_df[metric].tolist()
        
        if 'Type' in results_df.columns:
            colors = [type_colors.get(t, 'gray') for t in results_df['Type'].tolist()]
        else:
            colors = ['steelblue'] * len(agents)
        
        x = np.arange(len(agents))
        bars = ax.bar(x, values, color=colors, alpha=0.8, edgecolor='black')
        
        ax.set_xticks(x)
        ax.set_xticklabels(agents, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel(metric)
        ax.set_title(metric, fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
    
    fig.suptitle(title, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    return fig


# =============================================================================
# Allocation Visualizations
# =============================================================================

def plot_allocation_evolution(
    strategy: StrategyTimeSeries,
    top_n: int = 10,
    title: Optional[str] = None,
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (14, 8),
) -> plt.Figure:
    """
    Plot portfolio allocation evolution as stacked area chart.
    
    Parameters
    ----------
    strategy : StrategyTimeSeries
        Time series data from a single strategy
    top_n : int
        Number of top assets to show individually
    title : str, optional
        Chart title (default: f'Portfolio Allocation - {strategy.name}')
    output_path : Path, optional
        If provided, save figure to this path
        
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    # Collect all assets and compute average weights
    all_assets = set()
    for weights_dict in strategy.weights_history:
        all_assets.update(weights_dict.keys())
    
    asset_avg_weights = {}
    for asset in all_assets:
        weights = [w.get(asset, 0.0) for w in strategy.weights_history]
        asset_avg_weights[asset] = np.mean(weights)
    
    sorted_assets = sorted(asset_avg_weights.keys(), 
                          key=lambda x: asset_avg_weights[x], reverse=True)
    
    top_assets = sorted_assets[:top_n]
    other_assets = sorted_assets[top_n:] if len(sorted_assets) > top_n else []
    
    # Build DataFrame
    steps = np.arange(len(strategy.weights_history))
    data = {}
    
    for asset in top_assets:
        data[asset] = [w.get(asset, 0.0) for w in strategy.weights_history]
    
    if other_assets:
        data['Other'] = [sum(w.get(a, 0.0) for a in other_assets) 
                        for w in strategy.weights_history]
    
    df = pd.DataFrame(data, index=steps)
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.tab20(np.linspace(0, 1, len(df.columns)))
    
    df.plot.area(ax=ax, stacked=True, alpha=0.8, color=colors)
    
    ax.set_xlabel('Trading Day (Step)')
    ax.set_ylabel('Portfolio Weight')
    ax.set_title(title or f'Portfolio Allocation - {strategy.name}', fontweight='bold')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, len(steps) - 1)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    return fig


def plot_turnover_comparison(
    strategies: List[StrategyTimeSeries],
    title: str = "Portfolio Turnover Comparison",
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (14, 5),
) -> plt.Figure:
    """
    Plot turnover over time for multiple strategies.
    
    Parameters
    ----------
    strategies : List[StrategyTimeSeries]
        Time series data from different strategies
    title : str
        Chart title
    output_path : Path, optional
        If provided, save figure to this path
        
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    colors = get_agent_colors(len(strategies))
    
    for strat, color in zip(strategies, colors):
        steps = np.arange(1, len(strat.turnovers) + 1)
        ax.plot(steps, strat.turnovers, label=strat.name, 
                linewidth=1.5, alpha=0.7, color=color)
    
    ax.set_xlabel('Trading Day (Step)')
    ax.set_ylabel('Turnover')
    ax.set_title(title, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    return fig


# =============================================================================
# Regime/Heatmap Visualizations
# =============================================================================

def plot_metric_heatmap(
    results_df: pd.DataFrame,
    agents: List[str],
    metrics: List[str],
    title: str = "Performance Metrics Heatmap",
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (14, 7),
    cmap: str = 'RdYlGn',
    annotate: bool = True,
    invert_metrics: Optional[List[str]] = None,
) -> plt.Figure:
    """
    Create heatmap of metrics across agents with per-column normalization.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Results DataFrame with Agent and metric columns
    agents : List[str]
        Agent names (rows)
    metrics : List[str]
        Metric names (columns)
    title : str
        Chart title
    output_path : Path, optional
        If provided, save figure to this path
    cmap : str
        Colormap name
    annotate : bool
        Whether to show values in cells
    invert_metrics : List[str], optional
        Metrics where lower is better (e.g., Max DD, Volatility).
        These will have inverted color scale (green for low values).
        
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    # Default metrics where lower is better
    if invert_metrics is None:
        invert_metrics = ['Max DD (%)', 'Volatility (%)', 'Turnover (%)']
    
    # Build matrix with raw values
    data = []
    for agent in agents:
        row = results_df[results_df['Agent'] == agent]
        if len(row) == 0:
            data.append([np.nan] * len(metrics))
        else:
            row_data = []
            for m in metrics:
                val = row[m].values[0] if m in row.columns else np.nan
                row_data.append(val)
            data.append(row_data)
    
    matrix = np.array(data)
    
    # Normalize each column independently for coloring
    normalized_matrix = np.zeros_like(matrix, dtype=float)
    for j, metric in enumerate(metrics):
        col = matrix[:, j]
        col_min, col_max = np.nanmin(col), np.nanmax(col)
        
        if col_max - col_min > 1e-8:
            if metric in invert_metrics:
                # Invert: lower values get higher normalized score (green)
                normalized_matrix[:, j] = (col_max - col) / (col_max - col_min)
            else:
                # Standard: higher values get higher normalized score (green)
                normalized_matrix[:, j] = (col - col_min) / (col_max - col_min)
        else:
            normalized_matrix[:, j] = 0.5  # All same value
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Use normalized matrix for coloring (0-1 scale per column)
    im = ax.imshow(normalized_matrix, cmap=cmap, aspect='auto', vmin=0, vmax=1)
    
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_yticks(np.arange(len(agents)))
    ax.set_xticklabels(metrics, rotation=45, ha='right')
    ax.set_yticklabels(agents)
    
    # Annotate with ORIGINAL values (not normalized)
    if annotate:
        for i in range(len(agents)):
            for j in range(len(metrics)):
                val = matrix[i, j]
                if not np.isnan(val):
                    text = f'{val:.2f}' if abs(val) < 10 else f'{val:.1f}'
                    # Choose text color based on background brightness
                    bg_val = normalized_matrix[i, j]
                    text_color = 'white' if bg_val < 0.3 or bg_val > 0.7 else 'black'
                    ax.text(j, i, text, ha='center', va='center', 
                           color=text_color, fontsize=9, fontweight='bold')
    
    ax.set_title(title, fontweight='bold', pad=10)
    
    # Add color interpretation note instead of colorbar
    ax.text(0.5, -0.15, "Color scale per column: Green = Best, Red = Worst", 
            transform=ax.transAxes, ha='center', fontsize=10, style='italic')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    return fig


# =============================================================================
# New Publication-Quality Visualizations (Academic Paper Standards)
# =============================================================================

def plot_cumulative_returns_comparison(
    multi_agent_pvs: Dict[str, List[float]],
    title: str = "Cumulative Returns Comparison",
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (16, 7),
    log_scale: bool = False,
    show_drawdown: bool = True,
) -> plt.Figure:
    """
    Plot cumulative returns (as percentages) for multiple agents.
    
    This is the main chart for academic papers, showing portfolio growth
    over the evaluation period.
    
    Parameters
    ----------
    multi_agent_pvs : Dict[str, List[float]]
        Mapping agent_name -> portfolio_values list
    title : str
        Chart title
    output_path : Path, optional
        If provided, save figure to this path
    log_scale : bool
        Whether to use log scale on y-axis
    show_drawdown : bool
        Whether to show a drawdown subplot
        
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    if show_drawdown:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(figsize[0], figsize[1] * 1.3),
                                        gridspec_kw={'height_ratios': [3, 1]})
    else:
        fig, ax1 = plt.subplots(figsize=figsize)
        ax2 = None
    
    colors = get_agent_colors(len(multi_agent_pvs))
    
    for (agent_name, pvs), color in zip(multi_agent_pvs.items(), colors):
        pvs_arr = np.array(pvs)
        returns_pct = (pvs_arr - 1) * 100  # Convert to percentage returns
        steps = np.arange(len(pvs_arr))
        
        ax1.plot(steps, returns_pct, label=agent_name, linewidth=2, color=color)
        
        # Drawdown subplot
        if show_drawdown and ax2 is not None:
            running_max = np.maximum.accumulate(pvs_arr)
            drawdown = (pvs_arr - running_max) / running_max * 100
            ax2.fill_between(steps, drawdown, 0, alpha=0.3, color=color)
            ax2.plot(steps, drawdown, linewidth=1, color=color, alpha=0.7)
    
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Trading Day')
    ax1.set_ylabel('Cumulative Return (%)')
    if log_scale:
        ax1.set_yscale('symlog')
    ax1.set_title(title, fontweight='bold', fontsize=14)
    ax1.legend(loc='upper left', bbox_to_anchor=(1.02, 1), framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    
    if show_drawdown and ax2 is not None:
        ax2.set_xlabel('Trading Day')
        ax2.set_ylabel('Drawdown (%)')
        ax2.set_title('Drawdown', fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(bottom=ax2.get_ylim()[0] * 1.1)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    return fig


def plot_rolling_sharpe(
    multi_agent_pvs: Dict[str, List[float]],
    window: int = 30,
    title: str = "Rolling Sharpe Ratio (30-day)",
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (16, 5),
    risk_free_rate: float = 0.0,
) -> plt.Figure:
    """
    Plot rolling Sharpe ratio over time for multiple agents.
    
    This visualization shows how risk-adjusted performance varies over time,
    highlighting periods of strong/weak performance.
    
    Parameters
    ----------
    multi_agent_pvs : Dict[str, List[float]]
        Mapping agent_name -> portfolio_values list
    window : int
        Rolling window size in days (default: 30)
    title : str
        Chart title
    output_path : Path, optional
        If provided, save figure to this path
    risk_free_rate : float
        Annualized risk-free rate (default: 0.0)
        
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    colors = get_agent_colors(len(multi_agent_pvs))
    
    daily_rf = risk_free_rate / 252  # Daily risk-free rate
    
    for (agent_name, pvs), color in zip(multi_agent_pvs.items(), colors):
        pvs_arr = np.array(pvs)
        
        # Compute daily returns
        daily_returns = np.diff(pvs_arr) / pvs_arr[:-1]
        excess_returns = daily_returns - daily_rf
        
        # Rolling Sharpe
        if len(excess_returns) >= window:
            rolling_mean = pd.Series(excess_returns).rolling(window).mean()
            rolling_std = pd.Series(excess_returns).rolling(window).std()
            rolling_sharpe = (rolling_mean / (rolling_std + 1e-10)) * np.sqrt(252)
            
            steps = np.arange(len(rolling_sharpe))
            ax.plot(steps, rolling_sharpe, label=agent_name, linewidth=1.5, 
                   color=color, alpha=0.8)
    
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=1, color='green', linestyle=':', alpha=0.4, label='Sharpe = 1')
    ax.axhline(y=-1, color='red', linestyle=':', alpha=0.4, label='Sharpe = -1')
    
    ax.set_xlabel('Trading Day')
    ax.set_ylabel('Rolling Sharpe Ratio')
    ax.set_title(title, fontweight='bold')
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-5, 5)  # Clip extreme values
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    return fig


def plot_daily_returns_distribution(
    multi_agent_pvs: Dict[str, List[float]],
    title: str = "Daily Returns Distribution",
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (14, 5),
    bins: int = 50,
    show_stats: bool = True,
) -> plt.Figure:
    """
    Plot histogram of daily returns with normal distribution overlay.
    
    This visualization shows the return distribution, highlighting
    skewness and kurtosis (fat tails) typical of financial returns.
    
    Parameters
    ----------
    multi_agent_pvs : Dict[str, List[float]]
        Mapping agent_name -> portfolio_values list
    title : str
        Chart title
    output_path : Path, optional
        If provided, save figure to this path
    bins : int
        Number of histogram bins
    show_stats : bool
        Whether to show skewness/kurtosis statistics
        
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    n_agents = len(multi_agent_pvs)
    fig, axes = plt.subplots(1, n_agents, figsize=(figsize[0], figsize[1]), 
                             sharey=True, squeeze=False)
    axes = axes.flatten()
    
    colors = get_agent_colors(n_agents)
    
    for idx, ((agent_name, pvs), color) in enumerate(zip(multi_agent_pvs.items(), colors)):
        ax = axes[idx]
        pvs_arr = np.array(pvs)
        daily_returns = np.diff(pvs_arr) / pvs_arr[:-1] * 100  # Percentage
        
        # Histogram
        ax.hist(daily_returns, bins=bins, density=True, alpha=0.7, 
               color=color, edgecolor='white', linewidth=0.5)
        
        # Normal distribution overlay
        mu, sigma = np.mean(daily_returns), np.std(daily_returns)
        x = np.linspace(mu - 4*sigma, mu + 4*sigma, 100)
        normal_pdf = stats.norm.pdf(x, mu, sigma)
        ax.plot(x, normal_pdf, 'k--', linewidth=2, alpha=0.7, label='Normal')
        
        ax.axvline(x=0, color='gray', linestyle='-', alpha=0.5)
        ax.set_xlabel('Daily Return (%)')
        ax.set_title(agent_name, fontweight='bold')
        
        if show_stats:
            skew = stats.skew(daily_returns)
            kurt = stats.kurtosis(daily_returns)
            stats_text = f'μ={mu:.2f}%\nσ={sigma:.2f}%\nSkew={skew:.2f}\nKurt={kurt:.2f}'
            ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        if idx == 0:
            ax.set_ylabel('Density')
    
    fig.suptitle(title, fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    return fig


def plot_weight_evolution_selected(
    strategy: StrategyTimeSeries,
    top_n: int = 5,
    title: Optional[str] = None,
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (14, 6),
) -> plt.Figure:
    """
    Plot weight evolution for top N assets as line chart.
    
    Unlike stacked area charts, this shows individual weight trajectories
    more clearly, making it easier to see rebalancing behavior.
    
    Parameters
    ----------
    strategy : StrategyTimeSeries
        Time series data from a single strategy
    top_n : int
        Number of top assets to show (by average weight)
    title : str, optional
        Chart title
    output_path : Path, optional
        If provided, save figure to this path
        
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    # Identify top assets by average weight
    all_assets = set()
    for weights_dict in strategy.weights_history:
        all_assets.update(weights_dict.keys())
    
    asset_avg_weights = {}
    for asset in all_assets:
        weights = [w.get(asset, 0.0) for w in strategy.weights_history]
        asset_avg_weights[asset] = np.mean(weights)
    
    top_assets = sorted(asset_avg_weights.keys(), 
                       key=lambda x: asset_avg_weights[x], reverse=True)[:top_n]
    
    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.tab10(np.linspace(0, 1, top_n))
    
    steps = np.arange(len(strategy.weights_history))
    
    for asset, color in zip(top_assets, colors):
        weights = [w.get(asset, 0.0) for w in strategy.weights_history]
        ax.plot(steps, weights, label=asset, linewidth=2, color=color, alpha=0.8)
    
    ax.set_xlabel('Trading Day')
    ax.set_ylabel('Portfolio Weight')
    ax.set_title(title or f'Weight Evolution - {strategy.name} (Top {top_n} Assets)', 
                fontweight='bold')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, None)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    return fig


def plot_multi_agent_allocation_comparison(
    strategies: List[StrategyTimeSeries],
    top_n: int = 8,
    title: str = "Portfolio Allocation Comparison",
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (16, 10),
) -> plt.Figure:
    """
    Plot allocation evolution for multiple agents as stacked area subplots.
    
    Creates a grid of subplots, one per agent, allowing visual comparison
    of allocation strategies. Uses a GLOBAL color mapping so each asset
    has the same color across all subplots for accurate comparison.
    
    Parameters
    ----------
    strategies : List[StrategyTimeSeries]
        Time series data from different strategies
    top_n : int
        Number of top assets to show individually
    title : str
        Main chart title
    output_path : Path, optional
        If provided, save figure to this path
        
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    # STEP 1: Compute GLOBAL asset ordering across ALL strategies
    # This ensures consistent colors/labels across all subplots
    global_asset_totals = {}
    for strat in strategies:
        for weights_dict in strat.weights_history:
            for asset, weight in weights_dict.items():
                global_asset_totals[asset] = global_asset_totals.get(asset, 0.0) + weight
    
    # Sort by total weight across all strategies
    global_sorted_assets = sorted(global_asset_totals.keys(),
                                   key=lambda x: global_asset_totals[x], reverse=True)
    
    # Top N assets globally + "Other" category
    global_top_assets = global_sorted_assets[:top_n]
    global_other_assets = set(global_sorted_assets[top_n:]) if len(global_sorted_assets) > top_n else set()
    
    # Create a FIXED color mapping for all assets
    all_labels = global_top_assets + (['Other'] if global_other_assets else [])
    color_map = {label: plt.cm.tab20(i / len(all_labels)) for i, label in enumerate(all_labels)}
    
    n_agents = len(strategies)
    n_cols = min(2, n_agents)
    n_rows = (n_agents + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    axes = axes.flatten()
    
    for idx, strat in enumerate(strategies):
        ax = axes[idx]
        
        # Build data using the GLOBAL asset ordering
        steps = np.arange(len(strat.weights_history))
        data = {}
        
        for asset in global_top_assets:
            data[asset] = [w.get(asset, 0.0) for w in strat.weights_history]
        
        if global_other_assets:
            data['Other'] = [sum(w.get(a, 0.0) for a in global_other_assets) 
                           for w in strat.weights_history]
        
        # Use fixed column order to match color_map
        df = pd.DataFrame(data, index=steps)[all_labels]
        colors = [color_map[col] for col in df.columns]
        
        df.plot.area(ax=ax, stacked=True, alpha=0.8, color=colors, legend=False)
        
        ax.set_xlabel('Day')
        ax.set_ylabel('Weight')
        ax.set_title(strat.name, fontweight='bold')
        ax.set_ylim(0, 1.05)
        ax.set_xlim(0, len(steps) - 1)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    
    # Hide unused axes
    for idx in range(len(strategies), len(axes)):
        axes[idx].set_visible(False)
    
    # Create legend using the global color mapping
    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=color_map[label], alpha=0.8, label=label) for label in all_labels]
    fig.legend(handles=legend_handles, loc='center right', bbox_to_anchor=(1.12, 0.5), fontsize=9)
    
    fig.suptitle(title, fontweight='bold', fontsize=14)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    return fig


# =============================================================================
# Helper Functions
# =============================================================================

def _find_window_boundaries(
    dates: List[np.datetime64], 
    validation_windows: List[Dict]
) -> List[Tuple[int, str]]:
    """Find step indices where validation windows change."""
    boundaries = []
    dates_pd = pd.to_datetime(dates)
    
    for window in validation_windows:
        window_start = pd.to_datetime(window['start'])
        window_end = pd.to_datetime(window['end'])
        
        for i, d in enumerate(dates_pd):
            if window_start <= d <= window_end:
                boundaries.append((i, window['name']))
                break
    
    boundaries.sort(key=lambda x: x[0])
    return boundaries


def save_all_figures(
    figures: Dict[str, plt.Figure],
    output_dir: Path,
    prefix: str = "",
    formats: List[str] = ['png', 'pdf'],
) -> List[Path]:
    """
    Save multiple figures to disk.
    
    Parameters
    ----------
    figures : Dict[str, plt.Figure]
        Mapping name -> figure
    output_dir : Path
        Output directory
    prefix : str
        Filename prefix
    formats : List[str]
        Output formats (e.g., ['png', 'pdf'])
        
    Returns
    -------
    List[Path]
        Saved file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved = []
    for name, fig in figures.items():
        for fmt in formats:
            filename = f"{prefix}{name}.{fmt}" if prefix else f"{name}.{fmt}"
            path = output_dir / filename
            fig.savefig(path, dpi=300, bbox_inches='tight', format=fmt)
            saved.append(path)
    
    print(f"Saved {len(saved)} figures to {output_dir}")
    return saved
