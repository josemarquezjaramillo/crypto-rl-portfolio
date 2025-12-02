#!/usr/bin/env python
"""
Full Evaluation Pipeline for Portfolio Agents.

Runs comprehensive evaluation comparing DQN/DDQN agents against baselines,
generates publication-ready tables and visualizations.

Usage:
    python -m evaluation.run_full_evaluation --split val
    python -m evaluation.run_full_evaluation --split test
    python -m evaluation.run_full_evaluation --split both
    python -m evaluation.run_full_evaluation --split test --detailed  # Full viz

Options:
    --split      : 'val', 'test', or 'both' (default: both)
    --seeds      : Number of seeds for evaluation (default: 5)
    --output     : Output directory (default: results)
    --no-viz     : Skip visualization generation
    --save-latex : Save LaTeX tables for paper
    --detailed   : Collect time-series data for enhanced visualizations
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt

# Ensure project root is in path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from evaluation.evaluator import (
    EvaluationConfig,
    Evaluator,
    DetailedAgentResult,
    create_equal_weight_agent,
    create_market_cap_agent,
    create_mean_variance_agent,
    create_dqn_agent,
    run_detailed_evaluation,
)
from evaluation.visualizer import (
    StrategyTimeSeries,
    plot_bar_comparison_with_ci,
    plot_multi_metric_comparison,
    plot_metric_heatmap,
    plot_portfolio_values,
    plot_drawdown,
    plot_allocation_evolution,
    plot_cumulative_returns_comparison,
    plot_rolling_sharpe,
    plot_daily_returns_distribution,
    plot_weight_evolution_selected,
    plot_multi_agent_allocation_comparison,
)
from evaluation.tables import (
    TableConfig,
    generate_latex_table,
    generate_markdown_table,
    save_all_tables,
)


def setup_evaluator(config: EvaluationConfig) -> Evaluator:
    """Create and configure the evaluator with all agents."""
    evaluator = Evaluator(config)
    
    # Register baselines
    evaluator.register_baseline("Equal Weight", create_equal_weight_agent)
    evaluator.register_baseline("Market Cap", 
        lambda seed, env: create_market_cap_agent(seed, env, max_weight=0.35))
    evaluator.register_baseline("Mean-Variance", 
        lambda seed, env: create_mean_variance_agent(seed, env, risk_aversion=1.0, max_weight=0.35))
    
    # Register RL agents (check if checkpoints exist)
    dqn_path = Path("checkpoints/dqn_production/best")
    ddqn_path = Path("checkpoints/ddqn_production/best")
    
    if dqn_path.exists():
        evaluator.register_rl_agent("DQN", 
            lambda seed, env, cp, dev: create_dqn_agent(seed, env, cp, dev, use_double=False),
            dqn_path)
        print(f"  ✓ Registered DQN agent (checkpoint: {dqn_path})")
    else:
        print(f"  ✗ DQN checkpoint not found: {dqn_path}")
    
    if ddqn_path.exists():
        evaluator.register_rl_agent("DDQN",
            lambda seed, env, cp, dev: create_dqn_agent(seed, env, cp, dev, use_double=True),
            ddqn_path)
        print(f"  ✓ Registered DDQN agent (checkpoint: {ddqn_path})")
    else:
        print(f"  ✗ DDQN checkpoint not found: {ddqn_path}")
    
    return evaluator


def create_visualizations(
    results_df: pd.DataFrame,
    output_dir: Path,
    split: str,
    timestamp: str,
):
    """Generate all visualizations from results."""
    viz_dir = output_dir / "visualizations" / split
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nGenerating visualizations in: {viz_dir}")
    
    # Filter to combined/test_full window
    target_window = "combined" if split == "val" else "test_full"
    df = results_df[results_df['Window'] == target_window].copy()
    
    if df.empty:
        print("  No results for target window, skipping visualizations")
        return
    
    # 1. Return comparison bar chart
    try:
        fig = plot_bar_comparison_with_ci(
            df,
            metric='Return (%)',
            ci_lower_col='Return CI Low (%)',
            ci_upper_col='Return CI High (%)',
            title=f'Cumulative Return Comparison ({split.upper()} Set)',
            figsize=(10, 6),
        )
        fig.savefig(viz_dir / f'return_comparison_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ Saved return comparison chart")
    except Exception as e:
        print(f"  ✗ Return chart failed: {e}")
    
    # 2. Sharpe comparison bar chart
    try:
        fig = plot_bar_comparison_with_ci(
            df,
            metric='Sharpe',
            ci_lower_col='Sharpe CI Low',
            ci_upper_col='Sharpe CI High',
            title=f'Sharpe Ratio Comparison ({split.upper()} Set)',
            figsize=(10, 6),
        )
        fig.savefig(viz_dir / f'sharpe_comparison_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ Saved Sharpe comparison chart")
    except Exception as e:
        print(f"  ✗ Sharpe chart failed: {e}")
    
    # 3. Multi-metric comparison
    try:
        fig = plot_multi_metric_comparison(
            df,
            metrics=['Return (%)', 'Sharpe', 'Max DD (%)', 'Sortino'],
            title=f'Multi-Metric Comparison ({split.upper()} Set)',
            figsize=(16, 5),
        )
        fig.savefig(viz_dir / f'multi_metric_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ Saved multi-metric comparison chart")
    except Exception as e:
        print(f"  ✗ Multi-metric chart failed: {e}")
    
    # 4. Metrics heatmap
    try:
        agents = df['Agent'].tolist()
        metrics = ['Return (%)', 'CAGR (%)', 'Sharpe', 'Sortino', 'Calmar', 'Max DD (%)', 'Volatility (%)', 'Hit Rate (%)']
        available_metrics = [m for m in metrics if m in df.columns]
        
        fig = plot_metric_heatmap(
            df,
            agents=agents,
            metrics=available_metrics,
            title=f'Performance Metrics Heatmap ({split.upper()} Set)',
            figsize=(12, 6),
        )
        fig.savefig(viz_dir / f'metrics_heatmap_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ Saved metrics heatmap")
    except Exception as e:
        print(f"  ✗ Heatmap failed: {e}")
    
    print(f"  Visualizations saved to: {viz_dir}")


def create_timeseries_visualizations(
    detailed_results: Dict[str, DetailedAgentResult],
    output_dir: Path,
    split: str,
    timestamp: str,
):
    """
    Generate time-series visualizations from detailed evaluation results.
    
    These are the publication-quality charts that require per-step data:
    - Cumulative returns evolution
    - Drawdown evolution
    - Rolling Sharpe ratio
    - Daily returns distribution
    - Asset allocation charts
    """
    viz_dir = output_dir / "visualizations" / split
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nGenerating time-series visualizations in: {viz_dir}")
    
    if not detailed_results:
        print("  No detailed results available, skipping time-series visualizations")
        return
    
    # Extract portfolio values dict for multi-agent charts
    multi_agent_pvs = {
        name: result.portfolio_values
        for name, result in detailed_results.items()
    }
    
    # 1. Cumulative Returns Comparison (main chart for papers)
    try:
        fig = plot_cumulative_returns_comparison(
            multi_agent_pvs,
            title=f'Cumulative Returns Comparison ({split.upper()} Set)',
            show_drawdown=True,
        )
        fig.savefig(viz_dir / f'cumulative_returns_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ Saved cumulative returns chart")
    except Exception as e:
        print(f"  ✗ Cumulative returns chart failed: {e}")
    
    # 2. Rolling Sharpe Ratio
    try:
        fig = plot_rolling_sharpe(
            multi_agent_pvs,
            window=30,
            title=f'Rolling 30-Day Sharpe Ratio ({split.upper()} Set)',
        )
        fig.savefig(viz_dir / f'rolling_sharpe_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ Saved rolling Sharpe chart")
    except Exception as e:
        print(f"  ✗ Rolling Sharpe chart failed: {e}")
    
    # 3. Daily Returns Distribution
    try:
        fig = plot_daily_returns_distribution(
            multi_agent_pvs,
            title=f'Daily Returns Distribution ({split.upper()} Set)',
            bins=40,
        )
        fig.savefig(viz_dir / f'returns_distribution_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ Saved returns distribution chart")
    except Exception as e:
        print(f"  ✗ Returns distribution chart failed: {e}")
    
    # 4. Allocation charts for RL agents
    rl_agents = [name for name, r in detailed_results.items() if r.agent_type == "rl"]
    
    for agent_name in rl_agents:
        try:
            result = detailed_results[agent_name]
            strategy_ts = result.to_strategy_timeseries()
            
            # Stacked area allocation chart
            fig = plot_allocation_evolution(
                strategy_ts,
                top_n=10,
                title=f'Portfolio Allocation - {agent_name} ({split.upper()} Set)',
            )
            safe_name = agent_name.replace(" ", "_").lower()
            fig.savefig(viz_dir / f'allocation_{safe_name}_{timestamp}.png', dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"  ✓ Saved allocation chart for {agent_name}")
            
            # Weight evolution line chart
            fig = plot_weight_evolution_selected(
                strategy_ts,
                top_n=5,
                title=f'Top Asset Weights - {agent_name} ({split.upper()} Set)',
            )
            fig.savefig(viz_dir / f'weights_{safe_name}_{timestamp}.png', dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"  ✓ Saved weight evolution chart for {agent_name}")
            
        except Exception as e:
            print(f"  ✗ Allocation charts for {agent_name} failed: {e}")
    
    # 5. Multi-agent allocation comparison (if we have multiple strategies with weights)
    strategies_with_weights = []
    for name, result in detailed_results.items():
        if result.weights_history and len(result.weights_history) > 1:
            strategies_with_weights.append(result.to_strategy_timeseries())
    
    if len(strategies_with_weights) >= 2:
        try:
            fig = plot_multi_agent_allocation_comparison(
                strategies_with_weights[:4],  # Limit to 4 for layout
                top_n=8,
                title=f'Allocation Strategies Comparison ({split.upper()} Set)',
            )
            fig.savefig(viz_dir / f'allocation_comparison_{timestamp}.png', dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"  ✓ Saved allocation comparison chart")
        except Exception as e:
            print(f"  ✗ Allocation comparison chart failed: {e}")
    
    # 6. Portfolio value evolution (simple line chart)
    try:
        strategies_ts = [result.to_strategy_timeseries() for result in detailed_results.values()]
        fig = plot_portfolio_values(
            strategies_ts,
            title=f'Portfolio Value Evolution ({split.upper()} Set)',
        )
        fig.savefig(viz_dir / f'portfolio_values_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ Saved portfolio values chart")
    except Exception as e:
        print(f"  ✗ Portfolio values chart failed: {e}")
    
    # 7. Drawdown chart
    try:
        fig = plot_drawdown(
            strategies_ts,
            title=f'Drawdown Evolution ({split.upper()} Set)',
        )
        fig.savefig(viz_dir / f'drawdown_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ Saved drawdown chart")
    except Exception as e:
        print(f"  ✗ Drawdown chart failed: {e}")
    
    print(f"  Time-series visualizations saved to: {viz_dir}")


def create_tables(
    results_df: pd.DataFrame,
    output_dir: Path,
    split: str,
    timestamp: str,
    save_latex: bool = False,
):
    """Generate publication-ready tables."""
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nGenerating tables in: {tables_dir}")
    
    # Filter to combined/test_full window
    target_window = "combined" if split == "val" else "test_full"
    df = results_df[results_df['Window'] == target_window].copy()
    
    if df.empty:
        print("  No results for target window, skipping tables")
        return
    
    # Table config
    config = TableConfig(
        title=f"Agent Performance Comparison ({split.upper()} Set)",
        metrics=['Return (%)', 'CAGR (%)', 'Sharpe', 'Sortino', 'Max DD (%)', 'Turnover (%)'],
        highlight_best=True,
    )
    
    # Markdown table (for README)
    md_path = tables_dir / f'{split}_results_{timestamp}.md'
    generate_markdown_table(df, config, md_path)
    print(f"  ✓ Saved Markdown table: {md_path.name}")
    
    # LaTeX table (for paper)
    if save_latex:
        tex_path = tables_dir / f'{split}_results_{timestamp}.tex'
        generate_latex_table(df, config, tex_path)
        print(f"  ✓ Saved LaTeX table: {tex_path.name}")


def run_evaluation_pipeline(
    split: str,
    n_seeds: int,
    output_dir: Path,
    generate_viz: bool = True,
    save_latex: bool = False,
    verbose: bool = True,
    detailed: bool = False,
):
    """Run the full evaluation pipeline for a given split."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("\n" + "=" * 80)
    print(f"  EVALUATION PIPELINE: {split.upper()} SET")
    print("=" * 80)
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    
    # Seeds
    seeds = list(range(42, 42 + n_seeds * 111, 111))[:n_seeds]  # [42, 153, 264, ...]
    print(f"Seeds: {seeds}")
    
    # Config
    config = EvaluationConfig(
        seeds=seeds,
        output_dir=output_dir,
        device=device,
    )
    
    # Setup evaluator
    print("\nRegistering agents...")
    evaluator = setup_evaluator(config)
    
    # Run evaluation
    print(f"\nRunning evaluation on {split} set...")
    results = evaluator.run_evaluation(split=split, verbose=verbose)
    
    # Print publication table
    evaluator.print_publication_table(results)
    
    # Save CSV results
    summary_path, raw_path = evaluator.save_results(results, prefix=split)
    
    # Load summary for visualizations/tables
    summary_df = pd.read_csv(summary_path)
    
    # Generate visualizations
    if generate_viz:
        create_visualizations(summary_df, output_dir, split, timestamp)
    
    # Generate tables
    create_tables(summary_df, output_dir, split, timestamp, save_latex)
    
    # Run detailed evaluation for time-series charts if requested
    detailed_results = None
    if detailed and generate_viz:
        print(f"\nRunning detailed evaluation for time-series charts...")
        detailed_results = run_detailed_pipeline(
            evaluator=evaluator,
            split=split,
            seed=seeds[0],  # Use first seed for detailed viz
            device=device,
        )
        
        if detailed_results:
            create_timeseries_visualizations(
                detailed_results, output_dir, split, timestamp
            )
    
    return results, summary_df, detailed_results


def run_detailed_pipeline(
    evaluator: Evaluator,
    split: str,
    seed: int,
    device: str,
) -> Dict[str, DetailedAgentResult]:
    """
    Run detailed evaluation for a single seed to collect time-series data.
    
    This runs all registered agents once with full data collection for
    generating publication-quality time-series visualizations.
    """
    from data.dataset_loader import load_exported_dataset
    from data.dataset_backend import DatasetBackend
    from environment.environment import PortfolioEnv, EnvConfig
    
    detailed_results = {}
    
    # Load dataset
    ds = evaluator.ds_test if split == "test" else evaluator.ds_dev
    
    # Create backend
    if split == "test":
        backend = DatasetBackend(ds)
    else:
        # Combine all validation windows for detailed eval
        all_tags = [f"val_window_{w['name']}" for w in evaluator.validation_windows]
        backend = DatasetBackend(ds, split_tag_filter=all_tags)
    
    # Create environment - use continuous mode (default) for baselines
    env_cfg = EnvConfig(
        split=split,
        cost_rate=evaluator.config.cost_rate,
        turnover_cap=evaluator.config.turnover_cap,
        max_weight_per_asset=evaluator.config.max_weight_per_asset,
        strict_projection=True,
        random_seed=seed,
    )
    env = PortfolioEnv(env_cfg, backend)
    
    # Run each baseline
    for name, factory in evaluator._baselines.items():
        try:
            print(f"  Running detailed eval: {name}...")
            agent = factory(seed, env)
            result = run_detailed_evaluation(
                agent=agent,
                env=env,
                seed=seed,
                agent_name=name,
                agent_type="baseline",
                window_name=f"{split}_detailed",
            )
            detailed_results[name] = result
            env.reset(seed=seed)  # Reset for next agent
        except Exception as e:
            print(f"    ✗ {name} failed: {e}")
    
    # Run each RL agent
    for name, (factory, checkpoint_path) in evaluator._rl_agents.items():
        try:
            print(f"  Running detailed eval: {name}...")
            agent = factory(seed, env, checkpoint_path, device)
            result = run_detailed_evaluation(
                agent=agent,
                env=env,
                seed=seed,
                agent_name=name,
                agent_type="rl",
                window_name=f"{split}_detailed",
            )
            detailed_results[name] = result
            env.reset(seed=seed)  # Reset for next agent
        except Exception as e:
            print(f"    ✗ {name} failed: {e}")
    
    print(f"  Detailed evaluation complete: {len(detailed_results)} agents")
    return detailed_results
    
    print(f"  Detailed evaluation complete: {len(detailed_results)} agents")
    return detailed_results


def main():
    parser = argparse.ArgumentParser(
        description="Run full evaluation pipeline for portfolio agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Evaluate on validation set
    python -m evaluation.run_full_evaluation --split val
    
    # Evaluate on test set with 5 seeds
    python -m evaluation.run_full_evaluation --split test --seeds 5
    
    # Full evaluation with LaTeX tables
    python -m evaluation.run_full_evaluation --split both --save-latex
    
    # Quick evaluation without visualizations
    python -m evaluation.run_full_evaluation --split val --seeds 3 --no-viz
    
    # Detailed evaluation with time-series charts (allocation, cumulative returns, etc.)
    python -m evaluation.run_full_evaluation --split test --detailed
        """
    )
    
    parser.add_argument(
        "--split", type=str, default="both",
        choices=["val", "test", "both"],
        help="Dataset split to evaluate (default: both)"
    )
    parser.add_argument(
        "--seeds", type=int, default=5,
        help="Number of seeds for evaluation (default: 5)"
    )
    parser.add_argument(
        "--output", type=str, default="results",
        help="Output directory (default: results)"
    )
    parser.add_argument(
        "--no-viz", action="store_true",
        help="Skip visualization generation"
    )
    parser.add_argument(
        "--save-latex", action="store_true",
        help="Save LaTeX tables for paper"
    )
    parser.add_argument(
        "--detailed", action="store_true",
        help="Generate detailed time-series visualizations (allocation, cumulative returns, etc.)"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Reduce output verbosity"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("  PORTFOLIO AGENT EVALUATION PIPELINE")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Split: {args.split}")
    print(f"  Seeds: {args.seeds}")
    print(f"  Output: {output_dir}")
    print(f"  Visualizations: {'No' if args.no_viz else 'Yes'}")
    print(f"  Detailed charts: {'Yes' if args.detailed else 'No'}")
    print(f"  LaTeX tables: {'Yes' if args.save_latex else 'No'}")
    
    # Run evaluations
    all_results = {}
    
    if args.split in ["val", "both"]:
        results, df, detailed = run_evaluation_pipeline(
            split="val",
            n_seeds=args.seeds,
            output_dir=output_dir,
            generate_viz=not args.no_viz,
            save_latex=args.save_latex,
            verbose=not args.quiet,
            detailed=args.detailed,
        )
        all_results["val"] = results
    
    if args.split in ["test", "both"]:
        results, df, detailed = run_evaluation_pipeline(
            split="test",
            n_seeds=args.seeds,
            output_dir=output_dir,
            generate_viz=not args.no_viz,
            save_latex=args.save_latex,
            verbose=not args.quiet,
            detailed=args.detailed,
        )
        all_results["test"] = results
    
    # Final summary
    print("\n" + "=" * 80)
    print("  EVALUATION COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}")
    print(f"\nOutput files:")
    for f in sorted(output_dir.glob("*_summary_*.csv")):
        print(f"  - {f.name}")
    
    if not args.no_viz:
        print(f"\nVisualizations:")
        for f in sorted(output_dir.glob("visualizations/**/*.png")):
            print(f"  - {f.relative_to(output_dir)}")
    
    print(f"\nTables:")
    for f in sorted(output_dir.glob("tables/*")):
        print(f"  - {f.relative_to(output_dir)}")


if __name__ == "__main__":
    main()
