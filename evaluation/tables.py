"""
Publication-ready table generation for portfolio evaluation.

Generates tables in multiple formats:
- LaTeX (for academic papers)
- Markdown (for documentation/README)
- Plain text (for console output)
- HTML (for web reports)

Table styles follow academic paper conventions:
- Jiang et al. (2017) - EIIE paper table format
- Lucarelli & Borrotti (2020) - DQN crypto paper format
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np
import pandas as pd
from datetime import datetime


# =============================================================================
# Table Configuration
# =============================================================================

@dataclass
class TableConfig:
    """
    Configuration for table formatting.
    
    Attributes
    ----------
    title : str
        Table title/caption
    metrics : List[str]
        Metrics to include
    metric_formats : Dict[str, str]
        Format strings for each metric
    show_ci : bool
        Whether to show confidence intervals
    highlight_best : bool
        Whether to bold the best value in each metric column
    sort_by : str
        Column to sort by
    ascending : bool
        Sort order
    """
    title: str = "Agent Performance Comparison"
    metrics: List[str] = None
    metric_formats: Dict[str, str] = None
    show_ci: bool = True
    highlight_best: bool = True
    sort_by: Optional[str] = None
    ascending: bool = False
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = [
                'Return (%)', 'CAGR (%)', 'Sharpe', 'Sortino',
                'Calmar', 'Max DD (%)', 'Vol (%)', 'Hit Rate (%)'
            ]
        if self.metric_formats is None:
            self.metric_formats = {
                'Return (%)': '.2f',
                'CAGR (%)': '.2f',
                'Sharpe': '.3f',
                'Sortino': '.3f',
                'Calmar': '.3f',
                'Max DD (%)': '.2f',
                'Vol (%)': '.2f',
                'Hit Rate (%)': '.1f',
                'Turnover (%)': '.2f',
                'Costs (%)': '.4f',
            }


# =============================================================================
# Main Table Generation Functions
# =============================================================================

def generate_latex_table(
    results_df: pd.DataFrame,
    config: Optional[TableConfig] = None,
    output_path: Optional[Path] = None,
) -> str:
    """
    Generate LaTeX table from results DataFrame.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Results with Agent, Type, and metric columns
    config : TableConfig, optional
        Table configuration
    output_path : Path, optional
        If provided, save to file
        
    Returns
    -------
    str
        LaTeX table code
    """
    config = config or TableConfig()
    df = _prepare_dataframe(results_df, config)
    
    # Determine column alignment
    n_metrics = len([m for m in config.metrics if m in df.columns])
    col_spec = 'l' + 'r' * n_metrics  # Agent left-aligned, metrics right-aligned
    
    lines = [
        r'\begin{table}[htbp]',
        r'\centering',
        f'\\caption{{{config.title}}}',
        r'\label{tab:agent_comparison}',
        f'\\begin{{tabular}}{{{col_spec}}}',
        r'\toprule',
    ]
    
    # Header row
    header_cols = ['Agent'] + [m for m in config.metrics if m in df.columns]
    lines.append(' & '.join(header_cols) + r' \\')
    lines.append(r'\midrule')
    
    # Determine best values for highlighting
    best_vals = {}
    if config.highlight_best:
        for m in config.metrics:
            if m in df.columns:
                col = df[m]
                # Higher is better for most metrics, lower for Max DD, Vol
                if 'DD' in m or 'Vol' in m:
                    best_vals[m] = col.min()
                else:
                    best_vals[m] = col.max()
    
    # Data rows
    for _, row in df.iterrows():
        cells = [row['Agent']]
        for m in config.metrics:
            if m not in df.columns:
                continue
            val = row[m]
            fmt = config.metric_formats.get(m, '.2f')
            formatted = f'{val:{fmt}}'
            
            # Highlight best
            if config.highlight_best and m in best_vals:
                if np.isclose(val, best_vals[m], rtol=1e-5):
                    formatted = r'\textbf{' + formatted + '}'
            
            cells.append(formatted)
        
        lines.append(' & '.join(cells) + r' \\')
    
    lines.extend([
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}',
    ])
    
    latex = '\n'.join(lines)
    
    if output_path:
        Path(output_path).write_text(latex)
        print(f"Saved LaTeX table to: {output_path}")
    
    return latex


def generate_markdown_table(
    results_df: pd.DataFrame,
    config: Optional[TableConfig] = None,
    output_path: Optional[Path] = None,
) -> str:
    """
    Generate Markdown table from results DataFrame.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Results with Agent, Type, and metric columns
    config : TableConfig, optional
        Table configuration
    output_path : Path, optional
        If provided, save to file
        
    Returns
    -------
    str
        Markdown table
    """
    config = config or TableConfig()
    df = _prepare_dataframe(results_df, config)
    
    header_cols = ['Agent'] + [m for m in config.metrics if m in df.columns]
    
    # Header
    lines = [
        f'## {config.title}',
        '',
        '| ' + ' | '.join(header_cols) + ' |',
        '|' + '|'.join(['---:' if i > 0 else ':---' for i in range(len(header_cols))]) + '|',
    ]
    
    # Best values
    best_vals = {}
    if config.highlight_best:
        for m in config.metrics:
            if m in df.columns:
                col = df[m]
                if 'DD' in m or 'Vol' in m:
                    best_vals[m] = col.min()
                else:
                    best_vals[m] = col.max()
    
    # Data rows
    for _, row in df.iterrows():
        cells = [row['Agent']]
        for m in config.metrics:
            if m not in df.columns:
                continue
            val = row[m]
            fmt = config.metric_formats.get(m, '.2f')
            formatted = f'{val:{fmt}}'
            
            if config.highlight_best and m in best_vals:
                if np.isclose(val, best_vals[m], rtol=1e-5):
                    formatted = f'**{formatted}**'
            
            cells.append(formatted)
        
        lines.append('| ' + ' | '.join(cells) + ' |')
    
    markdown = '\n'.join(lines)
    
    if output_path:
        Path(output_path).write_text(markdown)
        print(f"Saved Markdown table to: {output_path}")
    
    return markdown


def generate_text_table(
    results_df: pd.DataFrame,
    config: Optional[TableConfig] = None,
    output_path: Optional[Path] = None,
) -> str:
    """
    Generate plain text table for console output.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Results with Agent, Type, and metric columns
    config : TableConfig, optional
        Table configuration
    output_path : Path, optional
        If provided, save to file
        
    Returns
    -------
    str
        Plain text table
    """
    config = config or TableConfig()
    df = _prepare_dataframe(results_df, config)
    
    # Column widths
    header_cols = ['Agent'] + [m for m in config.metrics if m in df.columns]
    widths = {col: max(len(col), 10) for col in header_cols}
    widths['Agent'] = max(len('Agent'), df['Agent'].str.len().max())
    
    # Build table
    border = '+' + '+'.join(['-' * (widths[c] + 2) for c in header_cols]) + '+'
    
    lines = [
        '',
        f'  {config.title}',
        '  ' + '=' * (sum(widths.values()) + 3 * len(header_cols) - 1),
        '',
        border,
        '|' + '|'.join([f' {c:^{widths[c]}} ' for c in header_cols]) + '|',
        border.replace('-', '='),
    ]
    
    # Best values
    best_vals = {}
    if config.highlight_best:
        for m in config.metrics:
            if m in df.columns:
                col = df[m]
                if 'DD' in m or 'Vol' in m:
                    best_vals[m] = col.min()
                else:
                    best_vals[m] = col.max()
    
    for _, row in df.iterrows():
        cells = [f' {row["Agent"]:<{widths["Agent"]}} ']
        for m in config.metrics:
            if m not in df.columns:
                continue
            val = row[m]
            fmt = config.metric_formats.get(m, '.2f')
            formatted = f'{val:{fmt}}'
            
            # Mark best with asterisk
            if config.highlight_best and m in best_vals:
                if np.isclose(val, best_vals[m], rtol=1e-5):
                    formatted = f'{formatted}*'
            
            cells.append(f' {formatted:>{widths[m]}} ')
        
        lines.append('|' + '|'.join(cells) + '|')
    
    lines.extend([border, '', '  * indicates best value in column', ''])
    
    text = '\n'.join(lines)
    
    if output_path:
        Path(output_path).write_text(text)
        print(f"Saved text table to: {output_path}")
    
    return text


def generate_html_table(
    results_df: pd.DataFrame,
    config: Optional[TableConfig] = None,
    output_path: Optional[Path] = None,
) -> str:
    """
    Generate HTML table from results DataFrame.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Results with Agent, Type, and metric columns
    config : TableConfig, optional
        Table configuration
    output_path : Path, optional
        If provided, save to file
        
    Returns
    -------
    str
        HTML table code
    """
    config = config or TableConfig()
    df = _prepare_dataframe(results_df, config)
    
    header_cols = ['Agent'] + [m for m in config.metrics if m in df.columns]
    
    # CSS styles
    style = """
    <style>
        .results-table {
            border-collapse: collapse;
            font-family: Arial, sans-serif;
            margin: 20px 0;
        }
        .results-table th, .results-table td {
            border: 1px solid #ddd;
            padding: 8px 12px;
            text-align: right;
        }
        .results-table th {
            background-color: #4472C4;
            color: white;
            font-weight: bold;
        }
        .results-table td:first-child {
            text-align: left;
            font-weight: bold;
        }
        .results-table tr:nth-child(even) {
            background-color: #f2f2f2;
        }
        .results-table tr:hover {
            background-color: #ddd;
        }
        .best-value {
            font-weight: bold;
            color: #2E7D32;
        }
        .baseline-row {
            background-color: #E3F2FD !important;
        }
        .rl-row {
            background-color: #FFF3E0 !important;
        }
    </style>
    """
    
    lines = [
        style,
        f'<h3>{config.title}</h3>',
        '<table class="results-table">',
        '<thead><tr>',
    ]
    
    # Header
    for col in header_cols:
        lines.append(f'<th>{col}</th>')
    lines.append('</tr></thead>')
    lines.append('<tbody>')
    
    # Best values
    best_vals = {}
    if config.highlight_best:
        for m in config.metrics:
            if m in df.columns:
                col = df[m]
                if 'DD' in m or 'Vol' in m:
                    best_vals[m] = col.min()
                else:
                    best_vals[m] = col.max()
    
    # Rows
    for _, row in df.iterrows():
        row_class = 'baseline-row' if row.get('Type') == 'baseline' else 'rl-row'
        lines.append(f'<tr class="{row_class}">')
        lines.append(f'<td>{row["Agent"]}</td>')
        
        for m in config.metrics:
            if m not in df.columns:
                continue
            val = row[m]
            fmt = config.metric_formats.get(m, '.2f')
            formatted = f'{val:{fmt}}'
            
            cell_class = ''
            if config.highlight_best and m in best_vals:
                if np.isclose(val, best_vals[m], rtol=1e-5):
                    cell_class = ' class="best-value"'
            
            lines.append(f'<td{cell_class}>{formatted}</td>')
        
        lines.append('</tr>')
    
    lines.extend(['</tbody>', '</table>'])
    
    html = '\n'.join(lines)
    
    if output_path:
        Path(output_path).write_text(html)
        print(f"Saved HTML table to: {output_path}")
    
    return html


# =============================================================================
# Specialized Table Generators
# =============================================================================

def generate_comparison_table_with_ci(
    results_df: pd.DataFrame,
    main_metrics: List[str] = None,
    output_format: str = 'latex',
    output_path: Optional[Path] = None,
) -> str:
    """
    Generate table showing metrics with 95% confidence intervals.
    
    Format: "value (±margin)" or "value [ci_low, ci_high]"
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Results with metric and CI columns
    main_metrics : List[str]
        Main metrics (CI columns assumed to be f"{metric} CI Low/High")
    output_format : str
        'latex', 'markdown', or 'text'
    output_path : Path, optional
        If provided, save to file
        
    Returns
    -------
    str
        Table in specified format
    """
    main_metrics = main_metrics or ['Return (%)', 'Sharpe']
    
    lines = []
    
    if output_format == 'latex':
        col_spec = 'l' + 'c' * len(main_metrics)
        lines = [
            r'\begin{table}[htbp]',
            r'\centering',
            r'\caption{Agent Performance with 95\% Confidence Intervals}',
            r'\label{tab:agent_ci}',
            f'\\begin{{tabular}}{{{col_spec}}}',
            r'\toprule',
            'Agent & ' + ' & '.join(main_metrics) + r' \\',
            r'\midrule',
        ]
        
        for _, row in results_df.iterrows():
            cells = [row['Agent']]
            for m in main_metrics:
                val = row[m]
                ci_low_col = f"{m.replace(' (%)', '')} CI Low" if '(%)' in m else f"{m} CI Low"
                ci_high_col = f"{m.replace(' (%)', '')} CI High" if '(%)' in m else f"{m} CI High"
                
                # Try different naming conventions
                if ci_low_col not in row.index:
                    ci_low_col = m.replace(' (%)', ' CI Low (%)').replace('Sharpe', 'Sharpe CI Low')
                    ci_high_col = m.replace(' (%)', ' CI High (%)').replace('Sharpe', 'Sharpe CI High')
                
                if ci_low_col in row.index and ci_high_col in row.index:
                    ci_low = row[ci_low_col]
                    ci_high = row[ci_high_col]
                    margin = (ci_high - ci_low) / 2
                    fmt = '.2f' if '%' in m else '.3f'
                    cells.append(f'${val:{fmt}} \\pm {margin:{fmt}}$')
                else:
                    fmt = '.2f' if '%' in m else '.3f'
                    cells.append(f'{val:{fmt}}')
            
            lines.append(' & '.join(cells) + r' \\')
        
        lines.extend([r'\bottomrule', r'\end{tabular}', r'\end{table}'])
        
    elif output_format == 'markdown':
        lines = [
            '## Agent Performance with 95% Confidence Intervals',
            '',
            '| Agent | ' + ' | '.join(main_metrics) + ' |',
            '|:---' + '|:---:' * len(main_metrics) + '|',
        ]
        
        for _, row in results_df.iterrows():
            cells = [row['Agent']]
            for m in main_metrics:
                val = row[m]
                ci_low_col = f"{m.replace(' (%)', '')} CI Low" if '(%)' in m else f"{m} CI Low"
                ci_high_col = f"{m.replace(' (%)', '')} CI High" if '(%)' in m else f"{m} CI High"
                
                if ci_low_col not in row.index:
                    ci_low_col = m.replace(' (%)', ' CI Low (%)').replace('Sharpe', 'Sharpe CI Low')
                    ci_high_col = m.replace(' (%)', ' CI High (%)').replace('Sharpe', 'Sharpe CI High')
                
                if ci_low_col in row.index and ci_high_col in row.index:
                    ci_low = row[ci_low_col]
                    ci_high = row[ci_high_col]
                    margin = (ci_high - ci_low) / 2
                    fmt = '.2f' if '%' in m else '.3f'
                    cells.append(f'{val:{fmt}} ± {margin:{fmt}}')
                else:
                    fmt = '.2f' if '%' in m else '.3f'
                    cells.append(f'{val:{fmt}}')
            
            lines.append('| ' + ' | '.join(cells) + ' |')
    
    else:  # text
        lines = [
            '',
            '  Agent Performance with 95% Confidence Intervals',
            '  ' + '=' * 50,
            '',
        ]
        
        header = 'Agent'.ljust(20) + ''.join([m.center(15) for m in main_metrics])
        lines.extend([header, '-' * len(header)])
        
        for _, row in results_df.iterrows():
            cells = [row['Agent'].ljust(20)]
            for m in main_metrics:
                val = row[m]
                ci_low_col = f"{m.replace(' (%)', '')} CI Low" if '(%)' in m else f"{m} CI Low"
                ci_high_col = f"{m.replace(' (%)', '')} CI High" if '(%)' in m else f"{m} CI High"
                
                if ci_low_col not in row.index:
                    ci_low_col = m.replace(' (%)', ' CI Low (%)').replace('Sharpe', 'Sharpe CI Low')
                    ci_high_col = m.replace(' (%)', ' CI High (%)').replace('Sharpe', 'Sharpe CI High')
                
                if ci_low_col in row.index and ci_high_col in row.index:
                    margin = (row[ci_high_col] - row[ci_low_col]) / 2
                    fmt = '.2f' if '%' in m else '.3f'
                    cells.append(f'{val:{fmt}}±{margin:{fmt}}'.center(15))
                else:
                    fmt = '.2f' if '%' in m else '.3f'
                    cells.append(f'{val:{fmt}}'.center(15))
            
            lines.append(''.join(cells))
    
    result = '\n'.join(lines)
    
    if output_path:
        Path(output_path).write_text(result)
        print(f"Saved table to: {output_path}")
    
    return result


def generate_per_window_table(
    results_df: pd.DataFrame,
    agent_name: str,
    metrics: List[str] = None,
    output_format: str = 'latex',
    output_path: Optional[Path] = None,
) -> str:
    """
    Generate table showing one agent's performance across validation windows.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Results with Window column
    agent_name : str
        Agent to generate table for
    metrics : List[str]
        Metrics to include
    output_format : str
        'latex', 'markdown', or 'text'
    output_path : Path, optional
        If provided, save to file
        
    Returns
    -------
    str
        Table in specified format
    """
    metrics = metrics or ['Return (%)', 'Sharpe', 'Max DD (%)']
    
    agent_df = results_df[results_df['Agent'] == agent_name].copy()
    if agent_df.empty:
        return f"No results found for agent: {agent_name}"
    
    lines = []
    
    if output_format == 'latex':
        col_spec = 'l' + 'r' * len(metrics)
        lines = [
            r'\begin{table}[htbp]',
            r'\centering',
            f'\\caption{{Performance by Validation Window: {agent_name}}}',
            r'\label{tab:per_window}',
            f'\\begin{{tabular}}{{{col_spec}}}',
            r'\toprule',
            'Window & ' + ' & '.join(metrics) + r' \\',
            r'\midrule',
        ]
        
        for _, row in agent_df.iterrows():
            cells = [row['Window'].replace('_', '\\_')]
            for m in metrics:
                if m in row.index:
                    fmt = '.2f' if '%' in m else '.3f'
                    cells.append(f'{row[m]:{fmt}}')
                else:
                    cells.append('--')
            lines.append(' & '.join(cells) + r' \\')
        
        lines.extend([r'\bottomrule', r'\end{tabular}', r'\end{table}'])
        
    elif output_format == 'markdown':
        lines = [
            f'## Performance by Validation Window: {agent_name}',
            '',
            '| Window | ' + ' | '.join(metrics) + ' |',
            '|:---' + '|---:' * len(metrics) + '|',
        ]
        
        for _, row in agent_df.iterrows():
            cells = [row['Window']]
            for m in metrics:
                if m in row.index:
                    fmt = '.2f' if '%' in m else '.3f'
                    cells.append(f'{row[m]:{fmt}}')
                else:
                    cells.append('--')
            lines.append('| ' + ' | '.join(cells) + ' |')
    
    else:  # text
        lines = [
            '',
            f'  Performance by Validation Window: {agent_name}',
            '  ' + '=' * 50,
            '',
        ]
        
        header = 'Window'.ljust(15) + ''.join([m.rjust(12) for m in metrics])
        lines.extend([header, '-' * len(header)])
        
        for _, row in agent_df.iterrows():
            cells = [row['Window'].ljust(15)]
            for m in metrics:
                if m in row.index:
                    fmt = '.2f' if '%' in m else '.3f'
                    cells.append(f'{row[m]:{fmt}}'.rjust(12))
                else:
                    cells.append('--'.rjust(12))
            lines.append(''.join(cells))
    
    result = '\n'.join(lines)
    
    if output_path:
        Path(output_path).write_text(result)
        print(f"Saved table to: {output_path}")
    
    return result


# =============================================================================
# Helper Functions
# =============================================================================

def _prepare_dataframe(
    results_df: pd.DataFrame,
    config: TableConfig,
) -> pd.DataFrame:
    """Prepare DataFrame for table generation."""
    df = results_df.copy()
    
    # Filter to available metrics
    available_metrics = [m for m in config.metrics if m in df.columns]
    keep_cols = ['Agent', 'Type'] + available_metrics
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols]
    
    # Sort if specified
    if config.sort_by and config.sort_by in df.columns:
        df = df.sort_values(config.sort_by, ascending=config.ascending)
    
    return df


def save_all_tables(
    results_df: pd.DataFrame,
    output_dir: Path,
    prefix: str = "",
    formats: List[str] = ['latex', 'markdown', 'text'],
) -> List[Path]:
    """
    Save results tables in multiple formats.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Results DataFrame
    output_dir : Path
        Output directory
    prefix : str
        Filename prefix
    formats : List[str]
        Output formats
        
    Returns
    -------
    List[Path]
        Saved file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config = TableConfig()
    saved = []
    
    generators = {
        'latex': (generate_latex_table, '.tex'),
        'markdown': (generate_markdown_table, '.md'),
        'text': (generate_text_table, '.txt'),
        'html': (generate_html_table, '.html'),
    }
    
    for fmt in formats:
        if fmt in generators:
            gen_func, ext = generators[fmt]
            filename = f"{prefix}results{ext}" if prefix else f"results{ext}"
            path = output_dir / filename
            gen_func(results_df, config, path)
            saved.append(path)
    
    print(f"Saved {len(saved)} tables to {output_dir}")
    return saved
