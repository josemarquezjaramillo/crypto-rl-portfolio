#!/usr/bin/env python
"""
Metrics module for portfolio evaluation.

Provides pure functions for computing standard portfolio performance metrics
used in academic finance and RL for trading literature.

References:
- Jiang et al. (2017): CAGR, Sharpe, Max Drawdown
- Lucarelli & Borrotti (2020): Sortino, Calmar, Hit Rate
- Standard finance: Annualized volatility, risk-adjusted returns

All functions are stateless and operate on numpy arrays for efficiency.
"""

import numpy as np
from typing import List, Tuple, Optional
import warnings


# =============================================================================
# Profitability Metrics
# =============================================================================

def compute_cumulative_return(portfolio_values: np.ndarray) -> float:
    """
    Compute cumulative return from portfolio value series.
    
    Parameters
    ----------
    portfolio_values : np.ndarray
        Portfolio values starting at 1.0
        
    Returns
    -------
    float
        Cumulative return (e.g., 0.5 for 50% gain)
    """
    if len(portfolio_values) < 2:
        return 0.0
    return float(portfolio_values[-1] / portfolio_values[0] - 1.0)


def compute_cagr(portfolio_values: np.ndarray, periods_per_year: float = 365.0) -> float:
    """
    Compute Compound Annual Growth Rate (CAGR).
    
    Parameters
    ----------
    portfolio_values : np.ndarray
        Portfolio values starting at 1.0
    periods_per_year : float
        Number of periods per year (365 for daily, 252 for trading days, 52 for weekly)
        
    Returns
    -------
    float
        Annualized return (e.g., 0.15 for 15% annual return)
    """
    if len(portfolio_values) < 2:
        return 0.0
    
    n_periods = len(portfolio_values) - 1
    years = n_periods / periods_per_year
    
    if years <= 0:
        return 0.0
    
    final_value = portfolio_values[-1]
    initial_value = portfolio_values[0]
    
    if initial_value <= 0 or final_value <= 0:
        return 0.0
    
    # CAGR = (P_T / P_0)^(1/years) - 1
    return float((final_value / initial_value) ** (1.0 / years) - 1.0)


def compute_excess_return(
    agent_return: float,
    baseline_return: float
) -> float:
    """
    Compute excess return relative to a baseline.
    
    Parameters
    ----------
    agent_return : float
        Agent's cumulative or annualized return
    baseline_return : float
        Baseline's cumulative or annualized return
        
    Returns
    -------
    float
        Excess return (agent - baseline)
    """
    return agent_return - baseline_return


# =============================================================================
# Risk Metrics
# =============================================================================

def compute_daily_returns(portfolio_values: np.ndarray) -> np.ndarray:
    """
    Compute daily simple returns from portfolio values.
    
    Parameters
    ----------
    portfolio_values : np.ndarray
        Portfolio values
        
    Returns
    -------
    np.ndarray
        Daily returns (length = len(portfolio_values) - 1)
    """
    if len(portfolio_values) < 2:
        return np.array([])
    
    pv = np.array(portfolio_values)
    # Simple returns: (P_t - P_{t-1}) / P_{t-1}
    returns = np.diff(pv) / np.maximum(pv[:-1], 1e-9)
    return returns


def compute_annualized_volatility(
    portfolio_values: np.ndarray,
    periods_per_year: float = 365.0
) -> float:
    """
    Compute annualized volatility (standard deviation of returns).
    
    Parameters
    ----------
    portfolio_values : np.ndarray
        Portfolio values
    periods_per_year : float
        Number of periods per year (365 for daily crypto)
        
    Returns
    -------
    float
        Annualized volatility (e.g., 0.3 for 30%)
    """
    returns = compute_daily_returns(portfolio_values)
    
    if len(returns) < 2:
        return 0.0
    
    daily_vol = np.std(returns, ddof=1)
    annualized_vol = daily_vol * np.sqrt(periods_per_year)
    
    return float(annualized_vol)


def compute_sharpe_ratio(
    portfolio_values: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: float = 365.0
) -> float:
    """
    Compute annualized Sharpe ratio.
    
    Parameters
    ----------
    portfolio_values : np.ndarray
        Portfolio values
    risk_free_rate : float
        Annual risk-free rate (default 0 for crypto)
    periods_per_year : float
        Number of periods per year
        
    Returns
    -------
    float
        Sharpe ratio
    """
    returns = compute_daily_returns(portfolio_values)
    
    if len(returns) < 2:
        return 0.0
    
    mean_return = np.mean(returns)
    std_return = np.std(returns, ddof=1)
    
    if std_return < 1e-9:
        return 0.0
    
    # Daily risk-free rate
    daily_rf = risk_free_rate / periods_per_year
    
    # Sharpe = sqrt(periods) * (mean - rf) / std
    sharpe = np.sqrt(periods_per_year) * (mean_return - daily_rf) / std_return
    
    return float(sharpe)


def compute_sortino_ratio(
    portfolio_values: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: float = 365.0,
    target_return: float = 0.0
) -> float:
    """
    Compute annualized Sortino ratio (uses downside deviation).
    
    The Sortino ratio is similar to Sharpe but only penalizes downside
    volatility, which better captures investor preferences.
    
    Parameters
    ----------
    portfolio_values : np.ndarray
        Portfolio values
    risk_free_rate : float
        Annual risk-free rate
    periods_per_year : float
        Number of periods per year
    target_return : float
        Target return threshold (default 0, i.e., any loss is downside)
        
    Returns
    -------
    float
        Sortino ratio
    """
    returns = compute_daily_returns(portfolio_values)
    
    if len(returns) < 2:
        return 0.0
    
    mean_return = np.mean(returns)
    
    # Daily target
    daily_target = target_return / periods_per_year
    
    # Downside returns (only negative deviations from target)
    downside_returns = returns[returns < daily_target] - daily_target
    
    if len(downside_returns) < 1:
        # No downside - return high Sortino (capped)
        return 10.0 if mean_return > 0 else 0.0
    
    # Downside deviation (semi-standard deviation)
    downside_std = np.sqrt(np.mean(downside_returns ** 2))
    
    if downside_std < 1e-9:
        return 10.0 if mean_return > 0 else 0.0
    
    # Daily risk-free rate
    daily_rf = risk_free_rate / periods_per_year
    
    # Sortino = sqrt(periods) * (mean - rf) / downside_std
    sortino = np.sqrt(periods_per_year) * (mean_return - daily_rf) / downside_std
    
    return float(np.clip(sortino, -10.0, 10.0))


def compute_max_drawdown(portfolio_values: np.ndarray) -> float:
    """
    Compute maximum drawdown (largest peak-to-trough decline).
    
    Parameters
    ----------
    portfolio_values : np.ndarray
        Portfolio values
        
    Returns
    -------
    float
        Maximum drawdown as positive fraction (e.g., 0.25 for 25% drawdown)
    """
    if len(portfolio_values) < 2:
        return 0.0
    
    pv = np.array(portfolio_values)
    running_max = np.maximum.accumulate(pv)
    drawdowns = (running_max - pv) / np.maximum(running_max, 1e-9)
    
    return float(np.max(drawdowns))


def compute_drawdown_series(portfolio_values: np.ndarray) -> np.ndarray:
    """
    Compute drawdown at each time point.
    
    Parameters
    ----------
    portfolio_values : np.ndarray
        Portfolio values
        
    Returns
    -------
    np.ndarray
        Drawdown series (same length as input)
    """
    if len(portfolio_values) < 1:
        return np.array([])
    
    pv = np.array(portfolio_values)
    running_max = np.maximum.accumulate(pv)
    drawdowns = (running_max - pv) / np.maximum(running_max, 1e-9)
    
    return drawdowns


def compute_calmar_ratio(
    portfolio_values: np.ndarray,
    periods_per_year: float = 365.0
) -> float:
    """
    Compute Calmar ratio (CAGR / Max Drawdown).
    
    Higher is better - measures return per unit of drawdown risk.
    
    Parameters
    ----------
    portfolio_values : np.ndarray
        Portfolio values
    periods_per_year : float
        Number of periods per year
        
    Returns
    -------
    float
        Calmar ratio
    """
    cagr = compute_cagr(portfolio_values, periods_per_year)
    max_dd = compute_max_drawdown(portfolio_values)
    
    if max_dd < 1e-9:
        return 10.0 if cagr > 0 else 0.0
    
    calmar = cagr / max_dd
    
    return float(np.clip(calmar, -10.0, 10.0))


# =============================================================================
# Efficiency Metrics
# =============================================================================

def compute_hit_rate(portfolio_values: np.ndarray) -> float:
    """
    Compute hit rate (fraction of profitable days).
    
    Parameters
    ----------
    portfolio_values : np.ndarray
        Portfolio values
        
    Returns
    -------
    float
        Hit rate (e.g., 0.55 for 55% winning days)
    """
    returns = compute_daily_returns(portfolio_values)
    
    if len(returns) < 1:
        return 0.0
    
    profitable_days = np.sum(returns > 0)
    hit_rate = profitable_days / len(returns)
    
    return float(hit_rate)


def compute_profit_factor(portfolio_values: np.ndarray) -> float:
    """
    Compute profit factor (gross profits / gross losses).
    
    Parameters
    ----------
    portfolio_values : np.ndarray
        Portfolio values
        
    Returns
    -------
    float
        Profit factor (> 1 means profitable)
    """
    returns = compute_daily_returns(portfolio_values)
    
    if len(returns) < 1:
        return 0.0
    
    gross_profits = np.sum(returns[returns > 0])
    gross_losses = np.abs(np.sum(returns[returns < 0]))
    
    if gross_losses < 1e-9:
        return 10.0 if gross_profits > 0 else 0.0
    
    return float(np.clip(gross_profits / gross_losses, 0.0, 10.0))


def compute_average_turnover(turnovers: List[float]) -> float:
    """
    Compute average daily turnover.
    
    Parameters
    ----------
    turnovers : List[float]
        Daily turnover values (L1 norm of weight changes)
        
    Returns
    -------
    float
        Average turnover
    """
    if not turnovers:
        return 0.0
    return float(np.mean(turnovers))


# =============================================================================
# Rolling / Time-Series Metrics
# =============================================================================

def compute_rolling_sharpe(
    portfolio_values: np.ndarray,
    window: int = 30,
    periods_per_year: float = 365.0
) -> np.ndarray:
    """
    Compute rolling Sharpe ratio.
    
    Parameters
    ----------
    portfolio_values : np.ndarray
        Portfolio values
    window : int
        Rolling window size in periods
    periods_per_year : float
        Number of periods per year
        
    Returns
    -------
    np.ndarray
        Rolling Sharpe values (NaN for first window-1 values)
    """
    returns = compute_daily_returns(portfolio_values)
    
    if len(returns) < window:
        return np.full(len(portfolio_values), np.nan)
    
    rolling_sharpe = np.full(len(portfolio_values), np.nan)
    
    for i in range(window - 1, len(returns)):
        window_returns = returns[i - window + 1:i + 1]
        mean_ret = np.mean(window_returns)
        std_ret = np.std(window_returns, ddof=1)
        
        if std_ret > 1e-9:
            rolling_sharpe[i + 1] = np.sqrt(periods_per_year) * mean_ret / std_ret
        else:
            rolling_sharpe[i + 1] = 0.0
    
    return rolling_sharpe


def compute_rolling_volatility(
    portfolio_values: np.ndarray,
    window: int = 30,
    periods_per_year: float = 365.0
) -> np.ndarray:
    """
    Compute rolling annualized volatility.
    
    Parameters
    ----------
    portfolio_values : np.ndarray
        Portfolio values
    window : int
        Rolling window size in periods
    periods_per_year : float
        Number of periods per year
        
    Returns
    -------
    np.ndarray
        Rolling volatility values (NaN for first window-1 values)
    """
    returns = compute_daily_returns(portfolio_values)
    
    if len(returns) < window:
        return np.full(len(portfolio_values), np.nan)
    
    rolling_vol = np.full(len(portfolio_values), np.nan)
    
    for i in range(window - 1, len(returns)):
        window_returns = returns[i - window + 1:i + 1]
        daily_vol = np.std(window_returns, ddof=1)
        rolling_vol[i + 1] = daily_vol * np.sqrt(periods_per_year)
    
    return rolling_vol


# =============================================================================
# Statistical Utilities
# =============================================================================

def compute_confidence_interval(
    values: List[float],
    confidence: float = 0.95
) -> Tuple[float, float]:
    """
    Compute confidence interval using t-distribution for small samples.
    
    Parameters
    ----------
    values : List[float]
        Sample values
    confidence : float
        Confidence level (default 0.95 for 95% CI)
        
    Returns
    -------
    Tuple[float, float]
        (lower_bound, upper_bound)
    """
    n = len(values)
    if n < 2:
        mean_val = values[0] if values else 0.0
        return (mean_val, mean_val)
    
    mean_val = np.mean(values)
    std_val = np.std(values, ddof=1)
    
    from scipy import stats
    t_value = stats.t.ppf((1 + confidence) / 2, df=n-1)
    
    margin = t_value * std_val / np.sqrt(n)
    return (float(mean_val - margin), float(mean_val + margin))


def compute_statistical_test(
    agent_values: List[float],
    baseline_values: List[float],
    test: str = "ttest"
) -> Tuple[float, float]:
    """
    Compute statistical significance test between agent and baseline.
    
    Parameters
    ----------
    agent_values : List[float]
        Agent metric values across seeds
    baseline_values : List[float]
        Baseline metric values across seeds
    test : str
        Test type: "ttest" (paired t-test) or "wilcoxon" (non-parametric)
        
    Returns
    -------
    Tuple[float, float]
        (test_statistic, p_value)
    """
    from scipy import stats
    
    if len(agent_values) < 2 or len(baseline_values) < 2:
        return (0.0, 1.0)
    
    if test == "ttest":
        # Independent samples t-test
        stat, p_value = stats.ttest_ind(agent_values, baseline_values)
    elif test == "wilcoxon":
        # Mann-Whitney U test (non-parametric)
        stat, p_value = stats.mannwhitneyu(
            agent_values, baseline_values, alternative='two-sided'
        )
    else:
        raise ValueError(f"Unknown test: {test}")
    
    return (float(stat), float(p_value))


# =============================================================================
# Comprehensive Metrics Computation
# =============================================================================

def compute_all_metrics(
    portfolio_values: np.ndarray,
    turnovers: Optional[List[float]] = None,
    costs: Optional[List[float]] = None,
    periods_per_year: float = 365.0,
    risk_free_rate: float = 0.0
) -> dict:
    """
    Compute all standard portfolio metrics at once.
    
    Parameters
    ----------
    portfolio_values : np.ndarray
        Portfolio values starting at 1.0
    turnovers : List[float], optional
        Daily turnover values
    costs : List[float], optional
        Daily transaction costs
    periods_per_year : float
        Number of periods per year
    risk_free_rate : float
        Annual risk-free rate
        
    Returns
    -------
    dict
        Dictionary with all computed metrics
    """
    pv = np.array(portfolio_values)
    n_steps = len(pv) - 1 if len(pv) > 1 else 0
    
    return {
        # Profitability
        'cumulative_return': compute_cumulative_return(pv),
        'cagr': compute_cagr(pv, periods_per_year),
        'final_value': float(pv[-1]) if len(pv) > 0 else 1.0,
        
        # Risk
        'annualized_volatility': compute_annualized_volatility(pv, periods_per_year),
        'sharpe_ratio': compute_sharpe_ratio(pv, risk_free_rate, periods_per_year),
        'sortino_ratio': compute_sortino_ratio(pv, risk_free_rate, periods_per_year),
        'max_drawdown': compute_max_drawdown(pv),
        'calmar_ratio': compute_calmar_ratio(pv, periods_per_year),
        
        # Efficiency
        'hit_rate': compute_hit_rate(pv),
        'profit_factor': compute_profit_factor(pv),
        'mean_turnover': compute_average_turnover(turnovers) if turnovers else 0.0,
        'total_costs': sum(costs) if costs else 0.0,
        
        # Meta
        'n_steps': n_steps,
    }
