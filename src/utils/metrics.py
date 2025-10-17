import pandas as pd
import numpy as np

def calculate_sharpe_ratio(returns, risk_free_rate=0.02, periods_per_year=252):
    """Calculate Sharpe ratio"""
    if len(returns) == 0:
        return 0
    excess_returns = returns - (risk_free_rate / periods_per_year)
    if excess_returns.std() == 0:
        return 0
    return (excess_returns.mean() * periods_per_year) / (excess_returns.std() * np.sqrt(periods_per_year))

def calculate_sortino_ratio(returns, risk_free_rate=0.02, periods_per_year=252):
    """Calculate Sortino ratio"""
    if len(returns) == 0:
        return 0
    excess_returns = returns - (risk_free_rate / periods_per_year)
    downside_returns = excess_returns[excess_returns < 0]
    if len(downside_returns) == 0 or downside_returns.std() == 0:
        return 0
    return (excess_returns.mean() * periods_per_year) / (downside_returns.std() * np.sqrt(periods_per_year))

def calculate_max_drawdown(portfolio_values):
    """Calculate maximum drawdown"""
    if len(portfolio_values) == 0:
        return 0
    portfolio_series = pd.Series(portfolio_values)
    running_max = portfolio_series.expanding().max()
    drawdown = (portfolio_series - running_max) / running_max
    return drawdown.min()

def calculate_calmar_ratio(portfolio_values, periods_per_year=252):
    """Calculate Calmar ratio"""
    if len(portfolio_values) < 2:
        return 0
    total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
    if total_return <= -1:  # Total loss
        return -np.inf
    annualized_return = (1 + total_return) ** (periods_per_year / len(portfolio_values)) - 1
    max_dd = abs(calculate_max_drawdown(portfolio_values))
    if max_dd == 0:
        return np.inf
    return annualized_return / max_dd

def calculate_enhanced_metrics(portfolio_values, trade_history, initial_cash, periods_per_year=252):
    """Calculate comprehensive performance metrics"""
    
    returns = pd.Series(portfolio_values).pct_change().dropna()
    
    # Trading statistics
    if len(trade_history) > 0:
        trade_returns = trade_history['Return_Pct'].dropna() if 'Return_Pct' in trade_history.columns else pd.Series()
        if len(trade_returns) > 0:
            avg_trade_return = trade_returns.mean()
            winning_trades = trade_returns[trade_returns > 0]
            losing_trades = trade_returns[trade_returns < 0]
            
            if len(losing_trades) > 0:
                profit_factor = -winning_trades.sum() / losing_trades.sum()
            else:
                profit_factor = float('inf') if len(winning_trades) > 0 else 0
                
            win_rate = len(winning_trades) / len(trade_returns)
        else:
            avg_trade_return = 0
            profit_factor = 0
            win_rate = 0
    else:
        avg_trade_return = 0
        profit_factor = 0
        win_rate = 0
    
    metrics = {
        'Total Return': (portfolio_values[-1] - initial_cash) / initial_cash,
        'Annualized Return': ((portfolio_values[-1] / initial_cash) ** (periods_per_year / len(portfolio_values))) - 1,
        'Annualized Volatility': returns.std() * np.sqrt(periods_per_year) if len(returns) > 0 else 0,
        'Sharpe Ratio': calculate_sharpe_ratio(returns),
        'Sortino Ratio': calculate_sortino_ratio(returns),
        'Max Drawdown': calculate_max_drawdown(portfolio_values),
        'Calmar Ratio': calculate_calmar_ratio(portfolio_values, periods_per_year),
        'Win Rate': win_rate,
        'Profit Factor': profit_factor,
        'Avg Trade Return': avg_trade_return,
        'Total Trades': len(trade_history),
        'Final Portfolio Value': portfolio_values[-1],
        'Total P/L': portfolio_values[-1] - initial_cash
    }
    
    return metrics