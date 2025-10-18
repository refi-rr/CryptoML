# core/backtester.py
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class Backtester:
    def __init__(self, initial_capital=10000):
        self.initial_capital = initial_capital
    
    def calculate_metrics(self, predictions, actuals, returns):
        """Calculate comprehensive trading metrics"""
        if len(predictions) == 0 or len(actuals) == 0:
            return {
                'total_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'accuracy': 0,
                'precision': 0,
                'recall': 0,
                'f1_score': 0,
                'win_rate': 0
            }
        
        try:
            # Generate trading signals (1 for buy, -1 for sell, 0 for hold)
            signals = np.where(predictions > actuals * 1.02, 1, 
                              np.where(predictions < actuals * 0.98, -1, 0))
            
            # Ensure signals match returns length
            min_len = min(len(signals), len(returns))
            signals = signals[:min_len]
            returns_adj = returns[:min_len]
            
            # Calculate strategy returns
            strategy_returns = returns_adj * signals
            
            # Metrics
            total_return = np.prod(1 + strategy_returns) - 1 if len(strategy_returns) > 0 else 0
            sharpe_ratio = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252) if np.std(strategy_returns) > 0 else 0
            max_drawdown = self.calculate_max_drawdown(strategy_returns)
            
            # Accuracy metrics
            if len(returns_adj) > 1:
                actual_direction = np.where(returns_adj > 0, 1, 0)
                predicted_direction = np.where(predictions[1:min_len+1] > actuals[:min_len-1], 1, 0)
                
                # Ensure same length
                min_dir_len = min(len(actual_direction), len(predicted_direction))
                actual_direction = actual_direction[:min_dir_len]
                predicted_direction = predicted_direction[:min_dir_len]
                
                accuracy = accuracy_score(actual_direction, predicted_direction) if min_dir_len > 0 else 0
                precision = precision_score(actual_direction, predicted_direction, zero_division=0) if min_dir_len > 0 else 0
                recall = recall_score(actual_direction, predicted_direction, zero_division=0) if min_dir_len > 0 else 0
                f1 = f1_score(actual_direction, predicted_direction, zero_division=0) if min_dir_len > 0 else 0
            else:
                accuracy = precision = recall = f1 = 0
            
            return {
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'win_rate': np.mean(strategy_returns > 0) if len(strategy_returns) > 0 else 0
            }
            
        except Exception as e:
            print(f"Error in backtesting: {e}")
            return {
                'total_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'accuracy': 0,
                'precision': 0,
                'recall': 0,
                'f1_score': 0,
                'win_rate': 0
            }
    
    def calculate_max_drawdown(self, returns):
        """Calculate maximum drawdown"""
        if len(returns) == 0:
            return 0
        
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min() if len(drawdown) > 0 else 0
