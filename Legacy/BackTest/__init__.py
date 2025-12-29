"""
TrickTrade 回测模块
提供完整的量化交易回测框架
"""

from .backtest_framework import BaseBacktestEngine, BaseStrategy, run_backtest
from .baseline_models import STRATEGIES
from .data_provider import get_data, data_manager, DataManager

__version__ = "1.0.0"
__author__ = "TrickTrade Team"

__all__ = [
    'BaseBacktestEngine',
    'BaseStrategy', 
    'run_backtest',
    'STRATEGIES',
    'get_data',
    'data_manager',
    'DataManager'
]
