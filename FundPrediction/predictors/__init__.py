"""
预测器模块
包含各种基金价格预测器
"""

from .lstm_predictor import LSTMPredictor
from .xgboost_predictor import XGBoostPredictor
from .prophet_predictor import ProphetPredictor
from .ensemble_predictor import EnsemblePredictor

__all__ = [
    'LSTMPredictor',
    'XGBoostPredictor',
    'ProphetPredictor',
    'EnsemblePredictor'
]
