"""
基金预测模块
提供基金价格预测、趋势分析等功能
"""

from .predictors.lstm_predictor import LSTMPredictor
from .predictors.xgboost_predictor import XGBoostPredictor
from .predictors.prophet_predictor import ProphetPredictor
from .predictors.ensemble_predictor import EnsemblePredictor
from .evaluators.prediction_evaluator import PredictionEvaluator
from .visualizers.prediction_visualizer import PredictionVisualizer

__all__ = [
    'LSTMPredictor',
    'XGBoostPredictor', 
    'ProphetPredictor',
    'EnsemblePredictor',
    'PredictionEvaluator',
    'PredictionVisualizer'
]
