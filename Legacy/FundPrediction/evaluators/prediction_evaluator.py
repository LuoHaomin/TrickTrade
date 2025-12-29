"""
预测评估器
评估基金价格预测的准确性
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class PredictionEvaluator:
    """预测评估器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化预测评估器
        
        Args:
            config: 配置参数
        """
        self.config = config or {}
        
    def evaluate_point_predictions(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        评估点预测结果
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            
        Returns:
            评估指标
        """
        # 移除缺失值
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]
        
        if len(y_true_clean) == 0:
            return {'error': 'No valid data for evaluation'}
        
        # 计算评估指标
        mse = mean_squared_error(y_true_clean, y_pred_clean)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true_clean, y_pred_clean)
        r2 = r2_score(y_true_clean, y_pred_clean)
        
        # 计算百分比误差
        mape = np.mean(np.abs((y_true_clean - y_pred_clean) / y_true_clean)) * 100
        
        # 计算方向准确性（涨跌方向预测准确率）
        direction_accuracy = self._calculate_direction_accuracy(y_true_clean, y_pred_clean)
        
        # 计算最大误差
        max_error = np.max(np.abs(y_true_clean - y_pred_clean))
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape,
            'direction_accuracy': direction_accuracy,
            'max_error': max_error,
            'n_samples': len(y_true_clean)
        }
    
    def evaluate_probability_predictions(self, y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
        """
        评估概率预测结果
        
        Args:
            y_true: 真实值（0或1）
            y_prob: 预测概率
            
        Returns:
            评估指标
        """
        # 移除缺失值
        mask = ~(np.isnan(y_true) | np.isnan(y_prob))
        y_true_clean = y_true[mask]
        y_prob_clean = y_prob[mask]
        
        if len(y_true_clean) == 0:
            return {'error': 'No valid data for evaluation'}
        
        # 计算AUC
        from sklearn.metrics import roc_auc_score
        try:
            auc = roc_auc_score(y_true_clean, y_prob_clean)
        except ValueError:
            auc = 0.5  # 如果只有一个类别
        
        # 计算对数损失
        from sklearn.metrics import log_loss
        try:
            logloss = log_loss(y_true_clean, y_prob_clean)
        except ValueError:
            logloss = float('inf')
        
        # 计算Brier分数
        brier_score = np.mean((y_prob_clean - y_true_clean) ** 2)
        
        # 计算校准误差
        calibration_error = self._calculate_calibration_error(y_true_clean, y_prob_clean)
        
        return {
            'auc': auc,
            'log_loss': logloss,
            'brier_score': brier_score,
            'calibration_error': calibration_error,
            'n_samples': len(y_true_clean)
        }
    
    def evaluate_interval_predictions(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                    y_lower: np.ndarray, y_upper: np.ndarray) -> Dict[str, float]:
        """
        评估区间预测结果
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            y_lower: 预测下界
            y_upper: 预测上界
            
        Returns:
            评估指标
        """
        # 移除缺失值
        mask = ~(np.isnan(y_true) | np.isnan(y_pred) | np.isnan(y_lower) | np.isnan(y_upper))
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]
        y_lower_clean = y_lower[mask]
        y_upper_clean = y_upper[mask]
        
        if len(y_true_clean) == 0:
            return {'error': 'No valid data for evaluation'}
        
        # 计算区间覆盖率
        coverage = np.mean((y_true_clean >= y_lower_clean) & (y_true_clean <= y_upper_clean))
        
        # 计算区间宽度
        interval_width = np.mean(y_upper_clean - y_lower_clean)
        
        # 计算区间得分
        interval_score = self._calculate_interval_score(y_true_clean, y_lower_clean, y_upper_clean)
        
        # 计算Winkler得分
        winkler_score = self._calculate_winkler_score(y_true_clean, y_lower_clean, y_upper_clean)
        
        return {
            'coverage': coverage,
            'interval_width': interval_width,
            'interval_score': interval_score,
            'winkler_score': winkler_score,
            'n_samples': len(y_true_clean)
        }
    
    def evaluate_time_series_predictions(self, data: pd.DataFrame, predictions: List[float], 
                                      prediction_dates: List[str], target_column: str = 'Close') -> Dict[str, Any]:
        """
        评估时间序列预测结果
        
        Args:
            data: 原始数据
            predictions: 预测值列表
            prediction_dates: 预测日期列表
            target_column: 目标列名
            
        Returns:
            评估结果
        """
        # 获取真实值
        pred_dates = pd.to_datetime(prediction_dates)
        true_values = []
        
        for date in pred_dates:
            if date in data.index:
                true_values.append(data.loc[date, target_column])
            else:
                # 如果预测日期不在数据中，尝试找到最近的值
                closest_date = data.index[data.index.get_indexer([date], method='nearest')[0]]
                true_values.append(data.loc[closest_date, target_column])
        
        if len(true_values) != len(predictions):
            return {'error': 'Mismatch between predictions and true values'}
        
        # 转换为numpy数组
        y_true = np.array(true_values)
        y_pred = np.array(predictions)
        
        # 评估点预测
        point_metrics = self.evaluate_point_predictions(y_true, y_pred)
        
        # 计算时间序列特定指标
        ts_metrics = self._calculate_time_series_metrics(y_true, y_pred)
        
        # 合并结果
        result = {**point_metrics, **ts_metrics}
        
        return result
    
    def _calculate_direction_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """计算方向准确性"""
        if len(y_true) < 2:
            return 0.0
        
        # 计算真实值的变化方向
        true_direction = np.diff(y_true) > 0
        
        # 计算预测值的变化方向
        pred_direction = np.diff(y_pred) > 0
        
        # 计算方向匹配率
        direction_accuracy = np.mean(true_direction == pred_direction)
        
        return direction_accuracy
    
    def _calculate_calibration_error(self, y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
        """计算校准误差"""
        # 将概率分成n_bins个区间
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        calibration_error = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # 找到在这个区间内的样本
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                # 计算这个区间内的平均真实值
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_prob[in_bin].mean()
                calibration_error += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return calibration_error
    
    def _calculate_interval_score(self, y_true: np.ndarray, y_lower: np.ndarray, y_upper: np.ndarray) -> float:
        """计算区间得分"""
        # 简化的区间得分计算
        penalty = np.where(y_true < y_lower, 2 * (y_lower - y_true),
                          np.where(y_true > y_upper, 2 * (y_true - y_upper), 0))
        interval_score = np.mean(penalty)
        return interval_score
    
    def _calculate_winkler_score(self, y_true: np.ndarray, y_lower: np.ndarray, y_upper: np.ndarray) -> float:
        """计算Winkler得分"""
        # 简化的Winkler得分计算
        width = y_upper - y_lower
        penalty = np.where(y_true < y_lower, 2 * (y_lower - y_true),
                          np.where(y_true > y_upper, 2 * (y_true - y_upper), 0))
        winkler_score = np.mean(width + penalty)
        return winkler_score
    
    def _calculate_time_series_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """计算时间序列特定指标"""
        # 计算趋势准确性
        trend_accuracy = self._calculate_direction_accuracy(y_true, y_pred)
        
        # 计算滞后相关性
        lag_correlation = self._calculate_lag_correlation(y_true, y_pred)
        
        # 计算预测稳定性
        prediction_stability = self._calculate_prediction_stability(y_pred)
        
        return {
            'trend_accuracy': trend_accuracy,
            'lag_correlation': lag_correlation,
            'prediction_stability': prediction_stability
        }
    
    def _calculate_lag_correlation(self, y_true: np.ndarray, y_pred: np.ndarray, max_lag: int = 5) -> float:
        """计算滞后相关性"""
        max_corr = 0
        for lag in range(max_lag + 1):
            if lag == 0:
                corr = np.corrcoef(y_true, y_pred)[0, 1]
            else:
                if len(y_true) > lag:
                    corr = np.corrcoef(y_true[lag:], y_pred[:-lag])[0, 1]
                else:
                    corr = 0
            
            if not np.isnan(corr):
                max_corr = max(max_corr, abs(corr))
        
        return max_corr
    
    def _calculate_prediction_stability(self, y_pred: np.ndarray) -> float:
        """计算预测稳定性"""
        if len(y_pred) < 2:
            return 0.0
        
        # 计算预测值的变化率
        changes = np.diff(y_pred) / y_pred[:-1]
        
        # 计算变化率的稳定性（标准差越小越稳定）
        stability = 1 / (1 + np.std(changes))
        
        return stability
    
    def compare_predictors(self, evaluation_results: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """
        比较多个预测器的性能
        
        Args:
            evaluation_results: 各预测器的评估结果
            
        Returns:
            比较结果
        """
        if not evaluation_results:
            return {'error': 'No evaluation results provided'}
        
        # 提取指标
        metrics = list(evaluation_results.values())[0].keys()
        comparison = {}
        
        for metric in metrics:
            if metric == 'n_samples':
                continue
                
            metric_values = {}
            for predictor_name, results in evaluation_results.items():
                if metric in results and not np.isnan(results[metric]):
                    metric_values[predictor_name] = results[metric]
            
            if metric_values:
                # 找到最佳预测器
                if metric in ['mse', 'rmse', 'mae', 'mape', 'max_error', 'log_loss', 'brier_score', 'calibration_error']:
                    best_predictor = min(metric_values, key=metric_values.get)
                else:
                    best_predictor = max(metric_values, key=metric_values.get)
                
                comparison[metric] = {
                    'values': metric_values,
                    'best_predictor': best_predictor,
                    'best_value': metric_values[best_predictor]
                }
        
        return comparison
    
    def generate_evaluation_report(self, evaluation_results: Dict[str, Dict[str, float]]) -> str:
        """
        生成评估报告
        
        Args:
            evaluation_results: 评估结果
            
        Returns:
            评估报告字符串
        """
        report = "=== 预测评估报告 ===\n\n"
        
        for predictor_name, results in evaluation_results.items():
            report += f"预测器: {predictor_name}\n"
            report += "-" * 30 + "\n"
            
            for metric, value in results.items():
                if metric == 'n_samples':
                    report += f"样本数量: {value}\n"
                elif metric == 'error':
                    report += f"错误: {value}\n"
                else:
                    report += f"{metric}: {value:.4f}\n"
            
            report += "\n"
        
        # 添加比较结果
        comparison = self.compare_predictors(evaluation_results)
        if 'error' not in comparison:
            report += "=== 性能比较 ===\n"
            for metric, comp_data in comparison.items():
                report += f"{metric}:\n"
                report += f"  最佳预测器: {comp_data['best_predictor']}\n"
                report += f"  最佳值: {comp_data['best_value']:.4f}\n"
                report += f"  所有值: {comp_data['values']}\n\n"
        
        return report
