"""
集成基金价格预测器
结合多个预测器进行集成预测
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union
from .lstm_predictor import LSTMPredictor
from .xgboost_predictor import XGBoostPredictor
from .prophet_predictor import ProphetPredictor
import warnings
warnings.filterwarnings('ignore')

class EnsemblePredictor:
    """集成基金价格预测器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化集成预测器
        
        Args:
            config: 配置参数
                - predictors: 预测器配置列表
                - weights: 权重列表
                - ensemble_method: 集成方法 ('weighted_average', 'voting', 'stacking')
                - use_probability: 是否使用概率预测
        """
        self.config = config
        self.predictors_config = config.get('predictors', [])
        self.weights = config.get('weights', None)
        self.ensemble_method = config.get('ensemble_method', 'weighted_average')
        self.use_probability = config.get('use_probability', False)
        
        self.predictors = []
        self.is_trained = False
        
        # 初始化预测器
        self._initialize_predictors()
        
        # 设置权重
        if self.weights is None:
            self.weights = [1.0 / len(self.predictors)] * len(self.predictors)
        
        if len(self.weights) != len(self.predictors):
            raise ValueError("权重数量必须与预测器数量相等")
    
    def _initialize_predictors(self):
        """初始化预测器"""
        for pred_config in self.predictors_config:
            pred_type = pred_config.get('type')
            pred_params = pred_config.get('params', {})
            
            if pred_type == 'lstm':
                predictor = LSTMPredictor(pred_params)
            elif pred_type == 'xgboost':
                predictor = XGBoostPredictor(pred_params)
            elif pred_type == 'prophet':
                predictor = ProphetPredictor(pred_params)
            else:
                raise ValueError(f"不支持的预测器类型: {pred_type}")
            
            self.predictors.append(predictor)
    
    def train(self, data: pd.DataFrame, target_column: str = 'Close') -> None:
        """
        训练所有预测器
        
        Args:
            data: 训练数据
            target_column: 目标列名
        """
        print("开始训练集成预测器...")
        
        for i, predictor in enumerate(self.predictors):
            print(f"训练预测器 {i+1}/{len(self.predictors)}: {type(predictor).__name__}")
            predictor.train(data, target_column)
        
        self.is_trained = True
        print("集成预测器训练完成")
    
    def predict(self, data: pd.DataFrame, steps: int = 1) -> Dict[str, Any]:
        """
        集成预测基金价格
        
        Args:
            data: 输入数据
            steps: 预测步数
            
        Returns:
            集成预测结果
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练")
        
        # 获取所有预测器的预测结果
        predictions_list = []
        individual_results = []
        
        for i, predictor in enumerate(self.predictors):
            try:
                result = predictor.predict(data, steps)
                predictions_list.append(result['predictions'])
                individual_results.append({
                    'predictor': type(predictor).__name__,
                    'predictions': result['predictions'],
                    'confidence_interval': result.get('confidence_interval', 0)
                })
            except Exception as e:
                print(f"预测器 {type(predictor).__name__} 预测失败: {e}")
                continue
        
        if not predictions_list:
            raise ValueError("所有预测器都预测失败")
        
        # 集成预测
        if self.ensemble_method == 'weighted_average':
            ensemble_predictions = self._weighted_average(predictions_list)
        elif self.ensemble_method == 'voting':
            ensemble_predictions = self._voting(predictions_list)
        elif self.ensemble_method == 'stacking':
            ensemble_predictions = self._stacking(predictions_list, data, steps)
        else:
            raise ValueError(f"不支持的集成方法: {self.ensemble_method}")
        
        # 计算集成置信区间
        ensemble_confidence = self._calculate_ensemble_confidence(predictions_list)
        
        result = {
            'predictions': ensemble_predictions,
            'confidence_interval': ensemble_confidence,
            'prediction_dates': individual_results[0]['prediction_dates'] if individual_results else [],
            'model_type': 'Ensemble',
            'ensemble_method': self.ensemble_method,
            'steps': steps,
            'individual_results': individual_results
        }
        
        return result
    
    def _weighted_average(self, predictions_list: List[List[float]]) -> List[float]:
        """加权平均集成"""
        if not predictions_list:
            return []
        
        # 转换为numpy数组
        predictions_array = np.array(predictions_list)
        
        # 加权平均
        ensemble_predictions = np.average(predictions_array, axis=0, weights=self.weights)
        
        return ensemble_predictions.tolist()
    
    def _voting(self, predictions_list: List[List[float]]) -> List[float]:
        """投票集成"""
        if not predictions_list:
            return []
        
        # 转换为numpy数组
        predictions_array = np.array(predictions_list)
        
        # 简单平均（投票）
        ensemble_predictions = np.mean(predictions_array, axis=0)
        
        return ensemble_predictions.tolist()
    
    def _stacking(self, predictions_list: List[List[float]], data: pd.DataFrame, steps: int) -> List[float]:
        """堆叠集成（简化实现）"""
        if not predictions_list:
            return []
        
        # 转换为numpy数组
        predictions_array = np.array(predictions_list)
        
        # 简化的堆叠：使用线性回归作为元学习器
        from sklearn.linear_model import LinearRegression
        
        # 这里简化处理，实际应用中需要更复杂的实现
        # 使用简单平均作为堆叠结果
        ensemble_predictions = np.mean(predictions_array, axis=0)
        
        return ensemble_predictions.tolist()
    
    def _calculate_ensemble_confidence(self, predictions_list: List[List[float]]) -> float:
        """计算集成置信区间"""
        if not predictions_list:
            return 0.0
        
        # 转换为numpy数组
        predictions_array = np.array(predictions_list)
        
        # 计算预测的标准差作为不确定性度量
        std_dev = np.std(predictions_array, axis=0)
        avg_std = np.mean(std_dev)
        
        # 转换为置信区间
        confidence_interval = 1.96 * avg_std
        
        return confidence_interval
    
    def predict_probability(self, data: pd.DataFrame, steps: int = 1) -> Dict[str, Any]:
        """
        集成预测上涨/下跌概率
        
        Args:
            data: 输入数据
            steps: 预测步数
            
        Returns:
            集成概率预测结果
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练")
        
        # 获取所有预测器的概率预测结果
        probabilities_list = []
        individual_results = []
        
        for i, predictor in enumerate(self.predictors):
            try:
                result = predictor.predict_probability(data, steps)
                probabilities_list.append(result['probabilities'])
                individual_results.append({
                    'predictor': type(predictor).__name__,
                    'probabilities': result['probabilities']
                })
            except Exception as e:
                print(f"预测器 {type(predictor).__name__} 概率预测失败: {e}")
                continue
        
        if not probabilities_list:
            raise ValueError("所有预测器都概率预测失败")
        
        # 集成概率预测
        ensemble_probabilities = self._ensemble_probabilities(probabilities_list)
        
        result = {
            'probabilities': ensemble_probabilities,
            'prediction_dates': individual_results[0]['prediction_dates'] if individual_results else [],
            'model_type': 'Ensemble',
            'ensemble_method': self.ensemble_method,
            'individual_results': individual_results
        }
        
        return result
    
    def _ensemble_probabilities(self, probabilities_list: List[List[Dict]]) -> List[Dict]:
        """集成概率预测"""
        if not probabilities_list:
            return []
        
        ensemble_probabilities = []
        
        # 对每个时间步进行集成
        for step in range(len(probabilities_list[0])):
            step_probabilities = [prob_list[step] for prob_list in probabilities_list]
            
            # 计算平均概率
            avg_up_prob = np.mean([prob['up_probability'] for prob in step_probabilities])
            avg_down_prob = np.mean([prob['down_probability'] for prob in step_probabilities])
            avg_predicted_price = np.mean([prob['predicted_price'] for prob in step_probabilities])
            
            ensemble_probabilities.append({
                'up_probability': avg_up_prob,
                'down_probability': avg_down_prob,
                'predicted_price': avg_predicted_price,
                'current_price': step_probabilities[0]['current_price']
            })
        
        return ensemble_probabilities
    
    def get_feature_importance(self) -> Dict[str, Any]:
        """
        获取特征重要性（仅适用于支持特征重要性的预测器）
        
        Returns:
            特征重要性结果
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练")
        
        importance_results = {}
        
        for i, predictor in enumerate(self.predictors):
            if hasattr(predictor, 'get_feature_importance'):
                try:
                    importance = predictor.get_feature_importance()
                    importance_results[type(predictor).__name__] = importance
                except Exception as e:
                    print(f"获取预测器 {type(predictor).__name__} 特征重要性失败: {e}")
        
        return importance_results
    
    def evaluate_individual_predictors(self, data: pd.DataFrame, target_column: str = 'Close') -> Dict[str, Any]:
        """
        评估各个预测器的性能
        
        Args:
            data: 评估数据
            target_column: 目标列名
            
        Returns:
            评估结果
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练")
        
        evaluation_results = {}
        
        for i, predictor in enumerate(self.predictors):
            try:
                # 这里可以添加更详细的评估逻辑
                # 目前只是简单标记为可用
                evaluation_results[type(predictor).__name__] = {
                    'status': 'trained',
                    'weight': self.weights[i]
                }
            except Exception as e:
                evaluation_results[type(predictor).__name__] = {
                    'status': 'error',
                    'error': str(e),
                    'weight': self.weights[i]
                }
        
        return evaluation_results
    
    def save_model(self, path: str) -> None:
        """保存集成模型"""
        if not self.is_trained:
            raise ValueError("模型尚未训练")
        
        import joblib
        
        model_data = {
            'config': self.config,
            'weights': self.weights,
            'ensemble_method': self.ensemble_method,
            'predictors': []
        }
        
        # 保存各个预测器
        for i, predictor in enumerate(self.predictors):
            predictor_path = f"{path}_predictor_{i}.pkl"
            predictor.save_model(predictor_path)
            model_data['predictors'].append({
                'type': type(predictor).__name__,
                'path': predictor_path
            })
        
        joblib.dump(model_data, path)
    
    def load_model(self, path: str) -> None:
        """加载集成模型"""
        import joblib
        
        model_data = joblib.load(path)
        self.config = model_data['config']
        self.weights = model_data['weights']
        self.ensemble_method = model_data['ensemble_method']
        
        # 重新初始化预测器
        self._initialize_predictors()
        
        # 加载各个预测器
        for i, pred_info in enumerate(model_data['predictors']):
            if i < len(self.predictors):
                self.predictors[i].load_model(pred_info['path'])
        
        self.is_trained = True
