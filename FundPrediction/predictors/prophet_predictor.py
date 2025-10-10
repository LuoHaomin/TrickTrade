"""
Prophet基金价格预测器
使用Facebook Prophet进行基金价格预测
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')

class ProphetPredictor:
    """Prophet基金价格预测器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化Prophet预测器
        
        Args:
            config: 配置参数
                - yearly_seasonality: 年度季节性
                - weekly_seasonality: 周度季节性
                - daily_seasonality: 日度季节性
                - seasonality_mode: 季节性模式 ('additive' or 'multiplicative')
                - changepoint_prior_scale: 变点先验尺度
                - seasonality_prior_scale: 季节性先验尺度
                - holidays_prior_scale: 节假日先验尺度
        """
        self.config = config
        self.yearly_seasonality = config.get('yearly_seasonality', True)
        self.weekly_seasonality = config.get('weekly_seasonality', True)
        self.daily_seasonality = config.get('daily_seasonality', False)
        self.seasonality_mode = config.get('seasonality_mode', 'additive')
        self.changepoint_prior_scale = config.get('changepoint_prior_scale', 0.05)
        self.seasonality_prior_scale = config.get('seasonality_prior_scale', 10.0)
        self.holidays_prior_scale = config.get('holidays_prior_scale', 10.0)
        
        self.model = None
        self.is_trained = False
        
    def prepare_data(self, data: pd.DataFrame, target_column: str = 'Close') -> pd.DataFrame:
        """
        准备Prophet格式的数据
        
        Args:
            data: 原始数据
            target_column: 目标列名
            
        Returns:
            Prophet格式的数据
        """
        # Prophet需要ds和y列
        prophet_data = pd.DataFrame({
            'ds': data.index,
            'y': data[target_column]
        })
        
        # 重置索引
        prophet_data = prophet_data.reset_index(drop=True)
        
        # 移除缺失值
        prophet_data = prophet_data.dropna()
        
        return prophet_data
    
    def train(self, data: pd.DataFrame, target_column: str = 'Close') -> None:
        """
        训练Prophet模型
        
        Args:
            data: 训练数据
            target_column: 目标列名
        """
        print("开始训练Prophet模型...")
        
        # 准备数据
        prophet_data = self.prepare_data(data, target_column)
        
        # 创建Prophet模型
        self.model = Prophet(
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            seasonality_mode=self.seasonality_mode,
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_prior_scale=self.seasonality_prior_scale,
            holidays_prior_scale=self.holidays_prior_scale
        )
        
        # 训练模型
        self.model.fit(prophet_data)
        
        self.is_trained = True
        print("Prophet模型训练完成")
    
    def predict(self, data: pd.DataFrame, steps: int = 1) -> Dict[str, Any]:
        """
        预测基金价格
        
        Args:
            data: 输入数据
            steps: 预测步数
            
        Returns:
            预测结果
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练")
        
        # 创建未来日期
        future = self.model.make_future_dataframe(periods=steps)
        
        # 预测
        forecast = self.model.predict(future)
        
        # 提取预测结果
        predictions = forecast['yhat'].tail(steps).values
        lower_bound = forecast['yhat_lower'].tail(steps).values
        upper_bound = forecast['yhat_upper'].tail(steps).values
        
        # 计算置信区间
        confidence_interval = (upper_bound - lower_bound) / 2
        
        result = {
            'predictions': predictions.tolist(),
            'lower_bound': lower_bound.tolist(),
            'upper_bound': upper_bound.tolist(),
            'confidence_interval': confidence_interval.tolist(),
            'prediction_dates': forecast['ds'].tail(steps).dt.strftime('%Y-%m-%d').tolist(),
            'model_type': 'Prophet',
            'steps': steps
        }
        
        return result
    
    def predict_probability(self, data: pd.DataFrame, steps: int = 1) -> Dict[str, Any]:
        """
        预测上涨/下跌概率
        
        Args:
            data: 输入数据
            steps: 预测步数
            
        Returns:
            概率预测结果
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练")
        
        # 获取价格预测
        price_result = self.predict(data, steps)
        predictions = price_result['predictions']
        lower_bound = price_result['lower_bound']
        upper_bound = price_result['upper_bound']
        
        # 计算概率（基于置信区间）
        current_price = data['Close'].iloc[-1]
        probabilities = []
        
        for i, (pred, lower, upper) in enumerate(zip(predictions, lower_bound, upper_bound)):
            # 基于置信区间计算概率
            if pred > current_price:
                # 上涨概率
                prob = min(0.9, 0.5 + (pred - current_price) / current_price * 2)
            else:
                # 下跌概率
                prob = max(0.1, 0.5 - (current_price - pred) / current_price * 2)
            
            # 调整基于置信区间的概率
            confidence_width = upper - lower
            if confidence_width > 0:
                # 置信区间越宽，概率越接近0.5
                confidence_factor = min(1.0, confidence_width / current_price)
                prob = prob * (1 - confidence_factor) + 0.5 * confidence_factor
            
            probabilities.append({
                'up_probability': prob,
                'down_probability': 1 - prob,
                'predicted_price': pred,
                'current_price': current_price,
                'confidence_lower': lower,
                'confidence_upper': upper
            })
        
        return {
            'probabilities': probabilities,
            'prediction_dates': price_result['prediction_dates'],
            'model_type': 'Prophet'
        }
    
    def get_components(self, data: pd.DataFrame, steps: int = 1) -> Dict[str, Any]:
        """
        获取预测组件（趋势、季节性等）
        
        Args:
            data: 输入数据
            steps: 预测步数
            
        Returns:
            预测组件
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练")
        
        # 创建未来日期
        future = self.model.make_future_dataframe(periods=steps)
        
        # 预测
        forecast = self.model.predict(future)
        
        # 提取组件
        components = {
            'trend': forecast['trend'].tail(steps).values.tolist(),
            'yearly': forecast.get('yearly', pd.Series(0, index=forecast.index)).tail(steps).values.tolist(),
            'weekly': forecast.get('weekly', pd.Series(0, index=forecast.index)).tail(steps).values.tolist(),
            'daily': forecast.get('daily', pd.Series(0, index=forecast.index)).tail(steps).values.tolist(),
            'holidays': forecast.get('holidays', pd.Series(0, index=forecast.index)).tail(steps).values.tolist(),
            'prediction_dates': forecast['ds'].tail(steps).dt.strftime('%Y-%m-%d').tolist()
        }
        
        return components
    
    def plot_forecast(self, data: pd.DataFrame, steps: int = 1):
        """
        绘制预测结果
        
        Args:
            data: 输入数据
            steps: 预测步数
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练")
        
        # 创建未来日期
        future = self.model.make_future_dataframe(periods=steps)
        
        # 预测
        forecast = self.model.predict(future)
        
        # 绘制
        fig = self.model.plot(forecast)
        return fig
    
    def plot_components(self, data: pd.DataFrame, steps: int = 1):
        """
        绘制预测组件
        
        Args:
            data: 输入数据
            steps: 预测步数
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练")
        
        # 创建未来日期
        future = self.model.make_future_dataframe(periods=steps)
        
        # 预测
        forecast = self.model.predict(future)
        
        # 绘制组件
        fig = self.model.plot_components(forecast)
        return fig
    
    def save_model(self, path: str) -> None:
        """保存模型"""
        if self.model is None:
            raise ValueError("没有可保存的模型")
        
        import joblib
        
        model_data = {
            'model': self.model,
            'config': self.config
        }
        
        joblib.dump(model_data, path)
    
    def load_model(self, path: str) -> None:
        """加载模型"""
        import joblib
        
        model_data = joblib.load(path)
        self.model = model_data['model']
        self.config = model_data['config']
        self.is_trained = True
