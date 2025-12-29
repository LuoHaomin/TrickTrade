"""
XGBoost基金价格预测器
使用XGBoost进行基金价格预测
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

class XGBoostPredictor:
    """XGBoost基金价格预测器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化XGBoost预测器
        
        Args:
            config: 配置参数
                - n_estimators: 树的数量
                - max_depth: 最大深度
                - learning_rate: 学习率
                - subsample: 子样本比例
                - colsample_bytree: 特征采样比例
                - random_state: 随机种子
                - lookback_periods: 回望期数列表
        """
        self.config = config
        self.n_estimators = config.get('n_estimators', 100)
        self.max_depth = config.get('max_depth', 6)
        self.learning_rate = config.get('learning_rate', 0.1)
        self.subsample = config.get('subsample', 0.8)
        self.colsample_bytree = config.get('colsample_bytree', 0.8)
        self.random_state = config.get('random_state', 42)
        self.lookback_periods = config.get('lookback_periods', [1, 2, 3, 5, 10])
        
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_columns = []
        
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        准备特征数据
        
        Args:
            data: 原始数据
            
        Returns:
            特征数据
        """
        features = data.copy()
        
        # 基础价格特征
        features['Price_Change'] = features['Close'].pct_change()
        features['Price_Change_5'] = features['Close'].pct_change(5)
        features['Price_Change_10'] = features['Close'].pct_change(10)
        features['Price_Change_20'] = features['Close'].pct_change(20)
        
        # 技术指标
        features['SMA_5'] = features['Close'].rolling(5).mean()
        features['SMA_10'] = features['Close'].rolling(10).mean()
        features['SMA_20'] = features['Close'].rolling(20).mean()
        features['SMA_50'] = features['Close'].rolling(50).mean()
        
        features['EMA_5'] = features['Close'].ewm(span=5).mean()
        features['EMA_10'] = features['Close'].ewm(span=10).mean()
        features['EMA_20'] = features['Close'].ewm(span=20).mean()
        
        # RSI
        features['RSI_14'] = self._calculate_rsi(features['Close'], 14)
        features['RSI_21'] = self._calculate_rsi(features['Close'], 21)
        
        # MACD
        macd_data = self._calculate_macd(features['Close'])
        features['MACD'] = macd_data['macd']
        features['MACD_Signal'] = macd_data['signal']
        features['MACD_Histogram'] = macd_data['histogram']
        
        # 布林带
        bb_data = self._calculate_bollinger_bands(features['Close'])
        features['BB_Upper'] = bb_data['upper']
        features['BB_Middle'] = bb_data['middle']
        features['BB_Lower'] = bb_data['lower']
        features['BB_Width'] = (features['BB_Upper'] - features['BB_Lower']) / features['BB_Middle']
        features['BB_Position'] = (features['Close'] - features['BB_Lower']) / (features['BB_Upper'] - features['BB_Lower'])
        
        # 波动率
        features['Volatility_5'] = features['Close'].rolling(5).std()
        features['Volatility_10'] = features['Close'].rolling(10).std()
        features['Volatility_20'] = features['Close'].rolling(20).std()
        
        # 成交量特征
        features['Volume_MA_5'] = features['Volume'].rolling(5).mean()
        features['Volume_MA_10'] = features['Volume'].rolling(10).mean()
        features['Volume_MA_20'] = features['Volume'].rolling(20).mean()
        features['Volume_Ratio_5'] = features['Volume'] / features['Volume_MA_5']
        features['Volume_Ratio_10'] = features['Volume'] / features['Volume_MA_10']
        
        # 滞后特征
        for period in self.lookback_periods:
            features[f'Close_lag_{period}'] = features['Close'].shift(period)
            features[f'Volume_lag_{period}'] = features['Volume'].shift(period)
            features[f'Price_Change_lag_{period}'] = features['Price_Change'].shift(period)
        
        # 滚动统计特征
        for window in [5, 10, 20]:
            features[f'Close_mean_{window}'] = features['Close'].rolling(window).mean()
            features[f'Close_std_{window}'] = features['Close'].rolling(window).std()
            features[f'Close_min_{window}'] = features['Close'].rolling(window).min()
            features[f'Close_max_{window}'] = features['Close'].rolling(window).max()
            features[f'Close_skew_{window}'] = features['Close'].rolling(window).skew()
            features[f'Close_kurt_{window}'] = features['Close'].rolling(window).kurt()
        
        # 选择数值特征
        numeric_features = features.select_dtypes(include=[np.number]).columns
        features = features[numeric_features]
        
        # 处理缺失值
        features = features.fillna(method='ffill').fillna(method='bfill')
        
        self.feature_columns = features.columns.tolist()
        return features
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """计算RSI指标"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """计算MACD指标"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        histogram = macd - signal_line
        
        return {
            'macd': macd,
            'signal': signal_line,
            'histogram': histogram
        }
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: float = 2) -> Dict[str, pd.Series]:
        """计算布林带指标"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        
        return {
            'upper': sma + (std * std_dev),
            'middle': sma,
            'lower': sma - (std * std_dev)
        }
    
    def train(self, data: pd.DataFrame, target_column: str = 'Close') -> None:
        """
        训练XGBoost模型
        
        Args:
            data: 训练数据
            target_column: 目标列名
        """
        print("开始训练XGBoost模型...")
        
        # 准备特征
        features = self.prepare_features(data)
        
        # 准备标签
        labels = data[target_column].values
        
        # 移除缺失值
        valid_mask = ~(features.isnull().any(axis=1) | pd.isnull(labels))
        features = features[valid_mask]
        labels = labels[valid_mask]
        
        # 数据标准化
        features_scaled = self.scaler.fit_transform(features)
        
        # 分割训练和验证集
        split_idx = int(len(features_scaled) * 0.8)
        X_train, X_val = features_scaled[:split_idx], features_scaled[split_idx:]
        y_train, y_val = labels[:split_idx], labels[split_idx:]
        
        # 创建XGBoost模型
        self.model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        # 训练模型
        self.model.fit(X_train, y_train)
        
        # 验证
        y_pred = self.model.predict(X_val)
        mse = mean_squared_error(y_val, y_pred)
        mae = mean_absolute_error(y_val, y_pred)
        
        print(f"验证集 MSE: {mse:.4f}")
        print(f"验证集 MAE: {mae:.4f}")
        
        self.is_trained = True
        print("XGBoost模型训练完成")
    
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
        
        # 准备特征
        features = self.prepare_features(data)
        
        # 数据标准化
        features_scaled = self.scaler.transform(features)
        
        # 预测
        predictions = self.model.predict(features_scaled[-steps:])
        
        # 计算置信区间（基于模型不确定性）
        # 这里使用简化的方法，实际应用中可以使用更复杂的方法
        std_dev = np.std(predictions) if len(predictions) > 1 else 0.1
        confidence_interval = 1.96 * std_dev
        
        result = {
            'predictions': predictions.tolist(),
            'confidence_interval': confidence_interval,
            'prediction_dates': pd.date_range(start=data.index[-1], periods=steps+1, freq='D')[1:].strftime('%Y-%m-%d').tolist(),
            'model_type': 'XGBoost',
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
        
        # 计算概率（简化实现）
        current_price = data['Close'].iloc[-1]
        probabilities = []
        
        for pred in predictions:
            if pred > current_price:
                # 上涨概率
                prob = min(0.9, 0.5 + (pred - current_price) / current_price * 2)
            else:
                # 下跌概率
                prob = max(0.1, 0.5 - (current_price - pred) / current_price * 2)
            
            probabilities.append({
                'up_probability': prob,
                'down_probability': 1 - prob,
                'predicted_price': pred,
                'current_price': current_price
            })
        
        return {
            'probabilities': probabilities,
            'prediction_dates': price_result['prediction_dates'],
            'model_type': 'XGBoost'
        }
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        获取特征重要性
        
        Returns:
            特征重要性字典
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练")
        
        importance = self.model.feature_importances_
        feature_importance = dict(zip(self.feature_columns, importance))
        
        # 按重要性排序
        sorted_importance = dict(sorted(feature_importance.items(), 
                                      key=lambda x: x[1], reverse=True))
        
        return sorted_importance
    
    def save_model(self, path: str) -> None:
        """保存模型"""
        if self.model is None:
            raise ValueError("没有可保存的模型")
        
        import joblib
        
        model_data = {
            'model': self.model,
            'config': self.config,
            'feature_columns': self.feature_columns,
            'scaler': self.scaler
        }
        
        joblib.dump(model_data, path)
    
    def load_model(self, path: str) -> None:
        """加载模型"""
        import joblib
        
        model_data = joblib.load(path)
        self.model = model_data['model']
        self.config = model_data['config']
        self.feature_columns = model_data['feature_columns']
        self.scaler = model_data['scaler']
        self.is_trained = True
