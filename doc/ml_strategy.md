# TrckTrade 机器学习策略设计

## 概述

本文档详细描述了TrckTrade平台中机器学习策略的设计与实现，包括传统机器学习、深度学习和强化学习策略的完整框架。

## 机器学习策略架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    策略接口层                                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │ 策略基类     │  │ 信号生成     │  │ 风险管理     │            │
│  │ (BaseML)    │  │ (Signals)   │  │ (RiskMgmt)  │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    模型管理层                                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │ 模型训练     │  │ 模型预测     │  │ 模型评估     │            │
│  │ (Training)  │  │ (Prediction)│  │ (Evaluation)│            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    算法实现层                                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │ 传统ML      │  │ 深度学习     │  │ 强化学习     │            │
│  │ (Sklearn)   │  │ (PyTorch)   │  │ (SB3)       │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
└─────────────────────────────────────────────────────────────────┘
```

## 策略基类设计

### 1. 机器学习策略基类

```python
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
import joblib
import torch

class BaseMLStrategy(ABC):
    """机器学习策略基类"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.feature_columns = []
        self.is_trained = False
        self.performance_metrics = {}
        
    @abstractmethod
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """准备特征数据"""
        pass
    
    @abstractmethod
    def prepare_labels(self, data: pd.DataFrame) -> pd.Series:
        """准备标签数据"""
        pass
    
    @abstractmethod
    def train_model(self, X: pd.DataFrame, y: pd.Series) -> None:
        """训练模型"""
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """模型预测"""
        pass
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """生成交易信号"""
        if not self.is_trained:
            raise ValueError("模型尚未训练")
        
        features = self.prepare_features(data)
        predictions = self.predict(features)
        
        # 将预测结果转换为交易信号
        signals = self.predictions_to_signals(predictions, data)
        return signals
    
    def predictions_to_signals(self, predictions: np.ndarray, data: pd.DataFrame) -> pd.Series:
        """将预测结果转换为交易信号"""
        # 默认实现：基于预测值生成信号
        signals = pd.Series(0, index=data.index)
        
        # 可以根据具体策略重写此方法
        if hasattr(self, 'signal_threshold'):
            signals[predictions > self.signal_threshold] = 1   # 买入
            signals[predictions < -self.signal_threshold] = -1  # 卖出
        
        return signals
    
    def save_model(self, path: str) -> None:
        """保存模型"""
        if self.model is None:
            raise ValueError("没有可保存的模型")
        
        model_data = {
            'model': self.model,
            'feature_columns': self.feature_columns,
            'config': self.config,
            'performance_metrics': self.performance_metrics
        }
        
        joblib.dump(model_data, path)
    
    def load_model(self, path: str) -> None:
        """加载模型"""
        model_data = joblib.load(path)
        self.model = model_data['model']
        self.feature_columns = model_data['feature_columns']
        self.config = model_data['config']
        self.performance_metrics = model_data['performance_metrics']
        self.is_trained = True
```

### 2. 特征工程基类

```python
class BaseFeatureEngineer:
    """特征工程基类"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.feature_columns = []
    
    def extract_technical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """提取技术指标特征"""
        features = data.copy()
        
        # 移动平均线
        for period in [5, 10, 20, 50]:
            features[f'SMA_{period}'] = data['Close'].rolling(period).mean()
            features[f'EMA_{period}'] = data['Close'].ewm(span=period).mean()
        
        # RSI
        features['RSI_14'] = self.calculate_rsi(data['Close'], 14)
        
        # MACD
        macd_data = self.calculate_macd(data['Close'])
        features['MACD'] = macd_data['macd']
        features['MACD_Signal'] = macd_data['signal']
        features['MACD_Histogram'] = macd_data['histogram']
        
        # 布林带
        bb_data = self.calculate_bollinger_bands(data['Close'])
        features['BB_Upper'] = bb_data['upper']
        features['BB_Middle'] = bb_data['middle']
        features['BB_Lower'] = bb_data['lower']
        
        return features
    
    def extract_statistical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """提取统计特征"""
        features = data.copy()
        
        # 价格变化率
        features['Price_Change'] = data['Close'].pct_change()
        features['Price_Change_5'] = data['Close'].pct_change(5)
        features['Price_Change_10'] = data['Close'].pct_change(10)
        
        # 波动率
        features['Volatility_5'] = data['Close'].rolling(5).std()
        features['Volatility_20'] = data['Close'].rolling(20).std()
        
        # 成交量特征
        features['Volume_MA_5'] = data['Volume'].rolling(5).mean()
        features['Volume_MA_20'] = data['Volume'].rolling(20).mean()
        features['Volume_Ratio'] = data['Volume'] / features['Volume_MA_20']
        
        return features
    
    def extract_lag_features(self, data: pd.DataFrame, lags: list = [1, 2, 3, 5, 10]) -> pd.DataFrame:
        """提取滞后特征"""
        features = data.copy()
        
        for lag in lags:
            features[f'Close_lag_{lag}'] = data['Close'].shift(lag)
            features[f'Volume_lag_{lag}'] = data['Volume'].shift(lag)
            features[f'Price_Change_lag_{lag}'] = data['Close'].pct_change().shift(lag)
        
        return features
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """计算RSI指标"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
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
    
    def calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: float = 2) -> Dict[str, pd.Series]:
        """计算布林带指标"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        
        return {
            'upper': sma + (std * std_dev),
            'middle': sma,
            'lower': sma - (std * std_dev)
        }
```

## 传统机器学习策略

### 1. 价格预测策略

```python
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class PricePredictionStrategy(BaseMLStrategy):
    """价格预测策略"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_type = config.get('model_type', 'random_forest')
        self.prediction_horizon = config.get('prediction_horizon', 1)
        self.feature_engineer = BaseFeatureEngineer(config)
        self.scaler = StandardScaler()
        
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """准备特征数据"""
        # 提取技术指标特征
        features = self.feature_engineer.extract_technical_features(data)
        
        # 提取统计特征
        features = self.feature_engineer.extract_statistical_features(features)
        
        # 提取滞后特征
        features = self.feature_engineer.extract_lag_features(features)
        
        # 选择数值特征
        numeric_features = features.select_dtypes(include=[np.number]).columns
        features = features[numeric_features]
        
        # 处理缺失值
        features = features.fillna(method='ffill').fillna(method='bfill')
        
        self.feature_columns = features.columns.tolist()
        return features
    
    def prepare_labels(self, data: pd.DataFrame) -> pd.Series:
        """准备标签数据"""
        # 预测未来价格变化率
        labels = data['Close'].pct_change(self.prediction_horizon).shift(-self.prediction_horizon)
        return labels
    
    def train_model(self, X: pd.DataFrame, y: pd.Series) -> None:
        """训练模型"""
        # 数据预处理
        X_scaled = self.scaler.fit_transform(X)
        
        # 分割训练和验证集
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        # 选择模型
        if self.model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
        elif self.model_type == 'linear_regression':
            self.model = LinearRegression()
        elif self.model_type == 'ridge':
            self.model = Ridge(alpha=1.0)
        elif self.model_type == 'lasso':
            self.model = Lasso(alpha=0.1)
        elif self.model_type == 'svr':
            self.model = SVR(kernel='rbf', C=1.0, gamma='scale')
        
        # 训练模型
        self.model.fit(X_train, y_train)
        
        # 模型评估
        y_pred = self.model.predict(X_val)
        self.performance_metrics = {
            'mse': mean_squared_error(y_val, y_pred),
            'mae': mean_absolute_error(y_val, y_pred),
            'r2': r2_score(y_val, y_pred)
        }
        
        self.is_trained = True
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """模型预测"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predictions_to_signals(self, predictions: np.ndarray, data: pd.DataFrame) -> pd.Series:
        """将预测结果转换为交易信号"""
        signals = pd.Series(0, index=data.index)
        
        # 基于预测的价格变化率生成信号
        threshold = self.config.get('signal_threshold', 0.02)  # 2%阈值
        
        signals[predictions > threshold] = 1   # 买入信号
        signals[predictions < -threshold] = -1  # 卖出信号
        
        return signals
```

### 2. 涨跌分类策略

```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class DirectionPredictionStrategy(BaseMLStrategy):
    """涨跌分类策略"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_type = config.get('model_type', 'random_forest')
        self.prediction_horizon = config.get('prediction_horizon', 1)
        self.feature_engineer = BaseFeatureEngineer(config)
        self.scaler = StandardScaler()
        
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """准备特征数据"""
        # 提取技术指标特征
        features = self.feature_engineer.extract_technical_features(data)
        
        # 提取统计特征
        features = self.feature_engineer.extract_statistical_features(features)
        
        # 提取滞后特征
        features = self.feature_engineer.extract_lag_features(features)
        
        # 选择数值特征
        numeric_features = features.select_dtypes(include=[np.number]).columns
        features = features[numeric_features]
        
        # 处理缺失值
        features = features.fillna(method='ffill').fillna(method='bfill')
        
        self.feature_columns = features.columns.tolist()
        return features
    
    def prepare_labels(self, data: pd.DataFrame) -> pd.Series:
        """准备标签数据"""
        # 预测未来价格方向
        future_returns = data['Close'].pct_change(self.prediction_horizon).shift(-self.prediction_horizon)
        labels = (future_returns > 0).astype(int)  # 1表示上涨，0表示下跌
        return labels
    
    def train_model(self, X: pd.DataFrame, y: pd.Series) -> None:
        """训练模型"""
        # 数据预处理
        X_scaled = self.scaler.fit_transform(X)
        
        # 分割训练和验证集
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        # 选择模型
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
        elif self.model_type == 'logistic_regression':
            self.model = LogisticRegression(random_state=42)
        elif self.model_type == 'svc':
            self.model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
        
        # 训练模型
        self.model.fit(X_train, y_train)
        
        # 模型评估
        y_pred = self.model.predict(X_val)
        self.performance_metrics = {
            'accuracy': accuracy_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred, average='weighted'),
            'recall': recall_score(y_val, y_pred, average='weighted'),
            'f1': f1_score(y_val, y_pred, average='weighted')
        }
        
        self.is_trained = True
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """模型预测"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """预测概率"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def predictions_to_signals(self, predictions: np.ndarray, data: pd.DataFrame) -> pd.Series:
        """将预测结果转换为交易信号"""
        signals = pd.Series(0, index=data.index)
        
        # 基于分类结果生成信号
        signals[predictions == 1] = 1   # 买入信号
        signals[predictions == 0] = -1  # 卖出信号
        
        return signals
```

## 深度学习策略

### 1. LSTM价格预测策略

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler

class LSTMModel(nn.Module):
    """LSTM模型"""
    
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, output_size: int = 1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM前向传播
        out, _ = self.lstm(x, (h0, c0))
        
        # 取最后一个时间步的输出
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        
        return out

class LSTMStrategy(BaseMLStrategy):
    """LSTM价格预测策略"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.sequence_length = config.get('sequence_length', 60)
        self.hidden_size = config.get('hidden_size', 64)
        self.num_layers = config.get('num_layers', 2)
        self.learning_rate = config.get('learning_rate', 0.001)
        self.batch_size = config.get('batch_size', 32)
        self.epochs = config.get('epochs', 100)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = MinMaxScaler()
        
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """准备特征数据"""
        # 选择特征
        feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        features = data[feature_columns].copy()
        
        # 添加技术指标
        features['SMA_20'] = data['Close'].rolling(20).mean()
        features['RSI_14'] = self.calculate_rsi(data['Close'], 14)
        features['Price_Change'] = data['Close'].pct_change()
        
        # 处理缺失值
        features = features.fillna(method='ffill').fillna(method='bfill')
        
        self.feature_columns = features.columns.tolist()
        return features
    
    def prepare_labels(self, data: pd.DataFrame) -> pd.Series:
        """准备标签数据"""
        # 预测未来价格
        labels = data['Close'].shift(-1)
        return labels
    
    def create_sequences(self, data: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """创建时间序列数据"""
        X, y = [], []
        
        for i in range(self.sequence_length, len(data)):
            X.append(data[i-self.sequence_length:i])
            y.append(labels[i])
        
        return np.array(X), np.array(y)
    
    def train_model(self, X: pd.DataFrame, y: pd.Series) -> None:
        """训练模型"""
        # 数据标准化
        X_scaled = self.scaler.fit_transform(X)
        y_scaled = self.scaler.fit_transform(y.values.reshape(-1, 1)).flatten()
        
        # 创建时间序列
        X_seq, y_seq = self.create_sequences(X_scaled, y_scaled)
        
        # 转换为PyTorch张量
        X_tensor = torch.FloatTensor(X_seq).to(self.device)
        y_tensor = torch.FloatTensor(y_seq).to(self.device)
        
        # 创建数据加载器
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # 初始化模型
        self.model = LSTMModel(
            input_size=X_scaled.shape[1],
            hidden_size=self.hidden_size,
            num_layers=self.num_layers
        ).to(self.device)
        
        # 定义损失函数和优化器
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # 训练模型
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if epoch % 10 == 0:
                print(f'Epoch {epoch}, Loss: {total_loss/len(dataloader):.4f}')
        
        self.is_trained = True
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """模型预测"""
        # 数据标准化
        X_scaled = self.scaler.transform(X)
        
        # 创建时间序列
        X_seq, _ = self.create_sequences(X_scaled, np.zeros(len(X_scaled)))
        
        # 转换为PyTorch张量
        X_tensor = torch.FloatTensor(X_seq).to(self.device)
        
        # 模型预测
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_tensor).cpu().numpy()
        
        # 反标准化
        predictions = self.scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
        
        return predictions
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """计算RSI指标"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
```

### 2. CNN-LSTM混合策略

```python
class CNNLSTMModel(nn.Module):
    """CNN-LSTM混合模型"""
    
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, output_size: int = 1):
        super(CNNLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # CNN层
        self.conv1d = nn.Conv1d(input_size, 64, kernel_size=3, padding=1)
        self.conv1d2 = nn.Conv1d(64, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        
        # LSTM层
        self.lstm = nn.LSTM(32, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        # CNN前向传播
        x = x.transpose(1, 2)  # (batch, seq, features) -> (batch, features, seq)
        x = torch.relu(self.conv1d(x))
        x = self.pool(x)
        x = torch.relu(self.conv1d2(x))
        x = self.pool(x)
        x = x.transpose(1, 2)  # (batch, features, seq) -> (batch, seq, features)
        
        # LSTM前向传播
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        
        return out

class CNNLSTMStrategy(BaseMLStrategy):
    """CNN-LSTM混合策略"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.sequence_length = config.get('sequence_length', 60)
        self.hidden_size = config.get('hidden_size', 64)
        self.num_layers = config.get('num_layers', 2)
        self.learning_rate = config.get('learning_rate', 0.001)
        self.batch_size = config.get('batch_size', 32)
        self.epochs = config.get('epochs', 100)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = MinMaxScaler()
        
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """准备特征数据"""
        # 选择特征
        feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        features = data[feature_columns].copy()
        
        # 添加技术指标
        features['SMA_20'] = data['Close'].rolling(20).mean()
        features['RSI_14'] = self.calculate_rsi(data['Close'], 14)
        features['Price_Change'] = data['Close'].pct_change()
        
        # 处理缺失值
        features = features.fillna(method='ffill').fillna(method='bfill')
        
        self.feature_columns = features.columns.tolist()
        return features
    
    def prepare_labels(self, data: pd.DataFrame) -> pd.Series:
        """准备标签数据"""
        # 预测未来价格
        labels = data['Close'].shift(-1)
        return labels
    
    def create_sequences(self, data: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """创建时间序列数据"""
        X, y = [], []
        
        for i in range(self.sequence_length, len(data)):
            X.append(data[i-self.sequence_length:i])
            y.append(labels[i])
        
        return np.array(X), np.array(y)
    
    def train_model(self, X: pd.DataFrame, y: pd.Series) -> None:
        """训练模型"""
        # 数据标准化
        X_scaled = self.scaler.fit_transform(X)
        y_scaled = self.scaler.fit_transform(y.values.reshape(-1, 1)).flatten()
        
        # 创建时间序列
        X_seq, y_seq = self.create_sequences(X_scaled, y_scaled)
        
        # 转换为PyTorch张量
        X_tensor = torch.FloatTensor(X_seq).to(self.device)
        y_tensor = torch.FloatTensor(y_seq).to(self.device)
        
        # 创建数据加载器
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # 初始化模型
        self.model = CNNLSTMModel(
            input_size=X_scaled.shape[1],
            hidden_size=self.hidden_size,
            num_layers=self.num_layers
        ).to(self.device)
        
        # 定义损失函数和优化器
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # 训练模型
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if epoch % 10 == 0:
                print(f'Epoch {epoch}, Loss: {total_loss/len(dataloader):.4f}')
        
        self.is_trained = True
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """模型预测"""
        # 数据标准化
        X_scaled = self.scaler.transform(X)
        
        # 创建时间序列
        X_seq, _ = self.create_sequences(X_scaled, np.zeros(len(X_scaled)))
        
        # 转换为PyTorch张量
        X_tensor = torch.FloatTensor(X_seq).to(self.device)
        
        # 模型预测
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_tensor).cpu().numpy()
        
        # 反标准化
        predictions = self.scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
        
        return predictions
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """计算RSI指标"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
```

## 强化学习策略

### 1. DQN策略

```python
import gym
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

class TradingEnvironment(gym.Env):
    """交易环境"""
    
    def __init__(self, data: pd.DataFrame, initial_balance: float = 100000):
        super(TradingEnvironment, self).__init__()
        
        self.data = data
        self.initial_balance = initial_balance
        self.current_step = 0
        self.balance = initial_balance
        self.position = 0
        self.shares = 0
        
        # 动作空间：0=持有, 1=买入, 2=卖出
        self.action_space = gym.spaces.Discrete(3)
        
        # 状态空间：价格、技术指标、账户信息
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
        )
        
    def reset(self):
        """重置环境"""
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0
        self.shares = 0
        return self._get_observation()
    
    def step(self, action):
        """执行动作"""
        current_price = self.data.iloc[self.current_step]['Close']
        
        if action == 1 and self.balance > current_price:  # 买入
            self.shares = self.balance // current_price
            self.balance -= self.shares * current_price
            self.position = 1
        elif action == 2 and self.shares > 0:  # 卖出
            self.balance += self.shares * current_price
            self.shares = 0
            self.position = 0
        
        # 计算奖励
        reward = self._calculate_reward(action, current_price)
        
        # 更新步骤
        self.current_step += 1
        
        # 检查是否结束
        done = self.current_step >= len(self.data) - 1
        
        return self._get_observation(), reward, done, {}
    
    def _get_observation(self):
        """获取当前状态"""
        if self.current_step >= len(self.data):
            return np.zeros(10)
        
        row = self.data.iloc[self.current_step]
        
        # 状态特征
        features = [
            row['Close'] / 100,  # 归一化价格
            row['Volume'] / 1000000,  # 归一化成交量
            row.get('SMA_20', 0) / 100,
            row.get('RSI_14', 50) / 100,
            row.get('MACD', 0),
            self.balance / self.initial_balance,  # 账户余额比例
            self.position,  # 持仓状态
            self.shares / 1000,  # 持股数量
            (self.balance + self.shares * row['Close']) / self.initial_balance,  # 总资产比例
            self.current_step / len(self.data)  # 时间进度
        ]
        
        return np.array(features, dtype=np.float32)
    
    def _calculate_reward(self, action, current_price):
        """计算奖励"""
        # 基础奖励：资产变化
        total_value = self.balance + self.shares * current_price
        reward = (total_value - self.initial_balance) / self.initial_balance
        
        # 交易成本惩罚
        if action != 0:  # 有交易行为
            reward -= 0.001  # 0.1%的交易成本
        
        return reward

class DQNStrategy(BaseMLStrategy):
    """DQN强化学习策略"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.initial_balance = config.get('initial_balance', 100000)
        self.model = None
        self.env = None
        
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """准备特征数据"""
        # 添加技术指标
        features = data.copy()
        features['SMA_20'] = data['Close'].rolling(20).mean()
        features['RSI_14'] = self.calculate_rsi(data['Close'], 14)
        features['MACD'] = self.calculate_macd(data['Close'])
        
        # 处理缺失值
        features = features.fillna(method='ffill').fillna(method='bfill')
        
        self.feature_columns = features.columns.tolist()
        return features
    
    def prepare_labels(self, data: pd.DataFrame) -> pd.Series:
        """强化学习不需要标签"""
        return pd.Series()
    
    def train_model(self, X: pd.DataFrame, y: pd.Series) -> None:
        """训练DQN模型"""
        # 创建交易环境
        self.env = TradingEnvironment(X, self.initial_balance)
        
        # 创建DQN模型
        self.model = DQN(
            "MlpPolicy",
            self.env,
            learning_rate=0.0001,
            buffer_size=50000,
            learning_starts=1000,
            batch_size=32,
            tau=1.0,
            gamma=0.99,
            train_freq=4,
            target_update_interval=1000,
            exploration_fraction=0.1,
            exploration_final_eps=0.02,
            verbose=1
        )
        
        # 训练模型
        self.model.learn(total_timesteps=100000)
        
        self.is_trained = True
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """DQN预测"""
        if not self.is_trained:
            raise ValueError("模型尚未训练")
        
        # 创建环境
        env = TradingEnvironment(X, self.initial_balance)
        
        # 预测动作
        obs = env.reset()
        actions = []
        
        for _ in range(len(X)):
            action, _ = self.model.predict(obs, deterministic=True)
            actions.append(action)
            obs, _, done, _ = env.step(action)
            if done:
                break
        
        return np.array(actions)
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """计算RSI指标"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, prices: pd.Series) -> pd.Series:
        """计算MACD指标"""
        ema_12 = prices.ewm(span=12).mean()
        ema_26 = prices.ewm(span=26).mean()
        macd = ema_12 - ema_26
        return macd
```

### 2. PPO策略

```python
from stable_baselines3 import PPO

class PPOStrategy(BaseMLStrategy):
    """PPO强化学习策略"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.initial_balance = config.get('initial_balance', 100000)
        self.model = None
        self.env = None
        
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """准备特征数据"""
        # 添加技术指标
        features = data.copy()
        features['SMA_20'] = data['Close'].rolling(20).mean()
        features['RSI_14'] = self.calculate_rsi(data['Close'], 14)
        features['MACD'] = self.calculate_macd(data['Close'])
        
        # 处理缺失值
        features = features.fillna(method='ffill').fillna(method='bfill')
        
        self.feature_columns = features.columns.tolist()
        return features
    
    def prepare_labels(self, data: pd.DataFrame) -> pd.Series:
        """强化学习不需要标签"""
        return pd.Series()
    
    def train_model(self, X: pd.DataFrame, y: pd.Series) -> None:
        """训练PPO模型"""
        # 创建交易环境
        self.env = TradingEnvironment(X, self.initial_balance)
        
        # 创建PPO模型
        self.model = PPO(
            "MlpPolicy",
            self.env,
            learning_rate=0.0003,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1
        )
        
        # 训练模型
        self.model.learn(total_timesteps=100000)
        
        self.is_trained = True
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """PPO预测"""
        if not self.is_trained:
            raise ValueError("模型尚未训练")
        
        # 创建环境
        env = TradingEnvironment(X, self.initial_balance)
        
        # 预测动作
        obs = env.reset()
        actions = []
        
        for _ in range(len(X)):
            action, _ = self.model.predict(obs, deterministic=True)
            actions.append(action)
            obs, _, done, _ = env.step(action)
            if done:
                break
        
        return np.array(actions)
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """计算RSI指标"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, prices: pd.Series) -> pd.Series:
        """计算MACD指标"""
        ema_12 = prices.ewm(span=12).mean()
        ema_26 = prices.ewm(span=26).mean()
        macd = ema_12 - ema_26
        return macd
```

## 策略集成与管理

### 1. 策略管理器

```python
class MLStrategyManager:
    """机器学习策略管理器"""
    
    def __init__(self):
        self.strategies = {}
        self.performance_tracker = {}
        
    def register_strategy(self, name: str, strategy: BaseMLStrategy):
        """注册策略"""
        self.strategies[name] = strategy
        
    def train_strategy(self, name: str, data: pd.DataFrame):
        """训练策略"""
        if name not in self.strategies:
            raise ValueError(f"策略 {name} 未注册")
        
        strategy = self.strategies[name]
        features = strategy.prepare_features(data)
        labels = strategy.prepare_labels(data)
        
        # 移除缺失值
        valid_mask = ~(features.isnull().any(axis=1) | labels.isnull())
        features = features[valid_mask]
        labels = labels[valid_mask]
        
        strategy.train_model(features, labels)
        
        print(f"策略 {name} 训练完成")
        
    def evaluate_strategy(self, name: str, data: pd.DataFrame) -> Dict[str, float]:
        """评估策略"""
        if name not in self.strategies:
            raise ValueError(f"策略 {name} 未注册")
        
        strategy = self.strategies[name]
        if not strategy.is_trained:
            raise ValueError(f"策略 {name} 尚未训练")
        
        features = strategy.prepare_features(data)
        signals = strategy.generate_signals(data)
        
        # 计算策略性能
        performance = self._calculate_performance(signals, data)
        
        self.performance_tracker[name] = performance
        return performance
    
    def _calculate_performance(self, signals: pd.Series, data: pd.DataFrame) -> Dict[str, float]:
        """计算策略性能"""
        returns = data['Close'].pct_change()
        strategy_returns = signals.shift(1) * returns
        
        # 计算性能指标
        total_return = (1 + strategy_returns).prod() - 1
        annual_return = (1 + total_return) ** (252 / len(strategy_returns)) - 1
        volatility = strategy_returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio
        }
    
    def get_strategy_signals(self, name: str, data: pd.DataFrame) -> pd.Series:
        """获取策略信号"""
        if name not in self.strategies:
            raise ValueError(f"策略 {name} 未注册")
        
        strategy = self.strategies[name]
        if not strategy.is_trained:
            raise ValueError(f"策略 {name} 尚未训练")
        
        return strategy.generate_signals(data)
    
    def save_strategy(self, name: str, path: str):
        """保存策略"""
        if name not in self.strategies:
            raise ValueError(f"策略 {name} 未注册")
        
        strategy = self.strategies[name]
        strategy.save_model(path)
        
    def load_strategy(self, name: str, path: str):
        """加载策略"""
        # 这里需要根据策略类型创建相应的实例
        # 简化实现，实际使用时需要更复杂的逻辑
        pass
```

### 2. 策略配置管理

```python
import yaml

class StrategyConfigManager:
    """策略配置管理器"""
    
    def __init__(self, config_path: str = "config/strategies.yaml"):
        self.config_path = config_path
        self.configs = self._load_configs()
        
    def _load_configs(self) -> Dict[str, Dict]:
        """加载配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            return {}
    
    def get_strategy_config(self, strategy_name: str) -> Dict[str, Any]:
        """获取策略配置"""
        return self.configs.get(strategy_name, {})
    
    def save_strategy_config(self, strategy_name: str, config: Dict[str, Any]):
        """保存策略配置"""
        self.configs[strategy_name] = config
        self._save_configs()
    
    def _save_configs(self):
        """保存配置文件"""
        with open(self.config_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.configs, f, default_flow_style=False, allow_unicode=True)
```

## 超参数优化

### 1. Optuna集成

```python
import optuna
from optuna.integration import PyTorchLightningPruningCallback

class HyperparameterOptimizer:
    """超参数优化器"""
    
    def __init__(self, strategy_class, data: pd.DataFrame):
        self.strategy_class = strategy_class
        self.data = data
        
    def optimize(self, n_trials: int = 100) -> Dict[str, Any]:
        """优化超参数"""
        def objective(trial):
            # 定义超参数搜索空间
            config = self._suggest_hyperparameters(trial)
            
            # 创建策略实例
            strategy = self.strategy_class(config)
            
            # 准备数据
            features = strategy.prepare_features(self.data)
            labels = strategy.prepare_labels(self.data)
            
            # 移除缺失值
            valid_mask = ~(features.isnull().any(axis=1) | labels.isnull())
            features = features[valid_mask]
            labels = labels[valid_mask]
            
            # 训练模型
            strategy.train_model(features, labels)
            
            # 评估性能
            performance = self._evaluate_strategy(strategy, features, labels)
            
            return performance['sharpe_ratio']
        
        # 创建优化器
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        return study.best_params
    
    def _suggest_hyperparameters(self, trial) -> Dict[str, Any]:
        """建议超参数"""
        # 这里需要根据具体策略类型定义超参数搜索空间
        # 示例实现
        config = {
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
            'epochs': trial.suggest_int('epochs', 50, 200),
        }
        
        return config
    
    def _evaluate_strategy(self, strategy, features: pd.DataFrame, labels: pd.Series) -> Dict[str, float]:
        """评估策略性能"""
        # 简化的评估实现
        predictions = strategy.predict(features)
        
        # 计算性能指标
        mse = np.mean((predictions - labels.values) ** 2)
        mae = np.mean(np.abs(predictions - labels.values))
        
        return {
            'mse': mse,
            'mae': mae,
            'sharpe_ratio': -mse  # 简化实现
        }
```

## 总结

TrckTrade的机器学习策略设计采用分层架构，从基础策略类到具体的算法实现，形成了完整的机器学习策略框架。通过统一的接口设计，支持传统机器学习、深度学习和强化学习策略的无缝集成。模块化的设计使得各个策略可以独立开发和测试，为平台的扩展性提供了坚实的基础。

关键特性：
1. **统一接口**: 所有策略遵循相同的接口规范
2. **模块化设计**: 特征工程、模型训练、预测分离
3. **多算法支持**: 支持传统ML、深度学习、强化学习
4. **超参数优化**: 集成Optuna进行自动调参
5. **策略管理**: 统一的策略注册、训练、评估机制
6. **性能监控**: 完整的策略性能跟踪和评估
