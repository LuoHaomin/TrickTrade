"""
LSTM基金价格预测器
使用LSTM神经网络进行基金价格预测
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

class LSTMModel(nn.Module):
    """LSTM模型"""
    
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, 
                 output_size: int = 1, dropout: float = 0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        
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

class LSTMPredictor:
    """LSTM基金价格预测器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化LSTM预测器
        
        Args:
            config: 配置参数
                - sequence_length: 序列长度
                - hidden_size: 隐藏层大小
                - num_layers: LSTM层数
                - learning_rate: 学习率
                - batch_size: 批次大小
                - epochs: 训练轮数
                - dropout: Dropout比例
        """
        self.config = config
        self.sequence_length = config.get('sequence_length', 60)
        self.hidden_size = config.get('hidden_size', 64)
        self.num_layers = config.get('num_layers', 2)
        self.learning_rate = config.get('learning_rate', 0.001)
        self.batch_size = config.get('batch_size', 32)
        self.epochs = config.get('epochs', 100)
        self.dropout = config.get('dropout', 0.2)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = MinMaxScaler()
        self.model = None
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
        
        # 技术指标
        features['SMA_5'] = features['Close'].rolling(5).mean()
        features['SMA_10'] = features['Close'].rolling(10).mean()
        features['SMA_20'] = features['Close'].rolling(20).mean()
        
        features['EMA_5'] = features['Close'].ewm(span=5).mean()
        features['EMA_10'] = features['Close'].ewm(span=10).mean()
        
        # RSI
        features['RSI_14'] = self._calculate_rsi(features['Close'], 14)
        
        # 波动率
        features['Volatility_5'] = features['Close'].rolling(5).std()
        features['Volatility_20'] = features['Close'].rolling(20).std()
        
        # 成交量特征
        features['Volume_MA_5'] = features['Volume'].rolling(5).mean()
        features['Volume_Ratio'] = features['Volume'] / features['Volume_MA_5']
        
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
    
    def create_sequences(self, data: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        创建时间序列数据
        
        Args:
            data: 特征数据
            labels: 标签数据
            
        Returns:
            序列数据和标签
        """
        X, y = [], []
        
        for i in range(self.sequence_length, len(data)):
            X.append(data[i-self.sequence_length:i])
            y.append(labels[i])
        
        return np.array(X), np.array(y)
    
    def train(self, data: pd.DataFrame, target_column: str = 'Close') -> None:
        """
        训练LSTM模型
        
        Args:
            data: 训练数据
            target_column: 目标列名
        """
        print("开始训练LSTM模型...")
        
        # 准备特征
        features = self.prepare_features(data)
        
        # 准备标签
        labels = data[target_column].values
        
        # 数据标准化
        features_scaled = self.scaler.fit_transform(features)
        labels_scaled = self.scaler.fit_transform(labels.reshape(-1, 1)).flatten()
        
        # 创建时间序列
        X_seq, y_seq = self.create_sequences(features_scaled, labels_scaled)
        
        # 分割训练和验证集
        split_idx = int(len(X_seq) * 0.8)
        X_train, X_val = X_seq[:split_idx], X_seq[split_idx:]
        y_train, y_val = y_seq[:split_idx], y_seq[split_idx:]
        
        # 转换为PyTorch张量
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_val_tensor = torch.FloatTensor(y_val).to(self.device)
        
        # 创建数据加载器
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        # 初始化模型
        self.model = LSTMModel(
            input_size=features_scaled.shape[1],
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)
        
        # 定义损失函数和优化器
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        # 训练模型
        self.model.train()
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.epochs):
            total_loss = 0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            # 验证
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_tensor)
                val_loss = criterion(val_outputs.squeeze(), y_val_tensor).item()
            
            scheduler.step(val_loss)
            
            # 早停
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= 20:
                print(f"早停于第 {epoch+1} 轮")
                break
            
            if epoch % 10 == 0:
                print(f'Epoch {epoch+1}/{self.epochs}, Loss: {total_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}')
        
        self.is_trained = True
        print("LSTM模型训练完成")
    
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
        
        self.model.eval()
        
        # 准备特征
        features = self.prepare_features(data)
        
        # 数据标准化
        features_scaled = self.scaler.transform(features)
        
        predictions = []
        current_sequence = features_scaled[-self.sequence_length:].copy()
        
        with torch.no_grad():
            for _ in range(steps):
                # 转换为张量
                X_tensor = torch.FloatTensor(current_sequence.reshape(1, self.sequence_length, -1)).to(self.device)
                
                # 预测
                pred = self.model(X_tensor).cpu().numpy()[0, 0]
                predictions.append(pred)
                
                # 更新序列（简化处理，实际应用中需要更复杂的更新策略）
                current_sequence = np.roll(current_sequence, -1, axis=0)
                current_sequence[-1, 0] = pred  # 假设第一个特征是价格
        
        # 反标准化
        predictions = self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
        
        # 计算置信区间（简化实现）
        std_dev = np.std(predictions) if len(predictions) > 1 else 0.1
        confidence_interval = 1.96 * std_dev
        
        result = {
            'predictions': predictions.tolist(),
            'confidence_interval': confidence_interval,
            'prediction_dates': pd.date_range(start=data.index[-1], periods=steps+1, freq='D')[1:].strftime('%Y-%m-%d').tolist(),
            'model_type': 'LSTM',
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
            'model_type': 'LSTM'
        }
    
    def save_model(self, path: str) -> None:
        """保存模型"""
        if self.model is None:
            raise ValueError("没有可保存的模型")
        
        model_data = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'feature_columns': self.feature_columns,
            'scaler': self.scaler
        }
        
        torch.save(model_data, path)
    
    def load_model(self, path: str) -> None:
        """加载模型"""
        model_data = torch.load(path, map_location=self.device)
        
        # 重建模型
        self.model = LSTMModel(
            input_size=len(model_data['feature_columns']),
            hidden_size=self.config.get('hidden_size', 64),
            num_layers=self.config.get('num_layers', 2),
            dropout=self.config.get('dropout', 0.2)
        ).to(self.device)
        
        self.model.load_state_dict(model_data['model_state_dict'])
        self.config = model_data['config']
        self.feature_columns = model_data['feature_columns']
        self.scaler = model_data['scaler']
        self.is_trained = True
