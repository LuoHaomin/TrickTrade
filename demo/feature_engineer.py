"""
特征工程模块
计算技术指标并进行归一化处理，为强化学习模型提供特征
"""

import pandas as pd
import numpy as np
import talib
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """特征工程器"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化特征工程器
        
        Args:
            config: 配置字典，包含技术指标参数
        """
        self.config = config or self._get_default_config()
        
    def _get_default_config(self) -> Dict:
        """获取默认配置"""
        return {
            'ma_periods': [5, 10, 20, 60],
            'rsi_period': 14,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'bb_period': 20,
            'bb_std': 2,
            'momentum_period': 10,
            'price_change_periods': [1, 3, 5, 10],
            'volume_periods': [5, 10, 20]
        }
    
    def calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算技术指标
        
        Args:
            data: 包含OHLCV数据的DataFrame
            
        Returns:
            添加了技术指标的DataFrame
        """
        df = data.copy()
        
        # 确保数据按日期排序
        df = df.sort_index()
        
        # 计算移动平均线
        df = self._calculate_moving_averages(df)
        
        # 计算RSI
        df = self._calculate_rsi(df)
        
        # 计算MACD
        df = self._calculate_macd(df)
        
        # 计算布林带
        df = self._calculate_bollinger_bands(df)
        
        # 计算动量指标
        df = self._calculate_momentum(df)
        
        # 计算价格变化率
        df = self._calculate_price_changes(df)
        
        # 计算成交量指标
        df = self._calculate_volume_indicators(df)
        
        # 计算波动率
        df = self._calculate_volatility(df)
        
        return df
    
    def _calculate_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算移动平均线"""
        periods = self.config['ma_periods']
        
        for period in periods:
            # 简单移动平均
            df[f'MA_{period}'] = talib.SMA(df['Close'].values.astype(np.float64), timeperiod=period)
            
            # 指数移动平均
            df[f'EMA_{period}'] = talib.EMA(df['Close'].values.astype(np.float64), timeperiod=period)
        
        return df
    
    def _calculate_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算RSI"""
        period = self.config['rsi_period']
        df['RSI'] = talib.RSI(df['Close'].values.astype(np.float64), timeperiod=period)
        return df
    
    def _calculate_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算MACD"""
        fast = self.config['macd_fast']
        slow = self.config['macd_slow']
        signal = self.config['macd_signal']
        
        macd, macd_signal, macd_hist = talib.MACD(
            df['Close'].values.astype(np.float64), 
            fastperiod=fast, 
            slowperiod=slow, 
            signalperiod=signal
        )
        
        df['MACD'] = macd
        df['MACD_Signal'] = macd_signal
        df['MACD_Hist'] = macd_hist
        
        return df
    
    def _calculate_bollinger_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算布林带"""
        period = self.config['bb_period']
        std = self.config['bb_std']
        
        upper, middle, lower = talib.BBANDS(
            df['Close'].values.astype(np.float64),
            timeperiod=period,
            nbdevup=std,
            nbdevdn=std,
            matype=0
        )
        
        df['BB_Upper'] = upper
        df['BB_Middle'] = middle
        df['BB_Lower'] = lower
        
        # 布林带宽度
        df['BB_Width'] = (upper - lower) / middle
        
        # 布林带位置
        df['BB_Position'] = (df['Close'] - lower) / (upper - lower)
        
        return df
    
    def _calculate_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算动量指标"""
        period = self.config['momentum_period']
        
        # 动量
        df['Momentum'] = talib.MOM(df['Close'].values.astype(np.float64), timeperiod=period)
        
        # 变化率
        df['ROC'] = talib.ROC(df['Close'].values.astype(np.float64), timeperiod=period)
        
        return df
    
    def _calculate_price_changes(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算价格变化率"""
        periods = self.config['price_change_periods']
        
        for period in periods:
            df[f'Price_Change_{period}'] = df['Close'].pct_change(period)
            df[f'Price_Change_Abs_{period}'] = df[f'Price_Change_{period}'].abs()
        
        return df
    
    def _calculate_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算成交量指标"""
        periods = self.config['volume_periods']
        
        for period in periods:
            # 成交量移动平均
            df[f'Volume_MA_{period}'] = df['Volume'].rolling(window=period).mean()
            
            # 成交量比率
            df[f'Volume_Ratio_{period}'] = df['Volume'] / df[f'Volume_MA_{period}']
        
        # OBV (On Balance Volume)
        df['OBV'] = talib.OBV(df['Close'].values.astype(np.float64), df['Volume'].values.astype(np.float64))
        
        return df
    
    def _calculate_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算波动率"""
        periods = [5, 10, 20]
        
        for period in periods:
            # 历史波动率
            df[f'Volatility_{period}'] = df['Close'].pct_change().rolling(window=period).std() * np.sqrt(252)
            
            # ATR (Average True Range)
            df[f'ATR_{period}'] = talib.ATR(
                df['High'].values.astype(np.float64), 
                df['Low'].values.astype(np.float64), 
                df['Close'].values.astype(np.float64), 
                timeperiod=period
            )
        
        return df
    
    def normalize_features(self, df: pd.DataFrame, feature_columns: List[str], 
                          method: str = 'zscore') -> pd.DataFrame:
        """
        归一化特征
        
        Args:
            df: 包含特征的DataFrame
            feature_columns: 需要归一化的列名列表
            method: 归一化方法 ('zscore', 'minmax', 'robust')
            
        Returns:
            归一化后的DataFrame
        """
        df_normalized = df.copy()
        
        for col in feature_columns:
            if col in df.columns:
                if method == 'zscore':
                    # Z-score标准化
                    mean_val = df[col].mean()
                    std_val = df[col].std()
                    if std_val > 0:
                        df_normalized[col] = (df[col] - mean_val) / std_val
                    else:
                        df_normalized[col] = 0
                        
                elif method == 'minmax':
                    # Min-Max归一化
                    min_val = df[col].min()
                    max_val = df[col].max()
                    if max_val > min_val:
                        df_normalized[col] = (df[col] - min_val) / (max_val - min_val)
                    else:
                        df_normalized[col] = 0
                        
                elif method == 'robust':
                    # 鲁棒归一化（使用中位数和四分位距）
                    median_val = df[col].median()
                    q75, q25 = df[col].quantile([0.75, 0.25])
                    iqr = q75 - q25
                    if iqr > 0:
                        df_normalized[col] = (df[col] - median_val) / iqr
                    else:
                        df_normalized[col] = 0
        
        return df_normalized
    
    def get_feature_columns(self) -> List[str]:
        """获取所有特征列名"""
        feature_columns = []
        
        # 移动平均线
        for period in self.config['ma_periods']:
            feature_columns.extend([f'MA_{period}', f'EMA_{period}'])
        
        # RSI
        feature_columns.append('RSI')
        
        # MACD
        feature_columns.extend(['MACD', 'MACD_Signal', 'MACD_Hist'])
        
        # 布林带
        feature_columns.extend(['BB_Upper', 'BB_Middle', 'BB_Lower', 'BB_Width', 'BB_Position'])
        
        # 动量
        feature_columns.extend(['Momentum', 'ROC'])
        
        # 价格变化
        for period in self.config['price_change_periods']:
            feature_columns.extend([f'Price_Change_{period}', f'Price_Change_Abs_{period}'])
        
        # 成交量
        for period in self.config['volume_periods']:
            feature_columns.extend([f'Volume_MA_{period}', f'Volume_Ratio_{period}'])
        feature_columns.append('OBV')
        
        # 波动率
        for period in [5, 10, 20]:
            feature_columns.extend([f'Volatility_{period}', f'ATR_{period}'])
        
        return feature_columns
    
    def prepare_features_for_rl(self, data: pd.DataFrame, 
                                lookback_window: int = 20) -> Tuple[np.ndarray, List[str]]:
        """
        为强化学习准备特征数据
        
        Args:
            data: 包含技术指标的DataFrame
            lookback_window: 回望窗口大小
            
        Returns:
            特征数组和特征名称列表
        """
        # 获取特征列
        feature_columns = self.get_feature_columns()
        
        # 过滤存在的列
        available_features = [col for col in feature_columns if col in data.columns]
        
        # 添加基础价格特征
        basic_features = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in basic_features:
            if col in data.columns and col not in available_features:
                available_features.append(col)
        
        # 选择特征数据
        feature_data = data[available_features].copy()
        
        # 处理缺失值
        feature_data = feature_data.ffill().fillna(0)
        
        # 归一化特征
        feature_data = self.normalize_features(feature_data, available_features, method='zscore')
        
        # 创建滑动窗口特征
        windowed_features = []
        feature_names = []
        
        for i in range(lookback_window, len(feature_data)):
            window_data = feature_data.iloc[i-lookback_window:i]
            
            # 对每个特征计算统计量
            for col in available_features:
                values = window_data[col].values
                
                # 添加当前值
                windowed_features.append(values[-1])
                feature_names.append(f'{col}_current')
                
                # 添加均值
                windowed_features.append(np.mean(values))
                feature_names.append(f'{col}_mean')
                
                # 添加标准差
                windowed_features.append(np.std(values))
                feature_names.append(f'{col}_std')
                
                # 添加最大值和最小值
                windowed_features.append(np.max(values))
                feature_names.append(f'{col}_max')
                windowed_features.append(np.min(values))
                feature_names.append(f'{col}_min')
        
        # 转换为numpy数组
        feature_array = np.array(windowed_features).reshape(-1, len(available_features) * 5)
        
        return feature_array, feature_names
    
    def create_price_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        创建价格相关特征
        
        Args:
            data: 原始OHLCV数据
            
        Returns:
            添加了价格特征的DataFrame
        """
        df = data.copy()
        
        # 价格位置特征
        df['Price_Position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
        df['Price_Position'] = df['Price_Position'].fillna(0.5)  # 处理除零情况
        
        # 价格变化
        df['Price_Change'] = df['Close'].pct_change()
        df['Price_Change_Abs'] = df['Price_Change'].abs()
        
        # 高低价差
        df['High_Low_Diff'] = (df['High'] - df['Low']) / df['Close']
        
        # 开收价差
        df['Open_Close_Diff'] = (df['Close'] - df['Open']) / df['Open']
        
        # 成交量价格关系
        df['Volume_Price_Ratio'] = df['Volume'] / df['Close']
        
        return df


def main():
    """测试函数"""
    # 创建测试数据
    dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')
    np.random.seed(42)
    
    test_data = pd.DataFrame({
        'Open': 100 + np.cumsum(np.random.randn(len(dates)) * 0.5),
        'High': 100 + np.cumsum(np.random.randn(len(dates)) * 0.5) + np.random.rand(len(dates)) * 2,
        'Low': 100 + np.cumsum(np.random.randn(len(dates)) * 0.5) - np.random.rand(len(dates)) * 2,
        'Close': 100 + np.cumsum(np.random.randn(len(dates)) * 0.5),
        'Volume': np.random.randint(1000000, 10000000, len(dates))
    }, index=dates)
    
    # 确保High >= Low
    test_data['High'] = np.maximum(test_data['High'], test_data['Low'])
    test_data['High'] = np.maximum(test_data['High'], test_data['Close'])
    test_data['Low'] = np.minimum(test_data['Low'], test_data['Close'])
    
    print("原始数据:")
    print(test_data.head())
    
    # 创建特征工程器
    fe = FeatureEngineer()
    
    # 计算技术指标
    print("\n计算技术指标...")
    features_data = fe.calculate_technical_indicators(test_data)
    
    print(f"特征数据形状: {features_data.shape}")
    print(f"特征列数: {len(features_data.columns)}")
    
    # 准备RL特征
    print("\n准备RL特征...")
    rl_features, feature_names = fe.prepare_features_for_rl(features_data, lookback_window=10)
    
    print(f"RL特征形状: {rl_features.shape}")
    print(f"特征名称数量: {len(feature_names)}")
    print(f"前5个特征名称: {feature_names[:5]}")


if __name__ == "__main__":
    main()
