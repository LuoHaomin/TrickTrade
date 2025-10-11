# TrickTrade 数据流设计 (优化版)

## 概述

本文档描述了基于新架构的TrickTrade平台数据流设计，采用事件驱动架构，支持MVP优先的开发策略，确保数据流的简洁性和可扩展性。

## 数据流架构图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          数据源层 (Data Sources)                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │
│  │   AKShare   │  │ Yahoo Finance│  │   模拟数据   │  │   自定义数据 │      │
│  │   (A股/基金) │  │  (海外市场)  │  │   (测试)    │  │   (扩展)   │      │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘      │
└─────────────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      数据源适配器层 (Data Source Adapters)                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │
│  │ AKShare     │  │ Yahoo       │  │ Mock        │  │ Custom      │      │
│  │ Adapter     │  │ Adapter     │  │ Adapter     │  │ Adapter     │      │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘      │
└─────────────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        数据访问层 (Data Access Layer)                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │
│  │ 数据适配器   │  │ 数据验证器   │  │ 数据转换器   │  │ 数据仓库     │      │
│  │ (Data       │  │ (Data       │  │ (Data       │  │ (Data       │      │
│  │  Adapter)   │  │  Validator) │  │  Transformer)│  │  Warehouse) │      │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘      │
└─────────────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          事件总线 (Event Bus)                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │
│  │ 数据事件     │  │ 策略事件     │  │ 交易事件     │  │ 风险事件     │      │
│  │ (Data       │  │ (Strategy   │  │ (Trade      │  │ (Risk       │      │
│  │  Events)    │  │  Events)    │  │  Events)    │  │  Events)    │      │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘      │
└─────────────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        业务逻辑层 (Business Logic)                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │
│  │ 策略管理器   │  │ 回测引擎     │  │ 风险管理器   │  │ 投资组合管理 │      │
│  │ (Strategy   │  │ (Backtest   │  │ (Risk       │  │ (Portfolio  │      │
│  │  Manager)   │  │  Engine)    │  │  Manager)   │  │  Manager)   │      │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘      │
└─────────────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        应用服务层 (Application Services)                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │
│  │ 策略服务     │  │ 回测服务     │  │ 预测服务     │  │ 分析服务     │      │
│  │ (Strategy   │  │ (Backtest   │  │ (Prediction │  │ (Analysis   │      │
│  │  Service)   │  │  Service)   │  │  Service)   │  │  Service)   │      │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘      │
└─────────────────────────────────────────────────────────────────────────────┘
```

## MVP阶段数据流设计

### 1. 简化数据流 (MVP阶段)

#### 1.1 核心数据流
```
数据源 → 数据适配器 → 数据验证 → 数据存储 → 策略处理 → 回测引擎 → 结果输出
```

#### 1.2 MVP数据源
- **AKShare**: 主要数据源，支持A股和基金数据
- **模拟数据**: 用于测试和演示
- **文件数据**: 支持CSV格式的历史数据导入

#### 1.3 数据格式标准
```python
# 统一的数据格式
{
    'symbol': str,           # 标的代码
    'date': datetime,        # 日期
    'open': float,          # 开盘价
    'high': float,          # 最高价
    'low': float,           # 最低价
    'close': float,         # 收盘价
    'volume': int,          # 成交量
    'source': str           # 数据源
}
```

## 迭代阶段数据流设计

### 1. 扩展数据源

#### 1.1 多数据源支持
- **AKShare**: A股、基金、指数数据
- **Yahoo Finance**: 海外市场数据
- **Wind**: 专业金融数据 (企业版)
- **自定义API**: 支持第三方数据源

#### 1.2 实时数据流
```
实时数据源 → 数据适配器 → 数据验证 → 事件总线 → 策略处理 → 信号生成 → 交易执行
```

#### 1.3 数据质量监控
- **数据完整性检查**: 缺失值、异常值检测
- **数据一致性验证**: 价格逻辑、时间序列连续性
- **数据质量评分**: 自动质量评估和报告

### 2. 数据格式标准

#### 2.1 基础数据格式
```python
# OHLCV数据格式
{
    'Open': float,      # 开盘价
    'High': float,      # 最高价
    'Low': float,       # 最低价
    'Close': float,     # 收盘价
    'Volume': int,      # 成交量
    'Date': datetime    # 日期
}
```

#### 2.2 扩展数据格式
```python
# 包含技术指标的数据格式
{
    'Open': float,
    'High': float,
    'Low': float,
    'Close': float,
    'Volume': int,
    'Date': datetime,
    'SMA_20': float,    # 20日移动平均
    'RSI_14': float,    # 14日RSI
    'MACD': float,      # MACD指标
    # ... 其他技术指标
}
```

## 数据获取流程

### 1. 数据获取接口

#### 1.1 统一数据接口
```python
class DataProvider:
    def get_data(self, symbol: str, start_date: str, end_date: str, **kwargs):
        """获取历史数据"""
        pass
    
    def get_realtime_data(self, symbol: str):
        """获取实时数据"""
        pass
    
    def get_fundamental_data(self, symbol: str):
        """获取基本面数据"""
        pass
```

#### 1.2 数据验证机制
```python
class DataValidator:
    def validate_ohlcv(self, data: pd.DataFrame) -> bool:
        """验证OHLCV数据完整性"""
        pass
    
    def validate_price_range(self, data: pd.DataFrame) -> bool:
        """验证价格范围合理性"""
        pass
    
    def validate_volume(self, data: pd.DataFrame) -> bool:
        """验证成交量合理性"""
        pass
```

### 2. 数据获取策略

#### 2.1 增量更新策略
- **首次获取**: 获取全部历史数据
- **增量更新**: 只获取新增数据
- **数据补全**: 自动补全缺失数据

#### 2.2 缓存策略
- **内存缓存**: 最近访问的数据
- **磁盘缓存**: 持久化缓存
- **缓存更新**: 定期更新机制

#### 2.3 错误处理策略
- **重试机制**: 网络错误自动重试
- **降级策略**: 主数据源失败时使用备用源
- **异常记录**: 详细记录错误信息

## 数据处理流程

### 1. 数据清洗

#### 1.1 缺失值处理
```python
class DataCleaner:
    def handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """处理缺失值"""
        # 前向填充
        data = data.fillna(method='ffill')
        # 后向填充
        data = data.fillna(method='bfill')
        # 删除仍缺失的行
        data = data.dropna()
        return data
    
    def handle_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """处理异常值"""
        # 使用IQR方法检测异常值
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # 替换异常值
        data = data.clip(lower_bound, upper_bound, axis=1)
        return data
```

#### 1.2 数据标准化
```python
class DataNormalizer:
    def normalize_prices(self, data: pd.DataFrame) -> pd.DataFrame:
        """价格数据标准化"""
        # Min-Max标准化
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        price_columns = ['Open', 'High', 'Low', 'Close']
        data[price_columns] = scaler.fit_transform(data[price_columns])
        return data
    
    def standardize_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """特征标准化"""
        # Z-score标准化
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        feature_columns = data.select_dtypes(include=[np.number]).columns
        data[feature_columns] = scaler.fit_transform(data[feature_columns])
        return data
```

### 2. 特征工程

#### 2.1 技术指标计算
```python
class TechnicalIndicators:
    def calculate_sma(self, data: pd.DataFrame, period: int) -> pd.DataFrame:
        """计算简单移动平均"""
        data[f'SMA_{period}'] = data['Close'].rolling(window=period).mean()
        return data
    
    def calculate_rsi(self, data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """计算RSI指标"""
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        data[f'RSI_{period}'] = 100 - (100 / (1 + rs))
        return data
    
    def calculate_macd(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算MACD指标"""
        ema_12 = data['Close'].ewm(span=12).mean()
        ema_26 = data['Close'].ewm(span=26).mean()
        data['MACD'] = ema_12 - ema_26
        data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
        data['MACD_Histogram'] = data['MACD'] - data['MACD_Signal']
        return data
```

#### 2.2 时序特征提取
```python
class TimeSeriesFeatures:
    def extract_lag_features(self, data: pd.DataFrame, lags: list) -> pd.DataFrame:
        """提取滞后特征"""
        for lag in lags:
            data[f'Close_lag_{lag}'] = data['Close'].shift(lag)
            data[f'Volume_lag_{lag}'] = data['Volume'].shift(lag)
        return data
    
    def extract_rolling_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """提取滚动统计特征"""
        windows = [5, 10, 20, 50]
        for window in windows:
            data[f'Close_mean_{window}'] = data['Close'].rolling(window).mean()
            data[f'Close_std_{window}'] = data['Close'].rolling(window).std()
            data[f'Volume_mean_{window}'] = data['Volume'].rolling(window).mean()
        return data
    
    def extract_momentum_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """提取动量特征"""
        data['Price_change'] = data['Close'].pct_change()
        data['Volume_change'] = data['Volume'].pct_change()
        data['Price_volatility'] = data['Price_change'].rolling(20).std()
        return data
```

## 数据存储设计

### 1. 存储架构

#### 1.1 本地文件存储
```python
class LocalFileStorage:
    def __init__(self, base_path: str = "./data"):
        self.base_path = base_path
        self.ensure_directory()
    
    def save_data(self, data: pd.DataFrame, symbol: str, data_type: str):
        """保存数据到本地文件"""
        file_path = f"{self.base_path}/{data_type}/{symbol}.parquet"
        data.to_parquet(file_path)
    
    def load_data(self, symbol: str, data_type: str) -> pd.DataFrame:
        """从本地文件加载数据"""
        file_path = f"{self.base_path}/{data_type}/{symbol}.parquet"
        return pd.read_parquet(file_path)
```

#### 1.2 HDF5存储
```python
class HDF5Storage:
    def __init__(self, file_path: str = "./data/market_data.h5"):
        self.file_path = file_path
    
    def save_data(self, data: pd.DataFrame, symbol: str, data_type: str):
        """保存数据到HDF5文件"""
        with pd.HDFStore(self.file_path) as store:
            store.put(f"{data_type}/{symbol}", data, format='table')
    
    def load_data(self, symbol: str, data_type: str) -> pd.DataFrame:
        """从HDF5文件加载数据"""
        with pd.HDFStore(self.file_path) as store:
            return store.get(f"{data_type}/{symbol}")
```

### 2. 数据索引设计

#### 2.1 时间索引
```python
class TimeIndex:
    def __init__(self):
        self.index = {}
    
    def add_data(self, symbol: str, start_date: datetime, end_date: datetime):
        """添加数据时间范围索引"""
        self.index[symbol] = {
            'start_date': start_date,
            'end_date': end_date,
            'last_update': datetime.now()
        }
    
    def get_available_range(self, symbol: str) -> tuple:
        """获取可用数据范围"""
        if symbol in self.index:
            return self.index[symbol]['start_date'], self.index[symbol]['end_date']
        return None, None
```

#### 2.2 数据版本管理
```python
class DataVersionManager:
    def __init__(self):
        self.versions = {}
    
    def create_version(self, symbol: str, version: str, data: pd.DataFrame):
        """创建数据版本"""
        self.versions[f"{symbol}_{version}"] = {
            'data': data,
            'created_at': datetime.now(),
            'size': len(data)
        }
    
    def get_version(self, symbol: str, version: str) -> pd.DataFrame:
        """获取指定版本数据"""
        key = f"{symbol}_{version}"
        if key in self.versions:
            return self.versions[key]['data']
        return None
```

## 数据应用流程

### 1. 策略回测数据流

#### 1.1 回测数据准备
```python
class BacktestDataFlow:
    def prepare_backtest_data(self, symbol: str, start_date: str, end_date: str):
        """准备回测数据"""
        # 1. 加载历史数据
        data = self.data_manager.load_data(symbol, 'ohlcv')
        
        # 2. 计算技术指标
        data = self.technical_indicators.calculate_all_indicators(data)
        
        # 3. 特征工程
        data = self.feature_engineer.extract_features(data)
        
        # 4. 数据验证
        data = self.data_validator.validate_data(data)
        
        return data
```

#### 1.2 实时数据流
```python
class RealtimeDataFlow:
    def __init__(self):
        self.data_buffer = {}
        self.update_frequency = 60  # 秒
    
    def start_realtime_update(self, symbols: list):
        """启动实时数据更新"""
        while True:
            for symbol in symbols:
                # 获取最新数据
                new_data = self.data_provider.get_realtime_data(symbol)
                
                # 更新缓存
                self.update_cache(symbol, new_data)
                
                # 触发策略更新
                self.strategy_manager.on_data_update(symbol, new_data)
            
            time.sleep(self.update_frequency)
```

### 2. 模型训练数据流

#### 2.1 训练数据准备
```python
class TrainingDataFlow:
    def prepare_training_data(self, symbols: list, start_date: str, end_date: str):
        """准备训练数据"""
        # 1. 加载多资产数据
        multi_asset_data = {}
        for symbol in symbols:
            data = self.data_manager.load_data(symbol, 'ohlcv')
            multi_asset_data[symbol] = data
        
        # 2. 数据对齐
        aligned_data = self.align_data(multi_asset_data)
        
        # 3. 特征工程
        features = self.feature_engineer.extract_multi_asset_features(aligned_data)
        
        # 4. 标签生成
        labels = self.generate_labels(aligned_data)
        
        return features, labels
```

#### 2.2 数据分割策略
```python
class DataSplitter:
    def split_data(self, data: pd.DataFrame, split_ratio: dict):
        """数据分割"""
        total_len = len(data)
        train_len = int(total_len * split_ratio['train'])
        val_len = int(total_len * split_ratio['val'])
        
        train_data = data[:train_len]
        val_data = data[train_len:train_len + val_len]
        test_data = data[train_len + val_len:]
        
        return train_data, val_data, test_data
```

## 数据质量保证

### 1. 数据验证

#### 1.1 完整性检查
```python
class DataIntegrityChecker:
    def check_data_completeness(self, data: pd.DataFrame) -> bool:
        """检查数据完整性"""
        # 检查必要列是否存在
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in data.columns for col in required_columns):
            return False
        
        # 检查数据是否为空
        if data.empty:
            return False
        
        # 检查是否有缺失值
        if data.isnull().any().any():
            return False
        
        return True
```

#### 1.2 一致性检查
```python
class DataConsistencyChecker:
    def check_price_consistency(self, data: pd.DataFrame) -> bool:
        """检查价格一致性"""
        # High >= Low
        if not (data['High'] >= data['Low']).all():
            return False
        
        # High >= Open, Close
        if not (data['High'] >= data['Open']).all():
            return False
        if not (data['High'] >= data['Close']).all():
            return False
        
        # Low <= Open, Close
        if not (data['Low'] <= data['Open']).all():
            return False
        if not (data['Low'] <= data['Close']).all():
            return False
        
        return True
```

### 2. 数据监控

#### 2.1 数据质量监控
```python
class DataQualityMonitor:
    def __init__(self):
        self.quality_metrics = {}
    
    def monitor_data_quality(self, data: pd.DataFrame, symbol: str):
        """监控数据质量"""
        metrics = {
            'completeness': self.check_completeness(data),
            'consistency': self.check_consistency(data),
            'accuracy': self.check_accuracy(data),
            'timeliness': self.check_timeliness(data)
        }
        
        self.quality_metrics[symbol] = metrics
        return metrics
```

#### 2.2 异常检测
```python
class DataAnomalyDetector:
    def detect_anomalies(self, data: pd.DataFrame) -> list:
        """检测数据异常"""
        anomalies = []
        
        # 价格异常检测
        price_anomalies = self.detect_price_anomalies(data)
        anomalies.extend(price_anomalies)
        
        # 成交量异常检测
        volume_anomalies = self.detect_volume_anomalies(data)
        anomalies.extend(volume_anomalies)
        
        # 时间序列异常检测
        ts_anomalies = self.detect_timeseries_anomalies(data)
        anomalies.extend(ts_anomalies)
        
        return anomalies
```

## 性能优化

### 1. 数据访问优化

#### 1.1 缓存策略
```python
class DataCache:
    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.max_size = max_size
        self.access_count = {}
    
    def get_data(self, key: str) -> pd.DataFrame:
        """获取缓存数据"""
        if key in self.cache:
            self.access_count[key] += 1
            return self.cache[key]
        return None
    
    def put_data(self, key: str, data: pd.DataFrame):
        """放入缓存数据"""
        if len(self.cache) >= self.max_size:
            self.evict_least_used()
        
        self.cache[key] = data
        self.access_count[key] = 1
```

#### 1.2 并行处理
```python
class ParallelDataProcessor:
    def __init__(self, n_workers: int = 4):
        self.n_workers = n_workers
    
    def process_multiple_symbols(self, symbols: list, func):
        """并行处理多个标的"""
        from concurrent.futures import ThreadPoolExecutor
        
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            futures = [executor.submit(func, symbol) for symbol in symbols]
            results = [future.result() for future in futures]
        
        return results
```

### 2. 存储优化

#### 2.1 数据压缩
```python
class DataCompressor:
    def compress_data(self, data: pd.DataFrame) -> bytes:
        """压缩数据"""
        import pickle
        import gzip
        
        serialized = pickle.dumps(data)
        compressed = gzip.compress(serialized)
        return compressed
    
    def decompress_data(self, compressed_data: bytes) -> pd.DataFrame:
        """解压数据"""
        import pickle
        import gzip
        
        decompressed = gzip.decompress(compressed_data)
        data = pickle.loads(decompressed)
        return data
```

#### 2.2 索引优化
```python
class DataIndexer:
    def __init__(self):
        self.indexes = {}
    
    def create_index(self, data: pd.DataFrame, columns: list):
        """创建数据索引"""
        for column in columns:
            if column in data.columns:
                self.indexes[column] = data[column].values
    
    def query_by_index(self, column: str, value):
        """通过索引查询"""
        if column in self.indexes:
            return np.where(self.indexes[column] == value)[0]
        return None
```
