# TrckTrade API 接口参考

## 概述

本文档详细描述了TrckTrade平台的所有API接口，包括数据获取、策略管理、回测引擎、机器学习等模块的接口规范。

## 目录

- [数据获取接口](#数据获取接口)
- [策略管理接口](#策略管理接口)
- [回测引擎接口](#回测引擎接口)
- [机器学习接口](#机器学习接口)
- [可视化接口](#可视化接口)
- [配置管理接口](#配置管理接口)

## 数据获取接口

### 1. DataManager 类

#### 1.1 基本方法

```python
class DataManager:
    """数据管理器"""
    
    def __init__(self):
        """初始化数据管理器"""
        pass
    
    def get_data(self, symbol: str, start_date: str, end_date: str, 
                provider: str = None, data_type: str = 'auto', **kwargs) -> Optional[pd.DataFrame]:
        """
        获取历史数据
        
        Args:
            symbol: 标的代码
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            provider: 数据提供者 ('akshare', 'yfinance', 'mock')
            data_type: 数据类型 ('stock', 'fund', 'index', 'auto')
            **kwargs: 其他参数
            
        Returns:
            包含OHLCV数据的DataFrame
            
        Raises:
            ValueError: 不支持的数据提供者或数据类型
            Exception: 数据获取失败
        """
        pass
    
    def get_realtime_data(self, symbol: str, provider: str = None) -> Optional[pd.DataFrame]:
        """
        获取实时数据
        
        Args:
            symbol: 标的代码
            provider: 数据提供者
            
        Returns:
            实时数据DataFrame
        """
        pass
    
    def add_provider(self, name: str, provider: DataProvider):
        """
        添加数据提供者
        
        Args:
            name: 提供者名称
            provider: 数据提供者实例
        """
        pass
    
    def set_default_provider(self, provider: str):
        """
        设置默认数据提供者
        
        Args:
            provider: 提供者名称
        """
        pass
```

#### 1.2 便捷函数

```python
def get_data(symbol: str, start_date: str, end_date: str, 
            provider: str = None, data_type: str = 'auto', **kwargs) -> Optional[pd.DataFrame]:
    """
    获取数据的便捷函数
    
    Args:
        symbol: 标的代码
        start_date: 开始日期
        end_date: 结束日期
        provider: 数据提供者
        data_type: 数据类型
        **kwargs: 其他参数
        
    Returns:
        包含OHLCV数据的DataFrame
    """
    pass
```

### 2. DataProvider 基类

```python
class DataProvider:
    """数据提供者基类"""
    
    def __init__(self):
        """初始化数据提供者"""
        pass
    
    def get_data(self, symbol: str, start_date: str, end_date: str, **kwargs) -> Optional[pd.DataFrame]:
        """
        获取数据
        
        Args:
            symbol: 标的代码
            start_date: 开始日期
            end_date: 结束日期
            **kwargs: 其他参数
            
        Returns:
            包含OHLCV数据的DataFrame
        """
        raise NotImplementedError
    
    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        数据预处理
        
        Args:
            data: 原始数据
            
        Returns:
            预处理后的数据
        """
        pass
```

### 3. AKShareDataProvider 类

```python
class AKShareDataProvider(DataProvider):
    """AKShare数据提供者"""
    
    def get_stock_data(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        获取股票数据
        
        Args:
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            包含OHLCV数据的DataFrame
        """
        pass
    
    def get_fund_data(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        获取基金数据
        
        Args:
            symbol: 基金代码
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            包含OHLCV数据的DataFrame
        """
        pass
    
    def get_index_data(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        获取指数数据
        
        Args:
            symbol: 指数代码
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            包含OHLCV数据的DataFrame
        """
        pass
```

## 策略管理接口

### 1. BaseStrategy 基类

```python
class BaseStrategy:
    """基础策略类"""
    
    def __init__(self):
        """初始化策略"""
        pass
    
    def next(self):
        """策略逻辑"""
        pass
    
    def notify_order(self, order):
        """订单状态通知"""
        pass
    
    def notify_trade(self, trade):
        """交易通知"""
        pass
    
    def log(self, txt, dt=None):
        """日志函数"""
        pass
    
    def stop(self):
        """策略结束时的处理"""
        pass
```

### 2. 传统策略类

#### 2.1 MovingAverageStrategy

```python
class MovingAverageStrategy(BaseStrategy):
    """移动平均策略"""
    
    params = (
        ('ma_period', 20),  # 移动平均周期
        ('min_trade_amount', 1000),  # 最小交易金额
    )
    
    def __init__(self):
        """初始化策略"""
        pass
    
    def next(self):
        """策略逻辑"""
        pass
```

#### 2.2 RSIStrategy

```python
class RSIStrategy(BaseStrategy):
    """RSI策略"""
    
    params = (
        ('rsi_period', 14),  # RSI周期
        ('rsi_oversold', 30),  # 超卖阈值
        ('rsi_overbought', 70),  # 超买阈值
        ('min_trade_amount', 1000),  # 最小交易金额
    )
    
    def __init__(self):
        """初始化策略"""
        pass
    
    def next(self):
        """策略逻辑"""
        pass
```

### 3. 策略字典

```python
STRATEGIES = {
    'buy_and_hold': BuyAndHoldStrategy,
    'moving_average': MovingAverageStrategy,
    'dual_moving_average': DualMovingAverageStrategy,
    'rsi': RSIStrategy,
    'macd': MACDStrategy,
    'bollinger_bands': BollingerBandsStrategy,
    'mean_reversion': MeanReversionStrategy,
    'momentum': MomentumStrategy,
}
```

## 回测引擎接口

### 1. BaseBacktestEngine 类

```python
class BaseBacktestEngine:
    """基础回测引擎"""
    
    def __init__(self, initial_cash: float = 100000.0, commission: float = 0.001):
        """
        初始化回测引擎
        
        Args:
            initial_cash: 初始资金
            commission: 手续费率
        """
        pass
    
    def add_strategy(self, strategy_class, **kwargs):
        """
        添加策略
        
        Args:
            strategy_class: 策略类
            **kwargs: 策略参数
        """
        pass
    
    def add_data(self, data: pd.DataFrame, name: str = "data"):
        """
        添加数据
        
        Args:
            data: 包含OHLCV数据的DataFrame
            name: 数据名称
            
        Raises:
            ValueError: 数据缺少必需的列
        """
        pass
    
    def run(self, show_log: bool = False):
        """
        运行回测
        
        Args:
            show_log: 是否显示详细日志
            
        Returns:
            回测结果
        """
        pass
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        获取性能指标
        
        Returns:
            包含各种性能指标的字典
            
        Raises:
            ValueError: 请先运行回测
        """
        pass
    
    def print_performance_summary(self):
        """打印性能摘要"""
        pass
    
    def plot(self, style: str = 'candlestick', show: bool = True):
        """
        绘制回测结果图表
        
        Args:
            style: 图表样式
            show: 是否显示图表
        """
        pass
```

### 2. 便捷回测函数

```python
def run_backtest(data: pd.DataFrame, strategy_class, strategy_params: Dict = None, 
                initial_cash: float = 100000.0, commission: float = 0.001,
                show_log: bool = False, show_plot: bool = True) -> BaseBacktestEngine:
    """
    运行回测的便捷函数
    
    Args:
        data: 包含OHLCV数据的DataFrame
        strategy_class: 策略类
        strategy_params: 策略参数
        initial_cash: 初始资金
        commission: 手续费率
        show_log: 是否显示详细日志
        show_plot: 是否显示图表
        
    Returns:
        回测引擎实例
    """
    pass
```

## 机器学习接口

### 1. BaseMLStrategy 基类

```python
class BaseMLStrategy(ABC):
    """机器学习策略基类"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化策略
        
        Args:
            config: 策略配置
        """
        pass
    
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
        """
        生成交易信号
        
        Args:
            data: 输入数据
            
        Returns:
            交易信号序列
            
        Raises:
            ValueError: 模型尚未训练
        """
        pass
    
    def save_model(self, path: str) -> None:
        """
        保存模型
        
        Args:
            path: 保存路径
            
        Raises:
            ValueError: 没有可保存的模型
        """
        pass
    
    def load_model(self, path: str) -> None:
        """
        加载模型
        
        Args:
            path: 模型路径
        """
        pass
```

### 2. 传统机器学习策略

#### 2.1 PricePredictionStrategy

```python
class PricePredictionStrategy(BaseMLStrategy):
    """价格预测策略"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化策略
        
        Args:
            config: 策略配置
                - model_type: 模型类型 ('random_forest', 'gradient_boosting', 'linear_regression', 'ridge', 'lasso', 'svr')
                - prediction_horizon: 预测时间跨度
                - signal_threshold: 信号阈值
        """
        pass
    
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """准备特征数据"""
        pass
    
    def prepare_labels(self, data: pd.DataFrame) -> pd.Series:
        """准备标签数据"""
        pass
    
    def train_model(self, X: pd.DataFrame, y: pd.Series) -> None:
        """训练模型"""
        pass
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """模型预测"""
        pass
```

#### 2.2 DirectionPredictionStrategy

```python
class DirectionPredictionStrategy(BaseMLStrategy):
    """涨跌分类策略"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化策略
        
        Args:
            config: 策略配置
                - model_type: 模型类型 ('random_forest', 'gradient_boosting', 'logistic_regression', 'svc')
                - prediction_horizon: 预测时间跨度
        """
        pass
    
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """准备特征数据"""
        pass
    
    def prepare_labels(self, data: pd.DataFrame) -> pd.Series:
        """准备标签数据"""
        pass
    
    def train_model(self, X: pd.DataFrame, y: pd.Series) -> None:
        """训练模型"""
        pass
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """模型预测"""
        pass
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """预测概率"""
        pass
```

### 3. 深度学习策略

#### 3.1 LSTMStrategy

```python
class LSTMStrategy(BaseMLStrategy):
    """LSTM价格预测策略"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化策略
        
        Args:
            config: 策略配置
                - sequence_length: 序列长度
                - hidden_size: 隐藏层大小
                - num_layers: LSTM层数
                - learning_rate: 学习率
                - batch_size: 批次大小
                - epochs: 训练轮数
        """
        pass
    
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """准备特征数据"""
        pass
    
    def prepare_labels(self, data: pd.DataFrame) -> pd.Series:
        """准备标签数据"""
        pass
    
    def train_model(self, X: pd.DataFrame, y: pd.Series) -> None:
        """训练模型"""
        pass
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """模型预测"""
        pass
```

### 4. 强化学习策略

#### 4.1 DQNStrategy

```python
class DQNStrategy(BaseMLStrategy):
    """DQN强化学习策略"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化策略
        
        Args:
            config: 策略配置
                - initial_balance: 初始资金
                - learning_rate: 学习率
                - buffer_size: 经验回放缓冲区大小
                - batch_size: 批次大小
                - gamma: 折扣因子
                - epsilon: 探索率
        """
        pass
    
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """准备特征数据"""
        pass
    
    def prepare_labels(self, data: pd.DataFrame) -> pd.Series:
        """强化学习不需要标签"""
        pass
    
    def train_model(self, X: pd.DataFrame, y: pd.Series) -> None:
        """训练DQN模型"""
        pass
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """DQN预测"""
        pass
```

### 5. 策略管理器

```python
class MLStrategyManager:
    """机器学习策略管理器"""
    
    def __init__(self):
        """初始化策略管理器"""
        pass
    
    def register_strategy(self, name: str, strategy: BaseMLStrategy):
        """
        注册策略
        
        Args:
            name: 策略名称
            strategy: 策略实例
        """
        pass
    
    def train_strategy(self, name: str, data: pd.DataFrame):
        """
        训练策略
        
        Args:
            name: 策略名称
            data: 训练数据
            
        Raises:
            ValueError: 策略未注册
        """
        pass
    
    def evaluate_strategy(self, name: str, data: pd.DataFrame) -> Dict[str, float]:
        """
        评估策略
        
        Args:
            name: 策略名称
            data: 评估数据
            
        Returns:
            策略性能指标
            
        Raises:
            ValueError: 策略未注册或尚未训练
        """
        pass
    
    def get_strategy_signals(self, name: str, data: pd.DataFrame) -> pd.Series:
        """
        获取策略信号
        
        Args:
            name: 策略名称
            data: 输入数据
            
        Returns:
            交易信号序列
            
        Raises:
            ValueError: 策略未注册或尚未训练
        """
        pass
    
    def save_strategy(self, name: str, path: str):
        """
        保存策略
        
        Args:
            name: 策略名称
            path: 保存路径
            
        Raises:
            ValueError: 策略未注册
        """
        pass
    
    def load_strategy(self, name: str, path: str):
        """
        加载策略
        
        Args:
            name: 策略名称
            path: 模型路径
        """
        pass
```

## 可视化接口

### 1. 基础可视化函数

```python
def visualize_fund_data(fund_data: pd.DataFrame, fund_code: str, fund_name: str = ""):
    """
    可视化基金数据
    
    Args:
        fund_data: 基金数据
        fund_code: 基金代码
        fund_name: 基金名称
    """
    pass

def compare_multiple_funds(fund_codes: list, fund_names: list, 
                          period: str = "daily", start_date: str = "20220101", 
                          end_date: str = "20230101"):
    """
    对比多个基金的价格走势
    
    Args:
        fund_codes: 基金代码列表
        fund_names: 基金名称列表
        period: 数据周期
        start_date: 开始日期
        end_date: 结束日期
    """
    pass
```

### 2. 回测结果可视化

```python
def plot_backtest_results(engine: BaseBacktestEngine, style: str = 'candlestick', show: bool = True):
    """
    绘制回测结果
    
    Args:
        engine: 回测引擎实例
        style: 图表样式
        show: 是否显示图表
    """
    pass

def plot_performance_metrics(metrics: Dict[str, Any]):
    """
    绘制性能指标
    
    Args:
        metrics: 性能指标字典
    """
    pass
```

### 3. 机器学习结果可视化

```python
def plot_model_predictions(y_true: np.ndarray, y_pred: np.ndarray, title: str = "模型预测结果"):
    """
    绘制模型预测结果
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        title: 图表标题
    """
    pass

def plot_feature_importance(feature_names: list, importance_scores: np.ndarray, 
                           top_n: int = 20):
    """
    绘制特征重要性
    
    Args:
        feature_names: 特征名称列表
        importance_scores: 重要性分数
        top_n: 显示前N个特征
    """
    pass

def plot_training_history(history: Dict[str, list]):
    """
    绘制训练历史
    
    Args:
        history: 训练历史字典
    """
    pass
```

## 配置管理接口

### 1. 配置文件格式

#### 1.1 系统配置 (config/system.yaml)

```yaml
system:
  data_path: "./data"
  model_path: "./models"
  log_level: "INFO"
  parallel_workers: 4

data:
  providers:
    - akshare
    - yfinance
  cache_size: 1000
  update_frequency: "daily"

backtest:
  initial_cash: 100000
  commission: 0.001
  slippage: 0.0005
```

#### 1.2 策略配置 (config/strategies.yaml)

```yaml
moving_average:
  ma_period: 20
  min_trade_amount: 1000

rsi:
  rsi_period: 14
  rsi_oversold: 30
  rsi_overbought: 70
  min_trade_amount: 1000

price_prediction:
  model_type: "random_forest"
  prediction_horizon: 1
  signal_threshold: 0.02

lstm:
  sequence_length: 60
  hidden_size: 64
  num_layers: 2
  learning_rate: 0.001
  batch_size: 32
  epochs: 100
```

### 2. 配置管理类

```python
class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_path: str = "config"):
        """
        初始化配置管理器
        
        Args:
            config_path: 配置文件路径
        """
        pass
    
    def load_config(self, config_name: str) -> Dict[str, Any]:
        """
        加载配置文件
        
        Args:
            config_name: 配置名称
            
        Returns:
            配置字典
        """
        pass
    
    def save_config(self, config_name: str, config: Dict[str, Any]):
        """
        保存配置文件
        
        Args:
            config_name: 配置名称
            config: 配置字典
        """
        pass
    
    def get_system_config(self) -> Dict[str, Any]:
        """获取系统配置"""
        pass
    
    def get_strategy_config(self, strategy_name: str) -> Dict[str, Any]:
        """获取策略配置"""
        pass
    
    def update_config(self, config_name: str, key: str, value: Any):
        """
        更新配置项
        
        Args:
            config_name: 配置名称
            key: 配置键
            value: 配置值
        """
        pass
```

## 错误处理

### 1. 自定义异常类

```python
class TrickTradeError(Exception):
    """TrckTrade基础异常类"""
    pass

class DataProviderError(TrickTradeError):
    """数据提供者异常"""
    pass

class StrategyError(TrickTradeError):
    """策略异常"""
    pass

class BacktestError(TrickTradeError):
    """回测异常"""
    pass

class ModelError(TrickTradeError):
    """模型异常"""
    pass
```

### 2. 错误处理装饰器

```python
def handle_errors(func):
    """错误处理装饰器"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"错误: {e}")
            return None
    return wrapper
```

## 使用示例

### 1. 基础使用示例

```python
# 获取数据
from TrickTrade import get_data
data = get_data("000001", "2023-01-01", "2023-12-31", data_type="stock")

# 运行回测
from TrickTrade import run_backtest
from TrickTrade.strategies import MovingAverageStrategy

engine = run_backtest(
    data=data,
    strategy_class=MovingAverageStrategy,
    strategy_params={'ma_period': 20},
    initial_cash=100000,
    show_log=True,
    show_plot=True
)

# 获取性能指标
metrics = engine.get_performance_metrics()
print(f"总收益率: {metrics['total_return']:.2f}%")
print(f"夏普比率: {metrics['sharpe_ratio']:.2f}")
```

### 2. 机器学习策略示例

```python
# 创建机器学习策略
from TrickTrade.ml_strategies import PricePredictionStrategy

strategy = PricePredictionStrategy({
    'model_type': 'random_forest',
    'prediction_horizon': 1,
    'signal_threshold': 0.02
})

# 训练策略
strategy.train_model(features, labels)

# 生成信号
signals = strategy.generate_signals(data)

# 保存模型
strategy.save_model("models/price_prediction_model.pkl")
```

### 3. 策略管理器示例

```python
# 创建策略管理器
from TrickTrade.ml_strategies import MLStrategyManager

manager = MLStrategyManager()

# 注册策略
manager.register_strategy("price_pred", strategy)

# 训练策略
manager.train_strategy("price_pred", data)

# 评估策略
performance = manager.evaluate_strategy("price_pred", test_data)

# 获取信号
signals = manager.get_strategy_signals("price_pred", data)
```

## 总结

TrckTrade提供了完整的API接口，涵盖了数据获取、策略管理、回测引擎、机器学习等核心功能。所有接口都遵循统一的规范，支持类型提示和详细的文档说明。通过模块化的设计，用户可以灵活地组合使用各个功能模块，构建自己的量化交易系统。
