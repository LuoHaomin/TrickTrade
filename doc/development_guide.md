# TrckTrade 开发指南

## 概述

本文档为TrckTrade量化投资平台的开发指南，包含环境搭建、项目结构、开发规范、测试指南等内容，帮助开发者快速上手并参与项目开发。

## 目录

- [环境搭建](#环境搭建)
- [项目结构](#项目结构)
- [开发规范](#开发规范)
- [模块开发指南](#模块开发指南)
- [测试指南](#测试指南)
- [部署指南](#部署指南)
- [贡献指南](#贡献指南)

## 环境搭建

### 1. 系统要求

- **操作系统**: Windows 10/11, macOS 10.14+, Ubuntu 18.04+
- **Python版本**: 3.12+
- **内存**: 8GB+ (推荐16GB+)
- **存储**: 10GB+ 可用空间

### 2. Python环境安装

#### 2.1 使用Conda (推荐)

```bash
# 创建虚拟环境
conda create -n tricktrade python=3.12

# 激活环境
conda activate tricktrade

# 安装基础依赖
conda install pandas numpy matplotlib seaborn scikit-learn

# 安装PyTorch (CPU版本)
conda install pytorch torchvision torchaudio cpuonly -c pytorch

# 安装其他依赖
pip install backtrader akshare yfinance ta-lib optuna stable-baselines3
```

#### 2.2 使用pip

```bash
# 创建虚拟环境
python -m venv tricktrade_env

# 激活环境
# Windows
tricktrade_env\Scripts\activate
# macOS/Linux
source tricktrade_env/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 3. 依赖文件

#### 3.1 requirements.txt

```txt
# 核心依赖
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0

# 回测框架
backtrader>=1.9.78

# 数据获取
akshare>=1.12.0
yfinance>=0.2.0

# 机器学习
scikit-learn>=1.3.0
xgboost>=1.7.0
lightgbm>=4.0.0
torch>=2.0.0
torchvision>=0.15.0
torchaudio>=2.0.0

# 强化学习
stable-baselines3>=2.0.0
gym>=0.26.0

# 技术指标
ta-lib>=0.4.0

# 特征工程
tsfresh>=0.20.0

# 超参数优化
optuna>=3.0.0

# 数据处理
h5py>=3.9.0
pyarrow>=12.0.0

# 配置管理
pyyaml>=6.0

# 可视化
plotly>=5.15.0

# 开发工具
pytest>=7.4.0
black>=23.0.0
flake8>=6.0.0
mypy>=1.5.0
```

#### 3.2 environment.yml (Conda)

```yaml
name: tricktrade
channels:
  - conda-forge
  - pytorch
dependencies:
  - python=3.12
  - pandas>=2.0.0
  - numpy>=1.24.0
  - matplotlib>=3.7.0
  - seaborn>=0.12.0
  - scikit-learn>=1.3.0
  - pytorch>=2.0.0
  - torchvision>=0.15.0
  - torchaudio>=2.0.0
  - pip
  - pip:
    - backtrader>=1.9.78
    - akshare>=1.12.0
    - yfinance>=0.2.0
    - xgboost>=1.7.0
    - lightgbm>=4.0.0
    - stable-baselines3>=2.0.0
    - gym>=0.26.0
    - ta-lib>=0.4.0
    - tsfresh>=0.20.0
    - optuna>=3.0.0
    - h5py>=3.9.0
    - pyarrow>=12.0.0
    - pyyaml>=6.0
    - plotly>=5.15.0
    - pytest>=7.4.0
    - black>=23.0.0
    - flake8>=6.0.0
    - mypy>=1.5.0
```

### 4. 开发工具配置

#### 4.1 VS Code配置

创建 `.vscode/settings.json`:

```json
{
    "python.defaultInterpreterPath": "./tricktrade_env/bin/python",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": false,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": [
        "tests"
    ],
    "files.exclude": {
        "**/__pycache__": true,
        "**/*.pyc": true
    }
}
```

#### 4.2 PyCharm配置

1. 设置Python解释器为项目虚拟环境
2. 配置代码格式化工具为Black
3. 启用Pytest测试框架
4. 配置代码检查工具

## 项目结构

### 1. 目录结构

```
TrickTrade/
├── BackTest/                 # 回测模块
│   ├── __init__.py
│   ├── backtest_framework.py
│   ├── baseline_models.py
│   ├── data_provider.py
│   ├── example.py
│   └── test.py
├── CorrelationAnalysis/      # 相关性分析
│   └── test.py
├── DataSource/              # 数据源
│   └── AKShareAPI.py
├── RLStrategy/              # 强化学习策略
│   ├── DQN/
│   └── PPO/
├── Startegy/                # 策略模块
├── Visualize/               # 可视化模块
├── doc/                     # 文档
│   ├── architecture.md
│   ├── data_flow.md
│   ├── ml_strategy.md
│   ├── api_reference.md
│   └── development_guide.md
├── config/                  # 配置文件
│   ├── system.yaml
│   └── strategies.yaml
├── data/                    # 数据目录
├── models/                  # 模型目录
├── logs/                    # 日志目录
├── tests/                   # 测试目录
├── requirements.txt          # 依赖文件
├── environment.yml          # Conda环境文件
├── .gitignore              # Git忽略文件
├── .flake8                 # 代码检查配置
├── pyproject.toml          # 项目配置
└── README.md               # 项目说明
```

### 2. 模块说明

#### 2.1 BackTest模块
- `backtest_framework.py`: 回测框架核心
- `baseline_models.py`: 基础策略模型
- `data_provider.py`: 数据提供者

#### 2.2 DataSource模块
- `AKShareAPI.py`: AKShare数据接口

#### 2.3 RLStrategy模块
- `DQN/`: DQN强化学习策略
- `PPO/`: PPO强化学习策略

#### 2.4 其他模块
- `Startegy/`: 传统策略实现
- `Visualize/`: 可视化工具
- `CorrelationAnalysis/`: 相关性分析工具

## 开发规范

### 1. 代码风格

#### 1.1 Python代码规范

遵循PEP 8规范，使用Black进行代码格式化：

```python
# 好的代码风格示例
class DataProvider:
    """数据提供者基类"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.cache = {}
    
    def get_data(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        获取历史数据
        
        Args:
            symbol: 标的代码
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            包含OHLCV数据的DataFrame
        """
        pass
```

#### 1.2 命名规范

- **类名**: 使用PascalCase，如 `DataProvider`, `BaseStrategy`
- **函数名**: 使用snake_case，如 `get_data`, `train_model`
- **变量名**: 使用snake_case，如 `data_provider`, `strategy_config`
- **常量**: 使用UPPER_CASE，如 `MAX_RETRY_COUNT`, `DEFAULT_CONFIG`

#### 1.3 文档字符串

使用Google风格的文档字符串：

```python
def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    计算RSI指标
    
    Args:
        prices: 价格序列
        period: 计算周期，默认14
        
    Returns:
        RSI指标序列
        
    Raises:
        ValueError: 当period小于1时抛出
    """
    if period < 1:
        raise ValueError("period must be greater than 0")
    
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi
```

### 2. 类型提示

使用Python类型提示提高代码可读性：

```python
from typing import Dict, List, Optional, Union, Tuple, Any
import pandas as pd
import numpy as np

def process_data(
    data: pd.DataFrame,
    config: Dict[str, Any],
    columns: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    处理数据
    
    Args:
        data: 输入数据
        config: 配置参数
        columns: 要处理的列
        
    Returns:
        处理后的数据和统计信息
    """
    pass
```

### 3. 异常处理

使用自定义异常类和处理机制：

```python
class TrickTradeError(Exception):
    """TrckTrade基础异常类"""
    pass

class DataProviderError(TrickTradeError):
    """数据提供者异常"""
    pass

def get_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """获取数据"""
    try:
        # 数据获取逻辑
        pass
    except Exception as e:
        raise DataProviderError(f"Failed to get data for {symbol}: {e}")
```

### 4. 日志记录

使用Python logging模块：

```python
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/tricktrade.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def train_model(data: pd.DataFrame):
    """训练模型"""
    logger.info("Starting model training")
    try:
        # 训练逻辑
        logger.info("Model training completed successfully")
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        raise
```

## 模块开发指南

### 1. 数据模块开发

#### 1.1 创建新的数据提供者

```python
from abc import ABC, abstractmethod
import pandas as pd
from typing import Optional, Dict, Any

class CustomDataProvider(ABC):
    """自定义数据提供者"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.cache = {}
    
    @abstractmethod
    def get_data(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """获取数据"""
        pass
    
    def _validate_data(self, data: pd.DataFrame) -> bool:
        """验证数据格式"""
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        return all(col in data.columns for col in required_columns)
    
    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """数据预处理"""
        # 确保索引是日期类型
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)
        
        # 按日期排序
        data = data.sort_index()
        
        # 移除缺失值
        data = data.dropna()
        
        return data
```

#### 1.2 数据缓存机制

```python
import pickle
import os
from datetime import datetime, timedelta

class DataCache:
    """数据缓存类"""
    
    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def get_cache_key(self, symbol: str, start_date: str, end_date: str) -> str:
        """生成缓存键"""
        return f"{symbol}_{start_date}_{end_date}.pkl"
    
    def get(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """获取缓存数据"""
        cache_key = self.get_cache_key(symbol, start_date, end_date)
        cache_path = os.path.join(self.cache_dir, cache_key)
        
        if os.path.exists(cache_path):
            # 检查缓存是否过期（24小时）
            cache_time = datetime.fromtimestamp(os.path.getmtime(cache_path))
            if datetime.now() - cache_time < timedelta(hours=24):
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
        
        return None
    
    def set(self, symbol: str, start_date: str, end_date: str, data: pd.DataFrame):
        """设置缓存数据"""
        cache_key = self.get_cache_key(symbol, start_date, end_date)
        cache_path = os.path.join(self.cache_dir, cache_key)
        
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)
```

### 2. 策略模块开发

#### 2.1 创建传统策略

```python
import backtrader as bt
from typing import Dict, Any

class CustomStrategy(bt.Strategy):
    """自定义策略"""
    
    params = (
        ('param1', 20),
        ('param2', 0.02),
    )
    
    def __init__(self):
        """初始化策略"""
        self.dataclose = self.datas[0].close
        self.order = None
        
        # 添加技术指标
        self.sma = bt.indicators.SimpleMovingAverage(
            self.datas[0], period=self.params.param1
        )
    
    def next(self):
        """策略逻辑"""
        if self.order:
            return
        
        if not self.position:
            # 买入逻辑
            if self.dataclose[0] > self.sma[0]:
                self.order = self.buy()
        else:
            # 卖出逻辑
            if self.dataclose[0] < self.sma[0]:
                self.order = self.sell()
    
    def notify_order(self, order):
        """订单通知"""
        if order.status in [order.Submitted, order.Accepted]:
            return
        
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'买入: 价格 {order.executed.price:.2f}')
            else:
                self.log(f'卖出: 价格 {order.executed.price:.2f}')
        
        self.order = None
    
    def log(self, txt, dt=None):
        """日志函数"""
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()}: {txt}')
```

#### 2.2 创建机器学习策略

```python
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import joblib

class BaseMLStrategy(ABC):
    """机器学习策略基类"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.feature_columns = []
        self.is_trained = False
    
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
        signals = pd.Series(0, index=data.index)
        threshold = self.config.get('signal_threshold', 0.02)
        
        signals[predictions > threshold] = 1   # 买入
        signals[predictions < -threshold] = -1  # 卖出
        
        return signals
    
    def save_model(self, path: str) -> None:
        """保存模型"""
        if self.model is None:
            raise ValueError("没有可保存的模型")
        
        model_data = {
            'model': self.model,
            'feature_columns': self.feature_columns,
            'config': self.config
        }
        
        joblib.dump(model_data, path)
    
    def load_model(self, path: str) -> None:
        """加载模型"""
        model_data = joblib.load(path)
        self.model = model_data['model']
        self.feature_columns = model_data['feature_columns']
        self.config = model_data['config']
        self.is_trained = True
```

### 3. 回测模块开发

#### 3.1 扩展回测引擎

```python
import backtrader as bt
import pandas as pd
from typing import Dict, Any, List

class AdvancedBacktestEngine:
    """高级回测引擎"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.cerebro = bt.Cerebro()
        self.results = None
        
        # 设置回测参数
        self.cerebro.broker.setcash(config.get('initial_cash', 100000))
        self.cerebro.broker.setcommission(commission=config.get('commission', 0.001))
        
        # 添加分析器
        self._add_analyzers()
    
    def _add_analyzers(self):
        """添加分析器"""
        self.cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        self.cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        self.cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        self.cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        self.cerebro.addanalyzer(bt.analyzers.SQN, _name='sqn')
    
    def add_strategy(self, strategy_class, **kwargs):
        """添加策略"""
        self.cerebro.addstrategy(strategy_class, **kwargs)
    
    def add_data(self, data: pd.DataFrame, name: str = "data"):
        """添加数据"""
        data_feed = bt.feeds.PandasData(
            dataname=data,
            datetime=None,
            open='Open',
            high='High',
            low='Low',
            close='Close',
            volume='Volume',
            openinterest=-1
        )
        
        self.cerebro.adddata(data_feed, name=name)
    
    def run(self, show_log: bool = False):
        """运行回测"""
        if show_log:
            print(f'初始资金: {self.cerebro.broker.getvalue():.2f}')
        
        self.results = self.cerebro.run()
        
        if show_log:
            print(f'最终资金: {self.cerebro.broker.getvalue():.2f}')
        
        return self.results
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        if self.results is None:
            raise ValueError("请先运行回测")
        
        strat = self.results[0]
        
        # 获取分析结果
        sharpe_analysis = strat.analyzers.sharpe.get_analysis()
        drawdown_analysis = strat.analyzers.drawdown.get_analysis()
        returns_analysis = strat.analyzers.returns.get_analysis()
        trades_analysis = strat.analyzers.trades.get_analysis()
        sqn_analysis = strat.analyzers.sqn.get_analysis()
        
        # 计算基本指标
        final_value = self.cerebro.broker.getvalue()
        initial_cash = self.config.get('initial_cash', 100000)
        total_return = (final_value - initial_cash) / initial_cash * 100
        
        metrics = {
            'initial_cash': initial_cash,
            'final_value': final_value,
            'total_return': total_return,
            'sharpe_ratio': sharpe_analysis.get("sharperatio", 0),
            'max_drawdown': drawdown_analysis.get("max", {}).get("drawdown", 0),
            'total_trades': trades_analysis.get("total", {}).get("total", 0),
            'won_trades': trades_analysis.get("won", {}).get("total", 0),
            'lost_trades': trades_analysis.get("lost", {}).get("total", 0),
            'sqn': sqn_analysis.get("sqn", 0)
        }
        
        # 计算胜率
        if metrics['total_trades'] > 0:
            metrics['win_rate'] = metrics['won_trades'] / metrics['total_trades'] * 100
        else:
            metrics['win_rate'] = 0
        
        return metrics
```

### 4. 可视化模块开发

#### 4.1 创建自定义图表

```python
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, Any, List, Optional

class ChartGenerator:
    """图表生成器"""
    
    def __init__(self, style: str = 'seaborn-v0_8'):
        plt.style.use(style)
        self.set_chinese_font()
    
    def set_chinese_font(self):
        """设置中文字体"""
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False
    
    def plot_price_chart(self, data: pd.DataFrame, title: str = "价格走势图"):
        """绘制价格图表"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), 
                                      gridspec_kw={'height_ratios': [3, 1]})
        
        # 价格走势
        ax1.plot(data.index, data['Close'], label='收盘价', linewidth=2)
        ax1.set_title(title, fontsize=16, fontweight='bold')
        ax1.set_ylabel('价格 (元)', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 成交量
        ax2.bar(data.index, data['Volume'], alpha=0.7, color='orange')
        ax2.set_ylabel('成交量', fontsize=12)
        ax2.set_xlabel('日期', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_performance_metrics(self, metrics: Dict[str, Any]):
        """绘制性能指标"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # 收益率
        axes[0, 0].bar(['总收益率'], [metrics['total_return']], color='green')
        axes[0, 0].set_title('总收益率')
        axes[0, 0].set_ylabel('收益率 (%)')
        
        # 夏普比率
        axes[0, 1].bar(['夏普比率'], [metrics['sharpe_ratio']], color='blue')
        axes[0, 1].set_title('夏普比率')
        
        # 最大回撤
        axes[1, 0].bar(['最大回撤'], [metrics['max_drawdown']], color='red')
        axes[1, 0].set_title('最大回撤')
        axes[1, 0].set_ylabel('回撤 (%)')
        
        # 胜率
        axes[1, 1].bar(['胜率'], [metrics['win_rate']], color='purple')
        axes[1, 1].set_title('胜率')
        axes[1, 1].set_ylabel('胜率 (%)')
        
        plt.tight_layout()
        return fig
    
    def plot_correlation_matrix(self, data: pd.DataFrame, title: str = "相关性矩阵"):
        """绘制相关性矩阵"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 计算相关性矩阵
        corr_matrix = data.corr()
        
        # 绘制热力图
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, ax=ax)
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        return fig
```

## 测试指南

### 1. 测试框架设置

#### 1.1 pytest配置

创建 `pytest.ini`:

```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v --tb=short
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks tests as integration tests
    unit: marks tests as unit tests
```

#### 1.2 测试目录结构

```
tests/
├── __init__.py
├── conftest.py
├── test_data_provider.py
├── test_strategies.py
├── test_backtest.py
├── test_ml_strategies.py
├── test_visualization.py
└── fixtures/
    ├── sample_data.csv
    └── test_config.yaml
```

### 2. 单元测试

#### 2.1 数据提供者测试

```python
import pytest
import pandas as pd
from unittest.mock import Mock, patch
from TrickTrade.data_provider import AKShareDataProvider

class TestAKShareDataProvider:
    """AKShare数据提供者测试"""
    
    def setup_method(self):
        """测试前准备"""
        self.provider = AKShareDataProvider()
    
    def test_get_stock_data_success(self):
        """测试成功获取股票数据"""
        # 模拟数据
        mock_data = pd.DataFrame({
            '开盘': [100, 101, 102],
            '最高': [105, 106, 107],
            '最低': [95, 96, 97],
            '收盘': [103, 104, 105],
            '成交量': [1000000, 1100000, 1200000]
        })
        
        with patch('akshare.stock_zh_a_hist', return_value=mock_data):
            result = self.provider.get_stock_data("000001", "2023-01-01", "2023-01-03")
            
            assert result is not None
            assert 'Open' in result.columns
            assert 'High' in result.columns
            assert 'Low' in result.columns
            assert 'Close' in result.columns
            assert 'Volume' in result.columns
    
    def test_get_stock_data_empty(self):
        """测试获取空数据"""
        with patch('akshare.stock_zh_a_hist', return_value=pd.DataFrame()):
            result = self.provider.get_stock_data("000001", "2023-01-01", "2023-01-03")
            assert result is None
    
    def test_get_stock_data_exception(self):
        """测试异常情况"""
        with patch('akshare.stock_zh_a_hist', side_effect=Exception("Network error")):
            result = self.provider.get_stock_data("000001", "2023-01-01", "2023-01-03")
            assert result is None
```

#### 2.2 策略测试

```python
import pytest
import pandas as pd
import numpy as np
from TrickTrade.strategies import MovingAverageStrategy

class TestMovingAverageStrategy:
    """移动平均策略测试"""
    
    def setup_method(self):
        """测试前准备"""
        self.strategy = MovingAverageStrategy()
        
        # 创建测试数据
        self.test_data = pd.DataFrame({
            'Open': [100, 101, 102, 103, 104],
            'High': [105, 106, 107, 108, 109],
            'Low': [95, 96, 97, 98, 99],
            'Close': [103, 104, 105, 106, 107],
            'Volume': [1000000, 1100000, 1200000, 1300000, 1400000]
        })
    
    def test_strategy_initialization(self):
        """测试策略初始化"""
        assert self.strategy.params.ma_period == 20
        assert self.strategy.params.min_trade_amount == 1000
    
    def test_sma_calculation(self):
        """测试移动平均计算"""
        # 添加数据到策略
        self.strategy.add_data(self.test_data)
        
        # 检查移动平均线
        assert hasattr(self.strategy, 'sma')
        assert self.strategy.sma is not None
    
    def test_buy_signal(self):
        """测试买入信号"""
        # 模拟价格高于移动平均线的情况
        self.strategy.dataclose = Mock()
        self.strategy.sma = Mock()
        self.strategy.dataclose[0] = 105
        self.strategy.sma[0] = 100
        
        # 模拟没有持仓
        self.strategy.position = Mock()
        self.strategy.position.size = 0
        
        # 模拟没有未完成订单
        self.strategy.order = None
        
        # 模拟broker
        self.strategy.broker = Mock()
        self.strategy.broker.getcash.return_value = 10000
        
        # 执行策略逻辑
        self.strategy.next()
        
        # 检查是否生成买入订单
        assert self.strategy.order is not None
```

#### 2.3 回测引擎测试

```python
import pytest
import pandas as pd
from TrickTrade.backtest_framework import BaseBacktestEngine, MovingAverageStrategy

class TestBaseBacktestEngine:
    """回测引擎测试"""
    
    def setup_method(self):
        """测试前准备"""
        self.engine = BaseBacktestEngine(initial_cash=100000, commission=0.001)
        
        # 创建测试数据
        self.test_data = pd.DataFrame({
            'Open': [100, 101, 102, 103, 104] * 20,
            'High': [105, 106, 107, 108, 109] * 20,
            'Low': [95, 96, 97, 98, 99] * 20,
            'Close': [103, 104, 105, 106, 107] * 20,
            'Volume': [1000000, 1100000, 1200000, 1300000, 1400000] * 20
        })
    
    def test_engine_initialization(self):
        """测试引擎初始化"""
        assert self.engine.initial_cash == 100000
        assert self.engine.commission == 0.001
        assert self.engine.results is None
    
    def test_add_strategy(self):
        """测试添加策略"""
        self.engine.add_strategy(MovingAverageStrategy, ma_period=10)
        assert len(self.engine.cerebro.strats) == 1
    
    def test_add_data(self):
        """测试添加数据"""
        self.engine.add_data(self.test_data, "test_data")
        assert len(self.engine.cerebro.datas) == 1
    
    def test_add_data_invalid(self):
        """测试添加无效数据"""
        invalid_data = pd.DataFrame({'Invalid': [1, 2, 3]})
        
        with pytest.raises(ValueError, match="数据缺少必需的列"):
            self.engine.add_data(invalid_data)
    
    def test_run_backtest(self):
        """测试运行回测"""
        self.engine.add_strategy(MovingAverageStrategy, ma_period=5)
        self.engine.add_data(self.test_data)
        
        results = self.engine.run(show_log=False)
        
        assert results is not None
        assert len(results) > 0
    
    def test_get_performance_metrics(self):
        """测试获取性能指标"""
        self.engine.add_strategy(MovingAverageStrategy, ma_period=5)
        self.engine.add_data(self.test_data)
        self.engine.run(show_log=False)
        
        metrics = self.engine.get_performance_metrics()
        
        assert 'initial_cash' in metrics
        assert 'final_value' in metrics
        assert 'total_return' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'max_drawdown' in metrics
    
    def test_get_performance_metrics_before_run(self):
        """测试在运行回测前获取性能指标"""
        with pytest.raises(ValueError, match="请先运行回测"):
            self.engine.get_performance_metrics()
```

### 3. 集成测试

#### 3.1 端到端测试

```python
import pytest
import pandas as pd
from TrickTrade import get_data, run_backtest
from TrickTrade.strategies import MovingAverageStrategy

class TestIntegration:
    """集成测试"""
    
    def test_full_workflow(self):
        """测试完整工作流程"""
        # 1. 获取数据
        data = get_data("000001", "2023-01-01", "2023-03-31", provider="mock")
        
        assert data is not None
        assert len(data) > 0
        assert 'Open' in data.columns
        assert 'High' in data.columns
        assert 'Low' in data.columns
        assert 'Close' in data.columns
        assert 'Volume' in data.columns
        
        # 2. 运行回测
        engine = run_backtest(
            data=data,
            strategy_class=MovingAverageStrategy,
            strategy_params={'ma_period': 20},
            initial_cash=100000,
            show_log=False,
            show_plot=False
        )
        
        assert engine is not None
        
        # 3. 获取性能指标
        metrics = engine.get_performance_metrics()
        
        assert metrics['initial_cash'] == 100000
        assert metrics['final_value'] > 0
        assert 'total_return' in metrics
        assert 'sharpe_ratio' in metrics
```

### 4. 性能测试

#### 4.1 性能基准测试

```python
import pytest
import time
import pandas as pd
from TrickTrade.backtest_framework import BaseBacktestEngine, MovingAverageStrategy

class TestPerformance:
    """性能测试"""
    
    def test_backtest_performance(self):
        """测试回测性能"""
        # 创建大量数据
        large_data = pd.DataFrame({
            'Open': [100] * 1000,
            'High': [105] * 1000,
            'Low': [95] * 1000,
            'Close': [103] * 1000,
            'Volume': [1000000] * 1000
        })
        
        engine = BaseBacktestEngine()
        engine.add_strategy(MovingAverageStrategy, ma_period=20)
        engine.add_data(large_data)
        
        # 测量执行时间
        start_time = time.time()
        engine.run(show_log=False)
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        # 性能要求：1000个数据点应在10秒内完成
        assert execution_time < 10.0
    
    def test_data_loading_performance(self):
        """测试数据加载性能"""
        # 模拟大量数据加载
        start_time = time.time()
        
        # 这里可以测试数据加载的性能
        # 例如：从文件加载大量数据
        
        end_time = time.time()
        loading_time = end_time - start_time
        
        # 性能要求：数据加载应在合理时间内完成
        assert loading_time < 5.0
```

## 部署指南

### 1. 本地部署

#### 1.1 环境准备

```bash
# 1. 克隆项目
git clone https://github.com/your-username/TrickTrade.git
cd TrickTrade

# 2. 创建虚拟环境
python -m venv tricktrade_env
source tricktrade_env/bin/activate  # Linux/Mac
# 或
tricktrade_env\Scripts\activate  # Windows

# 3. 安装依赖
pip install -r requirements.txt

# 4. 创建必要目录
mkdir -p data models logs config

# 5. 复制配置文件
cp config/system.yaml.example config/system.yaml
cp config/strategies.yaml.example config/strategies.yaml
```

#### 1.2 配置文件设置

创建 `config/system.yaml`:

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

#### 1.3 运行测试

```bash
# 运行所有测试
pytest

# 运行特定测试
pytest tests/test_data_provider.py

# 运行性能测试
pytest tests/test_performance.py -m "not slow"
```

### 2. Docker部署

#### 2.1 Dockerfile

```dockerfile
FROM python:3.12-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目文件
COPY . .

# 创建必要目录
RUN mkdir -p data models logs config

# 设置环境变量
ENV PYTHONPATH=/app

# 暴露端口（如果需要）
EXPOSE 8000

# 运行命令
CMD ["python", "main.py"]
```

#### 2.2 docker-compose.yml

```yaml
version: '3.8'

services:
  tricktrade:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./logs:/app/logs
      - ./config:/app/config
    environment:
      - PYTHONPATH=/app
    command: python main.py
```

#### 2.3 构建和运行

```bash
# 构建镜像
docker build -t tricktrade .

# 运行容器
docker run -d --name tricktrade-app tricktrade

# 使用docker-compose
docker-compose up -d
```

### 3. 生产环境部署

#### 3.1 系统服务配置

创建 `tricktrade.service`:

```ini
[Unit]
Description=TrickTrade Quant Trading Platform
After=network.target

[Service]
Type=simple
User=tricktrade
WorkingDirectory=/opt/tricktrade
ExecStart=/opt/tricktrade/venv/bin/python main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

#### 3.2 部署脚本

```bash
#!/bin/bash
# deploy.sh

# 设置变量
APP_NAME="tricktrade"
APP_DIR="/opt/$APP_NAME"
VENV_DIR="$APP_DIR/venv"
SERVICE_USER="tricktrade"

# 创建用户
sudo useradd -r -s /bin/false $SERVICE_USER

# 创建目录
sudo mkdir -p $APP_DIR
sudo chown $SERVICE_USER:$SERVICE_USER $APP_DIR

# 复制文件
sudo cp -r . $APP_DIR/
sudo chown -R $SERVICE_USER:$SERVICE_USER $APP_DIR

# 创建虚拟环境
sudo -u $SERVICE_USER python3 -m venv $VENV_DIR

# 安装依赖
sudo -u $SERVICE_USER $VENV_DIR/bin/pip install -r $APP_DIR/requirements.txt

# 复制服务文件
sudo cp tricktrade.service /etc/systemd/system/

# 启用服务
sudo systemctl daemon-reload
sudo systemctl enable $APP_NAME
sudo systemctl start $APP_NAME
```

## 贡献指南

### 1. 开发流程

#### 1.1 分支管理

```bash
# 1. Fork项目
# 2. 克隆你的Fork
git clone https://github.com/your-username/TrickTrade.git
cd TrickTrade

# 3. 添加上游仓库
git remote add upstream https://github.com/original-username/TrickTrade.git

# 4. 创建功能分支
git checkout -b feature/new-strategy

# 5. 开发功能
# ... 编写代码 ...

# 6. 提交更改
git add .
git commit -m "feat: add new strategy"

# 7. 推送分支
git push origin feature/new-strategy

# 8. 创建Pull Request
```

#### 1.2 提交规范

使用Conventional Commits规范：

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

类型说明：
- `feat`: 新功能
- `fix`: 修复bug
- `docs`: 文档更新
- `style`: 代码格式调整
- `refactor`: 代码重构
- `test`: 测试相关
- `chore`: 构建过程或辅助工具的变动

示例：
```
feat(strategies): add momentum strategy

Add a new momentum-based trading strategy that uses price momentum
to generate buy/sell signals.

Closes #123
```

### 2. 代码审查

#### 2.1 审查清单

- [ ] 代码符合PEP 8规范
- [ ] 添加了适当的类型提示
- [ ] 包含完整的文档字符串
- [ ] 添加了单元测试
- [ ] 测试通过
- [ ] 没有引入新的警告
- [ ] 性能影响评估

#### 2.2 审查流程

1. **自动化检查**: CI/CD流水线自动运行测试和代码检查
2. **同行审查**: 至少一名核心开发者审查代码
3. **功能测试**: 在测试环境中验证功能
4. **性能测试**: 确保没有性能回归
5. **文档更新**: 更新相关文档

### 3. 问题报告

#### 3.1 Bug报告模板

```markdown
## Bug描述
简要描述bug的内容

## 重现步骤
1. 执行步骤1
2. 执行步骤2
3. 执行步骤3

## 预期行为
描述期望的正确行为

## 实际行为
描述实际发生的行为

## 环境信息
- Python版本: 3.12
- 操作系统: Windows 10
- TrickTrade版本: 1.0.0

## 附加信息
添加任何其他相关信息
```

#### 3.2 功能请求模板

```markdown
## 功能描述
简要描述请求的功能

## 使用场景
描述这个功能的使用场景

## 解决方案
描述你期望的解决方案

## 替代方案
描述你考虑过的替代方案

## 附加信息
添加任何其他相关信息
```

### 4. 开发工具

#### 4.1 代码格式化

```bash
# 使用Black格式化代码
black .

# 使用isort排序导入
isort .

# 使用flake8检查代码
flake8 .
```

#### 4.2 类型检查

```bash
# 使用mypy进行类型检查
mypy .
```

#### 4.3 测试覆盖率

```bash
# 运行测试并生成覆盖率报告
pytest --cov=. --cov-report=html

# 查看覆盖率报告
open htmlcov/index.html
```

## 总结

本开发指南涵盖了TrckTrade平台的完整开发流程，从环境搭建到部署上线。通过遵循这些指南，开发者可以：

1. **快速上手**: 通过详细的环境搭建指南快速开始开发
2. **规范开发**: 遵循统一的代码规范和开发流程
3. **质量保证**: 通过完善的测试体系确保代码质量
4. **协作开发**: 通过清晰的贡献指南促进团队协作
5. **稳定部署**: 通过部署指南确保系统稳定运行

希望这份指南能够帮助开发者更好地参与TrckTrade项目的开发，共同构建一个优秀的量化投资平台。
