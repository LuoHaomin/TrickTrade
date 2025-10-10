# TrickTrade 回测模块

这是一个重构后的量化交易回测框架，提供了完整的回测功能。

## 模块结构

```
BackTest/
├── __init__.py              # 模块初始化
├── backtest_framework.py     # 基本回测框架
├── baseline_models.py        # 常用基线模型
├── data_provider.py          # 数据引入接口
├── example.py               # 使用示例
└── README.md                # 说明文档
```

## 主要功能

### 1. 基本回测框架 (`backtest_framework.py`)
- `BaseBacktestEngine`: 基础回测引擎
- `BaseStrategy`: 基础策略类
- `run_backtest()`: 便捷回测函数

### 2. 常用基线模型 (`baseline_models.py`)
- 买入持有策略
- 移动平均策略
- 双移动平均策略
- RSI策略
- MACD策略
- 布林带策略
- 均值回归策略
- 动量策略

### 3. 数据引入接口 (`data_provider.py`)
- AKShare数据提供者
- Yahoo Finance数据提供者
- 模拟数据提供者
- 统一的数据管理器

## 快速开始

### 1. 安装依赖
```bash
pip install backtrader matplotlib pandas numpy akshare yfinance
```

### 2. 基本使用
```python
from BackTest import get_data, STRATEGIES, run_backtest

# 获取数据
data = get_data("510050", "2022-01-01", "2023-12-31", data_type='fund')

# 运行回测
engine = run_backtest(
    data=data,
    strategy_class=STRATEGIES['moving_average'],
    strategy_params={'ma_period': 20},
    initial_cash=100000.0,
    commission=0.001
)
```

### 3. 多策略对比
```python
from BackTest import BaseBacktestEngine, STRATEGIES, get_data

# 获取数据
data = get_data("510300", "2022-01-01", "2023-12-31", data_type='fund')

# 测试不同策略
strategies = ['buy_and_hold', 'moving_average', 'rsi', 'macd']
results = {}

for strategy_key in strategies:
    engine = BaseBacktestEngine()
    engine.add_strategy(STRATEGIES[strategy_key])
    engine.add_data(data)
    engine.run()
    
    metrics = engine.get_performance_metrics()
    results[strategy_key] = metrics
```

### 4. 自定义策略
```python
from BackTest import BaseStrategy
import backtrader as bt

class MyStrategy(BaseStrategy):
    def __init__(self):
        super().__init__()
        self.sma = bt.indicators.SimpleMovingAverage(self.datas[0], period=20)
    
    def next(self):
        if not self.position:
            if self.dataclose[0] > self.sma[0]:
                self.buy()
        else:
            if self.dataclose[0] < self.sma[0]:
                self.sell()
```

## 数据支持

### 支持的数据源
- **AKShare**: 中国股票、基金、指数数据
- **Yahoo Finance**: 全球股票数据
- **模拟数据**: 用于测试和演示

### 数据类型
- `stock`: 股票数据
- `fund`: 基金数据
- `index`: 指数数据
- `auto`: 自动判断

### 使用示例
```python
# 获取股票数据
stock_data = get_data("000001", "2022-01-01", "2023-12-31", data_type='stock')

# 获取基金数据
fund_data = get_data("510050", "2022-01-01", "2023-12-31", data_type='fund')

# 获取指数数据
index_data = get_data("sh000001", "2022-01-01", "2023-12-31", data_type='index')

# 使用模拟数据
mock_data = get_data("mock", "2022-01-01", "2023-12-31", provider='mock', 
                    initial_price=100, volatility=0.02)
```

## 性能指标

回测框架提供以下性能指标：
- 总收益率
- 夏普比率
- 最大回撤
- 总交易次数
- 盈利/亏损交易数
- 胜率

## 运行示例

```bash
cd BackTest
python example.py
```

## 注意事项

1. 确保网络连接正常，以便获取真实数据
2. 如果无法获取真实数据，系统会自动使用模拟数据
3. 建议先运行示例程序熟悉框架使用方法
4. 自定义策略时，请继承 `BaseStrategy` 类

## 扩展功能

- 添加新的数据提供者
- 实现新的策略模型
- 自定义性能分析器
- 支持多资产组合回测
