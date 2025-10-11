# TrckTrade 基金预测模块设计

## 概述

基金预测模块是TrckTrade平台的核心辅助投资工具之一，提供多种机器学习算法对基金价格进行预测，包括LSTM、XGBoost、Prophet和集成预测等方法。该模块旨在为投资决策提供科学的价格预测支持。

## 模块架构

```
FundPrediction/
├── __init__.py                 # 模块初始化
├── predictors/                 # 预测器模块
│   ├── __init__.py
│   ├── lstm_predictor.py       # LSTM预测器
│   ├── xgboost_predictor.py   # XGBoost预测器
│   ├── prophet_predictor.py    # Prophet预测器
│   └── ensemble_predictor.py  # 集成预测器
├── evaluators/                 # 评估器模块
│   ├── __init__.py
│   └── prediction_evaluator.py # 预测评估器
└── visualizers/                # 可视化模块
    ├── __init__.py
    └── prediction_visualizer.py # 预测可视化器
```

## 核心组件设计

### 1. LSTM预测器 (LSTMPredictor)

#### 1.1 功能特性
- **时序建模**: 使用LSTM神经网络捕捉时序依赖关系
- **多特征支持**: 支持技术指标、价格变化、成交量等多维特征
- **序列预测**: 支持多步预测
- **概率预测**: 提供上涨/下跌概率预测

#### 1.2 技术实现
```python
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out
```

#### 1.3 配置参数
- `sequence_length`: 序列长度 (默认60)
- `hidden_size`: 隐藏层大小 (默认64)
- `num_layers`: LSTM层数 (默认2)
- `learning_rate`: 学习率 (默认0.001)
- `batch_size`: 批次大小 (默认32)
- `epochs`: 训练轮数 (默认100)
- `dropout`: Dropout比例 (默认0.2)

#### 1.4 特征工程
- **价格特征**: 价格变化率、移动平均、EMA
- **技术指标**: RSI、MACD、布林带
- **波动率特征**: 滚动标准差、价格波动
- **成交量特征**: 成交量比率、成交量移动平均

### 2. XGBoost预测器 (XGBoostPredictor)

#### 2.1 功能特性
- **树模型**: 使用梯度提升树进行预测
- **特征重要性**: 提供特征重要性分析
- **非线性建模**: 捕捉非线性关系
- **快速训练**: 高效的训练和预测

#### 2.2 技术实现
```python
class XGBoostPredictor:
    def __init__(self, config):
        self.model = xgb.XGBRegressor(
            n_estimators=config.get('n_estimators', 100),
            max_depth=config.get('max_depth', 6),
            learning_rate=config.get('learning_rate', 0.1),
            subsample=config.get('subsample', 0.8),
            colsample_bytree=config.get('colsample_bytree', 0.8)
        )
```

#### 2.3 配置参数
- `n_estimators`: 树的数量 (默认100)
- `max_depth`: 最大深度 (默认6)
- `learning_rate`: 学习率 (默认0.1)
- `subsample`: 子样本比例 (默认0.8)
- `colsample_bytree`: 特征采样比例 (默认0.8)
- `lookback_periods`: 回望期数列表 (默认[1,2,3,5,10])

#### 2.4 特征工程
- **滞后特征**: 价格、成交量、价格变化率的滞后值
- **滚动统计**: 均值、标准差、最值、偏度、峰度
- **技术指标**: 全面的技术指标计算
- **交互特征**: 特征间的交互关系

### 3. Prophet预测器 (ProphetPredictor)

#### 3.1 功能特性
- **趋势分解**: 自动分解趋势、季节性、节假日效应
- **缺失值处理**: 自动处理缺失值和异常值
- **置信区间**: 提供预测置信区间
- **组件分析**: 分析各组件对预测的贡献

#### 3.2 技术实现
```python
class ProphetPredictor:
    def __init__(self, config):
        self.model = Prophet(
            yearly_seasonality=config.get('yearly_seasonality', True),
            weekly_seasonality=config.get('weekly_seasonality', True),
            daily_seasonality=config.get('daily_seasonality', False),
            seasonality_mode=config.get('seasonality_mode', 'additive')
        )
```

#### 3.3 配置参数
- `yearly_seasonality`: 年度季节性 (默认True)
- `weekly_seasonality`: 周度季节性 (默认True)
- `daily_seasonality`: 日度季节性 (默认False)
- `seasonality_mode`: 季节性模式 (默认'additive')
- `changepoint_prior_scale`: 变点先验尺度 (默认0.05)
- `seasonality_prior_scale`: 季节性先验尺度 (默认10.0)

#### 3.4 预测组件
- **趋势组件**: 长期趋势变化
- **年度季节性**: 年度周期性模式
- **周度季节性**: 周度周期性模式
- **节假日效应**: 节假日对价格的影响

### 4. 集成预测器 (EnsemblePredictor)

#### 4.1 功能特性
- **多模型集成**: 结合多个预测器的结果
- **权重优化**: 支持动态权重调整
- **集成方法**: 加权平均、投票、堆叠
- **不确定性量化**: 提供集成预测的不确定性

#### 4.2 技术实现
```python
class EnsemblePredictor:
    def __init__(self, config):
        self.predictors = []
        self.weights = config.get('weights', None)
        self.ensemble_method = config.get('ensemble_method', 'weighted_average')
        
    def _weighted_average(self, predictions_list):
        predictions_array = np.array(predictions_list)
        ensemble_predictions = np.average(predictions_array, axis=0, weights=self.weights)
        return ensemble_predictions.tolist()
```

#### 4.3 配置参数
- `predictors`: 预测器配置列表
- `weights`: 权重列表
- `ensemble_method`: 集成方法 ('weighted_average', 'voting', 'stacking')
- `use_probability`: 是否使用概率预测 (默认False)

#### 4.4 集成方法
- **加权平均**: 基于权重的线性组合
- **投票**: 简单平均或多数投票
- **堆叠**: 使用元学习器进行集成

## 预测评估体系

### 1. 预测评估器 (PredictionEvaluator)

#### 1.1 评估指标
- **点预测指标**: MSE、RMSE、MAE、R²、MAPE
- **方向准确性**: 涨跌方向预测准确率
- **概率预测指标**: AUC、对数损失、Brier分数
- **区间预测指标**: 覆盖率、区间宽度、区间得分

#### 1.2 评估方法
```python
def evaluate_point_predictions(self, y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    direction_accuracy = self._calculate_direction_accuracy(y_true, y_pred)
    
    return {
        'mse': mse, 'mae': mae, 'r2': r2,
        'mape': mape, 'direction_accuracy': direction_accuracy
    }
```

#### 1.3 时间序列特定指标
- **趋势准确性**: 趋势方向预测准确率
- **滞后相关性**: 预测与真实值的滞后相关性
- **预测稳定性**: 预测值的变化稳定性

### 2. 模型比较
- **性能对比**: 多模型性能指标对比
- **特征重要性**: 各模型特征重要性分析
- **预测一致性**: 模型间预测一致性分析
- **鲁棒性测试**: 不同市场条件下的表现

## 可视化系统

### 1. 预测可视化器 (PredictionVisualizer)

#### 1.1 基础图表
- **预测结果图**: 历史价格与预测价格对比
- **置信区间图**: 预测结果与置信区间
- **概率预测图**: 上涨/下跌概率可视化
- **误差分析图**: 预测误差分布分析

#### 1.2 高级图表
- **集成预测图**: 多模型预测结果对比
- **组件分析图**: Prophet组件分解图
- **特征重要性图**: XGBoost特征重要性
- **预测准确性图**: 模型性能对比图

#### 1.3 交互式可视化
- **时间范围选择**: 支持动态时间范围调整
- **模型切换**: 实时切换不同预测模型
- **参数调整**: 可视化参数调整效果
- **导出功能**: 支持图表导出和分享

## 数据流设计

### 1. 数据预处理流程
```
原始数据 → 数据清洗 → 特征工程 → 数据标准化 → 模型训练
```

### 2. 预测流程
```
历史数据 → 特征提取 → 模型预测 → 结果后处理 → 可视化输出
```

### 3. 评估流程
```
预测结果 → 真实值 → 指标计算 → 性能分析 → 报告生成
```

## 使用示例

### 1. 基础使用
```python
from FundPrediction import LSTMPredictor

# 创建LSTM预测器
predictor = LSTMPredictor({
    'sequence_length': 60,
    'hidden_size': 64,
    'epochs': 100
})

# 训练模型
predictor.train(data, target_column='Close')

# 预测
result = predictor.predict(data, steps=5)
print(f"预测结果: {result['predictions']}")
```

### 2. 集成预测
```python
from FundPrediction import EnsemblePredictor

# 创建集成预测器
ensemble = EnsemblePredictor({
    'predictors': [
        {'type': 'lstm', 'params': {'sequence_length': 60}},
        {'type': 'xgboost', 'params': {'n_estimators': 100}},
        {'type': 'prophet', 'params': {'yearly_seasonality': True}}
    ],
    'weights': [0.4, 0.4, 0.2],
    'ensemble_method': 'weighted_average'
})

# 训练和预测
ensemble.train(data)
result = ensemble.predict(data, steps=5)
```

### 3. 预测评估
```python
from FundPrediction import PredictionEvaluator

# 创建评估器
evaluator = PredictionEvaluator()

# 评估预测结果
metrics = evaluator.evaluate_point_predictions(y_true, y_pred)
print(f"RMSE: {metrics['rmse']:.4f}")
print(f"方向准确性: {metrics['direction_accuracy']:.4f}")
```

## 性能优化

### 1. 计算优化
- **GPU加速**: 支持CUDA加速训练
- **并行处理**: 多模型并行训练
- **内存优化**: 高效的内存使用策略
- **缓存机制**: 预测结果缓存

### 2. 模型优化
- **超参数优化**: 使用Optuna进行自动调参
- **早停机制**: 防止过拟合
- **正则化**: L1/L2正则化
- **交叉验证**: 时间序列交叉验证

### 3. 部署优化
- **模型压缩**: 模型量化与剪枝
- **批量预测**: 支持批量预测
- **异步处理**: 异步预测处理
- **负载均衡**: 多实例负载均衡

## 扩展性设计

### 1. 新预测器接口
```python
class BasePredictor:
    def train(self, data, target_column='Close'):
        raise NotImplementedError
    
    def predict(self, data, steps=1):
        raise NotImplementedError
    
    def save_model(self, path):
        raise NotImplementedError
    
    def load_model(self, path):
        raise NotImplementedError
```

### 2. 插件机制
- **预测器插件**: 支持自定义预测器
- **特征工程插件**: 支持自定义特征
- **评估指标插件**: 支持自定义评估指标
- **可视化插件**: 支持自定义可视化

### 3. 配置驱动
- **YAML配置**: 使用YAML文件配置模型
- **环境变量**: 支持环境变量配置
- **动态配置**: 运行时配置更新
- **配置验证**: 配置参数验证

## 总结

基金预测模块提供了完整的基金价格预测解决方案，支持多种机器学习算法，具备完善的评估体系和可视化功能。通过模块化设计和标准化接口，支持灵活扩展和定制化开发，为量化投资提供科学的价格预测支持。
