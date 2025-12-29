# 基金投资强化学习策略Demo

## 项目简介

这是一个基于强化学习的基金投资策略Demo，使用Stable-Baselines3的PPO算法训练交易策略，并通过BackTrader进行回测验证。该Demo验证了从数据获取、特征工程、模型训练到策略回测的完整技术栈可行性。

## 技术栈

- **数据获取**: AKShare API
- **强化学习**: Stable-Baselines3 (PPO算法)
- **回测框架**: BackTrader
- **特征工程**: TA-Lib, 自定义技术指标
- **可视化**: Matplotlib, Seaborn
- **数据处理**: Pandas, NumPy

## 项目结构

```
demo/
├── data_fetcher.py      # ETF数据获取模块
├── feature_engineer.py  # 特征工程模块
├── rl_env.py           # 强化学习环境
├── train_rl.py         # PPO模型训练
├── bt_strategy.py      # BackTrader策略集成
├── backtest_eval.py    # 回测与评估
├── visualizer.py       # 可视化模块
├── main.py             # 主程序入口
├── config.yaml         # 配置文件
├── requirements.txt    # 依赖包
└── README.md          # 说明文档
```

## 快速开始

### 1. 环境准备

```bash
# 创建虚拟环境
python -m venv demo_env
source demo_env/bin/activate  # Linux/Mac
# 或
demo_env\Scripts\activate     # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 运行完整Demo

```bash
# 运行完整的Demo流程
python main.py

# 使用自定义参数
python main.py --symbol 510500 --start-date 2020-01-01 --end-date 2024-12-31
```

### 3. 分步运行

```bash
# 仅训练模型
python main.py --train-only

# 仅运行回测（需要已有训练好的模型）
python main.py --test-only

# 强制重新训练
python main.py --force-retrain
```

## 功能模块

### 1. 数据获取 (`data_fetcher.py`)

- 从AKShare获取ETF历史数据
- 支持数据缓存机制
- 自动数据预处理和清洗

**主要功能:**
- `get_etf_data()`: 获取单个ETF数据
- `get_multiple_etf_data()`: 获取多个ETF数据
- 数据缓存和过期检查

### 2. 特征工程 (`feature_engineer.py`)

- 计算技术指标（MA、RSI、MACD、布林带等）
- 特征归一化处理
- 为强化学习准备特征数据

**主要功能:**
- `calculate_technical_indicators()`: 计算技术指标
- `normalize_features()`: 特征归一化
- `prepare_features_for_rl()`: 准备RL特征

### 3. 强化学习环境 (`rl_env.py`)

- 基于gymnasium的自定义交易环境
- 连续动作空间（持仓比例调整）
- 综合奖励函数设计

**环境特性:**
- 状态空间: 技术指标 + 账户状态
- 动作空间: [-1, 1] 连续动作
- 奖励函数: 收益率 + 风险惩罚 + 交易成本

### 4. 模型训练 (`train_rl.py`)

- 使用PPO算法训练交易策略
- 支持超参数调优
- 训练进度监控和模型保存

**训练特性:**
- 自动数据分割（训练/验证）
- 训练回调函数
- 模型性能评估

### 5. BackTrader集成 (`bt_strategy.py`)

- RL策略的BackTrader实现
- 基线策略（买入持有、移动平均）
- 统一回测接口

**策略类型:**
- `RLStrategy`: 强化学习策略
- `BuyAndHoldStrategy`: 买入持有策略
- `MovingAverageStrategy`: 移动平均策略

### 6. 回测评估 (`backtest_eval.py`)

- 多策略性能对比
- 详细性能指标计算
- 自动生成对比报告

**评估指标:**
- 总收益率、年化收益率
- 夏普比率、最大回撤
- 胜率、交易次数

### 7. 可视化 (`visualizer.py`)

- 策略性能对比图表
- 风险收益分析
- 综合性能热力图

**图表类型:**
- 性能对比柱状图
- 风险收益散点图
- 策略排名图
- 综合评分图

## 配置说明

### 主要配置参数

```yaml
# ETF配置
symbol: '510300'  # ETF代码
start_date: '2020-01-01'  # 数据开始日期
end_date: '2024-12-31'    # 数据结束日期

# RL训练配置
rl_training:
  learning_rate: 3e-4
  total_timesteps: 100000
  n_steps: 2048
  batch_size: 64

# 回测配置
backtest:
  initial_cash: 100000.0
  commission: 0.001
```

## 预期结果

### 成功运行后，您将获得：

1. **训练好的RL模型** (`demo/output/models/`)
2. **回测结果报告** (`demo/output/results/`)
3. **性能对比图表** (`demo/output/plots/`)
4. **详细日志** (`demo/output/logs/`)
5. **最终报告** (`demo/output/final_report.json`)

### 性能指标对比

Demo将对比以下策略的性能：
- **RL策略**: 基于PPO的强化学习策略
- **买入持有**: 传统买入持有策略
- **移动平均**: 技术指标策略

**关键指标:**
- 总收益率
- 夏普比率
- 最大回撤
- 胜率

## 技术亮点

### 1. 环境设计
- 连续动作空间，更符合实际交易
- 综合奖励函数，平衡收益和风险
- 状态空间包含技术指标和账户状态

### 2. 特征工程
- 丰富的技术指标计算
- 滑动窗口特征提取
- 多维度特征归一化

### 3. 模型集成
- 无缝集成RL模型到BackTrader
- 支持多种基线策略对比
- 统一的回测接口

### 4. 可视化分析
- 多维度性能对比
- 风险收益分析
- 策略排名和评分

## 注意事项

1. **数据获取**: 首次运行需要网络连接获取数据
2. **训练时间**: RL模型训练可能需要较长时间
3. **内存使用**: 特征工程和模型训练需要足够内存
4. **依赖安装**: 确保所有依赖包正确安装

## 故障排除

### 常见问题

1. **数据获取失败**
   - 检查网络连接
   - 确认ETF代码正确
   - 检查日期格式

2. **模型训练失败**
   - 检查数据完整性
   - 调整训练参数
   - 查看训练日志

3. **回测失败**
   - 确认模型文件存在
   - 检查数据格式
   - 验证策略参数

## 扩展建议

1. **更多ETF**: 尝试不同的ETF标的
2. **参数调优**: 使用Optuna进行超参数优化
3. **更多策略**: 添加更多基线策略对比
4. **实时交易**: 扩展到实时交易系统

## 联系方式

如有问题或建议，请查看项目文档或提交Issue。

---

**免责声明**: 本Demo仅用于技术验证和学习目的，不构成投资建议。实际投资请谨慎决策。
