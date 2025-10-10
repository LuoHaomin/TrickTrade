# TrckTrade

## 简介

TrckTrade是一个量化交易平台，主要面向公募基金提供量化交易相关的工具，如关联性分析、选基、回测、预测等。

## 技术栈

### 核心语言
- **Python**: 3.12

### 数据获取与处理
- **数据来源**: AKShare、Yahoo Finance
- **数据处理**: Pandas、NumPy
- **数据存储**: HDF5、Parquet（本地高性能存储）
- **技术指标**: TA-Lib

### 回测框架
- **回测引擎**: BackTrader
- **策略开发**: 传统技术指标策略、机器学习策略

### 机器学习与AI
- **传统机器学习**: Scikit-learn、XGBoost、LightGBM
- **深度学习**: PyTorch
- **强化学习**: Stable-Baselines3（DQN、PPO等算法）
- **特征工程**: TA-Lib、tsfresh（时序特征提取）
- **超参数优化**: Optuna
- **模型持久化**: Joblib、PyTorch模型文件

### 可视化与分析
- **图表绘制**: Matplotlib、Seaborn
- **交互式可视化**: Plotly（可选）

### 性能与优化
- **并行计算**: Joblib、multiprocessing
- **实验管理**: MLflow（可选）

### 配置与工具
- **配置文件**: YAML、JSON
- **日志管理**: Python logging

## 项目结构

### 数据获取与处理
- **DataSource/**: 数据源模块，支持AKShare、Yahoo Finance等
- **BackTest/data_provider.py**: 统一数据接口

### 回测
- **BackTest/**: 回测框架，基于BackTrader
- **BackTest/baseline_models.py**: 基础策略模型
- **BackTest/backtest_framework.py**: 回测引擎

### 基金预测
- **FundPrediction/**: 基金价格预测模块
- **FundPrediction/predictors/**: 预测器（LSTM、XGBoost、Prophet、集成）
- **FundPrediction/evaluators/**: 预测评估器
- **FundPrediction/visualizers/**: 预测可视化器

### 关联性分析
- **CorrelationAnalysis/**: 基金关联性分析模块
- **CorrelationAnalysis/correlation/**: 相关性分析（皮尔森、斯皮尔曼、动态）
- **CorrelationAnalysis/causality/**: 因果分析（格兰杰、协整）
- **CorrelationAnalysis/clustering/**: 聚类分析
- **CorrelationAnalysis/visualizers/**: 关联性可视化

### 机器学习策略
- **RLStrategy/**: 强化学习策略（DQN、PPO）
- **Startegy/**: 传统策略模块

### 可视化
- **Visualize/**: 可视化工具模块

### 文档
- **doc/**: 完整的设计文档
  - `architecture.md`: 系统架构设计
  - `data_flow.md`: 数据流设计
  - `ml_strategy.md`: 机器学习策略设计
  - `api_reference.md`: API接口参考
  - `development_guide.md`: 开发指南

