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

### 回测

### 预测

### 关联性分析

### 选基

### 模型训练

### 可视化

### 其他

