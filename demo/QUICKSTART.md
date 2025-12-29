# 基金强化学习策略Demo - 快速开始指南

## 环境要求

- Python 3.8+
- 8GB+ 内存（推荐16GB+）
- 网络连接（用于数据获取）

## 安装步骤

### 1. 克隆或下载项目
```bash
# 如果是从Git仓库克隆
git clone <repository-url>
cd TrickTrade/demo

# 或者直接下载demo文件夹
```

### 2. 创建虚拟环境
```bash
# 使用venv
python -m venv demo_env
source demo_env/bin/activate  # Linux/Mac
# 或
demo_env\Scripts\activate     # Windows

# 或使用conda
conda create -n demo_env python=3.9
conda activate demo_env
```

### 3. 安装依赖
```bash
pip install -r requirements.txt
```

**注意**: 如果安装TA-Lib失败，请参考以下解决方案：

**Windows:**
```bash
# 下载预编译的wheel文件
pip install https://download.lfd.uci.edu/pythonlibs/archived/TA_Lib-0.4.24-cp39-cp39-win_amd64.whl
```

**Linux/Mac:**
```bash
# 先安装系统依赖
# Ubuntu/Debian:
sudo apt-get install build-essential

# CentOS/RHEL:
sudo yum groupinstall "Development Tools"

# 然后安装TA-Lib
pip install TA-Lib
```

### 4. 验证安装
```bash
python test_demo.py
```

## 运行Demo

### 完整Demo（推荐）
```bash
python main.py
```

这将运行完整的流程：
1. 获取ETF数据
2. 训练RL模型
3. 运行回测
4. 生成可视化图表

### 自定义参数
```bash
# 使用不同的ETF
python main.py --symbol 510500 --start-date 2020-01-01 --end-date 2024-12-31

# 强制重新训练
python main.py --force-retrain

# 仅训练模型
python main.py --train-only

# 仅运行回测（需要已有模型）
python main.py --test-only
```

## 预期结果

运行成功后，您将在 `demo/output/` 目录下看到：

```
demo/output/
├── models/           # 训练好的RL模型
├── results/          # 回测结果
├── plots/            # 可视化图表
├── logs/             # 运行日志
└── final_report.json # 最终报告
```

## 常见问题

### 1. 数据获取失败
**问题**: `无法获取ETF数据`
**解决方案**:
- 检查网络连接
- 确认ETF代码正确（如510300）
- 尝试使用VPN或更换网络

### 2. TA-Lib安装失败
**问题**: `No module named 'talib'`
**解决方案**:
- Windows: 下载预编译wheel文件
- Linux: 安装系统依赖后重新安装
- Mac: 使用Homebrew安装 `brew install ta-lib`

### 3. 内存不足
**问题**: `MemoryError` 或训练缓慢
**解决方案**:
- 减少训练时间步数（修改config.yaml中的total_timesteps）
- 减少特征维度
- 使用更短的时间范围

### 4. 模型训练失败
**问题**: 训练过程中出错
**解决方案**:
- 检查数据完整性
- 调整学习率等超参数
- 查看详细日志文件

## 性能优化建议

### 1. 加速训练
- 减少total_timesteps（如50000）
- 使用GPU（如果可用）
- 调整batch_size和n_steps

### 2. 减少内存使用
- 使用更短的时间范围
- 减少特征数量
- 启用数据缓存

### 3. 提高回测速度
- 减少回测期间
- 简化策略逻辑
- 使用更少的数据点

## 扩展功能

### 1. 超参数优化
```python
# 在train_rl.py中启用
trainer.hyperparameter_tuning({
    'learning_rate': [1e-4, 3e-4, 1e-3],
    'n_steps': [1024, 2048, 4096]
})
```

### 2. 更多ETF
修改config.yaml中的symbol参数：
```yaml
symbol: '510500'  # 中证500ETF
# 或
symbol: '159919'  # 沪深300ETF（深交所）
```

### 3. 自定义策略
在bt_strategy.py中添加新的策略类，继承bt.Strategy。

## 技术支持

如果遇到问题：
1. 查看日志文件 `demo/output/logs/main.log`
2. 运行测试脚本 `python test_demo.py`
3. 检查依赖版本是否匹配
4. 参考项目文档

## 免责声明

本Demo仅用于技术验证和学习目的，不构成投资建议。实际投资请谨慎决策，风险自负。
