"""
RL策略调试脚本
用于定位RL策略回测卡住的具体问题
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_fetcher import ETFDataFetcher
from feature_engineer import FeatureEngineer
from bt_strategy import RLStrategy, BacktestRunner

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_rl_strategy():
    """调试RL策略"""
    logger.info("开始调试RL策略...")
    
    # 获取数据
    fetcher = ETFDataFetcher()
    data = fetcher.get_etf_data("510300", "2024-01-01", "2024-12-31")
    
    if data is None:
        logger.error("无法获取数据")
        return False
    
    logger.info(f"获取到 {len(data)} 条数据记录")
    
    # 测试特征工程
    logger.info("测试特征工程...")
    fe = FeatureEngineer()
    
    try:
        logger.info("计算技术指标...")
        features_data = fe.calculate_technical_indicators(data)
        logger.info(f"技术指标计算完成: {features_data.shape}")
        
        logger.info("准备RL特征...")
        rl_features, feature_names = fe.prepare_features_for_rl(features_data, lookback_window=20)
        logger.info(f"RL特征准备完成: {rl_features.shape}")
        
    except Exception as e:
        logger.error(f"特征工程失败: {e}")
        return False
    
    # 测试RL策略初始化
    logger.info("测试RL策略初始化...")
    try:
        # 创建回测运行器
        runner = BacktestRunner(initial_cash=100000, commission=0.001)
        runner.add_data(data)
        
        # 检查模型文件是否存在
        model_path = 'output/models/ppo_trading_model_20251021_082935.zip'
        if not os.path.exists(model_path):
            logger.error(f"模型文件不存在: {model_path}")
            return False
        
        logger.info(f"模型文件存在: {model_path}")
        
        # 添加RL策略
        runner.add_strategy(RLStrategy, 
                           model_path=model_path,
                           lookback_window=20,
                           position_threshold=0.1,
                           printlog=True)
        
        logger.info("RL策略添加成功，开始回测...")
        
        # 运行回测
        results = runner.run(show_log=True)
        
        logger.info(f"RL策略回测完成: {results['metrics']}")
        
    except Exception as e:
        logger.error(f"RL策略回测失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def main():
    """主函数"""
    success = debug_rl_strategy()
    if success:
        print("\n✅ RL策略调试成功！")
    else:
        print("\n❌ RL策略调试失败")
        sys.exit(1)

if __name__ == "__main__":
    main()
