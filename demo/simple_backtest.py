"""
简化的回测脚本
仅测试买入持有和移动平均策略，避免RL策略的复杂性
"""

import os
import sys
import logging
from datetime import datetime

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_fetcher import ETFDataFetcher
from bt_strategy import BuyAndHoldStrategy, MovingAverageStrategy, BacktestRunner

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """运行简化回测"""
    logger.info("开始简化回测测试...")
    
    # 获取数据
    fetcher = ETFDataFetcher()
    data = fetcher.get_etf_data("510300", "2024-01-01", "2024-12-31")
    
    if data is None:
        logger.error("无法获取数据")
        return False
    
    logger.info(f"获取到 {len(data)} 条数据记录")
    
    # 测试买入持有策略
    logger.info("测试买入持有策略...")
    runner1 = BacktestRunner(initial_cash=100000, commission=0.001)
    runner1.add_data(data)
    runner1.add_strategy(BuyAndHoldStrategy, printlog=False)
    
    results1 = runner1.run(show_log=False)
    logger.info(f"买入持有策略结果: 总收益率 {results1['metrics']['total_return']:.2f}%")
    
    # 测试移动平均策略
    logger.info("测试移动平均策略...")
    runner2 = BacktestRunner(initial_cash=100000, commission=0.001)
    runner2.add_data(data)
    runner2.add_strategy(MovingAverageStrategy, ma_period=20, printlog=False)
    
    results2 = runner2.run(show_log=False)
    logger.info(f"移动平均策略结果: 总收益率 {results2['metrics']['total_return']:.2f}%")
    
    # 对比结果
    logger.info("=== 策略对比 ===")
    logger.info(f"买入持有策略: {results1['metrics']['total_return']:.2f}%")
    logger.info(f"移动平均策略: {results2['metrics']['total_return']:.2f}%")
    
    if results2['metrics']['total_return'] > results1['metrics']['total_return']:
        logger.info("✅ 移动平均策略表现更好")
    else:
        logger.info("✅ 买入持有策略表现更好")
    
    logger.info("简化回测测试完成")
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ 简化回测测试成功！")
    else:
        print("\n❌ 简化回测测试失败")
        sys.exit(1)
