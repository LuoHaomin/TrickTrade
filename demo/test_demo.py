"""
基金强化学习策略Demo测试脚本
用于验证各个模块的基本功能
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_data_fetcher():
    """测试数据获取模块"""
    print("=== 测试数据获取模块 ===")
    
    try:
        from data_fetcher import ETFDataFetcher
        
        fetcher = ETFDataFetcher()
        
        # 测试获取ETF列表
        print("获取ETF列表...")
        etf_list = fetcher.get_etf_list()
        if not etf_list.empty:
            print(f"✅ 成功获取 {len(etf_list)} 个ETF")
        else:
            print("⚠️ ETF列表为空")
        
        # 测试获取单个ETF数据（使用较短的时间范围）
        print("获取ETF数据...")
        data = fetcher.get_etf_data("510300", "2024-01-01", "2024-01-31")
        
        if data is not None and not data.empty:
            print(f"✅ 成功获取数据: {len(data)} 条记录")
            print(f"   数据列: {data.columns.tolist()}")
            print(f"   日期范围: {data.index[0]} 到 {data.index[-1]}")
        else:
            print("❌ 数据获取失败")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ 数据获取模块测试失败: {e}")
        return False

def test_feature_engineer():
    """测试特征工程模块"""
    print("\n=== 测试特征工程模块 ===")
    
    try:
        from feature_engineer import FeatureEngineer
        
        # 创建测试数据
        dates = pd.date_range('2024-01-01', '2024-01-31', freq='D')
        np.random.seed(42)
        
        test_data = pd.DataFrame({
            'Open': 100 + np.cumsum(np.random.randn(len(dates)) * 0.5),
            'High': 100 + np.cumsum(np.random.randn(len(dates)) * 0.5) + np.random.rand(len(dates)) * 2,
            'Low': 100 + np.cumsum(np.random.randn(len(dates)) * 0.5) - np.random.rand(len(dates)) * 2,
            'Close': 100 + np.cumsum(np.random.randn(len(dates)) * 0.5),
            'Volume': np.random.randint(1000000, 10000000, len(dates))
        }, index=dates)
        
        # 确保High >= Low
        test_data['High'] = np.maximum(test_data['High'], test_data['Low'])
        test_data['High'] = np.maximum(test_data['High'], test_data['Close'])
        test_data['Low'] = np.minimum(test_data['Low'], test_data['Close'])
        
        # 创建特征工程器
        fe = FeatureEngineer()
        
        # 测试技术指标计算
        print("计算技术指标...")
        features_data = fe.calculate_technical_indicators(test_data)
        
        if not features_data.empty:
            print(f"✅ 成功计算技术指标: {len(features_data.columns)} 个特征")
        else:
            print("❌ 技术指标计算失败")
            return False
        
        # 测试RL特征准备
        print("准备RL特征...")
        rl_features, feature_names = fe.prepare_features_for_rl(features_data, lookback_window=10)
        
        if len(rl_features) > 0:
            print(f"✅ 成功准备RL特征: {rl_features.shape}")
        else:
            print("❌ RL特征准备失败")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ 特征工程模块测试失败: {e}")
        return False

def test_rl_env():
    """测试强化学习环境"""
    print("\n=== 测试强化学习环境 ===")
    
    try:
        from rl_env import TradingEnv
        from feature_engineer import FeatureEngineer
        
        # 创建测试数据
        dates = pd.date_range('2024-01-01', '2024-01-31', freq='D')
        np.random.seed(42)
        
        test_data = pd.DataFrame({
            'Open': 100 + np.cumsum(np.random.randn(len(dates)) * 0.5),
            'High': 100 + np.cumsum(np.random.randn(len(dates)) * 0.5) + np.random.rand(len(dates)) * 2,
            'Low': 100 + np.cumsum(np.random.randn(len(dates)) * 0.5) - np.random.rand(len(dates)) * 2,
            'Close': 100 + np.cumsum(np.random.randn(len(dates)) * 0.5),
            'Volume': np.random.randint(1000000, 10000000, len(dates))
        }, index=dates)
        
        # 确保High >= Low
        test_data['High'] = np.maximum(test_data['High'], test_data['Low'])
        test_data['High'] = np.maximum(test_data['High'], test_data['Close'])
        test_data['Low'] = np.minimum(test_data['Low'], test_data['Close'])
        
        # 创建环境
        fe = FeatureEngineer()
        env = TradingEnv(test_data, fe)
        
        print(f"✅ 环境创建成功")
        print(f"   状态空间: {env.observation_space}")
        print(f"   动作空间: {env.action_space}")
        
        # 测试环境运行
        print("测试环境运行...")
        obs, info = env.reset()
        
        if obs is not None:
            print(f"✅ 环境重置成功，观察维度: {obs.shape}")
        else:
            print("❌ 环境重置失败")
            return False
        
        # 运行几步
        for i in range(3):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"   步骤 {i+1}: 动作={action[0]:.3f}, 奖励={reward:.4f}")
            
            if terminated or truncated:
                break
        
        return True
        
    except Exception as e:
        print(f"❌ 强化学习环境测试失败: {e}")
        return False

def test_bt_strategy():
    """测试BackTrader策略"""
    print("\n=== 测试BackTrader策略 ===")
    
    try:
        from bt_strategy import BuyAndHoldStrategy, MovingAverageStrategy, BacktestRunner
        
        # 创建测试数据
        dates = pd.date_range('2024-01-01', '2024-01-31', freq='D')
        np.random.seed(42)
        
        test_data = pd.DataFrame({
            'Open': 100 + np.cumsum(np.random.randn(len(dates)) * 0.5),
            'High': 100 + np.cumsum(np.random.randn(len(dates)) * 0.5) + np.random.rand(len(dates)) * 2,
            'Low': 100 + np.cumsum(np.random.randn(len(dates)) * 0.5) - np.random.rand(len(dates)) * 2,
            'Close': 100 + np.cumsum(np.random.randn(len(dates)) * 0.5),
            'Volume': np.random.randint(1000000, 10000000, len(dates))
        }, index=dates)
        
        # 确保High >= Low
        test_data['High'] = np.maximum(test_data['High'], test_data['Low'])
        test_data['High'] = np.maximum(test_data['High'], test_data['Close'])
        test_data['Low'] = np.minimum(test_data['Low'], test_data['Close'])
        
        # 测试买入持有策略
        print("测试买入持有策略...")
        runner = BacktestRunner(initial_cash=100000, commission=0.001)
        runner.add_data(test_data)
        runner.add_strategy(BuyAndHoldStrategy)
        
        results = runner.run(show_log=False)
        
        if results and 'metrics' in results:
            print(f"✅ 买入持有策略回测成功")
            print(f"   总收益率: {results['metrics']['total_return']:.2f}%")
        else:
            print("❌ 买入持有策略回测失败")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ BackTrader策略测试失败: {e}")
        return False

def test_visualizer():
    """测试可视化模块"""
    print("\n=== 测试可视化模块 ===")
    
    try:
        from visualizer import StrategyVisualizer
        
        # 创建测试结果
        test_results = {
            'rl_strategy': {
                'metrics': {
                    'total_return': 15.2,
                    'sharpe_ratio': 1.8,
                    'max_drawdown': -8.5,
                    'win_rate': 65.0,
                    'total_trades': 45
                }
            },
            'buy_and_hold': {
                'metrics': {
                    'total_return': 12.1,
                    'sharpe_ratio': 1.2,
                    'max_drawdown': -12.3,
                    'win_rate': 100.0,
                    'total_trades': 1
                }
            }
        }
        
        # 创建可视化器
        visualizer = StrategyVisualizer({
            'output_dir': 'demo/output', 
            'figure_size': (15, 10),
            'title_size': 16,
            'label_size': 14,
            'font_size': 12,
            'color_palette': 'husl',
            'dpi': 300
        })
        
        # 测试创建图表
        print("创建性能对比图...")
        fig = visualizer.plot_performance_comparison(test_results)
        
        if fig is not None:
            print("✅ 可视化模块测试成功")
            # 不显示图表，直接关闭
            import matplotlib.pyplot as plt
            plt.close(fig)
        else:
            print("❌ 可视化模块测试失败")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ 可视化模块测试失败: {e}")
        return False

def main():
    """运行所有测试"""
    print("基金强化学习策略Demo - 模块测试")
    print("=" * 50)
    
    # 创建输出目录
    os.makedirs('demo/output', exist_ok=True)
    
    # 运行测试
    tests = [
        test_data_fetcher,
        test_feature_engineer,
        test_rl_env,
        test_bt_strategy,
        test_visualizer
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ 测试异常: {e}")
    
    print("\n" + "=" * 50)
    print(f"测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有模块测试通过！Demo可以正常运行。")
        return True
    else:
        print("⚠️ 部分模块测试失败，请检查依赖和环境。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
