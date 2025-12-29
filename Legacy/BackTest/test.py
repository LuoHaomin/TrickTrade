"""
简单测试脚本
验证重构后的回测框架是否正常工作
"""

def test_imports():
    """测试模块导入"""
    try:
        from backtest_framework import BaseBacktestEngine, BaseStrategy
        from baseline_models import STRATEGIES
        from data_provider import get_data
        print("✓ 模块导入成功")
        return True
    except ImportError as e:
        print(f"✗ 模块导入失败: {e}")
        return False

def test_data_provider():
    """测试数据提供者"""
    try:
        from data_provider import get_data
        
        # 测试模拟数据
        data = get_data("test", "2022-01-01", "2022-12-31", provider='mock', 
                       initial_price=100, volatility=0.02)
        
        if data is not None and len(data) > 0:
            print("✓ 数据提供者测试成功")
            print(f"  数据形状: {data.shape}")
            print(f"  列名: {list(data.columns)}")
            return True
        else:
            print("✗ 数据提供者测试失败: 数据为空")
            return False
            
    except Exception as e:
        print(f"✗ 数据提供者测试失败: {e}")
        return False

def test_strategy():
    """测试策略"""
    try:
        from backtest_framework import BaseBacktestEngine
        from baseline_models import STRATEGIES
        from data_provider import get_data
        
        # 获取模拟数据
        data = get_data("test", "2022-01-01", "2022-12-31", provider='mock', 
                       initial_price=100, volatility=0.02)
        
        # 创建回测引擎
        engine = BaseBacktestEngine(initial_cash=100000, commission=0.001)
        
        # 添加买入持有策略
        engine.add_strategy(STRATEGIES['buy_and_hold'])
        
        # 添加数据
        engine.add_data(data)
        
        # 运行回测
        engine.run(show_log=False)
        
        # 获取性能指标
        metrics = engine.get_performance_metrics()
        
        if metrics['final_value'] > 0:
            print("✓ 策略测试成功")
            print(f"  初始资金: {metrics['initial_cash']:.2f}")
            print(f"  最终资金: {metrics['final_value']:.2f}")
            print(f"  总收益率: {metrics['total_return']:.2f}%")
            return True
        else:
            print("✗ 策略测试失败: 最终资金异常")
            return False
            
    except Exception as e:
        print(f"✗ 策略测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("=== TrickTrade 回测框架测试 ===")
    print()
    
    tests = [
        ("模块导入", test_imports),
        ("数据提供者", test_data_provider),
        ("策略回测", test_strategy)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"测试: {test_name}")
        if test_func():
            passed += 1
        print()
    
    print("=== 测试结果 ===")
    print(f"通过: {passed}/{total}")
    
    if passed == total:
        print("✓ 所有测试通过！回测框架工作正常。")
    else:
        print("✗ 部分测试失败，请检查相关模块。")

if __name__ == '__main__':
    main()
