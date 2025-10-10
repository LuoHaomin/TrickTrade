"""
回测系统使用示例
演示如何使用重构后的回测框架
"""

import pandas as pd
from datetime import datetime, timedelta
from backtest_framework import BaseBacktestEngine, run_backtest
from baseline_models import STRATEGIES
from data_provider import get_data, data_manager

def example_single_strategy():
    """单策略回测示例"""
    print("=== 单策略回测示例 ===")
    
    # 获取数据
    symbol = "510050"  # 上证50ETF
    start_date = "2022-01-01"
    end_date = "2023-12-31"
    
    print(f"正在获取 {symbol} 数据...")
    data = get_data(symbol, start_date, end_date, data_type='fund')
    
    if data is None:
        print("数据获取失败，使用模拟数据")
        data = get_data("mock_data", start_date, end_date, provider='mock', 
                       initial_price=3.0, volatility=0.02)
    
    print(f"数据获取成功，共 {len(data)} 个交易日")
    
    # 运行回测
    engine = run_backtest(
        data=data,
        strategy_class=STRATEGIES['moving_average'],
        strategy_params={'ma_period': 20, 'printlog': False},
        initial_cash=100000.0,
        commission=0.001,
        show_log=True,
        show_plot=False
    )
    
    return engine

def example_multiple_strategies():
    """多策略对比示例"""
    print("\n=== 多策略对比示例 ===")
    
    # 获取数据
    symbol = "510300"  # 沪深300ETF
    start_date = "2022-01-01"
    end_date = "2023-12-31"
    
    data = get_data(symbol, start_date, end_date, data_type='fund')
    
    if data is None:
        print("数据获取失败，使用模拟数据")
        data = get_data("mock_data", start_date, end_date, provider='mock', 
                       initial_price=4.0, volatility=0.025)
    
    # 测试不同策略
    strategies_to_test = [
        ('buy_and_hold', '买入持有'),
        ('moving_average', '移动平均'),
        ('dual_moving_average', '双移动平均'),
        ('rsi', 'RSI策略'),
        ('macd', 'MACD策略')
    ]
    
    results = {}
    
    for strategy_key, strategy_name in strategies_to_test:
        print(f"\n测试策略: {strategy_name}")
        
        try:
            engine = run_backtest(
                data=data,
                strategy_class=STRATEGIES[strategy_key],
                strategy_params={'printlog': False},
                initial_cash=100000.0,
                commission=0.001,
                show_log=False,
                show_plot=False
            )
            
            metrics = engine.get_performance_metrics()
            results[strategy_name] = metrics
            
        except Exception as e:
            print(f"策略 {strategy_name} 回测失败: {e}")
    
    # 打印对比结果
    print("\n=== 策略对比结果 ===")
    print(f"{'策略名称':<15} {'总收益率':<10} {'夏普比率':<10} {'最大回撤':<10} {'胜率':<10}")
    print("-" * 60)
    
    for strategy_name, metrics in results.items():
        print(f"{strategy_name:<15} {metrics['total_return']:<10.2f}% "
              f"{metrics['sharpe_ratio']:<10.2f} {metrics['max_drawdown']:<10.2f}% "
              f"{metrics['win_rate']:<10.2f}%")
    
    return results

def example_custom_strategy():
    """自定义策略示例"""
    print("\n=== 自定义策略示例 ===")
    
    from backtest_framework import BaseStrategy
    import backtrader as bt
    
    class CustomStrategy(BaseStrategy):
        """自定义策略：结合移动平均和RSI"""
        
        params = (
            ('ma_period', 20),
            ('rsi_period', 14),
            ('rsi_oversold', 30),
            ('rsi_overbought', 70),
        )
        
        def __init__(self):
            super().__init__()
            
            # 创建指标
            self.sma = bt.indicators.SimpleMovingAverage(
                self.datas[0], period=self.params.ma_period
            )
            self.rsi = bt.indicators.RSI(
                self.datas[0], period=self.params.rsi_period
            )
        
        def next(self):
            if self.order:
                return
            
            if not self.position:
                # 买入条件：价格在均线上方 且 RSI不在超买区域
                if (self.dataclose[0] > self.sma[0] and 
                    self.rsi[0] < self.params.rsi_overbought):
                    cash = self.broker.getcash()
                    if cash >= 1000:
                        size = int(cash * 0.9 / self.dataclose[0])
                        if size > 0:
                            self.log(f'买入: 价格 {self.dataclose[0]:.4f}, RSI {self.rsi[0]:.2f}')
                            self.order = self.buy(size=size)
            else:
                # 卖出条件：价格在均线下方 或 RSI超买
                if (self.dataclose[0] < self.sma[0] or 
                    self.rsi[0] > self.params.rsi_overbought):
                    self.log(f'卖出: 价格 {self.dataclose[0]:.4f}, RSI {self.rsi[0]:.2f}')
                    self.order = self.sell(size=self.position.size)
    
    # 获取数据
    data = get_data("159919", "2022-01-01", "2023-12-31", data_type='fund')
    
    if data is None:
        data = get_data("mock_data", "2022-01-01", "2023-12-31", provider='mock', 
                       initial_price=2.5, volatility=0.03)
    
    # 运行自定义策略
    engine = run_backtest(
        data=data,
        strategy_class=CustomStrategy,
        strategy_params={'printlog': True},
        initial_cash=100000.0,
        commission=0.001,
        show_log=True,
        show_plot=False
    )
    
    return engine

def example_data_providers():
    """数据提供者示例"""
    print("\n=== 数据提供者示例 ===")
    
    symbol = "510050"
    start_date = "2023-01-01"
    end_date = "2023-12-31"
    
    # 测试不同数据提供者
    providers = ['akshare', 'mock']
    
    for provider in providers:
        print(f"\n使用 {provider} 数据提供者:")
        
        try:
            if provider == 'akshare':
                data = get_data(symbol, start_date, end_date, 
                              provider=provider, data_type='fund')
            else:
                data = get_data("mock_data", start_date, end_date, 
                              provider=provider, initial_price=3.0, volatility=0.02)
            
            if data is not None:
                print(f"  数据获取成功: {len(data)} 个交易日")
                print(f"  价格范围: {data['Close'].min():.4f} - {data['Close'].max():.4f}")
                print(f"  数据预览:")
                print(data.head(3))
            else:
                print("  数据获取失败")
                
        except Exception as e:
            print(f"  错误: {e}")

def main():
    """主函数"""
    print("=== 回测系统使用示例 ===")
    print("本示例演示如何使用重构后的回测框架")
    print("=" * 50)
    
    try:
        # 单策略回测
        example_single_strategy()
        
        # 多策略对比
        example_multiple_strategies()
        
        # 自定义策略
        example_custom_strategy()
        
        # 数据提供者
        example_data_providers()
        
        print("\n=== 示例完成 ===")
        print("所有示例运行完成！")
        
    except Exception as e:
        print(f"示例运行出错: {e}")
        print("请确保已安装必要的依赖包: pip install backtrader matplotlib pandas numpy akshare yfinance")

if __name__ == '__main__':
    main()
