"""
基本回测框架
提供统一的回测接口和分析功能
"""

import backtrader as bt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, Any, Optional, List

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class BaseBacktestEngine:
    """基础回测引擎"""
    
    def __init__(self, initial_cash: float = 100000.0, commission: float = 0.001):
        """
        初始化回测引擎
        
        Args:
            initial_cash: 初始资金
            commission: 手续费率
        """
        self.cerebro = bt.Cerebro()
        self.initial_cash = initial_cash
        self.commission = commission
        self.results = None
        
        # 设置初始资金和手续费
        self.cerebro.broker.setcash(initial_cash)
        self.cerebro.broker.setcommission(commission=commission)
        
        # 添加分析器
        self._add_analyzers()
    
    def _add_analyzers(self):
        """添加分析器"""
        self.cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        self.cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        self.cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        self.cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        self.cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='timereturn')
    
    def add_strategy(self, strategy_class, **kwargs):
        """
        添加策略
        
        Args:
            strategy_class: 策略类
            **kwargs: 策略参数
        """
        self.cerebro.addstrategy(strategy_class, **kwargs)
    
    def add_data(self, data: pd.DataFrame, name: str = "data"):
        """
        添加数据
        
        Args:
            data: 包含OHLCV数据的DataFrame
            name: 数据名称
        """
        # 确保数据格式正确
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"数据缺少必需的列: {col}")
        
        # 转换为backtrader数据格式
        data_feed = bt.feeds.PandasData(
            dataname=data,
            datetime=None,  # 使用索引作为日期
            open='Open',
            high='High',
            low='Low',
            close='Close',
            volume='Volume',
            openinterest=-1
        )
        
        self.cerebro.adddata(data_feed, name=name)
    
    def run(self, show_log: bool = False):
        """
        运行回测
        
        Args:
            show_log: 是否显示详细日志
            
        Returns:
            回测结果
        """
        if show_log:
            print(f'初始资金: {self.cerebro.broker.getvalue():.2f}')
            print("开始回测...")
        
        self.results = self.cerebro.run()
        
        if show_log:
            print(f'最终资金: {self.cerebro.broker.getvalue():.2f}')
        
        return self.results
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        获取性能指标
        
        Returns:
            包含各种性能指标的字典
        """
        if self.results is None:
            raise ValueError("请先运行回测")
        
        strat = self.results[0]
        
        # 获取分析结果
        sharpe_analysis = strat.analyzers.sharpe.get_analysis()
        drawdown_analysis = strat.analyzers.drawdown.get_analysis()
        returns_analysis = strat.analyzers.returns.get_analysis()
        trades_analysis = strat.analyzers.trades.get_analysis()
        
        # 计算基本指标
        final_value = self.cerebro.broker.getvalue()
        total_return = (final_value - self.initial_cash) / self.initial_cash * 100
        
        metrics = {
            'initial_cash': self.initial_cash,
            'final_value': final_value,
            'total_return': total_return,
            'sharpe_ratio': sharpe_analysis.get("sharperatio", 0),
            'max_drawdown': drawdown_analysis.get("max", {}).get("drawdown", 0),
            'total_trades': trades_analysis.get("total", {}).get("total", 0),
            'won_trades': trades_analysis.get("won", {}).get("total", 0),
            'lost_trades': trades_analysis.get("lost", {}).get("total", 0),
            'win_rate': 0
        }
        
        # 计算胜率
        if metrics['total_trades'] > 0:
            metrics['win_rate'] = metrics['won_trades'] / metrics['total_trades'] * 100
        
        return metrics
    
    def print_performance_summary(self):
        """打印性能摘要"""
        metrics = self.get_performance_metrics()
        
        print('\n=== 回测结果分析 ===')
        print(f'初始资金: {metrics["initial_cash"]:.2f}')
        print(f'最终资金: {metrics["final_value"]:.2f}')
        print(f'总收益率: {metrics["total_return"]:.2f}%')
        print(f'夏普比率: {metrics["sharpe_ratio"]:.2f}')
        print(f'最大回撤: {metrics["max_drawdown"]:.2f}%')
        print(f'总交易次数: {metrics["total_trades"]}')
        print(f'盈利交易: {metrics["won_trades"]}')
        print(f'亏损交易: {metrics["lost_trades"]}')
        print(f'胜率: {metrics["win_rate"]:.2f}%')
    
    def plot(self, style: str = 'candlestick', show: bool = True):
        """
        绘制回测结果图表
        
        Args:
            style: 图表样式
            show: 是否显示图表
        """
        print("\n正在生成可视化图表...")
        self.cerebro.plot(style=style, barup='green', bardown='red')
        
        if show:
            plt.title('回测结果可视化')
            plt.show()

class BaseStrategy(bt.Strategy):
    """基础策略类"""
    
    params = (
        ('printlog', False),  # 是否打印日志
    )
    
    def __init__(self):
        """初始化策略"""
        self.dataclose = self.datas[0].close
        self.dataopen = self.datas[0].open
        self.datahigh = self.datas[0].high
        self.datalow = self.datas[0].low
        self.datavolume = self.datas[0].volume
        self.order = None
    
    def notify_order(self, order):
        """订单状态通知"""
        if order.status in [order.Submitted, order.Accepted]:
            return
        
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'买入: 价格 {order.executed.price:.4f}, '
                        f'数量 {order.executed.size:.2f}, '
                        f'金额 {order.executed.value:.2f}, '
                        f'手续费 {order.executed.comm:.2f}')
            else:
                self.log(f'卖出: 价格 {order.executed.price:.4f}, '
                        f'数量 {order.executed.size:.2f}, '
                        f'金额 {order.executed.value:.2f}, '
                        f'手续费 {order.executed.comm:.2f}')
        
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('订单取消/资金不足/拒绝')
        
        self.order = None
    
    def notify_trade(self, trade):
        """交易通知"""
        if not trade.isclosed:
            return
        
        self.log(f'交易完成: 利润 {trade.pnl:.2f}')
    
    def log(self, txt, dt=None):
        """日志函数"""
        if self.params.printlog:
            dt = dt or self.datas[0].datetime.date(0)
            print(f'{dt.isoformat()}: {txt}')
    
    def stop(self):
        """策略结束时的处理"""
        self.log(f'期末总价值: {self.broker.getvalue():.2f}')

def run_backtest(data: pd.DataFrame, strategy_class, strategy_params: Dict = None, 
                initial_cash: float = 100000.0, commission: float = 0.001,
                show_log: bool = False, show_plot: bool = True) -> BaseBacktestEngine:
    """
    运行回测的便捷函数
    
    Args:
        data: 包含OHLCV数据的DataFrame
        strategy_class: 策略类
        strategy_params: 策略参数
        initial_cash: 初始资金
        commission: 手续费率
        show_log: 是否显示详细日志
        show_plot: 是否显示图表
        
    Returns:
        回测引擎实例
    """
    # 创建回测引擎
    engine = BaseBacktestEngine(initial_cash=initial_cash, commission=commission)
    
    # 添加策略
    if strategy_params:
        engine.add_strategy(strategy_class, **strategy_params)
    else:
        engine.add_strategy(strategy_class)
    
    # 添加数据
    engine.add_data(data)
    
    # 运行回测
    engine.run(show_log=show_log)
    
    # 打印结果
    engine.print_performance_summary()
    
    # 绘制图表
    if show_plot:
        engine.plot()
    
    return engine
