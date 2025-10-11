"""
常用基线模型
提供各种常用的量化交易策略
"""

import backtrader as bt
import numpy as np
from backtest_framework import BaseStrategy

class BuyAndHoldStrategy(BaseStrategy):
    """买入持有策略（基准策略）"""
    
    def __init__(self):
        super().__init__()
    
    def next(self):
        """策略逻辑"""
        if not self.position:
            # 在第一天买入
            cash = self.broker.getcash()
            size = int(cash * 0.95 / self.dataclose[0])
            if size > 0:
                self.order = self.buy(size=size)

class MovingAverageStrategy(BaseStrategy):
    """移动平均策略"""
    
    params = (
        ('ma_period', 20),  # 移动平均周期
        ('min_trade_amount', 1000),  # 最小交易金额
    )
    
    def __init__(self):
        super().__init__()
        
        # 创建移动平均线指标
        self.sma = bt.indicators.SimpleMovingAverage(
            self.datas[0], period=self.params.ma_period
        )
    
    def next(self):
        """策略逻辑"""
        # 检查是否有未完成的订单
        if self.order:
            return
        
        # 检查是否持有仓位
        if not self.position:
            # 没有仓位，检查买入信号
            if self.dataclose[0] > self.sma[0]:
                cash = self.broker.getcash()
                if cash >= self.params.min_trade_amount:
                    size = int(cash * 0.9 / self.dataclose[0])
                    if size > 0:
                        self.log(f'买入信号: 价格 {self.dataclose[0]:.4f}, MA {self.sma[0]:.4f}')
                        self.order = self.buy(size=size)
        else:
            # 有仓位，检查卖出信号
            if self.dataclose[0] < self.sma[0]:
                self.log(f'卖出信号: 价格 {self.dataclose[0]:.4f}, MA {self.sma[0]:.4f}')
                self.order = self.sell(size=self.position.size)

class DualMovingAverageStrategy(BaseStrategy):
    """双移动平均策略"""
    
    params = (
        ('fast_period', 10),  # 快速移动平均周期
        ('slow_period', 30),  # 慢速移动平均周期
        ('min_trade_amount', 1000),  # 最小交易金额
    )
    
    def __init__(self):
        super().__init__()
        
        # 创建双移动平均线指标
        self.fast_ma = bt.indicators.SimpleMovingAverage(
            self.datas[0], period=self.params.fast_period
        )
        self.slow_ma = bt.indicators.SimpleMovingAverage(
            self.datas[0], period=self.params.slow_period
        )
    
    def next(self):
        """策略逻辑"""
        if self.order:
            return
        
        if not self.position:
            # 买入信号：快速均线上穿慢速均线
            if self.fast_ma[0] > self.slow_ma[0] and self.fast_ma[-1] <= self.slow_ma[-1]:
                cash = self.broker.getcash()
                if cash >= self.params.min_trade_amount:
                    size = int(cash * 0.9 / self.dataclose[0])
                    if size > 0:
                        self.log(f'买入信号: 快MA {self.fast_ma[0]:.4f}, 慢MA {self.slow_ma[0]:.4f}')
                        self.order = self.buy(size=size)
        else:
            # 卖出信号：快速均线下穿慢速均线
            if self.fast_ma[0] < self.slow_ma[0] and self.fast_ma[-1] >= self.slow_ma[-1]:
                self.log(f'卖出信号: 快MA {self.fast_ma[0]:.4f}, 慢MA {self.slow_ma[0]:.4f}')
                self.order = self.sell(size=self.position.size)

class RSIStrategy(BaseStrategy):
    """RSI策略"""
    
    params = (
        ('rsi_period', 14),  # RSI周期
        ('rsi_oversold', 30),  # 超卖阈值
        ('rsi_overbought', 70),  # 超买阈值
        ('min_trade_amount', 1000),  # 最小交易金额
    )
    
    def __init__(self):
        super().__init__()
        
        # 创建RSI指标
        self.rsi = bt.indicators.RSI(self.datas[0], period=self.params.rsi_period)
    
    def next(self):
        """策略逻辑"""
        if self.order:
            return
        
        if not self.position:
            # 买入信号：RSI从超卖区域回升
            if self.rsi[0] > self.params.rsi_oversold and self.rsi[-1] <= self.params.rsi_oversold:
                cash = self.broker.getcash()
                if cash >= self.params.min_trade_amount:
                    size = int(cash * 0.9 / self.dataclose[0])
                    if size > 0:
                        self.log(f'买入信号: RSI {self.rsi[0]:.2f}')
                        self.order = self.buy(size=size)
        else:
            # 卖出信号：RSI从超买区域回落
            if self.rsi[0] < self.params.rsi_overbought and self.rsi[-1] >= self.params.rsi_overbought:
                self.log(f'卖出信号: RSI {self.rsi[0]:.2f}')
                self.order = self.sell(size=self.position.size)

class MACDStrategy(BaseStrategy):
    """MACD策略"""
    
    params = (
        ('macd_fast', 12),  # MACD快线周期
        ('macd_slow', 26),  # MACD慢线周期
        ('macd_signal', 9),  # MACD信号线周期
        ('min_trade_amount', 1000),  # 最小交易金额
    )
    
    def __init__(self):
        super().__init__()
        
        # 创建MACD指标
        self.macd = bt.indicators.MACD(
            self.datas[0], 
            period_me1=self.params.macd_fast,
            period_me2=self.params.macd_slow,
            period_signal=self.params.macd_signal
        )
    
    def next(self):
        """策略逻辑"""
        if self.order:
            return
        
        if not self.position:
            # 买入信号：MACD线上穿信号线
            if (self.macd.macd[0] > self.macd.signal[0] and 
                self.macd.macd[-1] <= self.macd.signal[-1]):
                cash = self.broker.getcash()
                if cash >= self.params.min_trade_amount:
                    size = int(cash * 0.9 / self.dataclose[0])
                    if size > 0:
                        self.log(f'买入信号: MACD {self.macd.macd[0]:.4f}, Signal {self.macd.signal[0]:.4f}')
                        self.order = self.buy(size=size)
        else:
            # 卖出信号：MACD线下穿信号线
            if (self.macd.macd[0] < self.macd.signal[0] and 
                self.macd.macd[-1] >= self.macd.signal[-1]):
                self.log(f'卖出信号: MACD {self.macd.macd[0]:.4f}, Signal {self.macd.signal[0]:.4f}')
                self.order = self.sell(size=self.position.size)

class BollingerBandsStrategy(BaseStrategy):
    """布林带策略"""
    
    params = (
        ('bb_period', 20),  # 布林带周期
        ('bb_dev', 2),  # 布林带标准差倍数
        ('min_trade_amount', 1000),  # 最小交易金额
    )
    
    def __init__(self):
        super().__init__()
        
        # 创建布林带指标
        self.bollinger = bt.indicators.BollingerBands(
            self.datas[0], 
            period=self.params.bb_period,
            devfactor=self.params.bb_dev
        )
    
    def next(self):
        """策略逻辑"""
        if self.order:
            return
        
        if not self.position:
            # 买入信号：价格触及下轨后反弹
            if (self.dataclose[0] <= self.bollinger.lines.bot[0] and 
                self.dataclose[-1] > self.bollinger.lines.bot[-1]):
                cash = self.broker.getcash()
                if cash >= self.params.min_trade_amount:
                    size = int(cash * 0.9 / self.dataclose[0])
                    if size > 0:
                        self.log(f'买入信号: 价格 {self.dataclose[0]:.4f}, 下轨 {self.bollinger.lines.bot[0]:.4f}')
                        self.order = self.buy(size=size)
        else:
            # 卖出信号：价格触及上轨后回落
            if (self.dataclose[0] >= self.bollinger.lines.top[0] and 
                self.dataclose[-1] < self.bollinger.lines.top[-1]):
                self.log(f'卖出信号: 价格 {self.dataclose[0]:.4f}, 上轨 {self.bollinger.lines.top[0]:.4f}')
                self.order = self.sell(size=self.position.size)

class MeanReversionStrategy(BaseStrategy):
    """均值回归策略"""
    
    params = (
        ('lookback_period', 20),  # 回望周期
        ('deviation_threshold', 2),  # 偏离阈值（标准差倍数）
        ('min_trade_amount', 1000),  # 最小交易金额
    )
    
    def __init__(self):
        super().__init__()
        
        # 创建移动平均和标准差指标
        self.sma = bt.indicators.SimpleMovingAverage(
            self.datas[0], period=self.params.lookback_period
        )
        self.std = bt.indicators.StandardDeviation(
            self.datas[0], period=self.params.lookback_period
        )
    
    def next(self):
        """策略逻辑"""
        if self.order:
            return
        
        if not self.position:
            # 买入信号：价格显著低于均值
            lower_bound = self.sma[0] - self.params.deviation_threshold * self.std[0]
            if self.dataclose[0] < lower_bound:
                cash = self.broker.getcash()
                if cash >= self.params.min_trade_amount:
                    size = int(cash * 0.9 / self.dataclose[0])
                    if size > 0:
                        self.log(f'买入信号: 价格 {self.dataclose[0]:.4f}, 下界 {lower_bound:.4f}')
                        self.order = self.buy(size=size)
        else:
            # 卖出信号：价格回归到均值附近
            if self.dataclose[0] >= self.sma[0]:
                self.log(f'卖出信号: 价格 {self.dataclose[0]:.4f}, 均值 {self.sma[0]:.4f}')
                self.order = self.sell(size=self.position.size)

class MomentumStrategy(BaseStrategy):
    """动量策略"""
    
    params = (
        ('momentum_period', 10),  # 动量周期
        ('momentum_threshold', 0.02),  # 动量阈值（2%）
        ('min_trade_amount', 1000),  # 最小交易金额
    )
    
    def __init__(self):
        super().__init__()
        
        # 创建动量指标
        self.momentum = bt.indicators.Momentum(
            self.datas[0], period=self.params.momentum_period
        )
    
    def next(self):
        """策略逻辑"""
        if self.order:
            return
        
        if not self.position:
            # 买入信号：正动量且超过阈值
            if self.momentum[0] > self.params.momentum_threshold:
                cash = self.broker.getcash()
                if cash >= self.params.min_trade_amount:
                    size = int(cash * 0.9 / self.dataclose[0])
                    if size > 0:
                        self.log(f'买入信号: 动量 {self.momentum[0]:.4f}')
                        self.order = self.buy(size=size)
        else:
            # 卖出信号：动量转负
            if self.momentum[0] < 0:
                self.log(f'卖出信号: 动量 {self.momentum[0]:.4f}')
                self.order = self.sell(size=self.position.size)

# 策略字典，方便使用
STRATEGIES = {
    'buy_and_hold': BuyAndHoldStrategy,
    'moving_average': MovingAverageStrategy,
    'dual_moving_average': DualMovingAverageStrategy,
    'rsi': RSIStrategy,
    'macd': MACDStrategy,
    'bollinger_bands': BollingerBandsStrategy,
    'mean_reversion': MeanReversionStrategy,
    'momentum': MomentumStrategy,
}
