"""
BackTrader策略集成模块
将训练好的RL模型集成到BackTrader框架中，实现统一的回测接口
"""

import os
import numpy as np
import pandas as pd
import backtrader as bt
from typing import Dict, Optional, Any, List
import logging
import pickle
from datetime import datetime

from stable_baselines3 import PPO
from feature_engineer import FeatureEngineer
from rl_env import TradingEnv

logger = logging.getLogger(__name__)

class RLStrategy(bt.Strategy):
    """基于强化学习的交易策略"""
    
    params = (
        ('model_path', None),  # RL模型路径
        ('lookback_window', 20),  # 回望窗口
        ('position_threshold', 0.1),  # 仓位调整阈值
        ('printlog', False),  # 是否打印日志
    )
    
    def __init__(self):
        """初始化策略"""
        # 基础数据
        self.dataclose = self.datas[0].close
        self.dataopen = self.datas[0].open
        self.datahigh = self.datas[0].high
        self.datalow = self.datas[0].low
        self.datavolume = self.datas[0].volume
        
        # 订单管理
        self.order = None
        self.target_position = 0.0
        self.current_position = 0.0
        
        # RL模型相关
        self.model = None
        self.feature_engineer = None
        self.feature_data = None
        self.feature_array = None
        self.current_step = 0
        
        # 性能跟踪
        self.trade_count = 0
        self.position_history = []
        self.value_history = []
        
        # 初始化RL模型
        self._load_rl_model()
        
        # 延迟特征准备，在数据准备好后再进行
        self.feature_data = None
        self.feature_array = None
    
    def _load_rl_model(self):
        """加载RL模型"""
        if self.params.model_path and os.path.exists(self.params.model_path):
            try:
                logger.info(f"加载RL模型: {self.params.model_path}")
                self.model = PPO.load(self.params.model_path)
                
                # 加载特征工程器
                config_path = self.params.model_path.replace('.zip', '_config.pkl')
                if os.path.exists(config_path):
                    with open(config_path, 'rb') as f:
                        config = pickle.load(f)
                        self.feature_engineer = FeatureEngineer(config.get('feature_config'))
                else:
                    self.feature_engineer = FeatureEngineer()
                
                logger.info("RL模型加载成功")
            except Exception as e:
                logger.error(f"加载RL模型失败: {e}")
                self.model = None
        else:
            logger.warning("未提供有效的RL模型路径")
            self.model = None
    
    def _prepare_features(self):
        """准备特征数据"""
        if self.model is None:
            logger.warning("模型未加载，跳过特征准备")
            return
        
        try:
            logger.info("开始准备特征数据...")
            
            # 获取历史数据
            logger.info("获取历史数据...")
            data = pd.DataFrame({
                'Open': [self.dataopen[i] for i in range(len(self.datas[0]))],
                'High': [self.datahigh[i] for i in range(len(self.datas[0]))],
                'Low': [self.datalow[i] for i in range(len(self.datas[0]))],
                'Close': [self.dataclose[i] for i in range(len(self.datas[0]))],
                'Volume': [self.datavolume[i] for i in range(len(self.datas[0]))]
            })
            logger.info(f"历史数据获取完成: {data.shape}")
            
            # 计算技术指标
            logger.info("计算技术指标...")
            self.feature_data = self.feature_engineer.calculate_technical_indicators(data)
            logger.info(f"技术指标计算完成: {self.feature_data.shape}")
            
            # 准备RL特征
            logger.info("准备RL特征...")
            self.feature_array, _ = self.feature_engineer.prepare_features_for_rl(
                self.feature_data, 
                lookback_window=self.params.lookback_window
            )
            logger.info(f"RL特征准备完成: {self.feature_array.shape}")
            
            logger.info("特征数据准备完成")
            
        except Exception as e:
            logger.error(f"准备特征数据失败: {e}")
            import traceback
            traceback.print_exc()
            self.feature_array = None
    
    def next(self):
        """策略主逻辑"""
        # 检查是否有未完成的订单
        if self.order:
            return
        
        # 延迟准备特征数据（在第一次运行时）
        if self.feature_array is None:
            self._prepare_features()
            if self.feature_array is None:
                logger.warning("特征数据准备失败，跳过本次交易")
                return
        
        # 检查是否有足够的特征数据
        if self.current_step >= len(self.feature_array):
            return
        
        try:
            # 获取当前观察
            observation = self._get_current_observation()
            
            if observation is None:
                return
            
            # RL模型预测
            action, _ = self.model.predict(observation, deterministic=True)
            target_position = float(action[0])
            
            # 限制目标仓位
            target_position = np.clip(target_position, -1.0, 1.0)
            
            # 检查是否需要调整仓位
            position_diff = abs(target_position - self.current_position)
            
            if position_diff > self.params.position_threshold:
                self._adjust_position(target_position)
            
            # 更新当前步数
            self.current_step += 1
            
        except Exception as e:
            logger.error(f"策略执行错误: {e}")
    
    def _get_current_observation(self) -> Optional[np.ndarray]:
        """获取当前观察"""
        if self.feature_array is None or self.current_step >= len(self.feature_array):
            return None
        
        # 获取特征观察
        feature_obs = self.feature_array[self.current_step]
        
        # 添加账户状态特征
        balance = self.broker.getcash()
        position_value = self.broker.getvalue() - balance
        total_value = self.broker.getvalue()
        
        # 计算账户状态特征
        balance_ratio = balance / self.broker.startingcash
        position_ratio = self.current_position
        value_ratio = total_value / self.broker.startingcash
        
        account_state = np.array([balance_ratio, position_ratio, value_ratio], dtype=np.float32)
        
        # 组合观察
        observation = np.concatenate([feature_obs, account_state])
        
        return observation.astype(np.float32)
    
    def _adjust_position(self, target_position: float):
        """调整仓位"""
        current_value = self.broker.getvalue()
        current_price = self.dataclose[0]
        
        # 计算目标仓位价值
        target_position_value = target_position * current_value
        
        # 计算当前仓位价值
        current_position_value = self.current_position * current_value
        
        # 计算需要调整的金额
        position_change = target_position_value - current_position_value
        
        if abs(position_change) < current_value * 0.01:  # 小于1%的调整忽略
            return
        
        # 执行交易
        if position_change > 0:
            # 买入
            size = int(position_change / current_price)
            if size > 0:
                self.order = self.buy(size=size)
                self.log(f'RL买入信号: 目标仓位 {target_position:.3f}, 数量 {size}')
        else:
            # 卖出
            size = int(abs(position_change) / current_price)
            if size > 0 and self.current_position > 0:
                self.order = self.sell(size=size)
                self.log(f'RL卖出信号: 目标仓位 {target_position:.3f}, 数量 {size}')
        
        # 更新目标仓位
        self.target_position = target_position
    
    def notify_order(self, order):
        """订单状态通知"""
        if order.status in [order.Submitted, order.Accepted]:
            return
        
        if order.status in [order.Completed]:
            if order.isbuy():
                self.current_position += order.executed.size / (self.broker.getvalue() / self.dataclose[0])
                self.log(f'买入执行: 价格 {order.executed.price:.2f}, '
                        f'数量 {order.executed.size:.0f}, '
                        f'金额 {order.executed.value:.2f}')
            else:
                self.current_position -= order.executed.size / (self.broker.getvalue() / self.dataclose[0])
                self.log(f'卖出执行: 价格 {order.executed.price:.2f}, '
                        f'数量 {order.executed.size:.0f}, '
                        f'金额 {order.executed.value:.2f}')
            
            self.trade_count += 1
        
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('订单取消/资金不足/拒绝')
        
        self.order = None
    
    def notify_trade(self, trade):
        """交易通知"""
        if not trade.isclosed:
            return
        
        self.log(f'交易完成: 利润 {trade.pnl:.2f}, 净利润 {trade.pnlcomm:.2f}')
    
    def log(self, txt, dt=None):
        """日志函数"""
        if self.params.printlog:
            dt = dt or self.datas[0].datetime.date(0)
            print(f'{dt.isoformat()}: {txt}')
    
    def stop(self):
        """策略结束时的处理"""
        self.log(f'RL策略结束: 总价值 {self.broker.getvalue():.2f}, '
                f'交易次数 {self.trade_count}, '
                f'最终仓位 {self.current_position:.3f}')


class BuyAndHoldStrategy(bt.Strategy):
    """买入持有策略（基线策略）"""
    
    params = (
        ('printlog', False),
    )
    
    def __init__(self):
        """初始化策略"""
        self.dataclose = self.datas[0].close
        self.order = None
        self.bought = False
    
    def next(self):
        """策略逻辑"""
        if not self.bought:
            # 在第一天买入
            cash = self.broker.getcash()
            size = int(cash * 0.95 / self.dataclose[0])
            if size > 0:
                self.order = self.buy(size=size)
                self.bought = True
                self.log(f'买入持有: 价格 {self.dataclose[0]:.2f}, 数量 {size}')
    
    def notify_order(self, order):
        """订单状态通知"""
        if order.status in [order.Submitted, order.Accepted]:
            return
        
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'买入执行: 价格 {order.executed.price:.2f}, '
                        f'数量 {order.executed.size:.0f}, '
                        f'金额 {order.executed.value:.2f}')
        
        self.order = None
    
    def log(self, txt, dt=None):
        """日志函数"""
        if self.params.printlog:
            dt = dt or self.datas[0].datetime.date(0)
            print(f'{dt.isoformat()}: {txt}')
    
    def stop(self):
        """策略结束时的处理"""
        self.log(f'买入持有策略结束: 总价值 {self.broker.getvalue():.2f}')


class MovingAverageStrategy(bt.Strategy):
    """移动平均策略（另一个基线策略）"""
    
    params = (
        ('ma_period', 20),
        ('printlog', False),
    )
    
    def __init__(self):
        """初始化策略"""
        self.dataclose = self.datas[0].close
        self.order = None
        
        # 创建移动平均线指标
        self.sma = bt.indicators.SimpleMovingAverage(
            self.datas[0], period=self.params.ma_period
        )
    
    def next(self):
        """策略逻辑"""
        if self.order:
            return
        
        if not self.position:
            # 买入信号：价格上穿移动平均线
            if self.dataclose[0] > self.sma[0]:
                cash = self.broker.getcash()
                size = int(cash * 0.9 / self.dataclose[0])
                if size > 0:
                    self.order = self.buy(size=size)
                    self.log(f'MA买入: 价格 {self.dataclose[0]:.2f}, MA {self.sma[0]:.2f}')
        else:
            # 卖出信号：价格下穿移动平均线
            if self.dataclose[0] < self.sma[0]:
                self.order = self.sell(size=self.position.size)
                self.log(f'MA卖出: 价格 {self.dataclose[0]:.2f}, MA {self.sma[0]:.2f}')
    
    def notify_order(self, order):
        """订单状态通知"""
        if order.status in [order.Submitted, order.Accepted]:
            return
        
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'MA买入执行: 价格 {order.executed.price:.2f}')
            else:
                self.log(f'MA卖出执行: 价格 {order.executed.price:.2f}')
        
        self.order = None
    
    def log(self, txt, dt=None):
        """日志函数"""
        if self.params.printlog:
            dt = dt or self.datas[0].datetime.date(0)
            print(f'{dt.isoformat()}: {txt}')
    
    def stop(self):
        """策略结束时的处理"""
        self.log(f'MA策略结束: 总价值 {self.broker.getvalue():.2f}')


class BacktestRunner:
    """回测运行器"""
    
    def __init__(self, initial_cash: float = 100000.0, commission: float = 0.001):
        """
        初始化回测运行器
        
        Args:
            initial_cash: 初始资金
            commission: 手续费率
        """
        self.initial_cash = initial_cash
        self.commission = commission
        self.cerebro = bt.Cerebro()
        
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
    
    def add_data(self, data: pd.DataFrame, name: str = "data"):
        """添加数据"""
        # 确保数据格式正确
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"数据缺少必需的列: {col}")
        
        # 转换为backtrader数据格式
        data_feed = bt.feeds.PandasData(
            dataname=data,
            datetime=None,
            open='Open',
            high='High',
            low='Low',
            close='Close',
            volume='Volume',
            openinterest=-1
        )
        
        self.cerebro.adddata(data_feed, name=name)
    
    def add_strategy(self, strategy_class, **kwargs):
        """添加策略"""
        self.cerebro.addstrategy(strategy_class, **kwargs)
    
    def run(self, show_log: bool = False) -> Dict[str, Any]:
        """运行回测"""
        if show_log:
            initial_value = self.cerebro.broker.getvalue() or 0
            print(f'初始资金: {initial_value:.2f}')
            print("开始回测...")
        
        results = self.cerebro.run()
        
        if show_log:
            final_value = self.cerebro.broker.getvalue() or 0
            print(f'最终资金: {final_value:.2f}')
        
        # 提取结果
        strat = results[0]
        
        # 获取分析结果
        sharpe_analysis = strat.analyzers.sharpe.get_analysis()
        drawdown_analysis = strat.analyzers.drawdown.get_analysis()
        returns_analysis = strat.analyzers.returns.get_analysis()
        trades_analysis = strat.analyzers.trades.get_analysis()
        
        # 计算基本指标
        final_value = self.cerebro.broker.getvalue() or self.initial_cash
        total_return = (final_value - self.initial_cash) / self.initial_cash * 100
        
        metrics = {
            'initial_cash': self.initial_cash,
            'final_value': final_value,
            'total_return': total_return,
            'sharpe_ratio': sharpe_analysis.get("sharperatio") or 0,
            'max_drawdown': drawdown_analysis.get("max", {}).get("drawdown") or 0,
            'total_trades': trades_analysis.get("total", {}).get("total") or 0,
            'won_trades': trades_analysis.get("won", {}).get("total") or 0,
            'lost_trades': trades_analysis.get("lost", {}).get("total") or 0,
            'win_rate': 0
        }
        
        # 计算胜率
        if metrics['total_trades'] > 0:
            metrics['win_rate'] = metrics['won_trades'] / metrics['total_trades'] * 100
        
        return {
            'results': results,
            'metrics': metrics,
            'cerebro': self.cerebro
        }
    
    def plot(self, style: str = 'candlestick', show: bool = True):
        """绘制回测结果"""
        self.cerebro.plot(style=style, barup='green', bardown='red')
        
        if show:
            import matplotlib.pyplot as plt
            plt.title('回测结果可视化')
            plt.show()


def main():
    """测试函数"""
    # 创建测试数据
    dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')
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
    runner = BacktestRunner()
    runner.add_data(test_data)
    runner.add_strategy(BuyAndHoldStrategy, printlog=True)
    
    results = runner.run(show_log=True)
    print(f"买入持有策略结果: {results['metrics']}")
    
    # 测试移动平均策略
    print("\n测试移动平均策略...")
    runner2 = BacktestRunner()
    runner2.add_data(test_data)
    runner2.add_strategy(MovingAverageStrategy, printlog=True)
    
    results2 = runner2.run(show_log=True)
    print(f"移动平均策略结果: {results2['metrics']}")


if __name__ == "__main__":
    main()
