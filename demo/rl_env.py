"""
强化学习交易环境
基于gymnasium创建自定义交易环境，定义状态空间、动作空间和奖励函数
"""

import gymnasium as gym
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from gymnasium import spaces
from feature_engineer import FeatureEngineer

logger = logging.getLogger(__name__)

class TradingEnv(gym.Env):
    """交易环境"""
    
    metadata = {"render_modes": ["human"], "render_fps": 4}
    
    def __init__(self, data: pd.DataFrame, feature_engineer: FeatureEngineer, 
                 config: Optional[Dict] = None):
        """
        初始化交易环境
        
        Args:
            data: 包含OHLCV和技术指标的DataFrame
            feature_engineer: 特征工程器
            config: 环境配置
        """
        super().__init__()
        
        self.data = data.copy()
        self.feature_engineer = feature_engineer
        self.config = config or self._get_default_config()
        
        # 环境参数
        self.initial_balance = self.config['initial_balance']
        self.commission_rate = self.config['commission_rate']
        self.max_position = self.config['max_position']
        self.lookback_window = self.config['lookback_window']
        
        # 状态和动作空间
        self._setup_spaces()
        
        # 环境状态
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0.0  # 持仓比例 [-1, 1]
        self.position_value = 0.0
        self.total_value = self.initial_balance
        self.trade_count = 0
        self.total_reward = 0.0
        
        # 历史记录
        self.value_history = []
        self.position_history = []
        self.reward_history = []
        self.action_history = []
        
        # 准备特征数据
        self._prepare_features()
        
    def _get_default_config(self) -> Dict:
        """获取默认配置"""
        return {
            'initial_balance': 100000.0,
            'commission_rate': 0.001,
            'max_position': 1.0,
            'lookback_window': 20,
            'reward_scale': 1.0,
            'penalty_scale': 0.1,
            'transaction_penalty': 0.001
        }
    
    def _setup_spaces(self):
        """设置状态和动作空间"""
        # 动作空间：连续动作 [-1, 1]，表示目标持仓比例
        self.action_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(1,), 
            dtype=np.float32
        )
        
        # 状态空间维度（将在_prepare_features中确定）
        self.observation_space = None
    
    def _prepare_features(self):
        """准备特征数据"""
        logger.info("准备特征数据...")
        
        # 计算技术指标
        self.data_with_features = self.feature_engineer.calculate_technical_indicators(self.data)
        
        # 准备RL特征
        self.feature_array, self.feature_names = self.feature_engineer.prepare_features_for_rl(
            self.data_with_features, 
            lookback_window=self.lookback_window
        )
        
        # 设置观察空间
        feature_dim = self.feature_array.shape[1] if len(self.feature_array) > 0 else 10
        # 添加账户状态特征：余额比例、持仓比例、总价值比例
        state_dim = feature_dim + 3
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(state_dim,),
            dtype=np.float32
        )
        
        logger.info(f"状态空间维度: {state_dim}")
        logger.info(f"特征维度: {feature_dim}")
        logger.info(f"数据点数量: {len(self.feature_array)}")
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """重置环境"""
        super().reset(seed=seed)
        
        # 重置环境状态
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0.0
        self.position_value = 0.0
        self.total_value = self.initial_balance
        self.trade_count = 0
        self.total_reward = 0.0
        
        # 清空历史记录
        self.value_history = []
        self.position_history = []
        self.reward_history = []
        self.action_history = []
        
        # 获取初始状态
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """执行一步"""
        # 确保动作在有效范围内
        action = np.clip(action, -1.0, 1.0)
        target_position = float(action[0])
        
        # 记录动作
        self.action_history.append(target_position)
        
        # 执行交易
        reward = self._execute_trade(target_position)
        
        # 更新环境状态
        self.current_step += 1
        self.total_reward += reward
        
        # 记录历史
        self.value_history.append(self.total_value)
        self.position_history.append(self.position)
        self.reward_history.append(reward)
        
        # 检查是否结束
        terminated = self.current_step >= len(self.feature_array) - 1
        truncated = False
        
        # 获取新状态
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _execute_trade(self, target_position: float) -> float:
        """执行交易"""
        # 计算当前价格
        current_price = self._get_current_price()
        
        # 计算目标持仓价值
        target_position_value = target_position * self.total_value
        
        # 计算需要调整的持仓
        position_change = target_position_value - self.position_value
        
        # 计算交易成本
        transaction_cost = abs(position_change) * self.commission_rate
        
        # 更新余额
        self.balance -= position_change + transaction_cost
        
        # 更新持仓
        self.position_value = target_position_value
        self.position = target_position
        
        # 更新总价值
        self.total_value = self.balance + self.position_value
        
        # 计算奖励
        reward = self._calculate_reward(position_change, transaction_cost)
        
        # 记录交易
        if abs(position_change) > 0.01:  # 避免微小交易
            self.trade_count += 1
        
        return reward
    
    def _get_current_price(self) -> float:
        """获取当前价格"""
        if self.current_step < len(self.data_with_features):
            return float(self.data_with_features.iloc[self.current_step]['Close'])
        else:
            return float(self.data_with_features.iloc[-1]['Close'])
    
    def _calculate_reward(self, position_change: float, transaction_cost: float) -> float:
        """计算奖励"""
        # 基础奖励：总价值变化率
        if len(self.value_history) > 0:
            previous_value = self.value_history[-1]
            value_change = (self.total_value - previous_value) / previous_value
        else:
            value_change = 0.0
        
        # 奖励缩放
        reward = value_change * self.config['reward_scale']
        
        # 交易成本惩罚
        reward -= transaction_cost * self.config['penalty_scale']
        
        # 过度交易惩罚
        if abs(position_change) > 0.1:  # 大幅调整持仓
            reward -= self.config['transaction_penalty']
        
        # 持仓风险惩罚（鼓励适度持仓）
        position_penalty = abs(self.position) * 0.001
        reward -= position_penalty
        
        return reward
    
    def _get_observation(self) -> np.ndarray:
        """获取当前观察"""
        if self.current_step >= len(self.feature_array):
            # 如果超出数据范围，返回最后一个观察
            feature_obs = self.feature_array[-1] if len(self.feature_array) > 0 else np.zeros(10)
        else:
            feature_obs = self.feature_array[self.current_step]
        
        # 添加账户状态特征
        balance_ratio = self.balance / self.initial_balance
        position_ratio = self.position
        value_ratio = self.total_value / self.initial_balance
        
        account_state = np.array([balance_ratio, position_ratio, value_ratio], dtype=np.float32)
        
        # 组合特征
        observation = np.concatenate([feature_obs, account_state])
        
        return observation.astype(np.float32)
    
    def _get_info(self) -> Dict:
        """获取环境信息"""
        return {
            'step': self.current_step,
            'balance': self.balance,
            'position': self.position,
            'position_value': self.position_value,
            'total_value': self.total_value,
            'trade_count': self.trade_count,
            'total_reward': self.total_reward,
            'current_price': self._get_current_price()
        }
    
    def render(self, mode: str = "human"):
        """渲染环境"""
        if mode == "human":
            info = self._get_info()
            print(f"Step: {info['step']}, "
                  f"Balance: {info['balance']:.2f}, "
                  f"Position: {info['position']:.3f}, "
                  f"Total Value: {info['total_value']:.2f}, "
                  f"Trades: {info['trade_count']}")
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """获取性能指标"""
        if len(self.value_history) < 2:
            return {}
        
        values = np.array(self.value_history)
        returns = np.diff(values) / values[:-1]
        
        # 总收益率
        total_return = (values[-1] - values[0]) / values[0]
        
        # 年化收益率（假设252个交易日）
        days = len(values)
        annualized_return = (1 + total_return) ** (252 / days) - 1
        
        # 夏普比率
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
        else:
            sharpe_ratio = 0.0
        
        # 最大回撤
        peak = np.maximum.accumulate(values)
        drawdown = (values - peak) / peak
        max_drawdown = np.min(drawdown)
        
        # 胜率
        positive_returns = returns[returns > 0]
        win_rate = len(positive_returns) / len(returns) if len(returns) > 0 else 0.0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': self.trade_count,
            'final_value': values[-1]
        }


class TradingEnvWrapper:
    """交易环境包装器，提供额外的功能"""
    
    def __init__(self, env: TradingEnv):
        self.env = env
    
    def train_test_split(self, train_ratio: float = 0.8) -> Tuple[TradingEnv, TradingEnv]:
        """分割训练和测试环境"""
        split_idx = int(len(self.env.feature_array) * train_ratio)
        
        # 训练数据
        train_data = self.env.data_with_features.iloc[:split_idx + self.env.lookback_window]
        train_env = TradingEnv(
            train_data,
            self.env.feature_engineer,
            self.env.config
        )
        
        # 测试数据
        test_data = self.env.data_with_features.iloc[split_idx:]
        test_env = TradingEnv(
            test_data,
            self.env.feature_engineer,
            self.env.config
        )
        
        return train_env, test_env
    
    def get_data_summary(self) -> Dict:
        """获取数据摘要"""
        return {
            'total_steps': len(self.env.feature_array),
            'date_range': f"{self.env.data_with_features.index[0]} to {self.env.data_with_features.index[-1]}",
            'feature_dim': self.env.feature_array.shape[1] if len(self.env.feature_array) > 0 else 0,
            'price_range': f"{self.env.data_with_features['Close'].min():.2f} - {self.env.data_with_features['Close'].max():.2f}",
            'volatility': self.env.data_with_features['Close'].pct_change().std() * np.sqrt(252)
        }


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
    
    # 创建特征工程器
    fe = FeatureEngineer()
    
    # 创建交易环境
    env = TradingEnv(test_data, fe)
    
    print("环境信息:")
    print(f"状态空间: {env.observation_space}")
    print(f"动作空间: {env.action_space}")
    
    # 测试环境
    obs, info = env.reset()
    print(f"\n初始状态: {obs[:5]}...")
    print(f"初始信息: {info}")
    
    # 运行几步
    for i in range(5):
        action = env.action_space.sample()  # 随机动作
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1}: Action={action[0]:.3f}, Reward={reward:.4f}, Value={info['total_value']:.2f}")
        
        if terminated or truncated:
            break
    
    # 获取性能指标
    metrics = env.get_performance_metrics()
    print(f"\n性能指标: {metrics}")


if __name__ == "__main__":
    main()
