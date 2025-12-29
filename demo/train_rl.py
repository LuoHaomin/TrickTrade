"""
PPO强化学习策略训练模块
使用Stable-Baselines3的PPO算法训练基金投资策略
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List
import logging
from datetime import datetime
import pickle

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.results_plotter import load_results, ts2xy

import matplotlib.pyplot as plt
import seaborn as sns

from data_fetcher import ETFDataFetcher
from feature_engineer import FeatureEngineer
from rl_env import TradingEnv, TradingEnvWrapper

logger = logging.getLogger(__name__)

class TrainingCallback(BaseCallback):
    """训练回调函数"""
    
    def __init__(self, eval_env, eval_freq: int = 10000, verbose: int = 1):
        super(TrainingCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.best_mean_reward = -np.inf
        
    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            # 评估模型
            mean_reward, _ = self._evaluate_model()
            
            if self.verbose > 0:
                print(f"Step {self.n_calls}: Mean reward = {mean_reward:.2f}")
            
            # 保存最佳模型
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                if self.verbose > 0:
                    print(f"New best model! Mean reward: {mean_reward:.2f}")
        
        return True
    
    def _evaluate_model(self) -> Tuple[float, float]:
        """评估模型性能"""
        obs = self.eval_env.reset()
        total_reward = 0
        episode_count = 0
        
        for _ in range(100):  # 评估100步
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, done, _ = self.eval_env.step(action)
            total_reward += reward
            
            if done:
                episode_count += 1
                obs = self.eval_env.reset()
        
        mean_reward = float(total_reward / max(episode_count, 1))
        return mean_reward, float(total_reward)

class RLStrategyTrainer:
    """强化学习策略训练器"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化训练器
        
        Args:
            config: 训练配置
        """
        self.config = config or self._get_default_config()
        self.model = None
        self.training_env = None
        self.eval_env = None
        self.training_history = []
        
        # 创建输出目录
        self.output_dir = self.config['output_dir']
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'models'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'logs'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'plots'), exist_ok=True)
    
    def _get_default_config(self) -> Dict:
        """获取默认配置"""
        return {
            'symbol': '510300',  # 沪深300ETF
            'start_date': '2020-01-01',
            'end_date': '2024-12-31',
            'train_ratio': 0.8,
            'output_dir': 'demo/output',
            
            # PPO超参数
            'learning_rate': 3e-4,
            'n_steps': 2048,
            'batch_size': 64,
            'n_epochs': 10,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'ent_coef': 0.01,
            'vf_coef': 0.5,
            'max_grad_norm': 0.5,
            
            # 训练参数
            'total_timesteps': 100000,
            'eval_freq': 10000,
            'save_freq': 20000,
            'verbose': 1,
            
            # 环境参数
            'initial_balance': 100000.0,
            'commission_rate': 0.001,
            'max_position': 1.0,
            'lookback_window': 20
        }
    
    def prepare_data(self) -> Tuple[TradingEnv, TradingEnv]:
        """准备训练和评估数据"""
        logger.info("准备训练数据...")
        
        # 获取数据
        fetcher = ETFDataFetcher()
        data = fetcher.get_etf_data(
            self.config['symbol'],
            self.config['start_date'],
            self.config['end_date']
        )
        
        if data is None:
            raise ValueError(f"无法获取 {self.config['symbol']} 的数据")
        
        logger.info(f"获取到 {len(data)} 条数据记录")
        
        # 创建特征工程器
        fe = FeatureEngineer()
        
        # 创建交易环境
        env = TradingEnv(data, fe, {
            'initial_balance': self.config['initial_balance'],
            'commission_rate': self.config['commission_rate'],
            'max_position': self.config['max_position'],
            'lookback_window': self.config['lookback_window'],
            'reward_scale': self.config['reward_scale'],
            'penalty_scale': self.config['penalty_scale'],
            'transaction_penalty': self.config['transaction_penalty']
        })
        
        # 分割训练和评估环境
        wrapper = TradingEnvWrapper(env)
        train_env, eval_env = wrapper.train_test_split(self.config['train_ratio'])
        
        # 包装环境以支持向量化
        self.training_env = DummyVecEnv([lambda: Monitor(train_env)])
        self.eval_env = DummyVecEnv([lambda: Monitor(eval_env)])
        
        logger.info(f"训练环境步数: {len(train_env.feature_array)}")
        logger.info(f"评估环境步数: {len(eval_env.feature_array)}")
        
        return train_env, eval_env
    
    def create_model(self) -> PPO:
        """创建PPO模型"""
        logger.info("创建PPO模型...")
        
        model = PPO(
            "MlpPolicy",
            self.training_env,
            learning_rate=self.config['learning_rate'],
            n_steps=self.config['n_steps'],
            batch_size=self.config['batch_size'],
            n_epochs=self.config['n_epochs'],
            gamma=self.config['gamma'],
            gae_lambda=self.config['gae_lambda'],
            clip_range=self.config['clip_range'],
            ent_coef=self.config['ent_coef'],
            vf_coef=self.config['vf_coef'],
            max_grad_norm=self.config['max_grad_norm'],
            verbose=self.config['verbose'],
            tensorboard_log=os.path.join(self.output_dir, 'logs')
        )
        
        return model
    
    def train(self, model_path: Optional[str] = None) -> PPO:
        """训练模型"""
        logger.info("开始训练PPO模型...")
        
        # 准备数据
        train_env, eval_env = self.prepare_data()
        
        # 创建或加载模型
        if model_path and os.path.exists(model_path):
            logger.info(f"加载已有模型: {model_path}")
            self.model = PPO.load(model_path, env=self.training_env)
        else:
            self.model = self.create_model()
        
        # 设置回调函数
        callback = TrainingCallback(
            eval_env=self.eval_env,
            eval_freq=self.config['eval_freq'],
            verbose=self.config['verbose']
        )
        
        # 开始训练
        start_time = datetime.now()
        
        self.model.learn(
            total_timesteps=self.config['total_timesteps'],
            callback=callback,
            tb_log_name="ppo_trading"
        )
        
        training_time = datetime.now() - start_time
        logger.info(f"训练完成，耗时: {training_time}")
        
        # 保存模型
        self.save_model()
        
        return self.model
    
    def evaluate_model(self, model: Optional[PPO] = None, 
                      n_episodes: int = 10) -> Dict[str, float]:
        """评估模型性能"""
        if model is None:
            model = self.model
        
        if model is None:
            raise ValueError("没有可评估的模型")
        
        logger.info(f"评估模型性能，运行 {n_episodes} 个回合...")
        
        episode_rewards = []
        episode_metrics = []
        
        for episode in range(n_episodes):
            obs = self.eval_env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, _ = self.eval_env.step(action)
                episode_reward += reward
            
            episode_rewards.append(episode_reward)
            
            # 获取环境性能指标
            if hasattr(self.eval_env.envs[0], 'env'):
                env_instance = self.eval_env.envs[0].env
                if hasattr(env_instance, 'get_performance_metrics'):
                    metrics = env_instance.get_performance_metrics()
                    episode_metrics.append(metrics)
        
        # 计算平均性能
        avg_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        
        avg_metrics = {}
        if episode_metrics:
            for key in episode_metrics[0].keys():
                values = [m[key] for m in episode_metrics if key in m]
                avg_metrics[f'avg_{key}'] = np.mean(values)
                avg_metrics[f'std_{key}'] = np.std(values)
        
        evaluation_results = {
            'avg_reward': avg_reward,
            'std_reward': std_reward,
            'episode_rewards': episode_rewards,
            **avg_metrics
        }
        
        logger.info(f"评估结果: 平均奖励 = {float(avg_reward):.2f} ± {float(std_reward):.2f}")
        
        return evaluation_results
    
    def save_model(self, model_name: Optional[str] = None):
        """保存模型"""
        if self.model is None:
            logger.warning("没有可保存的模型")
            return
        
        if model_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"ppo_trading_model_{timestamp}"
        
        model_path = os.path.join(self.output_dir, 'models', f"{model_name}.zip")
        self.model.save(model_path)
        
        # 保存配置
        config_path = os.path.join(self.output_dir, 'models', f"{model_name}_config.pkl")
        with open(config_path, 'wb') as f:
            pickle.dump(self.config, f)
        
        logger.info(f"模型已保存到: {model_path}")
        logger.info(f"配置已保存到: {config_path}")
    
    def load_model(self, model_path: str) -> PPO:
        """加载模型"""
        logger.info(f"加载模型: {model_path}")
        
        # 加载模型
        self.model = PPO.load(model_path, env=self.training_env)
        
        # 尝试加载配置
        config_path = model_path.replace('.zip', '_config.pkl')
        if os.path.exists(config_path):
            with open(config_path, 'rb') as f:
                loaded_config = pickle.load(f)
                logger.info("已加载模型配置")
        
        return self.model
    
    def plot_training_progress(self, log_dir: Optional[str] = None):
        """绘制训练进度"""
        if log_dir is None:
            log_dir = os.path.join(self.output_dir, 'logs')
        
        try:
            # 加载训练结果
            results = load_results(log_dir)
            
            if len(results) == 0:
                logger.warning("没有找到训练日志")
                return
            
            # 绘制训练曲线
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # 奖励曲线
            x, y = ts2xy(results, 'timesteps')
            axes[0, 0].plot(x, y)
            axes[0, 0].set_title('Training Reward')
            axes[0, 0].set_xlabel('Timesteps')
            axes[0, 0].set_ylabel('Reward')
            
            # 奖励移动平均
            window = max(1, len(y) // 20)
            if window > 1:
                moving_avg = pd.Series(y).rolling(window=window).mean()
                axes[0, 1].plot(x, moving_avg)
                axes[0, 1].set_title(f'Training Reward (Moving Average, window={window})')
                axes[0, 1].set_xlabel('Timesteps')
                axes[0, 1].set_ylabel('Reward')
            
            # 损失曲线（如果有的话）
            try:
                loss_data = results['loss']
                axes[1, 0].plot(loss_data)
                axes[1, 0].set_title('Training Loss')
                axes[1, 0].set_xlabel('Episodes')
                axes[1, 0].set_ylabel('Loss')
            except:
                axes[1, 0].text(0.5, 0.5, 'Loss data not available', 
                              ha='center', va='center', transform=axes[1, 0].transAxes)
            
            # 性能指标
            axes[1, 1].text(0.1, 0.8, f'Total Timesteps: {x[-1]:,}', transform=axes[1, 1].transAxes)
            axes[1, 1].text(0.1, 0.6, f'Final Reward: {y[-1]:.2f}', transform=axes[1, 1].transAxes)
            axes[1, 1].text(0.1, 0.4, f'Max Reward: {np.max(y):.2f}', transform=axes[1, 1].transAxes)
            axes[1, 1].text(0.1, 0.2, f'Min Reward: {np.min(y):.2f}', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Training Summary')
            axes[1, 1].axis('off')
            
            plt.tight_layout()
            
            # 保存图片
            plot_path = os.path.join(self.output_dir, 'plots', 'training_progress.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.show()
            
            logger.info(f"训练进度图已保存到: {plot_path}")
            
        except Exception as e:
            logger.error(f"绘制训练进度失败: {e}")
    
    def hyperparameter_tuning(self, param_grid: Dict[str, List]) -> Dict:
        """超参数调优"""
        logger.info("开始超参数调优...")
        
        best_params = None
        best_score = -np.inf
        results = []
        
        # 生成参数组合
        import itertools
        keys, values = zip(*param_grid.items())
        param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        
        logger.info(f"总共 {len(param_combinations)} 个参数组合")
        
        for i, params in enumerate(param_combinations):
            logger.info(f"测试参数组合 {i+1}/{len(param_combinations)}: {params}")
            
            # 更新配置
            test_config = self.config.copy()
            test_config.update(params)
            
            # 创建临时训练器
            temp_trainer = RLStrategyTrainer(test_config)
            
            try:
                # 训练模型
                model = temp_trainer.train()
                
                # 评估模型
                eval_results = temp_trainer.evaluate_model(model, n_episodes=5)
                score = eval_results['avg_reward']
                
                results.append({
                    'params': params,
                    'score': score,
                    'eval_results': eval_results
                })
                
                if score > best_score:
                    best_score = score
                    best_params = params
                    logger.info(f"新的最佳参数! 分数: {score:.2f}")
                
            except Exception as e:
                logger.error(f"参数组合 {params} 训练失败: {e}")
                continue
        
        logger.info(f"超参数调优完成，最佳分数: {best_score:.2f}")
        logger.info(f"最佳参数: {best_params}")
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'all_results': results
        }


def main():
    """测试函数"""
    # 创建训练器
    trainer = RLStrategyTrainer()
    
    # 训练模型
    model = trainer.train()
    
    # 评估模型
    eval_results = trainer.evaluate_model(model)
    print(f"评估结果: {eval_results}")
    
    # 绘制训练进度
    trainer.plot_training_progress()


if __name__ == "__main__":
    main()
