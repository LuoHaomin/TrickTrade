"""
主程序入口
整合所有模块，提供完整的基金投资强化学习策略demo工作流
"""

import os
import sys
import argparse
import logging
from datetime import datetime
from typing import Dict, Optional, Any
import json

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_fetcher import ETFDataFetcher
from feature_engineer import FeatureEngineer
from rl_env import TradingEnv
from train_rl import RLStrategyTrainer
from bt_strategy import RLStrategy, BuyAndHoldStrategy, MovingAverageStrategy, BacktestRunner
from backtest_eval import StrategyEvaluator
from visualizer import StrategyVisualizer

# 创建必要的目录
os.makedirs('output/logs', exist_ok=True)

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('output/logs/main.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FundRLDemo:
    """基金强化学习策略Demo主类"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化Demo
        
        Args:
            config_path: 配置文件路径
        """
        self.config = self._load_config(config_path)
        self.setup_directories()
        
        # 初始化组件
        self.data_fetcher = ETFDataFetcher()
        self.feature_engineer = FeatureEngineer()
        self.trainer = None
        self.evaluator = None
        self.visualizer = None
        
        logger.info("基金强化学习策略Demo初始化完成")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """加载配置"""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'symbol': '510300',  # 沪深300ETF
            'start_date': '2020-01-01',
            'end_date': '2024-12-31',
            'train_ratio': 0.8,
            
            'data': {
                'cache_dir': 'data',
                'use_cache': True
            },
            
            'rl_training': {
                'learning_rate': 3e-4,
                'n_steps': 2048,
                'batch_size': 64,
                'n_epochs': 10,
                'gamma': 0.99,
                'total_timesteps': 100000,
                'eval_freq': 10000,
                'output_dir': 'output'
            },
            
            'backtest': {
                'initial_cash': 100000.0,
                'commission': 0.001,
                'test_start_date': '2024-01-01',
                'test_end_date': '2024-12-31'
            },
            
            'output': {
                'dir': 'output',
                'create_plots': True,
                'save_results': True
            }
        }
    
    def setup_directories(self):
        """创建必要的目录"""
        directories = [
            'data',
            'output',
            'output/models',
            'output/logs',
            'output/plots',
            'output/results'
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def step1_fetch_data(self) -> bool:
        """步骤1: 获取数据"""
        logger.info("=== 步骤1: 获取ETF数据 ===")
        
        try:
            # 获取训练数据
            train_data = self.data_fetcher.get_etf_data(
                self.config['symbol'],
                self.config['start_date'],
                self.config['end_date'],
                use_cache=self.config['data']['use_cache']
            )
            
            if train_data is None:
                logger.error(f"无法获取 {self.config['symbol']} 的数据")
                return False
            
            logger.info(f"成功获取数据: {len(train_data)} 条记录")
            logger.info(f"数据时间范围: {train_data.index[0]} 到 {train_data.index[-1]}")
            
            # 保存数据信息
            data_info = {
                'symbol': self.config['symbol'],
                'start_date': self.config['start_date'],
                'end_date': self.config['end_date'],
                'data_points': len(train_data),
                'date_range': f"{train_data.index[0]} to {train_data.index[-1]}",
                'price_range': f"{train_data['Close'].min():.2f} - {train_data['Close'].max():.2f}"
            }
            
            with open('output/data_info.json', 'w', encoding='utf-8') as f:
                json.dump(data_info, f, indent=2, ensure_ascii=False)
            
            return True
            
        except Exception as e:
            logger.error(f"获取数据失败: {e}")
            return False
    
    def step2_train_rl_model(self, force_retrain: bool = False) -> bool:
        """步骤2: 训练RL模型"""
        logger.info("=== 步骤2: 训练强化学习模型 ===")
        
        try:
            # 检查是否已有训练好的模型
            model_path = os.path.join('output', 'models', 'ppo_trading_model.zip')
            
            if os.path.exists(model_path) and not force_retrain:
                logger.info("发现已有训练好的模型，跳过训练步骤")
                return True
            
            # 创建训练器配置
            trainer_config = self.config['rl_training'].copy()
            trainer_config.update({
                'symbol': self.config['symbol'],
                'start_date': self.config['start_date'],
                'end_date': self.config['end_date'],
                'train_ratio': self.config['train_ratio'],
                'initial_balance': 100000.0,
                'commission_rate': 0.001,
                'max_position': 1.0,
                'lookback_window': 20,
                # PPO参数
                'gae_lambda': 0.95,
                'clip_range': 0.2,
                'ent_coef': 0.01,
                'vf_coef': 0.5,
                'max_grad_norm': 0.5,
                'verbose': 1,
                # 环境参数
                'reward_scale': 1.0,
                'penalty_scale': 0.1,
                'transaction_penalty': 0.001
            })
            
            # 创建训练器
            self.trainer = RLStrategyTrainer(trainer_config)
            
            # 训练模型
            logger.info("开始训练PPO模型...")
            model = self.trainer.train()
            
            if model is None:
                logger.error("模型训练失败")
                return False
            
            logger.info("模型训练完成")
            return True
            
        except Exception as e:
            logger.error(f"训练RL模型失败: {e}")
            return False
    
    def step3_run_backtest(self) -> bool:
        """步骤3: 运行回测"""
        logger.info("=== 步骤3: 运行策略回测 ===")
        
        try:
            # 创建评估器
            eval_config = {
                'symbol': self.config['symbol'],
                'start_date': self.config['backtest']['test_start_date'],
                'end_date': self.config['backtest']['test_end_date'],
                'initial_cash': self.config['backtest']['initial_cash'],
                'commission': self.config['backtest']['commission'],
                'output_dir': self.config['output']['dir'],
                'strategies': {
                    'rl_strategy': {
                        'class': RLStrategy,
                        'params': {
                            'model_path': os.path.join('output', 'models', 'ppo_trading_model_20251021_082935.zip'),
                            'lookback_window': 20,
                            'position_threshold': 0.1,
                            'printlog': False
                        }
                    },
                    'buy_and_hold': {
                        'class': BuyAndHoldStrategy,
                        'params': {
                            'printlog': False
                        }
                    },
                    'moving_average': {
                        'class': MovingAverageStrategy,
                        'params': {
                            'ma_period': 20,
                            'printlog': False
                        }
                    }
                }
            }
            
            self.evaluator = StrategyEvaluator(eval_config)
            
            # 运行回测
            results = self.evaluator.evaluate_all_strategies()
            
            if not results or not results['results']:
                logger.error("回测失败")
                return False
            
            logger.info("回测完成")
            
            # 打印对比报告
            self.evaluator.print_comparison_report()
            
            return True
            
        except Exception as e:
            logger.error(f"运行回测失败: {e}")
            return False
    
    def step4_create_visualizations(self) -> bool:
        """步骤4: 创建可视化图表"""
        logger.info("=== 步骤4: 创建可视化图表 ===")
        
        try:
            if not self.evaluator or not self.evaluator.results:
                logger.error("没有可用的回测结果")
                return False
            
            # 创建可视化器
            self.visualizer = StrategyVisualizer({
                'output_dir': self.config['output']['dir']
            })
            
            # 创建所有图表
            plot_files = self.visualizer.create_all_plots(self.evaluator.results)
            
            logger.info(f"成功创建 {len(plot_files)} 个可视化图表")
            
            return True
            
        except Exception as e:
            logger.error(f"创建可视化图表失败: {e}")
            return False
    
    def run_full_demo(self, force_retrain: bool = False) -> bool:
        """运行完整的Demo流程"""
        logger.info("开始运行基金强化学习策略Demo")
        logger.info(f"配置: {self.config}")
        
        start_time = datetime.now()
        
        try:
            # 步骤1: 获取数据
            if not self.step1_fetch_data():
                return False
            
            # 步骤2: 训练RL模型
            if not self.step2_train_rl_model(force_retrain):
                return False
            
            # 步骤3: 运行回测
            if not self.step3_run_backtest():
                return False
            
            # 步骤4: 创建可视化
            if self.config['output']['create_plots']:
                if not self.step4_create_visualizations():
                    logger.warning("可视化创建失败，但Demo继续")
            
            # 生成最终报告
            self._generate_final_report()
            
            end_time = datetime.now()
            duration = end_time - start_time
            
            logger.info(f"Demo运行完成，总耗时: {duration}")
            logger.info("所有结果已保存到 demo/output/ 目录")
            
            return True
            
        except Exception as e:
            logger.error(f"Demo运行失败: {e}")
            return False
    
    def _generate_final_report(self):
        """生成最终报告"""
        logger.info("生成最终报告...")
        
        try:
            report = {
                'demo_info': {
                    'timestamp': datetime.now().isoformat(),
                    'symbol': self.config['symbol'],
                    'data_period': f"{self.config['start_date']} to {self.config['end_date']}",
                    'test_period': f"{self.config['backtest']['test_start_date']} to {self.config['backtest']['test_end_date']}"
                },
                'config': self.config,
                'results_summary': {}
            }
            
            # 添加回测结果摘要
            if self.evaluator and self.evaluator.results:
                for strategy_name, result in self.evaluator.results.items():
                    if 'error' not in result and 'metrics' in result:
                        report['results_summary'][strategy_name] = {
                            'total_return': result['metrics'].get('total_return', 0),
                            'sharpe_ratio': result['metrics'].get('sharpe_ratio', 0),
                            'max_drawdown': result['metrics'].get('max_drawdown', 0),
                            'win_rate': result['metrics'].get('win_rate', 0)
                        }
            
            # 保存报告
            report_path = os.path.join('output', 'final_report.json')
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            logger.info(f"最终报告已保存到: {report_path}")
            
        except Exception as e:
            logger.error(f"生成最终报告失败: {e}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='基金强化学习策略Demo')
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--symbol', type=str, default='510300', help='ETF代码')
    parser.add_argument('--start-date', type=str, default='2020-01-01', help='开始日期')
    parser.add_argument('--end-date', type=str, default='2024-12-31', help='结束日期')
    parser.add_argument('--force-retrain', action='store_true', help='强制重新训练模型')
    parser.add_argument('--test-only', action='store_true', help='仅运行回测（跳过训练）')
    parser.add_argument('--train-only', action='store_true', help='仅训练模型（跳过回测）')
    
    args = parser.parse_args()
    
    # 创建Demo实例
    demo = FundRLDemo(args.config)
    
    # 更新配置
    if args.symbol:
        demo.config['symbol'] = args.symbol
    if args.start_date:
        demo.config['start_date'] = args.start_date
    if args.end_date:
        demo.config['end_date'] = args.end_date
    
    # 根据参数运行不同的流程
    if args.test_only:
        # 仅运行回测
        success = demo.step3_run_backtest()
        if success and demo.config['output']['create_plots']:
            demo.step4_create_visualizations()
    elif args.train_only:
        # 仅训练模型
        success = demo.step1_fetch_data() and demo.step2_train_rl_model(args.force_retrain)
    else:
        # 运行完整Demo
        success = demo.run_full_demo(args.force_retrain)
    
    if success:
        print("\n✅ Demo运行成功！")
        print("📊 查看结果:")
        print("  - 回测结果: output/results/")
        print("  - 可视化图表: output/plots/")
        print("  - 训练日志: output/logs/")
        print("  - 最终报告: output/final_report.json")
    else:
        print("\n❌ Demo运行失败，请检查日志")
        sys.exit(1)


if __name__ == "__main__":
    main()
