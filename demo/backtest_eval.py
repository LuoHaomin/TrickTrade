"""
回测与评估模块
对比强化学习策略与基线策略的性能，生成详细的评估报告
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime
import json

from data_fetcher import ETFDataFetcher
from feature_engineer import FeatureEngineer
from bt_strategy import RLStrategy, BuyAndHoldStrategy, MovingAverageStrategy, BacktestRunner
from train_rl import RLStrategyTrainer

logger = logging.getLogger(__name__)

class StrategyEvaluator:
    """策略评估器"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化评估器
        
        Args:
            config: 评估配置
        """
        self.config = config or self._get_default_config()
        self.results = {}
        
        # 创建输出目录
        self.output_dir = self.config['output_dir']
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'results'), exist_ok=True)
    
    def _get_default_config(self) -> Dict:
        """获取默认配置"""
        return {
            'symbol': '510300',
            'start_date': '2024-01-01',
            'end_date': '2024-12-31',
            'initial_cash': 100000.0,
            'commission': 0.001,
            'output_dir': 'demo/output',
            'strategies': {
                'rl_strategy': {
                    'class': RLStrategy,
                    'params': {
                        'model_path': 'demo/output/models/ppo_trading_model.zip',
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
    
    def prepare_data(self) -> pd.DataFrame:
        """准备回测数据"""
        logger.info("准备回测数据...")
        
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
        logger.info(f"数据时间范围: {data.index[0]} 到 {data.index[-1]}")
        
        return data
    
    def run_backtest(self, data: pd.DataFrame, strategy_name: str, 
                    strategy_config: Dict) -> Dict[str, Any]:
        """运行单个策略的回测"""
        logger.info(f"运行 {strategy_name} 策略回测...")
        
        try:
            # 创建回测运行器
            runner = BacktestRunner(
                initial_cash=self.config['initial_cash'],
                commission=self.config['commission']
            )
            
            # 添加数据
            runner.add_data(data)
            
            # 添加策略
            strategy_class = strategy_config['class']
            strategy_params = strategy_config['params']
            runner.add_strategy(strategy_class, **strategy_params)
            
            # 运行回测
            results = runner.run(show_log=False)
            
            # 添加策略名称
            results['strategy_name'] = strategy_name
            results['strategy_config'] = strategy_config
            
            logger.info(f"{strategy_name} 策略回测完成")
            logger.info(f"总收益率: {results['metrics'].get('total_return', 0):.2f}%")
            logger.info(f"夏普比率: {results['metrics'].get('sharpe_ratio', 0):.2f}")
            logger.info(f"最大回撤: {results['metrics'].get('max_drawdown', 0):.2f}%")
            
            return results
            
        except Exception as e:
            logger.error(f"{strategy_name} 策略回测失败: {e}")
            return {
                'strategy_name': strategy_name,
                'error': str(e),
                'metrics': {}
            }
    
    def evaluate_all_strategies(self) -> Dict[str, Any]:
        """评估所有策略"""
        logger.info("开始评估所有策略...")
        
        # 准备数据
        data = self.prepare_data()
        
        # 运行所有策略
        all_results = {}
        
        for strategy_name, strategy_config in self.config['strategies'].items():
            logger.info(f"评估策略: {strategy_name}")
            
            result = self.run_backtest(data, strategy_name, strategy_config)
            all_results[strategy_name] = result
        
        # 保存结果
        self.results = all_results
        self._save_results()
        
        # 生成对比报告
        comparison_report = self._generate_comparison_report()
        
        logger.info("所有策略评估完成")
        return {
            'results': all_results,
            'comparison': comparison_report
        }
    
    def _save_results(self):
        """保存结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存详细结果
        results_file = os.path.join(self.output_dir, 'results', f'backtest_results_{timestamp}.json')
        
        # 准备可序列化的结果
        serializable_results = {}
        for strategy_name, result in self.results.items():
            serializable_results[strategy_name] = {
                'strategy_name': result.get('strategy_name', strategy_name),
                'metrics': result.get('metrics', {}),
                'error': result.get('error', None)
            }
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"结果已保存到: {results_file}")
    
    def _generate_comparison_report(self) -> Dict[str, Any]:
        """生成对比报告"""
        logger.info("生成策略对比报告...")
        
        # 提取性能指标
        metrics_comparison = {}
        
        for strategy_name, result in self.results.items():
            if 'error' not in result:
                metrics = result['metrics']
                metrics_comparison[strategy_name] = {
                    'total_return': metrics.get('total_return', 0),
                    'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                    'max_drawdown': metrics.get('max_drawdown', 0),
                    'win_rate': metrics.get('win_rate', 0),
                    'total_trades': metrics.get('total_trades', 0),
                    'final_value': metrics.get('final_value', 0)
                }
        
        # 找出最佳策略
        best_strategies = {}
        
        if metrics_comparison:
            # 过滤掉有缺失指标的策略
            valid_strategies = {k: v for k, v in metrics_comparison.items() 
                              if all(key in v and v[key] is not None 
                                    for key in ['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate'])}
            
            if valid_strategies:
                # 最佳总收益率
                best_return = max(valid_strategies.items(), key=lambda x: x[1]['total_return'])
                best_strategies['best_return'] = {
                    'strategy': best_return[0],
                    'value': best_return[1]['total_return']
                }
                
                # 最佳夏普比率
                best_sharpe = max(valid_strategies.items(), key=lambda x: x[1]['sharpe_ratio'])
                best_strategies['best_sharpe'] = {
                    'strategy': best_sharpe[0],
                    'value': best_sharpe[1]['sharpe_ratio']
                }
                
                # 最小回撤
                min_drawdown = min(valid_strategies.items(), key=lambda x: x[1]['max_drawdown'])
                best_strategies['min_drawdown'] = {
                    'strategy': min_drawdown[0],
                    'value': min_drawdown[1]['max_drawdown']
                }
                
                # 最高胜率
                best_winrate = max(valid_strategies.items(), key=lambda x: x[1]['win_rate'])
                best_strategies['best_winrate'] = {
                    'strategy': best_winrate[0],
                    'value': best_winrate[1]['win_rate']
                }
        
        # 计算相对表现
        relative_performance = {}
        
        if 'buy_and_hold' in metrics_comparison:
            baseline_return = metrics_comparison['buy_and_hold']['total_return']
            baseline_sharpe = metrics_comparison['buy_and_hold']['sharpe_ratio']
            
            for strategy_name, metrics in metrics_comparison.items():
                if strategy_name != 'buy_and_hold':
                    relative_performance[strategy_name] = {
                        'return_vs_baseline': metrics['total_return'] - baseline_return,
                        'sharpe_vs_baseline': metrics['sharpe_ratio'] - baseline_sharpe,
                        'return_improvement_pct': (metrics['total_return'] - baseline_return) / abs(baseline_return) * 100 if baseline_return != 0 else 0
                    }
        
        comparison_report = {
            'metrics_comparison': metrics_comparison,
            'best_strategies': best_strategies,
            'relative_performance': relative_performance,
            'summary': self._generate_summary(metrics_comparison, best_strategies, relative_performance)
        }
        
        return comparison_report
    
    def _generate_summary(self, metrics_comparison: Dict, best_strategies: Dict, 
                         relative_performance: Dict) -> str:
        """生成总结报告"""
        summary_lines = []
        
        summary_lines.append("=== 策略回测对比报告 ===")
        summary_lines.append(f"回测时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        summary_lines.append(f"标的: {self.config['symbol']}")
        summary_lines.append(f"回测期间: {self.config['start_date']} 到 {self.config['end_date']}")
        summary_lines.append(f"初始资金: {self.config['initial_cash']:,.0f}")
        summary_lines.append("")
        
        # 策略表现排名
        summary_lines.append("=== 策略表现排名 ===")
        
        # 按总收益率排名
        return_ranking = sorted(metrics_comparison.items(), 
                              key=lambda x: x[1]['total_return'], reverse=True)
        summary_lines.append("总收益率排名:")
        for i, (strategy, metrics) in enumerate(return_ranking, 1):
            summary_lines.append(f"  {i}. {strategy}: {metrics['total_return']:.2f}%")
        
        summary_lines.append("")
        
        # 按夏普比率排名
        sharpe_ranking = sorted(metrics_comparison.items(), 
                              key=lambda x: x[1]['sharpe_ratio'], reverse=True)
        summary_lines.append("夏普比率排名:")
        for i, (strategy, metrics) in enumerate(sharpe_ranking, 1):
            summary_lines.append(f"  {i}. {strategy}: {metrics['sharpe_ratio']:.2f}")
        
        summary_lines.append("")
        
        # 最佳策略
        summary_lines.append("=== 最佳策略 ===")
        summary_lines.append(f"最佳总收益率: {best_strategies['best_return']['strategy']} ({best_strategies['best_return']['value']:.2f}%)")
        summary_lines.append(f"最佳夏普比率: {best_strategies['best_sharpe']['strategy']} ({best_strategies['best_sharpe']['value']:.2f})")
        summary_lines.append(f"最小回撤: {best_strategies['min_drawdown']['strategy']} ({best_strategies['min_drawdown']['value']:.2f}%)")
        summary_lines.append(f"最高胜率: {best_strategies['best_winrate']['strategy']} ({best_strategies['best_winrate']['value']:.2f}%)")
        
        summary_lines.append("")
        
        # 相对表现
        if relative_performance:
            summary_lines.append("=== 相对买入持有策略的表现 ===")
            for strategy, perf in relative_performance.items():
                summary_lines.append(f"{strategy}:")
                summary_lines.append(f"  收益率差异: {perf['return_vs_baseline']:+.2f}%")
                summary_lines.append(f"  夏普比率差异: {perf['sharpe_vs_baseline']:+.2f}")
                summary_lines.append(f"  收益率改善: {perf['return_improvement_pct']:+.1f}%")
                summary_lines.append("")
        
        # RL策略特别分析
        if 'rl_strategy' in metrics_comparison and 'rl_strategy' in relative_performance:
            rl_metrics = metrics_comparison['rl_strategy']
            rl_perf = relative_performance['rl_strategy']
            
            summary_lines.append("=== RL策略分析 ===")
            summary_lines.append(f"RL策略总收益率: {rl_metrics['total_return']:.2f}%")
            summary_lines.append(f"RL策略夏普比率: {rl_metrics['sharpe_ratio']:.2f}")
            summary_lines.append(f"RL策略最大回撤: {rl_metrics['max_drawdown']:.2f}%")
            summary_lines.append(f"RL策略交易次数: {rl_metrics['total_trades']}")
            summary_lines.append(f"相对买入持有策略收益率改善: {rl_perf['return_improvement_pct']:+.1f}%")
            
            if rl_perf['return_improvement_pct'] > 0:
                summary_lines.append("✅ RL策略表现优于买入持有策略")
            else:
                summary_lines.append("❌ RL策略表现不如买入持有策略")
        
        return "\n".join(summary_lines)
    
    def print_comparison_report(self):
        """打印对比报告"""
        if not self.results:
            logger.warning("没有可用的结果")
            return
        
        comparison_report = self._generate_comparison_report()
        print(comparison_report['summary'])
    
    def get_strategy_performance(self, strategy_name: str) -> Optional[Dict]:
        """获取特定策略的性能"""
        if strategy_name in self.results:
            return self.results[strategy_name]
        return None
    
    def export_results_to_csv(self, filename: Optional[str] = None):
        """导出结果到CSV文件"""
        if not self.results:
            logger.warning("没有可用的结果")
            return
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"strategy_comparison_{timestamp}.csv"
        
        filepath = os.path.join(self.output_dir, 'results', filename)
        
        # 准备数据
        data_rows = []
        for strategy_name, result in self.results.items():
            if 'error' not in result:
                metrics = result['metrics']
                row = {
                    'Strategy': strategy_name,
                    'Total_Return_Pct': metrics.get('total_return', 0),
                    'Sharpe_Ratio': metrics.get('sharpe_ratio', 0),
                    'Max_Drawdown_Pct': metrics.get('max_drawdown', 0),
                    'Win_Rate_Pct': metrics.get('win_rate', 0),
                    'Total_Trades': metrics.get('total_trades', 0),
                    'Final_Value': metrics.get('final_value', 0)
                }
                data_rows.append(row)
        
        # 创建DataFrame并保存
        df = pd.DataFrame(data_rows)
        df.to_csv(filepath, index=False, encoding='utf-8')
        
        logger.info(f"结果已导出到: {filepath}")
        return filepath


def main():
    """测试函数"""
    # 创建评估器
    evaluator = StrategyEvaluator()
    
    # 评估所有策略
    results = evaluator.evaluate_all_strategies()
    
    # 打印对比报告
    evaluator.print_comparison_report()
    
    # 导出结果
    evaluator.export_results_to_csv()


if __name__ == "__main__":
    main()
