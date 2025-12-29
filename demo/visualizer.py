"""
可视化模块
生成策略性能对比图表，包括收益曲线、回撤分析、交易信号等
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime
import warnings

# 忽略matplotlib警告
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置图表样式
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

logger = logging.getLogger(__name__)

class StrategyVisualizer:
    """策略可视化器"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化可视化器
        
        Args:
            config: 可视化配置
        """
        self.config = config or self._get_default_config()
        
        # 创建输出目录
        self.output_dir = self.config['output_dir']
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'plots'), exist_ok=True)
    
    def _get_default_config(self) -> Dict:
        """获取默认配置"""
        return {
            'output_dir': 'demo/output',
            'figure_size': (15, 10),
            'dpi': 300,
            'style': 'seaborn-v0_8',
            'color_palette': 'husl',
            'font_size': 12,
            'title_size': 16,
            'label_size': 14
        }
    
    def plot_performance_comparison(self, results: Dict[str, Any], 
                                 save_path: Optional[str] = None) -> plt.Figure:
        """绘制策略性能对比图"""
        logger.info("绘制策略性能对比图...")
        
        # 提取性能指标
        strategies = []
        returns = []
        sharpe_ratios = []
        max_drawdowns = []
        win_rates = []
        
        for strategy_name, result in results.items():
            if 'error' not in result and 'metrics' in result:
                metrics = result['metrics']
                strategies.append(strategy_name)
                returns.append(metrics.get('total_return', 0))
                sharpe_ratios.append(metrics.get('sharpe_ratio', 0))
                max_drawdowns.append(abs(metrics.get('max_drawdown', 0)))  # 取绝对值
                win_rates.append(metrics.get('win_rate', 0))
        
        if not strategies:
            logger.warning("没有可用的策略数据")
            return None
        
        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=tuple(self.config['figure_size']))
        fig.suptitle('策略性能对比', fontsize=self.config['title_size'], fontweight='bold')
        
        # 1. 总收益率对比
        bars1 = axes[0, 0].bar(strategies, returns, color=sns.color_palette(self.config['color_palette'], len(strategies)))
        axes[0, 0].set_title('总收益率对比', fontsize=self.config['label_size'])
        axes[0, 0].set_ylabel('收益率 (%)', fontsize=self.config['font_size'])
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 添加数值标签
        for bar, value in zip(bars1, returns):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                           f'{value:.1f}%', ha='center', va='bottom', fontsize=10)
        
        # 2. 夏普比率对比
        bars2 = axes[0, 1].bar(strategies, sharpe_ratios, color=sns.color_palette(self.config['color_palette'], len(strategies)))
        axes[0, 1].set_title('夏普比率对比', fontsize=self.config['label_size'])
        axes[0, 1].set_ylabel('夏普比率', fontsize=self.config['font_size'])
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 添加数值标签
        for bar, value in zip(bars2, sharpe_ratios):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.2f}', ha='center', va='bottom', fontsize=10)
        
        # 3. 最大回撤对比
        bars3 = axes[1, 0].bar(strategies, max_drawdowns, color=sns.color_palette(self.config['color_palette'], len(strategies)))
        axes[1, 0].set_title('最大回撤对比', fontsize=self.config['label_size'])
        axes[1, 0].set_ylabel('回撤 (%)', fontsize=self.config['font_size'])
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 添加数值标签
        for bar, value in zip(bars3, max_drawdowns):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                           f'{value:.1f}%', ha='center', va='bottom', fontsize=10)
        
        # 4. 胜率对比
        bars4 = axes[1, 1].bar(strategies, win_rates, color=sns.color_palette(self.config['color_palette'], len(strategies)))
        axes[1, 1].set_title('胜率对比', fontsize=self.config['label_size'])
        axes[1, 1].set_ylabel('胜率 (%)', fontsize=self.config['font_size'])
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # 添加数值标签
        for bar, value in zip(bars4, win_rates):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                           f'{value:.1f}%', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        # 保存图片
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.output_dir, 'plots', f'performance_comparison_{timestamp}.png')
        
        plt.savefig(save_path, dpi=self.config['dpi'], bbox_inches='tight')
        logger.info(f"性能对比图已保存到: {save_path}")
        
        return fig
    
    def plot_returns_comparison(self, results: Dict[str, Any], 
                              save_path: Optional[str] = None) -> plt.Figure:
        """绘制收益率对比图"""
        logger.info("绘制收益率对比图...")
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 提取收益率数据
        strategy_returns = {}
        for strategy_name, result in results.items():
            if 'error' not in result and 'metrics' in result:
                metrics = result['metrics']
                strategy_returns[strategy_name] = metrics.get('total_return', 0)
        
        if not strategy_returns:
            logger.warning("没有可用的收益率数据")
            return None
        
        # 绘制水平条形图
        strategies = list(strategy_returns.keys())
        returns = list(strategy_returns.values())
        
        # 根据收益率排序
        sorted_data = sorted(zip(strategies, returns), key=lambda x: x[1], reverse=True)
        strategies, returns = zip(*sorted_data)
        
        bars = ax.barh(strategies, returns, color=sns.color_palette(self.config['color_palette'], len(strategies)))
        
        # 设置标题和标签
        ax.set_title('策略收益率对比', fontsize=self.config['title_size'], fontweight='bold')
        ax.set_xlabel('总收益率 (%)', fontsize=self.config['label_size'])
        ax.set_ylabel('策略名称', fontsize=self.config['label_size'])
        
        # 添加数值标签
        for i, (bar, value) in enumerate(zip(bars, returns)):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                   f'{value:.2f}%', ha='left', va='center', fontsize=11, fontweight='bold')
        
        # 添加零线
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # 设置网格
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        # 保存图片
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.output_dir, 'plots', f'returns_comparison_{timestamp}.png')
        
        plt.savefig(save_path, dpi=self.config['dpi'], bbox_inches='tight')
        logger.info(f"收益率对比图已保存到: {save_path}")
        
        return fig
    
    def plot_risk_return_scatter(self, results: Dict[str, Any], 
                               save_path: Optional[str] = None) -> plt.Figure:
        """绘制风险收益散点图"""
        logger.info("绘制风险收益散点图...")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 提取数据
        strategies = []
        returns = []
        risks = []  # 使用最大回撤作为风险指标
        
        for strategy_name, result in results.items():
            if 'error' not in result and 'metrics' in result:
                metrics = result['metrics']
                strategies.append(strategy_name)
                returns.append(metrics.get('total_return', 0))
                risks.append(abs(metrics.get('max_drawdown', 0)))  # 取绝对值
        
        if not strategies:
            logger.warning("没有可用的数据")
            return None
        
        # 绘制散点图
        scatter = ax.scatter(risks, returns, s=100, alpha=0.7, 
                           c=sns.color_palette(self.config['color_palette'], len(strategies)))
        
        # 添加策略标签
        for i, strategy in enumerate(strategies):
            ax.annotate(strategy, (risks[i], returns[i]), 
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=10, fontweight='bold')
        
        # 设置标题和标签
        ax.set_title('风险收益散点图', fontsize=self.config['title_size'], fontweight='bold')
        ax.set_xlabel('最大回撤 (%)', fontsize=self.config['label_size'])
        ax.set_ylabel('总收益率 (%)', fontsize=self.config['label_size'])
        
        # 添加网格
        ax.grid(True, alpha=0.3)
        
        # 添加象限线
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.output_dir, 'plots', f'risk_return_scatter_{timestamp}.png')
        
        plt.savefig(save_path, dpi=self.config['dpi'], bbox_inches='tight')
        logger.info(f"风险收益散点图已保存到: {save_path}")
        
        return fig
    
    def plot_performance_heatmap(self, results: Dict[str, Any], 
                               save_path: Optional[str] = None) -> plt.Figure:
        """绘制性能指标热力图"""
        logger.info("绘制性能指标热力图...")
        
        # 准备数据
        metrics_data = []
        strategies = []
        
        for strategy_name, result in results.items():
            if 'error' not in result and 'metrics' in result:
                metrics = result['metrics']
                strategies.append(strategy_name)
                metrics_data.append([
                    metrics.get('total_return', 0),
                    metrics.get('sharpe_ratio', 0),
                    abs(metrics.get('max_drawdown', 0)),  # 取绝对值
                    metrics.get('win_rate', 0),
                    metrics.get('total_trades', 0)
                ])
        
        if not strategies:
            logger.warning("没有可用的数据")
            return None
        
        # 创建DataFrame
        metrics_df = pd.DataFrame(metrics_data, 
                                index=strategies,
                                columns=['总收益率(%)', '夏普比率', '最大回撤(%)', '胜率(%)', '交易次数'])
        
        # 创建热力图
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 标准化数据（除了交易次数）
        metrics_df_normalized = metrics_df.copy()
        for col in ['总收益率(%)', '夏普比率', '最大回撤(%)', '胜率(%)']:
            if metrics_df[col].max() != metrics_df[col].min():
                metrics_df_normalized[col] = (metrics_df[col] - metrics_df[col].min()) / (metrics_df[col].max() - metrics_df[col].min())
        
        # 交易次数单独处理
        if metrics_df['交易次数'].max() != metrics_df['交易次数'].min():
            metrics_df_normalized['交易次数'] = (metrics_df['交易次数'] - metrics_df['交易次数'].min()) / (metrics_df['交易次数'].max() - metrics_df['交易次数'].min())
        
        # 绘制热力图
        sns.heatmap(metrics_df_normalized.T, annot=True, cmap='RdYlBu_r', 
                   center=0.5, square=True, ax=ax, cbar_kws={'label': '标准化值'})
        
        ax.set_title('策略性能指标热力图', fontsize=self.config['title_size'], fontweight='bold')
        ax.set_xlabel('策略名称', fontsize=self.config['label_size'])
        ax.set_ylabel('性能指标', fontsize=self.config['label_size'])
        
        plt.tight_layout()
        
        # 保存图片
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.output_dir, 'plots', f'performance_heatmap_{timestamp}.png')
        
        plt.savefig(save_path, dpi=self.config['dpi'], bbox_inches='tight')
        logger.info(f"性能热力图已保存到: {save_path}")
        
        return fig
    
    def plot_strategy_summary(self, results: Dict[str, Any], 
                            save_path: Optional[str] = None) -> plt.Figure:
        """绘制策略总结图"""
        logger.info("绘制策略总结图...")
        
        fig = plt.figure(figsize=(16, 12))
        
        # 创建网格布局
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 提取数据
        strategies = []
        returns = []
        sharpe_ratios = []
        max_drawdowns = []
        win_rates = []
        trade_counts = []
        
        for strategy_name, result in results.items():
            if 'error' not in result and 'metrics' in result:
                metrics = result['metrics']
                strategies.append(strategy_name)
                returns.append(metrics.get('total_return', 0))
                sharpe_ratios.append(metrics.get('sharpe_ratio', 0))
                max_drawdowns.append(abs(metrics.get('max_drawdown', 0)))
                win_rates.append(metrics.get('win_rate', 0))
                trade_counts.append(metrics.get('total_trades', 0))
        
        if not strategies:
            logger.warning("没有可用的数据")
            return None
        
        # 1. 收益率对比（左上）
        ax1 = fig.add_subplot(gs[0, 0])
        bars1 = ax1.bar(strategies, returns, color=sns.color_palette(self.config['color_palette'], len(strategies)))
        ax1.set_title('总收益率', fontsize=12, fontweight='bold')
        ax1.set_ylabel('收益率 (%)')
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. 夏普比率对比（中上）
        ax2 = fig.add_subplot(gs[0, 1])
        bars2 = ax2.bar(strategies, sharpe_ratios, color=sns.color_palette(self.config['color_palette'], len(strategies)))
        ax2.set_title('夏普比率', fontsize=12, fontweight='bold')
        ax2.set_ylabel('夏普比率')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. 最大回撤对比（右上）
        ax3 = fig.add_subplot(gs[0, 2])
        bars3 = ax3.bar(strategies, max_drawdowns, color=sns.color_palette(self.config['color_palette'], len(strategies)))
        ax3.set_title('最大回撤', fontsize=12, fontweight='bold')
        ax3.set_ylabel('回撤 (%)')
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. 胜率对比（左中）
        ax4 = fig.add_subplot(gs[1, 0])
        bars4 = ax4.bar(strategies, win_rates, color=sns.color_palette(self.config['color_palette'], len(strategies)))
        ax4.set_title('胜率', fontsize=12, fontweight='bold')
        ax4.set_ylabel('胜率 (%)')
        ax4.tick_params(axis='x', rotation=45)
        
        # 5. 交易次数对比（中中）
        ax5 = fig.add_subplot(gs[1, 1])
        bars5 = ax5.bar(strategies, trade_counts, color=sns.color_palette(self.config['color_palette'], len(strategies)))
        ax5.set_title('交易次数', fontsize=12, fontweight='bold')
        ax5.set_ylabel('交易次数')
        ax5.tick_params(axis='x', rotation=45)
        
        # 6. 风险收益散点图（右中）
        ax6 = fig.add_subplot(gs[1, 2])
        scatter = ax6.scatter(max_drawdowns, returns, s=100, alpha=0.7,
                            c=sns.color_palette(self.config['color_palette'], len(strategies)))
        ax6.set_title('风险收益', fontsize=12, fontweight='bold')
        ax6.set_xlabel('最大回撤 (%)')
        ax6.set_ylabel('总收益率 (%)')
        ax6.grid(True, alpha=0.3)
        
        # 添加策略标签
        for i, strategy in enumerate(strategies):
            ax6.annotate(strategy, (max_drawdowns[i], returns[i]), 
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, fontweight='bold')
        
        # 7. 综合评分（左下，跨两列）
        ax7 = fig.add_subplot(gs[2, :2])
        
        # 计算综合评分（收益率权重40%，夏普比率权重30%，回撤权重20%，胜率权重10%）
        scores = []
        for i in range(len(strategies)):
            # 标准化各项指标
            norm_return = (returns[i] - min(returns)) / (max(returns) - min(returns)) if max(returns) != min(returns) else 0.5
            norm_sharpe = (sharpe_ratios[i] - min(sharpe_ratios)) / (max(sharpe_ratios) - min(sharpe_ratios)) if max(sharpe_ratios) != min(sharpe_ratios) else 0.5
            norm_drawdown = 1 - (max_drawdowns[i] - min(max_drawdowns)) / (max(max_drawdowns) - min(max_drawdowns)) if max(max_drawdowns) != min(max_drawdowns) else 0.5
            norm_winrate = (win_rates[i] - min(win_rates)) / (max(win_rates) - min(win_rates)) if max(win_rates) != min(win_rates) else 0.5
            
            score = norm_return * 0.4 + norm_sharpe * 0.3 + norm_drawdown * 0.2 + norm_winrate * 0.1
            scores.append(score)
        
        bars7 = ax7.bar(strategies, scores, color=sns.color_palette(self.config['color_palette'], len(strategies)))
        ax7.set_title('综合评分', fontsize=12, fontweight='bold')
        ax7.set_ylabel('评分')
        ax7.tick_params(axis='x', rotation=45)
        
        # 添加数值标签
        for bar, score in zip(bars7, scores):
            ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.2f}', ha='center', va='bottom', fontsize=10)
        
        # 8. 策略排名（右下）
        ax8 = fig.add_subplot(gs[2, 2])
        
        # 按综合评分排序
        sorted_data = sorted(zip(strategies, scores), key=lambda x: x[1], reverse=True)
        sorted_strategies, sorted_scores = zip(*sorted_data)
        
        bars8 = ax8.barh(range(len(sorted_strategies)), sorted_scores, 
                         color=sns.color_palette(self.config['color_palette'], len(sorted_strategies)))
        ax8.set_title('策略排名', fontsize=12, fontweight='bold')
        ax8.set_xlabel('综合评分')
        ax8.set_yticks(range(len(sorted_strategies)))
        ax8.set_yticklabels(sorted_strategies)
        
        # 添加排名标签
        for i, (bar, score) in enumerate(zip(bars8, sorted_scores)):
            ax8.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                    f'#{i+1}', ha='left', va='center', fontsize=10, fontweight='bold')
        
        # 设置总标题
        fig.suptitle('策略性能综合分析', fontsize=16, fontweight='bold', y=0.98)
        
        # 保存图片
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.output_dir, 'plots', f'strategy_summary_{timestamp}.png')
        
        plt.savefig(save_path, dpi=self.config['dpi'], bbox_inches='tight')
        logger.info(f"策略总结图已保存到: {save_path}")
        
        return fig
    
    def create_all_plots(self, results: Dict[str, Any]) -> List[str]:
        """创建所有图表"""
        logger.info("创建所有可视化图表...")
        
        plot_files = []
        
        try:
            # 1. 性能对比图
            fig1 = self.plot_performance_comparison(results)
            if fig1:
                plot_files.append("performance_comparison")
                plt.close(fig1)
            
            # 2. 收益率对比图
            fig2 = self.plot_returns_comparison(results)
            if fig2:
                plot_files.append("returns_comparison")
                plt.close(fig2)
            
            # 3. 风险收益散点图
            fig3 = self.plot_risk_return_scatter(results)
            if fig3:
                plot_files.append("risk_return_scatter")
                plt.close(fig3)
            
            # 4. 性能热力图
            fig4 = self.plot_performance_heatmap(results)
            if fig4:
                plot_files.append("performance_heatmap")
                plt.close(fig4)
            
            # 5. 策略总结图
            fig5 = self.plot_strategy_summary(results)
            if fig5:
                plot_files.append("strategy_summary")
                plt.close(fig5)
            
            logger.info(f"成功创建 {len(plot_files)} 个图表")
            
        except Exception as e:
            logger.error(f"创建图表时出错: {e}")
        
        return plot_files


def main():
    """测试函数"""
    # 创建测试数据
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
        },
        'moving_average': {
            'metrics': {
                'total_return': 8.7,
                'sharpe_ratio': 0.9,
                'max_drawdown': -15.2,
                'win_rate': 55.0,
                'total_trades': 23
            }
        }
    }
    
    # 创建可视化器
    visualizer = StrategyVisualizer()
    
    # 创建所有图表
    plot_files = visualizer.create_all_plots(test_results)
    
    print(f"创建了 {len(plot_files)} 个图表文件")


if __name__ == "__main__":
    main()
