"""
预测可视化器
可视化基金价格预测结果
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class PredictionVisualizer:
    """预测可视化器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化预测可视化器
        
        Args:
            config: 配置参数
        """
        self.config = config or {}
        self.style = self.config.get('style', 'seaborn-v0_8')
        plt.style.use(self.style)
        
    def plot_prediction_results(self, data: pd.DataFrame, predictions: List[float], 
                              prediction_dates: List[str], title: str = "基金价格预测结果") -> plt.Figure:
        """
        绘制预测结果
        
        Args:
            data: 历史数据
            predictions: 预测值
            prediction_dates: 预测日期
            title: 图表标题
            
        Returns:
            matplotlib图表对象
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 绘制历史数据
        ax.plot(data.index, data['Close'], label='历史价格', linewidth=2, color='blue')
        
        # 绘制预测数据
        pred_dates = pd.to_datetime(prediction_dates)
        ax.plot(pred_dates, predictions, label='预测价格', linewidth=2, color='red', linestyle='--')
        
        # 添加连接线
        last_historical_date = data.index[-1]
        last_historical_price = data['Close'].iloc[-1]
        first_pred_date = pred_dates[0]
        first_pred_price = predictions[0]
        
        ax.plot([last_historical_date, first_pred_date], 
                [last_historical_price, first_pred_price], 
                color='red', linestyle='--', alpha=0.5)
        
        # 设置图表属性
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('日期', fontsize=12)
        ax.set_ylabel('价格 (元)', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 格式化x轴
        ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        return fig
    
    def plot_prediction_with_confidence(self, data: pd.DataFrame, predictions: List[float], 
                                      prediction_dates: List[str], confidence_interval: float,
                                      title: str = "基金价格预测结果（含置信区间）") -> plt.Figure:
        """
        绘制带置信区间的预测结果
        
        Args:
            data: 历史数据
            predictions: 预测值
            prediction_dates: 预测日期
            confidence_interval: 置信区间
            title: 图表标题
            
        Returns:
            matplotlib图表对象
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 绘制历史数据
        ax.plot(data.index, data['Close'], label='历史价格', linewidth=2, color='blue')
        
        # 绘制预测数据
        pred_dates = pd.to_datetime(prediction_dates)
        ax.plot(pred_dates, predictions, label='预测价格', linewidth=2, color='red', linestyle='--')
        
        # 绘制置信区间
        upper_bound = np.array(predictions) + confidence_interval
        lower_bound = np.array(predictions) - confidence_interval
        
        ax.fill_between(pred_dates, lower_bound, upper_bound, 
                       alpha=0.3, color='red', label='置信区间')
        
        # 添加连接线
        last_historical_date = data.index[-1]
        last_historical_price = data['Close'].iloc[-1]
        first_pred_date = pred_dates[0]
        first_pred_price = predictions[0]
        
        ax.plot([last_historical_date, first_pred_date], 
                [last_historical_price, first_pred_price], 
                color='red', linestyle='--', alpha=0.5)
        
        # 设置图表属性
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('日期', fontsize=12)
        ax.set_ylabel('价格 (元)', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 格式化x轴
        ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        return fig
    
    def plot_probability_predictions(self, probabilities: List[Dict], prediction_dates: List[str],
                                   title: str = "涨跌概率预测") -> plt.Figure:
        """
        绘制概率预测结果
        
        Args:
            probabilities: 概率预测结果
            prediction_dates: 预测日期
            title: 图表标题
            
        Returns:
            matplotlib图表对象
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        pred_dates = pd.to_datetime(prediction_dates)
        up_probs = [prob['up_probability'] for prob in probabilities]
        down_probs = [prob['down_probability'] for prob in probabilities]
        predicted_prices = [prob['predicted_price'] for prob in probabilities]
        current_price = probabilities[0]['current_price']
        
        # 绘制概率
        ax1.plot(pred_dates, up_probs, label='上涨概率', linewidth=2, color='green')
        ax1.plot(pred_dates, down_probs, label='下跌概率', linewidth=2, color='red')
        ax1.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, label='中性线')
        
        ax1.set_title(title, fontsize=16, fontweight='bold')
        ax1.set_ylabel('概率', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # 绘制预测价格
        ax2.plot(pred_dates, predicted_prices, label='预测价格', linewidth=2, color='blue')
        ax2.axhline(y=current_price, color='black', linestyle='-', alpha=0.7, label='当前价格')
        
        ax2.set_xlabel('日期', fontsize=12)
        ax2.set_ylabel('价格 (元)', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 格式化x轴
        ax1.tick_params(axis='x', rotation=45)
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        return fig
    
    def plot_ensemble_predictions(self, data: pd.DataFrame, ensemble_result: Dict[str, Any],
                                title: str = "集成预测结果") -> plt.Figure:
        """
        绘制集成预测结果
        
        Args:
            data: 历史数据
            ensemble_result: 集成预测结果
            title: 图表标题
            
        Returns:
            matplotlib图表对象
        """
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # 绘制历史数据
        ax.plot(data.index, data['Close'], label='历史价格', linewidth=2, color='blue')
        
        # 绘制集成预测
        predictions = ensemble_result['predictions']
        prediction_dates = ensemble_result['prediction_dates']
        pred_dates = pd.to_datetime(prediction_dates)
        
        ax.plot(pred_dates, predictions, label='集成预测', linewidth=3, color='red', linestyle='--')
        
        # 绘制各个预测器的结果
        colors = ['orange', 'green', 'purple', 'brown', 'pink']
        for i, individual_result in enumerate(ensemble_result.get('individual_results', [])):
            pred_name = individual_result['predictor']
            pred_values = individual_result['predictions']
            color = colors[i % len(colors)]
            
            ax.plot(pred_dates, pred_values, label=f'{pred_name}预测', 
                   linewidth=1, color=color, alpha=0.7)
        
        # 绘制置信区间
        if 'confidence_interval' in ensemble_result:
            confidence_interval = ensemble_result['confidence_interval']
            upper_bound = np.array(predictions) + confidence_interval
            lower_bound = np.array(predictions) - confidence_interval
            
            ax.fill_between(pred_dates, lower_bound, upper_bound, 
                           alpha=0.2, color='red', label='置信区间')
        
        # 添加连接线
        last_historical_date = data.index[-1]
        last_historical_price = data['Close'].iloc[-1]
        first_pred_date = pred_dates[0]
        first_pred_price = predictions[0]
        
        ax.plot([last_historical_date, first_pred_date], 
                [last_historical_price, first_pred_price], 
                color='red', linestyle='--', alpha=0.5)
        
        # 设置图表属性
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('日期', fontsize=12)
        ax.set_ylabel('价格 (元)', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 格式化x轴
        ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        return fig
    
    def plot_prediction_components(self, components: Dict[str, Any], 
                                 title: str = "预测组件分析") -> plt.Figure:
        """
        绘制预测组件（趋势、季节性等）
        
        Args:
            components: 预测组件数据
            title: 图表标题
            
        Returns:
            matplotlib图表对象
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        pred_dates = pd.to_datetime(components['prediction_dates'])
        
        # 绘制趋势
        axes[0].plot(pred_dates, components['trend'], label='趋势', linewidth=2, color='blue')
        axes[0].set_title('趋势组件', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('趋势值')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 绘制年度季节性
        axes[1].plot(pred_dates, components['yearly'], label='年度季节性', linewidth=2, color='green')
        axes[1].set_title('年度季节性', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('季节性值')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # 绘制周度季节性
        axes[2].plot(pred_dates, components['weekly'], label='周度季节性', linewidth=2, color='orange')
        axes[2].set_title('周度季节性', fontsize=12, fontweight='bold')
        axes[2].set_ylabel('季节性值')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        # 绘制节假日效应
        axes[3].plot(pred_dates, components['holidays'], label='节假日效应', linewidth=2, color='red')
        axes[3].set_title('节假日效应', fontsize=12, fontweight='bold')
        axes[3].set_ylabel('效应值')
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)
        
        # 格式化x轴
        for ax in axes:
            ax.tick_params(axis='x', rotation=45)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def plot_feature_importance(self, feature_importance: Dict[str, float], 
                              title: str = "特征重要性", top_n: int = 20) -> plt.Figure:
        """
        绘制特征重要性
        
        Args:
            feature_importance: 特征重要性字典
            title: 图表标题
            top_n: 显示前N个特征
            
        Returns:
            matplotlib图表对象
        """
        # 排序并选择前N个特征
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        features, importance = zip(*sorted_features)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 绘制水平条形图
        y_pos = np.arange(len(features))
        bars = ax.barh(y_pos, importance, color='skyblue')
        
        # 设置标签
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.set_xlabel('重要性', fontsize=12)
        ax.set_title(title, fontsize=16, fontweight='bold')
        
        # 添加数值标签
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                   f'{width:.3f}', ha='left', va='center')
        
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig
    
    def plot_prediction_accuracy(self, evaluation_results: Dict[str, Dict[str, float]],
                               title: str = "预测准确性比较") -> plt.Figure:
        """
        绘制预测准确性比较
        
        Args:
            evaluation_results: 评估结果
            title: 图表标题
            
        Returns:
            matplotlib图表对象
        """
        # 提取指标
        metrics = ['rmse', 'mae', 'mape', 'r2', 'direction_accuracy']
        predictors = list(evaluation_results.keys())
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            if i >= len(axes):
                break
                
            ax = axes[i]
            values = []
            labels = []
            
            for predictor in predictors:
                if metric in evaluation_results[predictor]:
                    values.append(evaluation_results[predictor][metric])
                    labels.append(predictor)
            
            if values:
                bars = ax.bar(labels, values, color='lightcoral')
                ax.set_title(f'{metric.upper()}', fontsize=12, fontweight='bold')
                ax.set_ylabel(metric.upper())
                
                # 添加数值标签
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                           f'{height:.3f}', ha='center', va='bottom')
                
                ax.tick_params(axis='x', rotation=45)
                ax.grid(True, alpha=0.3)
        
        # 隐藏多余的子图
        for i in range(len(metrics), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def plot_prediction_error_analysis(self, y_true: np.ndarray, y_pred: np.ndarray,
                                     title: str = "预测误差分析") -> plt.Figure:
        """
        绘制预测误差分析
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            title: 图表标题
            
        Returns:
            matplotlib图表对象
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 计算误差
        errors = y_pred - y_true
        abs_errors = np.abs(errors)
        relative_errors = errors / y_true * 100
        
        # 误差分布直方图
        axes[0, 0].hist(errors, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('误差分布', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('预测误差')
        axes[0, 0].set_ylabel('频次')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 绝对误差分布
        axes[0, 1].hist(abs_errors, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
        axes[0, 1].set_title('绝对误差分布', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('绝对误差')
        axes[0, 1].set_ylabel('频次')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 散点图：真实值 vs 预测值
        axes[1, 0].scatter(y_true, y_pred, alpha=0.6, color='green')
        axes[1, 0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
                       'r--', linewidth=2, label='完美预测线')
        axes[1, 0].set_title('真实值 vs 预测值', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('真实值')
        axes[1, 0].set_ylabel('预测值')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 相对误差分布
        axes[1, 1].hist(relative_errors, bins=30, alpha=0.7, color='gold', edgecolor='black')
        axes[1, 1].set_title('相对误差分布', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('相对误差 (%)')
        axes[1, 1].set_ylabel('频次')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig
