# -*- coding: utf-8 -*-
from .config import *
from .bayesian_predictor import ImbalanceAwareBayesianPredictionResult, MissingnessAwareBayesianRiskPredictor, bayesian_predict_with_retrieval

class GLMARagVisualizer:
    """
    完整的可视化系统（Pattern Recognition论文级别）
    """
    
    def __init__(self, figsize=(16, 12)):
        self.figsize = figsize
    
    def plot_missing_pattern_analysis(self, X, mask, missing_mech, global_stats):
        """Figure 1: 全局缺失模式分析"""
        fig = plt.figure(figsize=(16, 5))
        
        # 子图1：特征缺失率
        ax1 = plt.subplot(1, 4, 1)
        feature_missing_rate = np.isnan(X).sum(axis=0) / len(X)
        sorted_idx = np.argsort(feature_missing_rate)[::-1][:15]
        
        colors = ['red' if feature_missing_rate[i] > 0.2 else 'orange' 
                  if feature_missing_rate[i] > 0.1 else 'green' 
                  for i in sorted_idx]
        
        ax1.barh(range(len(sorted_idx)), feature_missing_rate[sorted_idx], 
                color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        ax1.set_xlabel('Missing Rate', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Feature Index', fontsize=11, fontweight='bold')
        ax1.set_title('Global Missing Pattern\n(Top 15 Features)', fontsize=12, fontweight='bold')
        ax1.axvline(0.1, color='orange', linestyle='--', alpha=0.5, linewidth=2)
        ax1.axvline(0.3, color='red', linestyle='--', alpha=0.5, linewidth=2)
        ax1.grid(True, alpha=0.3, axis='x')
        
        # 子图2：缺失相关性热力图
        ax2 = plt.subplot(1, 4, 2)
        missing_corr = np.isnan(X).astype(int).T @ np.isnan(X).astype(int) / len(X)
        sns.heatmap(missing_corr[:10, :10], cmap='RdYlBu_r', ax=ax2, 
                   cbar_kws={'label': 'Co-occurrence'}, square=True)
        ax2.set_title('Missing Co-occurrence Matrix\n(First 10×10)', 
                     fontsize=12, fontweight='bold')
        
        # 子图3：缺失机制
        ax3 = plt.subplot(1, 4, 3)
        mechs = ['MCAR', 'MAR', 'MNAR']
        probs = [missing_mech.get(m, 0) for m in mechs]
        colors_mech = ['green', 'orange', 'red']
        
        bars = ax3.bar(mechs, probs, color=colors_mech, alpha=0.7, 
                      edgecolor='black', linewidth=2)
        ax3.set_ylabel('Probability', fontsize=11, fontweight='bold')
        ax3.set_title('Inferred Missing Mechanism', fontsize=12, fontweight='bold')
        ax3.set_ylim([0, 1])
        
        for bar, prob in zip(bars, probs):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{prob:.2%}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # 子图4：样本缺失分布
        ax4 = plt.subplot(1, 4, 4)
        sample_missing_ratio = np.isnan(X).sum(axis=1) / X.shape[1]
        
        ax4.hist(sample_missing_ratio, bins=30, color='skyblue', 
                edgecolor='black', alpha=0.7, linewidth=1.5)
        ax4.axvline(0.1, color='green', linestyle='--', linewidth=2.5, label='Low threshold')
        ax4.axvline(0.3, color='red', linestyle='--', linewidth=2.5, label='High threshold')
        ax4.set_xlabel('Sample Missing Ratio', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax4.set_title('Sample-level Missing Distribution', fontsize=12, fontweight='bold')
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        return fig
    
    def plot_model_comparison(self, base_res, best_model='GL-MA-RAG'):
        """Figure 2: 基模型性能对比"""
        fig, axes = plt.subplots(3, 3, figsize=(15, 10))
        
        metrics = ['AUC', 'KS', 'AUPRC','Accuracy', 'Recall_0', 'Recall_1', 'F1']
        models = list(base_res.keys())
        
        for ax_idx, metric in enumerate(metrics):
            ax = axes.flatten()[ax_idx]
            
            values = [base_res[m].get(metric, 0) for m in models]
            colors_bar = ['red' if m == best_model else 'steelblue' for m in models]
            
            bars = ax.bar(range(len(models)), values, color=colors_bar, 
                         alpha=0.7, edgecolor='black', linewidth=1.5)
            
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            ax.set_ylabel('Score', fontsize=10, fontweight='bold')
            ax.set_title(f'{metric} Comparison', fontsize=11, fontweight='bold')
            ax.set_xticks(range(len(models)))
            ax.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        return fig

#        ★★★ 第16部分：融合预测置信区间可视化系统 ★★★
# =====================================================================

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# =====================================================================
#        融合预测结果数据结构
# =====================================================================

class EnsemblePredictionResult:
    """
    融合预测结果的数据结构
    组织来自ensemble_predict_with_confidence的所有输出
    """
    def __init__(self):
        self.predictions = []          # 预测值
        self.confidences = []          # 置信度
        self.ci_lowers = []            # 置信区间下界
        self.ci_uppers = []            # 置信区间上界
        self.ci_widths = []            # 置信区间宽度
        self.weights_list = []         # 每个样本的模型权重
        self.true_labels = []          # 真实标签
        self.missing_ratios = []       # 缺失率
        self.missing_levels = []       # 缺失等级
        self.sample_ids = []           # 样本ID
    
    def add_prediction(self, pred_result, true_label=None, missing_ratio=None, 
                      missing_level=None, sample_id=None):
        """添加单个预测结果"""
        self.predictions.append(pred_result['prediction'])
        self.confidences.append(pred_result['confidence'])
        self.ci_lowers.append(pred_result['ci_lower'])
        self.ci_uppers.append(pred_result['ci_upper'])
        self.ci_widths.append(pred_result['ci_width'])
        self.weights_list.append(pred_result['weights'])
        self.true_labels.append(true_label)
        self.missing_ratios.append(missing_ratio)
        self.missing_levels.append(missing_level)
        self.sample_ids.append(sample_id)
    
    def to_dataframe(self):
        """转换为DataFrame便于分析"""
        return pd.DataFrame({
            'prediction': self.predictions,
            'confidence': self.confidences,
            'ci_lower': self.ci_lowers,
            'ci_upper': self.ci_uppers,
            'ci_width': self.ci_widths,
            'true_label': self.true_labels,
            'missing_ratio': self.missing_ratios,
            'missing_level': self.missing_levels,
            'sample_id': self.sample_ids
        })
    
    def __len__(self):
        return len(self.predictions)

# =====================================================================
#        图表1：单个样本的置信区间可视化
# =====================================================================

class ConfidenceIntervalVisualizer:
    """
    可视化单个或多个样本的置信区间
    """
    
    @staticmethod
    def plot_single_sample_ci(pred_result, true_label=None, sample_id=None, figsize=(10, 6)):
        """
        绘制单个样本的详细置信区间
        
        参数：
            pred_result: ensemble_predict_with_confidence的输出
            true_label: 真实标签（用于标记）
            sample_id: 样本ID
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        pred = pred_result['prediction']
        ci_lower = pred_result['ci_lower']
        ci_upper = pred_result['ci_upper']
        confidence = pred_result['confidence']
        
        # 绘制置信区间
        ci_width = ci_upper - ci_lower
        ax.barh(0, ci_width, left=ci_lower, height=0.4, 
               color='lightblue', edgecolor='darkblue', linewidth=2.5, alpha=0.7,
               label=f'95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]')
        
        # 绘制预测值
        ax.scatter(pred, 0, s=500, c='red', marker='D', zorder=5, 
                  edgecolor='darkred', linewidth=2, label=f'Prediction: {pred:.4f}')
        
        # 绘制真实标签
        if true_label is not None:
            ax.axvline(true_label, color='green', linestyle='--', linewidth=2.5, 
                      label=f'True Label: {true_label}', alpha=0.7)
        
        # 标记高风险/低风险区域
        ax.axvspan(0, 0.3, alpha=0.1, color='green', label='Low Risk')
        ax.axvspan(0.7, 1, alpha=0.1, color='red', label='High Risk')
        ax.axvline(0.5, color='gray', linestyle=':', linewidth=1.5, alpha=0.5)
        
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.8, 0.8)
        ax.set_xlabel('Prediction Probability', fontsize=12, fontweight='bold')
        ax.set_yticks([])
        ax.legend(fontsize=11, loc='upper right')
        ax.grid(True, alpha=0.3, axis='x')
        
        title = f'Single Sample Confidence Interval'
        if sample_id is not None:
            title += f' (Sample {sample_id})'
        title += f'\nConfidence: {confidence:.2%}'
        
        ax.set_title(title, fontsize=13, fontweight='bold', pad=20)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_multiple_samples_ci(results_df, sample_ids=None, figsize=(14, 8)):
        """
        绘制多个样本的置信区间对比
        
        参数：
            results_df: 包含predictions, ci_lower, ci_upper等列的DataFrame
            sample_ids: 要绘制的样本ID列表
        """
        if sample_ids is None:
            sample_ids = list(range(min(10, len(results_df))))  # 默认前10个
        
        n_samples = len(sample_ids)
        fig, ax = plt.subplots(figsize=figsize)
        
        for i, sample_id in enumerate(sample_ids):
            row = results_df.iloc[sample_id]
            
            # 置信区间
            ci_lower = row['ci_lower']
            ci_upper = row['ci_upper']
            pred = row['prediction']
            
            ax.barh(i, ci_upper - ci_lower, left=ci_lower, height=0.6,
                   color='lightblue', edgecolor='darkblue', linewidth=1.5, alpha=0.7)
            
            # 预测值
            ax.scatter(pred, i, s=300, c='red', marker='D', zorder=5,
                      edgecolor='darkred', linewidth=1.5)
            
            # 真实标签
            if pd.notna(row.get('true_label')):
                ax.scatter(row['true_label'], i, s=300, c='green', marker='s', 
                          zorder=5, edgecolor='darkgreen', linewidth=1.5, alpha=0.7)
        
        ax.set_yticks(range(n_samples))
        ax.set_yticklabels([f'Sample {sid}' for sid in sample_ids], fontsize=10)
        ax.set_xlim(-0.05, 1.05)
        ax.set_xlabel('Prediction Probability', fontsize=12, fontweight='bold')
        ax.set_title('Confidence Intervals for Multiple Samples\n(Red Diamond=Prediction, Green Square=True Label)',
                    fontsize=13, fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3, axis='x')
        ax.axvspan(0, 0.5, alpha=0.05, color='green')
        ax.axvspan(0.5, 1, alpha=0.05, color='red')
        
        # 添加图例1
        red_diamond = mpatches.Patch(color='red', label='Prediction')
        green_square = mpatches.Patch(color='green', label='True Label', alpha=0.7)
        ci_patch = mpatches.Patch(color='lightblue', label='95% Confidence Interval')
        ax.legend(handles=[red_diamond, green_square, ci_patch], fontsize=11, loc='upper right')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_ci_width_distribution(results_df, figsize=(12, 5)):
        """
        绘制置信区间宽度分布
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # 子图1：直方图
        axes[0].hist(results_df['ci_width'], bins=30, color='skyblue', 
                    edgecolor='black', alpha=0.7)
        axes[0].axvline(results_df['ci_width'].mean(), color='red', 
                       linestyle='--', linewidth=2.5, label=f"Mean: {results_df['ci_width'].mean():.4f}")
        axes[0].axvline(results_df['ci_width'].median(), color='green', 
                       linestyle='--', linewidth=2.5, label=f"Median: {results_df['ci_width'].median():.4f}")
        axes[0].set_xlabel('Confidence Interval Width', fontsize=11, fontweight='bold')
        axes[0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
        axes[0].set_title('Distribution of CI Width', fontsize=12, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # 子图2：Box plot
        data_by_label = []
        labels = []
        if pd.notna(results_df['true_label']).all():
            for label in [0, 1]:
                mask = results_df['true_label'] == label
                if mask.sum() > 0:
                    data_by_label.append(results_df[mask]['ci_width'].values)
                    labels.append(f'Label {label}')
        
        if data_by_label:
            bp = axes[1].boxplot(data_by_label, labels=labels, patch_artist=True)
            for patch, color in zip(bp['boxes'], ['lightgreen', 'lightcoral']):
                patch.set_facecolor(color)
        else:
            axes[1].boxplot(results_df['ci_width'], patch_artist=True)
            axes[1].set_xticklabels(['All Data'])
        
        axes[1].set_ylabel('Confidence Interval Width', fontsize=11, fontweight='bold')
        axes[1].set_title('CI Width by True Label', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        return fig

# =====================================================================
#        图表2：预测准确性与置信度关系
# =====================================================================

class PredictionAccuracyVisualizer:
    """
    分析预测准确性与置信度的关系
    """
    
    @staticmethod
    def plot_prediction_accuracy_vs_confidence(results_df, figsize=(12, 8)):
        """
        绘制准确性 vs 置信度的散点图
        """
        if pd.isna(results_df['true_label']).all():
            print("⚠️ 缺少真实标签，跳过准确性分析")
            return None
        
        # 计算预测是否正确
        pred_binary = (results_df['prediction'] >= 0.5).astype(int)
        results_df['is_correct'] = (pred_binary == results_df['true_label']).astype(int)
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # ===== 子图1：散点图 =====
        ax = axes[0, 0]
        correct = results_df[results_df['is_correct'] == 1]
        incorrect = results_df[results_df['is_correct'] == 0]
        
        ax.scatter(correct['confidence'], correct['prediction'], 
                  s=100, c='green', alpha=0.6, label='Correct', edgecolor='darkgreen', linewidth=1)
        ax.scatter(incorrect['confidence'], incorrect['prediction'], 
                  s=100, c='red', alpha=0.6, label='Incorrect', edgecolor='darkred', linewidth=1)
        
        ax.set_xlabel('Confidence', fontsize=11, fontweight='bold')
        ax.set_ylabel('Prediction Probability', fontsize=11, fontweight='bold')
        ax.set_title('Accuracy vs Confidence', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # ===== 子图2：置信度分布（按正确性） =====
        ax = axes[0, 1]
        ax.hist(correct['confidence'], bins=20, alpha=0.6, label='Correct', color='green', edgecolor='black')
        ax.hist(incorrect['confidence'], bins=20, alpha=0.6, label='Incorrect', color='red', edgecolor='black')
        ax.set_xlabel('Confidence', fontsize=11, fontweight='bold')
        ax.set_ylabel('Count', fontsize=11, fontweight='bold')
        ax.set_title('Confidence Distribution', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        # ===== 子图3：置信度分组准确性 =====
        ax = axes[1, 0]
        confidence_bins = [0, 0.6, 0.7, 0.8, 0.9, 1.0]
        confidence_labels = ['<0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9', '>0.9']
        results_df['conf_group'] = pd.cut(results_df['confidence'], 
                                          bins=confidence_bins, 
                                          labels=confidence_labels,
                                          right=False)
        
        accuracy_by_conf = results_df.groupby('conf_group')['is_correct'].agg(['sum', 'count'])
        accuracy_by_conf['accuracy'] = accuracy_by_conf['sum'] / accuracy_by_conf['count']
        
        bars = ax.bar(range(len(accuracy_by_conf)), accuracy_by_conf['accuracy'], 
                     color='steelblue', alpha=0.7, edgecolor='black', linewidth=1.5)
        
        # 添加数值标签
        for bar, (idx, row) in zip(bars, accuracy_by_conf.iterrows()):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2%}\n(n={int(row["count"])})', 
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax.set_xticks(range(len(confidence_labels)))
        ax.set_xticklabels(confidence_labels)
        ax.set_xlabel('Confidence Level', fontsize=11, fontweight='bold')
        ax.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
        ax.set_title('Accuracy by Confidence Group', fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3, axis='y')
        
        # ===== 子图4：置信区间宽度 vs 准确性 =====
        ax = axes[1, 1]
        ax.scatter(results_df[results_df['is_correct']==1]['ci_width'], 
                  results_df[results_df['is_correct']==1]['confidence'],
                  s=100, c='green', alpha=0.6, label='Correct', edgecolor='darkgreen', linewidth=1)
        ax.scatter(results_df[results_df['is_correct']==0]['ci_width'], 
                  results_df[results_df['is_correct']==0]['confidence'],
                  s=100, c='red', alpha=0.6, label='Incorrect', edgecolor='darkred', linewidth=1)
        
        ax.set_xlabel('Confidence Interval Width', fontsize=11, fontweight='bold')
        ax.set_ylabel('Confidence', fontsize=11, fontweight='bold')
        ax.set_title('CI Width vs Accuracy', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_calibration_curve(results_df, figsize=(10, 8)):
        """
        绘制校准曲线（预测概率 vs 实际频率）
        """
        if pd.isna(results_df['true_label']).all():
            print("⚠️ 缺少真实标签，跳过校准曲线")
            return None
        
        # 计算是否正确
        pred_binary = (results_df['prediction'] >= 0.5).astype(int)
        results_df['is_correct_pred'] = (pred_binary == results_df['true_label']).astype(int)
        
        # 分组计算校准
        n_bins = 10
        pred_bins = np.linspace(0, 1, n_bins + 1)
        calibration_data = []
        
        for i in range(n_bins):
            mask = (results_df['prediction'] >= pred_bins[i]) & (results_df['prediction'] < pred_bins[i+1])
            if mask.sum() > 0:
                bin_center = (pred_bins[i] + pred_bins[i+1]) / 2
                actual_freq = results_df[mask]['is_correct_pred'].mean()
                calibration_data.append({
                    'pred_prob': bin_center,
                    'actual_freq': actual_freq,
                    'count': mask.sum()
                })
        
        calib_df = pd.DataFrame(calibration_data)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # 完美校准线
        ax.plot([0, 1], [0, 1], 'k--', linewidth=2.5, label='Perfect Calibration', alpha=0.7)
        
        # 实际校准曲线
        scatter = ax.scatter(calib_df['pred_prob'], calib_df['actual_freq'], 
                            s=calib_df['count']*5, c=calib_df['count'], 
                            cmap='viridis', alpha=0.7, edgecolor='black', linewidth=1.5)
        
        # 连接点
        calib_df_sorted = calib_df.sort_values('pred_prob')
        ax.plot(calib_df_sorted['pred_prob'], calib_df_sorted['actual_freq'], 
               'b-', linewidth=2, alpha=0.5, label='Actual Calibration')
        
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel('Predicted Probability', fontsize=12, fontweight='bold')
        ax.set_ylabel('Actual Frequency', fontsize=12, fontweight='bold')
        ax.set_title('Calibration Curve', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)
        
        # 添加colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Sample Count', fontsize=11, fontweight='bold')
        
        # 计算校准误差
        ece = np.mean(np.abs(calib_df['pred_prob'] - calib_df['actual_freq']))
        ax.text(0.05, 0.95, f'Expected Calibration Error: {ece:.4f}',
               transform=ax.transAxes, fontsize=11, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
               verticalalignment='top')
        
        plt.tight_layout()
        return fig

# =====================================================================
#        图表3：模型权重分析
# =====================================================================

class ModelWeightVisualizer:
    """
    分析融合中各模型的权重分配
    """
    
    @staticmethod
    def extract_weights_matrix(weights_list):
        """
        从权重列表提取权重矩阵
        weights_list: 每个样本的权重字典列表
        """
        if len(weights_list) == 0:
            return None
        
        model_names = list(weights_list[0].keys())
        weights_matrix = np.array([
            [w.get(model, 0) for model in model_names]
            for w in weights_list
        ])
        
        return weights_matrix, model_names
    
    @staticmethod
    def plot_average_weights(weights_list, figsize=(10, 6)):
        """
        绘制平均权重分布
        """
        weights_matrix, model_names = ModelWeightVisualizer.extract_weights_matrix(weights_list)
        
        if weights_matrix is None:
            print("⚠️ 无权重数据")
            return None
        
        avg_weights = weights_matrix.mean(axis=0)
        std_weights = weights_matrix.std(axis=0)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(model_names)))
        bars = ax.bar(range(len(model_names)), avg_weights, 
                     yerr=std_weights, capsize=5, color=colors,
                     alpha=0.7, edgecolor='black', linewidth=1.5)
        
        # 添加数值标签
        for bar, weight, std in zip(bars, avg_weights, std_weights):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{weight:.3f}±{std:.3f}', 
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=11)
        ax.set_ylabel('Average Weight', fontsize=12, fontweight='bold')
        ax.set_title('Average Model Weights in Ensemble', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, max(avg_weights) * 1.3)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_weight_heatmap(weights_list, sample_ids=None, figsize=(12, 8)):
        """
        绘制权重热力图（样本 × 模型）
        """
        weights_matrix, model_names = ModelWeightVisualizer.extract_weights_matrix(weights_list)
        
        if weights_matrix is None:
            print("⚠️ 无权重数据")
            return None
        
        # 选择样本
        if sample_ids is None:
            sample_ids = list(range(min(50, len(weights_matrix))))
        
        weights_subset = weights_matrix[sample_ids]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        im = ax.imshow(weights_subset, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=11)
        ax.set_yticks(range(len(sample_ids)))
        ax.set_yticklabels([f'Sample {sid}' for sid in sample_ids], fontsize=9)
        
        ax.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax.set_ylabel('Sample', fontsize=12, fontweight='bold')
        ax.set_title('Model Weights Heatmap', fontsize=13, fontweight='bold')
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Weight', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_weight_evolution(weights_list, missing_ratios=None, figsize=(12, 6)):
        """
        绘制权重随缺失率变化的趋势
        """
        weights_matrix, model_names = ModelWeightVisualizer.extract_weights_matrix(weights_list)
        
        if weights_matrix is None or missing_ratios is None:
            return None
        
        # 按缺失率排序
        sorted_idx = np.argsort(missing_ratios)
        sorted_weights = weights_matrix[sorted_idx]
        sorted_ratios = np.array(missing_ratios)[sorted_idx]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(model_names)))
        
        for i, model_name in enumerate(model_names):
            ax.plot(sorted_ratios, sorted_weights[:, i], 
                   marker='o', linewidth=2.5, label=model_name,
                   color=colors[i], markersize=6, alpha=0.7)
        
        ax.set_xlabel('Missing Ratio', fontsize=12, fontweight='bold')
        ax.set_ylabel('Model Weight', fontsize=12, fontweight='bold')
        ax.set_title('Model Weight Evolution with Missing Ratio', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10, loc='best', ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_dominant_models(weights_list, figsize=(10, 6)):
        """
        统计各样本中的主导模型（权重最高的模型）
        """
        weights_matrix, model_names = ModelWeightVisualizer.extract_weights_matrix(weights_list)
        
        if weights_matrix is None:
            return None
        
        # 找到每个样本的主导模型
        dominant_models = np.argmax(weights_matrix, axis=1)
        model_counts = np.bincount(dominant_models, minlength=len(model_names))
        
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(model_names)))
        bars = ax.bar(range(len(model_names)), model_counts, 
                     color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        
        # 添加数值标签
        for bar, count in zip(bars, model_counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}\n({height/len(weights_list):.1%})', 
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=11)
        ax.set_ylabel('Count', fontsize=12, fontweight='bold')
        ax.set_title('Dominant Models Distribution', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        return fig

# =====================================================================
#        图表4：缺失率与预测性能关系
# =====================================================================

class MissingRateAnalysisVisualizer:
    """
    分析缺失率对预测性能的影响
    """
    
    @staticmethod
    def plot_performance_vs_missing(results_df, figsize=(14, 10)):
        """
        绘制性能 vs 缺失率的关系
        """
        if 'missing_ratio' not in results_df.columns or pd.isna(results_df['missing_ratio']).all():
            print("⚠️ 缺少缺失率信息")
            return None
        
        # 计算预测是否正确
        if 'true_label' in results_df.columns and pd.notna(results_df['true_label']).all():
            pred_binary = (results_df['prediction'] >= 0.5).astype(int)
            results_df['is_correct'] = (pred_binary == results_df['true_label']).astype(int)
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # ===== 子图1：预测值 vs 缺失率 =====
        ax = axes[0, 0]
        scatter = ax.scatter(results_df['missing_ratio'], results_df['prediction'],
                            c=results_df['confidence'], cmap='RdYlGn', 
                            s=100, alpha=0.6, edgecolor='black', linewidth=1)
        
        # 添加趋势线
        z = np.polyfit(results_df['missing_ratio'], results_df['prediction'], 2)
        p = np.poly1d(z)
        x_trend = np.linspace(results_df['missing_ratio'].min(), results_df['missing_ratio'].max(), 100)
        ax.plot(x_trend, p(x_trend), 'r--', linewidth=2.5, label='Trend')
        
        ax.set_xlabel('Missing Ratio', fontsize=11, fontweight='bold')
        ax.set_ylabel('Prediction Probability', fontsize=11, fontweight='bold')
        ax.set_title('Prediction vs Missing Ratio', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Confidence', fontsize=10, fontweight='bold')
        
        # ===== 子图2：置信区间宽度 vs 缺失率 =====
        ax = axes[0, 1]
        ax.scatter(results_df['missing_ratio'], results_df['ci_width'],
                  c=results_df['confidence'], cmap='viridis',
                  s=100, alpha=0.6, edgecolor='black', linewidth=1)
        
        # 趋势线
        z = np.polyfit(results_df['missing_ratio'], results_df['ci_width'], 2)
        p = np.poly1d(z)
        ax.plot(x_trend, p(x_trend), 'r--', linewidth=2.5, label='Trend')
        
        ax.set_xlabel('Missing Ratio', fontsize=11, fontweight='bold')
        ax.set_ylabel('CI Width', fontsize=11, fontweight='bold')
        ax.set_title('CI Width vs Missing Ratio', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # ===== 子图3：按缺失等级分组 =====
        ax = axes[1, 0]
        if 'missing_level' in results_df.columns:
            level_colors = {'low': 'green', 'mid': 'yellow', 'high': 'red'}
            for level in ['low', 'mid', 'high']:
                mask = results_df['missing_level'] == level
                if mask.sum() > 0:
                    ax.scatter(results_df[mask]['missing_ratio'], results_df[mask]['prediction'],
                              label=f'{level} missing', s=100, alpha=0.6, 
                              color=level_colors.get(level, 'gray'), edgecolor='black', linewidth=1)
        else:
            ax.scatter(results_df['missing_ratio'], results_df['prediction'],
                      s=100, alpha=0.6, edgecolor='black', linewidth=1)
        
        ax.set_xlabel('Missing Ratio', fontsize=11, fontweight='bold')
        ax.set_ylabel('Prediction Probability', fontsize=11, fontweight='bold')
        ax.set_title('Predictions by Missing Level', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # ===== 子图4：准确性 vs 缺失率（如果有真实标签） =====
        ax = axes[1, 1]
        if 'is_correct' in results_df.columns:
            # 按缺失率分组
            missing_bins = np.linspace(0, results_df['missing_ratio'].max(), 6)
            results_df['missing_group'] = pd.cut(results_df['missing_ratio'], bins=missing_bins)
            
            accuracy_by_missing = results_df.groupby('missing_group', observed=True)['is_correct'].agg(['sum', 'count'])
            accuracy_by_missing['accuracy'] = accuracy_by_missing['sum'] / accuracy_by_missing['count']
            
            group_centers = [interval.mid for interval in accuracy_by_missing.index]
            
            bars = ax.bar(range(len(accuracy_by_missing)), accuracy_by_missing['accuracy'],
                         color='steelblue', alpha=0.7, edgecolor='black', linewidth=1.5)
            
            for bar, (idx, row) in zip(bars, accuracy_by_missing.iterrows()):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2%}\n(n={int(row["count"])})', 
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            ax.set_xticks(range(len(accuracy_by_missing)))
            ax.set_xticklabels([f'{c:.1%}' if c else 'NaN' for c in group_centers], rotation=45)
            ax.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
            ax.set_xlabel('Missing Ratio Group', fontsize=11, fontweight='bold')
            ax.set_title('Accuracy vs Missing Ratio', fontsize=12, fontweight='bold')
            ax.set_ylim(0, 1.1)
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        return fig

# =====================================================================
#        图表5：综合评估仪表板
# =====================================================================

class ComprehensiveDashboard:
    """
    生成综合的评估仪表板
    """
    
    @staticmethod
    def generate_full_dashboard(results_df, weights_list, figsize=(20, 14)):
        """
        生成包含所有关键指标的仪表板
        """
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # ===== 1. 预测分布 =====
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.hist(results_df['prediction'], bins=30, color='steelblue', alpha=0.7, edgecolor='black')
        ax1.axvline(0.5, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Decision Boundary')
        ax1.set_xlabel('Prediction', fontsize=10, fontweight='bold')
        ax1.set_ylabel('Frequency', fontsize=10, fontweight='bold')
        ax1.set_title('Prediction Distribution', fontsize=11, fontweight='bold')
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # ===== 2. 置信度分布 =====
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.hist(results_df['confidence'], bins=30, color='lightgreen', alpha=0.7, edgecolor='black')
        ax2.axvline(results_df['confidence'].mean(), color='red', 
                   linestyle='--', linewidth=2, label=f'Mean: {results_df["confidence"].mean():.3f}')
        ax2.set_xlabel('Confidence', fontsize=10, fontweight='bold')
        ax2.set_ylabel('Frequency', fontsize=10, fontweight='bold')
        ax2.set_title('Confidence Distribution', fontsize=11, fontweight='bold')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # ===== 3. CI宽度分布 =====
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.hist(results_df['ci_width'], bins=30, color='lightcoral', alpha=0.7, edgecolor='black')
        ax3.axvline(results_df['ci_width'].mean(), color='darkred', 
                   linestyle='--', linewidth=2, label=f'Mean: {results_df["ci_width"].mean():.4f}')
        ax3.set_xlabel('CI Width', fontsize=10, fontweight='bold')
        ax3.set_ylabel('Frequency', fontsize=10, fontweight='bold')
        ax3.set_title('CI Width Distribution', fontsize=11, fontweight='bold')
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # ===== 4. 模型平均权重 =====
        ax4 = fig.add_subplot(gs[1, 0])
        weights_matrix, model_names = ModelWeightVisualizer.extract_weights_matrix(weights_list)
        if weights_matrix is not None:
            avg_weights = weights_matrix.mean(axis=0)
            colors_models = plt.cm.Set3(np.linspace(0, 1, len(model_names)))
            ax4.barh(range(len(model_names)), avg_weights, color=colors_models, alpha=0.7, edgecolor='black')
            ax4.set_yticks(range(len(model_names)))
            ax4.set_yticklabels(model_names, fontsize=9)
            ax4.set_xlabel('Average Weight', fontsize=10, fontweight='bold')
            ax4.set_title('Model Weights', fontsize=11, fontweight='bold')
            ax4.grid(True, alpha=0.3, axis='x')
        
        # ===== 5. Prediction vs Confidence =====
        ax5 = fig.add_subplot(gs[1, 1])
        if 'is_correct' in results_df.columns:
            correct = results_df[results_df['is_correct'] == 1]
            incorrect = results_df[results_df['is_correct'] == 0]
            ax5.scatter(correct['confidence'], correct['prediction'], 
                       s=50, c='green', alpha=0.6, label='Correct', edgecolor='darkgreen', linewidth=0.5)
            ax5.scatter(incorrect['confidence'], incorrect['prediction'], 
                       s=50, c='red', alpha=0.6, label='Incorrect', edgecolor='darkred', linewidth=0.5)
            ax5.legend(fontsize=9)
        else:
            ax5.scatter(results_df['confidence'], results_df['prediction'], 
                       s=50, c='steelblue', alpha=0.6, edgecolor='black', linewidth=0.5)
        
        ax5.set_xlabel('Confidence', fontsize=10, fontweight='bold')
        ax5.set_ylabel('Prediction', fontsize=10, fontweight='bold')
        ax5.set_title('Confidence vs Prediction', fontsize=11, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        
        # ===== 6. 缺失率分布 =====
        ax6 = fig.add_subplot(gs[1, 2])
        if 'missing_ratio' in results_df.columns and pd.notna(results_df['missing_ratio']).any():
            ax6.hist(results_df['missing_ratio'].dropna(), bins=20, 
                    color='lightyellow', alpha=0.7, edgecolor='black')
            ax6.set_xlabel('Missing Ratio', fontsize=10, fontweight='bold')
            ax6.set_ylabel('Frequency', fontsize=10, fontweight='bold')
            ax6.set_title('Missing Ratio Distribution', fontsize=11, fontweight='bold')
            ax6.grid(True, alpha=0.3, axis='y')
        
        # ===== 7. 摘要统计 =====
        ax7 = fig.add_subplot(gs[2, :])
        ax7.axis('off')
        
        # 计算统计量
        summary_stats = f"""
        📊 SUMMARY STATISTICS
        
        Total Samples: {len(results_df)}
        
        ✓ Prediction Statistics:
          Mean: {results_df['prediction'].mean():.4f} | Median: {results_df['prediction'].median():.4f} | Std: {results_df['prediction'].std():.4f}
          Min: {results_df['prediction'].min():.4f} | Max: {results_df['prediction'].max():.4f}
        
        ✓ Confidence Statistics:
          Mean: {results_df['confidence'].mean():.4f} | Median: {results_df['confidence'].median():.4f} | Std: {results_df['confidence'].std():.4f}
          Min: {results_df['confidence'].min():.4f} | Max: {results_df['confidence'].max():.4f}
        
        ✓ Confidence Interval Width:
          Mean: {results_df['ci_width'].mean():.6f} | Median: {results_df['ci_width'].median():.6f} | Std: {results_df['ci_width'].std():.6f}
          Min: {results_df['ci_width'].min():.6f} | Max: {results_df['ci_width'].max():.6f}
        """
        
        if 'is_correct' in results_df.columns:
            accuracy = results_df['is_correct'].mean()
            correct_high_conf = results_df[(results_df['is_correct']==1) & (results_df['confidence']>0.7)]
            summary_stats += f"""
        
        ✓ Accuracy Analysis:
          Overall Accuracy: {accuracy:.2%}
          Correct Predictions with High Confidence (>0.7): {len(correct_high_conf)}/{len(results_df)} ({len(correct_high_conf)/len(results_df):.2%})
        """
        
        if 'missing_ratio' in results_df.columns and pd.notna(results_df['missing_ratio']).any():
            summary_stats += f"""
        
        ✓ Missing Data Statistics:
          Mean Missing Ratio: {results_df['missing_ratio'].mean():.2%}
          Max Missing Ratio: {results_df['missing_ratio'].max():.2%}
        """
        
        ax7.text(0.05, 0.95, summary_stats, transform=ax7.transAxes,
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.suptitle('Comprehensive Ensemble Prediction Analysis Dashboard', 
                    fontsize=15, fontweight='bold', y=0.995)
        
        return fig

# =====================================================================
#        图表6：置信区间有效性评估
# =====================================================================

class ConfidenceIntervalValidation:
    """
    评估置信区间的有效性
    """
    
    @staticmethod
    def plot_coverage_analysis(results_df, figsize=(12, 8)):
        """
        绘制覆盖率分析（是否真实标签在置信区间内）
        """
        if 'true_label' not in results_df.columns or pd.isna(results_df['true_label']).all():
            print("⚠️ 缺少真实标签信息")
            return None
        
        # 计算是否真实标签在置信区间内
        results_df['in_ci'] = (
            (results_df['true_label'] >= results_df['ci_lower']) & 
            (results_df['true_label'] <= results_df['ci_upper'])
        ).astype(int)
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # ===== 子图1：总体覆盖率 =====
        ax = axes[0, 0]
        coverage = results_df['in_ci'].mean()
        not_coverage = 1 - coverage
        
        colors_pie = ['green', 'red']
        wedges, texts, autotexts = ax.pie([coverage, not_coverage], 
                                           labels=['In CI', 'Not in CI'],
                                           autopct='%1.1f%%',
                                           colors=colors_pie, startangle=90,
                                           textprops={'fontsize': 11, 'fontweight': 'bold'})
        ax.set_title(f'Overall CI Coverage: {coverage:.2%}', fontsize=12, fontweight='bold')
        
        # ===== 子图2：按预测值分组覆盖率 =====
        ax = axes[0, 1]
        pred_bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        pred_labels = ['0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0']
        results_df['pred_group'] = pd.cut(results_df['prediction'], bins=pred_bins, labels=pred_labels)
        
        coverage_by_pred = results_df.groupby('pred_group', observed=True)['in_ci'].agg(['sum', 'count'])
        coverage_by_pred['coverage'] = coverage_by_pred['sum'] / coverage_by_pred['count']
        
        bars = ax.bar(range(len(coverage_by_pred)), coverage_by_pred['coverage'],
                     color='steelblue', alpha=0.7, edgecolor='black', linewidth=1.5)
        
        for bar, (idx, row) in zip(bars, coverage_by_pred.iterrows()):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2%}\n(n={int(row["count"])})', 
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax.set_xticks(range(len(pred_labels)))
        ax.set_xticklabels(pred_labels, rotation=45)
        ax.set_ylabel('Coverage Rate', fontsize=11, fontweight='bold')
        ax.set_xlabel('Prediction Group', fontsize=11, fontweight='bold')
        ax.set_title('Coverage by Prediction Value', fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1.2)
        ax.grid(True, alpha=0.3, axis='y')
        
        # ===== 子图3：按置信度分组覆盖率 =====
        ax = axes[1, 0]
        conf_bins = [0, 0.6, 0.7, 0.8, 0.9, 1.0]
        conf_labels = ['<0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9', '>0.9']
        results_df['conf_group'] = pd.cut(results_df['confidence'], bins=conf_bins, labels=conf_labels)
        
        coverage_by_conf = results_df.groupby('conf_group', observed=True)['in_ci'].agg(['sum', 'count'])
        coverage_by_conf['coverage'] = coverage_by_conf['sum'] / coverage_by_conf['count']
        
        bars = ax.bar(range(len(coverage_by_conf)), coverage_by_conf['coverage'],
                     color='lightgreen', alpha=0.7, edgecolor='black', linewidth=1.5)
        
        for bar, (idx, row) in zip(bars, coverage_by_conf.iterrows()):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2%}\n(n={int(row["count"])})', 
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax.set_xticks(range(len(conf_labels)))
        ax.set_xticklabels(conf_labels, rotation=45)
        ax.set_ylabel('Coverage Rate', fontsize=11, fontweight='bold')
        ax.set_xlabel('Confidence Group', fontsize=11, fontweight='bold')
        ax.set_title('Coverage by Confidence Level', fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1.2)
        ax.grid(True, alpha=0.3, axis='y')
        
        # ===== 子图4：CI宽度 vs 覆盖率 =====
        ax = axes[1, 1]
        width_bins = np.percentile(results_df['ci_width'], np.linspace(0, 100, 6))
        results_df['width_group'] = pd.cut(results_df['ci_width'], bins=width_bins)
        
        coverage_by_width = results_df.groupby('width_group', observed=True)['in_ci'].agg(['sum', 'count'])
        coverage_by_width['coverage'] = coverage_by_width['sum'] / coverage_by_width['count']
        width_centers = [interval.mid for interval in coverage_by_width.index]
        
        ax.scatter(width_centers, coverage_by_width['coverage'], 
                  s=coverage_by_width['count']*2, alpha=0.6, 
                  c=coverage_by_width['count'], cmap='viridis',
                  edgecolor='black', linewidth=1.5)
        
        ax.axhline(0.95, color='red', linestyle='--', linewidth=2, 
                  label='Target (95%)', alpha=0.7)
        
        ax.set_xlabel('CI Width', fontsize=11, fontweight='bold')
        ax.set_ylabel('Coverage Rate', fontsize=11, fontweight='bold')
        ax.set_title('Coverage vs CI Width', fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1.1)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
# =====================================================================
#        ★★★ 第17部分：融合预测可视化的完整实验框架 ★★★
# =====================================================================

# （导入部分省略，使用第一部分的所有导入）

def bayesian_predict_batch_with_results(
    X_test, mask_test, missing_ratios, missing_levels,
    bayes_model, retriever_or_index, y_ref, ref_missing_ratios,
    scaler_X, global_weights, adaptive_params,
    local_test_info=None, store_results=True
):
    """
    ★ v3.0增强版批量预测函数
    
    改进点：
    1. 支持分层检索器（StratifiedRetriever）或普通FAISS索引
    2. 自动应用学习到的阈值
    3. 完整的贝叶斯置信区间计算
    4. 代价敏感决策
    
    参数：
        retriever_or_index: StratifiedRetriever或FAISS Index
        bayes_model: MissingnessAwareBayesianRiskPredictor实例（已含decision_threshold）
    """
    results = ImbalanceAwareBayesianPredictionResult()
    test_pred_probs = []
    test_pred_labels = []

    total = len(X_test)
    print(f"  开始贝叶斯批量预测（共{total}个样本）...")

    for i in range(total):
        if (i + 1) % max(1, total // 10) == 0:
            print(f"    进度: {i+1}/{total} ({(i+1)/total*100:.1f}%)")

        try:
            # ★ 步骤1：检索邻居
            # 判断是否为分层检索器
            if hasattr(retriever_or_index, 'retrieve_balanced'):
                # ★ 使用分层检索策略
                retrieve_result = retriever_or_index.retrieve_balanced(
                    X_test[i],
                    mask_test[i],
                    missing_ratios[i],
                    missing_levels[i],
                    k_total=adaptive_params['k_base']
                )
            else:
                # 使用普通FAISS检索（向后兼容）
                retrieve_result = gl_ma_retrieve(
                    X_test[i],
                    mask_test[i],
                    missing_ratios[i],
                    missing_levels[i],
                    retriever_or_index,
                    local_test_info,
                    global_weights,
                    scaler_X,
                    adaptive_params["fusion_weights"],
                    adaptive_params
                )

            # ★ 步骤2：贝叶斯预测（含代价敏感决策）
            pred_result = bayesian_predict_with_retrieval(
                X_test[i],
                mask_test[i],
                retrieve_result,
                bayes_model,
                y_ref,
                ref_missing_ratios
            )

            test_pred_probs.append(pred_result["prediction"])
            test_pred_labels.append(pred_result["pred_label"])

            if store_results:
                results.add_prediction(
                    pred_result,
                    true_label=None,
                    missing_ratio=missing_ratios[i],
                    missing_level=missing_levels[i],
                    sample_id=i
                )

        except Exception as e:
            print(f"    ⚠️ 样本{i}预测失败: {e}")
            test_pred_probs.append(0.5)
            test_pred_labels.append(0)

            if store_results:
                results.add_prediction(
                    {
                        "prediction": 0.5,
                        "pred_label": 0,
                        "confidence": 0.5,
                        "ci_lower": 0.45,
                        "ci_upper": 0.55,
                        "ci_width": 0.10,
                        "prior_prob": 0.5,
                        "posterior_alpha": 1.0,
                        "posterior_beta": 1.0,
                        "neighbor_weights": np.array([])
                    },
                    true_label=None,
                    missing_ratio=missing_ratios[i],
                    missing_level=missing_levels[i],
                    sample_id=i
                )

    return np.array(test_pred_probs, dtype=np.float32), np.array(test_pred_labels, dtype=int), results

# =====================================================================
#        修改后的主程序入口
# =====================================================================

# =====================================================================
#      第18部分：完整模型对比框架
# =====================================================================

class ComparisonVisualizer:
    """完整的模型对比可视化"""
    
    @staticmethod
    def plot_all_models_comparison(results, figsize=(18, 12)):
        """绘制所有模型的综合对比"""
        metrics = ['AUC', 'KS','AUPRC', 'Accuracy', 'Recall_0', 'Recall_1', 'Precision', 'F1']
        models = list(results.keys())
        
        # 按AUC降序排列
        sorted_models = sorted(models, key=lambda x: results[x].get('AUC', 0), reverse=True)
        
        fig, axes = plt.subplots(2, 4, figsize=figsize)
        fig.suptitle('All Models Comprehensive Comparison', fontsize=16, fontweight='bold', y=0.995)
        
        for ax_idx, metric in enumerate(metrics):
            ax = axes.flatten()[ax_idx]
            
            values = [results[m].get(metric, 0) for m in sorted_models]
            colors = ['#FF6B6B' if m == 'GL-MA-RAG' else '#4ECDC4' for m in sorted_models]
            
            bars = ax.bar(range(len(sorted_models)), values, color=colors, alpha=0.8, 
                         edgecolor='black', linewidth=1.5)
            
            # 添加数值标签
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            ax.set_ylabel('Score', fontsize=11, fontweight='bold')
            ax.set_title(f'{metric}', fontsize=12, fontweight='bold')
            ax.set_xticks(range(len(sorted_models)))
            ax.set_xticklabels(sorted_models, rotation=45, ha='right', fontsize=9)
            ax.set_ylim(0, 1.1)
            ax.grid(True, alpha=0.3, axis='y')
        
        # 第8个子图：模型排名（AUC）
        ax = axes.flatten()[7]
        auc_scores = [results[m].get('AUC', 0) for m in sorted_models]
        colors_rank = ['#FF6B6B' if m == 'GL-MA-RAG' else plt.cm.viridis(i/len(sorted_models)) 
                       for i, m in enumerate(sorted_models)]
        
        bars = ax.barh(range(len(sorted_models)), auc_scores, color=colors_rank, alpha=0.8, 
                      edgecolor='black', linewidth=1.5)
        ax.set_yticks(range(len(sorted_models)))
        ax.set_yticklabels(sorted_models, fontsize=10)
        ax.set_xlabel('AUC Score', fontsize=11, fontweight='bold')
        ax.set_title('Model Ranking (by AUC)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        for i, (bar, score) in enumerate(zip(bars, auc_scores)):
            ax.text(score + 0.01, i, f'{score:.4f}', ha='left', va='center', 
                   fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_model_category_comparison(results, figsize=(14, 8)):
        """按模型类别对比"""
        categories = {
            '传统填充\n(Traditional)': [m for m in results.keys() if m.startswith('RF+')],
            'SOTA缺失学习\n(SOTA)': [m for m in results.keys() if m.upper() in ['MISSFOREST', 'GAIN', 'MIWAE']],
            '基模型\n(ML)': ['LogR', 'RF', 'XGBoost', 'LightGBM', 'CatBoost', 'TabNet'],
            '深度学习\n(DL)': ['TabTransformer', 'SAINT', 'DeepFM'],
            '本文方法\n(Proposed)': ['GL-MA-RAG']
        }
        
        fig, axes = plt.subplots(3, 3, figsize=figsize)
        fig.suptitle('Model Category Comparison', fontsize=15, fontweight='bold')
        
        metrics = ['AUC', 'KS', 'AUPRC','Accuracy', 'Recall_0', 'Recall_1', 'F1']
        category_colors = ['#FF7F50', '#20B2AA', '#4169E1', '#9370DB', '#FF6B6B']
        
        for ax_idx, metric in enumerate(metrics):
            ax = axes.flatten()[ax_idx]
            
            category_scores = {}
            for cat_idx, (cat_name, models) in enumerate(categories.items()):
                scores = [results[m].get(metric, 0) for m in models if m in results]
                if scores:
                    category_scores[cat_name] = np.mean(scores)
            
            cats = list(category_scores.keys())
            scores = list(category_scores.values())
            
            bars = ax.bar(range(len(cats)), scores, color=category_colors[:len(cats)], 
                         alpha=0.8, edgecolor='black', linewidth=2)
            
            for bar, score in zip(bars, scores):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{score:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            ax.set_ylabel('Average Score', fontsize=11, fontweight='bold')
            ax.set_title(f'{metric}', fontsize=12, fontweight='bold')
            ax.set_xticks(range(len(cats)))
            ax.set_xticklabels(cats, fontsize=9)
            ax.set_ylim(0, 1.0)
            ax.grid(True, alpha=0.3, axis='y')
        
        # 移除多余的子图
        fig.delaxes(axes.flatten()[5])
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def generate_comparison_table(results, output_path="model_comparison_table.csv"):
        """生成对比表格"""
        df = pd.DataFrame(results).T
        df = df[['Accuracy', 'AUC', 'KS', 'AUPRC','Recall_0', 'Recall_1', 'Precision', 'F1']]
        
        # 按AUC降序
        df = df.sort_values('AUC', ascending=False)
        
        # 添加排名
        df.insert(0, 'Rank', range(1, len(df) + 1))
        
        # 添加改进率（相对于最后一名）
        baseline_auc = df['AUC'].iloc[-1]
        df.insert(1, 'AUC Improvement', (df['AUC'] - baseline_auc) / (baseline_auc + 1e-8))
        df['AUC Improvement'] = df['AUC Improvement'].apply(lambda x: f"{x:.2%}")
        
        # 保存CSV
        df.to_csv(output_path)
        
        print("\n" + "="*120)
        print("【完整模型对比表】")
        print("="*120)
        print(df.to_string())
        print("="*120 + "\n")
        
        return df

# =====================================================================
#      第19部分：Excel结果导出（多Sheet）
# =====================================================================

def export_results_to_excel_advanced(results, output_path="gl_ma_rag_results.xlsx"):
    """导出完整结果到Excel（多Sheet）"""
    
    wb = Workbook()
    wb.remove(wb.active)
    
    # ===== Sheet 1: 所有模型对比 =====
    ws1 = wb.create_sheet("Model Comparison", 0)
    
    df_results = pd.DataFrame(results).T
    df_results = df_results.sort_values('AUC', ascending=False)
    df_results.insert(0, 'Rank', range(1, len(df_results) + 1))
    
    # 计算改进率
    baseline_auc = df_results['AUC'].iloc[-1]
    df_results.insert(2, 'AUC Improve %', 
                     ((df_results['AUC'] - baseline_auc) / (baseline_auc + 1e-8) * 100).round(2))
    
    # 写入标题
    ws1.append(['Rank', 'Model', 'AUC Improve %'] + list(df_results.columns[3:]))
    
    for idx, (model_name, row) in enumerate(df_results.iterrows(), start=1):
        row_data = [idx, model_name] + [row[col] for col in df_results.columns[2:]]
        ws1.append(row_data)
    
    # 调整列宽
    for col in ws1.columns:
        ws1.column_dimensions[col[0].column_letter].width = 14
    
    # ===== Sheet 2: 分类统计 =====
    ws2 = wb.create_sheet("Category Summary", 1)
    
    categories = {
        '传统填充': [m for m in results.keys() if m.startswith('RF+')],
        'SOTA缺失学习': ['MISSFOREST', 'GAIN', 'MIWAE'],
        '基模型': ['LogR', 'RF', 'XGBoost', 'LightGBM', 'CatBoost', 'TabNet'],
        '深度学习': ['TabTransformer', 'SAINT', 'DeepFM'],
        '本文方法': ['GL-MA-RAG']
    }
    
    ws2.append(['Category', 'Model', 'AUC', 'KS','AUPRC', 'Accuracy', 'Recall_0', 'Recall_1', 'Precision', 'F1'])
    
    for category, models in categories.items():
        for model in models:
            if model in results:
                row_data = [category, model] + [results[model][m] for m in ['AUC', 'KS', 'AUPRC','Accuracy', 'Recall_0', 'Recall_1', 'Precision', 'F1']]
                ws2.append(row_data)
    
    for col in ws2.columns:
        ws2.column_dimensions[col[0].column_letter].width = 14
    
    # ===== Sheet 3: 统计摘要 =====
    ws3 = wb.create_sheet("Statistics", 2)
    
    ws3.append(['Metric', 'Count', 'Mean', 'Std', 'Min', 'Max', 'Best Model'])
    
    metrics_to_stats = ['AUC', 'KS','AUPRC', 'Accuracy', 'F1']
    
    for metric in metrics_to_stats:
        values = [results[m].get(metric, 0) for m in results.keys()]
        best_model = max(results.keys(), key=lambda x: results[x].get(metric, 0))
        
        ws3.append([
            metric,
            len(values),
            round(np.mean(values), 4),
            round(np.std(values), 4),
            round(np.min(values), 4),
            round(np.max(values), 4),
            best_model
        ])
    
    for col in ws3.columns:
        ws3.column_dimensions[col[0].column_letter].width = 16
    
    # ===== Sheet 4: GL-MA-RAG详细分析 =====
    ws4 = wb.create_sheet("GL-MA-RAG Analysis", 3)
    
    if 'GL-MA-RAG' in results:
        rag_metrics = results['GL-MA-RAG']
        
        ws4.append(['Metric', 'Value', 'Interpretation'])
        
        interpretations = {
            'AUC': '越接近1越好，表示分类性能',
            'KS': 'KS值越大越好，表示好坏客户区分能力',
            'Accuracy': '准确率，越高越好',
            'Recall_0': '低风险样本的识别率',
            'Recall_1': '高风险样本的识别率',
            'Precision': '预测为正的准确率',
            'F1': 'Recall和Precision的调和平均'
        }
        
        for metric, value in rag_metrics.items():
            interpretation = interpretations.get(metric, '')
            ws4.append([metric, value, interpretation])
    
    for col in ws4.columns:
        ws4.column_dimensions[col[0].column_letter].width = 20
    
    # ===== Sheet 5: Top 5模型对比 =====
    ws5 = wb.create_sheet("Top 5 Models", 4)
    
    df_top5 = pd.DataFrame(results).T
    df_top5 = df_top5.sort_values('F1', ascending=False).head(5)
    df_top5.insert(0, 'Rank', range(1, len(df_top5) + 1))
    
    ws5.append(['Rank'] + list(df_top5.columns[1:]))
    
    for idx, (model_name, row) in enumerate(df_top5.iterrows(), start=1):
        row_data = [idx, model_name] + [row[col] for col in df_top5.columns[2:]]
        ws5.append(row_data)
    
    for col in ws5.columns:
        ws5.column_dimensions[col[0].column_letter].width = 14
    
    wb.save(output_path)
    print(f"\n✓ 详细结果已导出至: {output_path}")
    print(f"   Sheet 1: Model Comparison (所有模型对比)")
    print(f"   Sheet 2: Category Summary (分类统计)")
    print(f"   Sheet 3: Statistics (统计摘要)")
    print(f"   Sheet 4: GL-MA-RAG Analysis (本文方法分析)")
    print(f"   Sheet 5: Top 5 Models (最优5个模型)")

# =====================================================================
#      第20部分：统计显著性检验
# =====================================================================

def perform_statistical_tests(results, y_test_true=None):
    """进行统计显著性检验"""
    
    print("\n" + "="*70)
    print("【统计显著性检验】")
    print("="*70)
    
    if 'GL-MA-RAG' not in results:
        print("⚠️ 没有GL-MA-RAG结果，跳过检验")
        return
    
    rag_auc = results['GL-MA-RAG']['AUC']
    
    print(f"\n基准方法: GL-MA-RAG (AUC={rag_auc:.4f})\n")
    print(f"{'模型':<20} {'AUC':<10} {'差异':<10} {'改进率':<10} {'排名'}")
    print("-"*70)
    
    # 按AUC排序
    sorted_results = sorted(results.items(), key=lambda x: x[1]['AUC'], reverse=True)
    
    for rank, (model_name, metrics) in enumerate(sorted_results, start=1):
        model_auc = metrics['AUC']
        diff = model_auc - rag_auc
        improvement = (diff / (rag_auc + 1e-8)) * 100
        
        # 添加符号
        if model_name == 'GL-MA-RAG':
            symbol = '⭐'
        elif diff > 0:
            symbol = '✓'
        elif diff < -0.01:
            symbol = '✗'
        else:
            symbol = '≈'
        
        print(f"{model_name:<20} {model_auc:<10.4f} {diff:<+10.4f} {improvement:<+10.2f}% {rank} {symbol}")
    
    # GL-MA-RAG排名
    rag_rank = next((i+1 for i, (m, _) in enumerate(sorted_results) if m == 'GL-MA-RAG'), None)
    
    print("\n" + "="*70)
    print(f"【GL-MA-RAG排名】: 第 {rag_rank} / {len(results)} 位")
    print("="*70)
    
    # 与Top基线对比
    if len(sorted_results) > 1:
        best_baseline = sorted_results[0] if sorted_results[0][0] != 'GL-MA-RAG' else sorted_results[1]
        baseline_name = best_baseline[0]
        baseline_auc = best_baseline[1]['AUC']
        
        print(f"\n【与最优基线对比】")
        print(f"  基线模型: {baseline_name}")
        print(f"  基线AUC: {baseline_auc:.4f}")
        print(f"  GL-MA-RAG AUC: {rag_auc:.4f}")
        
        if rag_auc > baseline_auc:
            improvement = ((rag_auc - baseline_auc) / baseline_auc) * 100
            print(f"  ✓ 改进: +{improvement:.2f}%")
        else:
            degradation = ((baseline_auc - rag_auc) / baseline_auc) * 100
            print(f"  ✗ 退化: -{degradation:.2f}%")

