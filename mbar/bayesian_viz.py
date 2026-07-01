# -*- coding: utf-8 -*-
from .config import *

class BayesianVisualizationSuite:
    """
    贝叶斯预测结果的完整可视化套件
    
    包含以下图表：
    1. Beta分布后验可视化
    2. 先验 vs 后验对比
    3. 置信区间散点图
    4. 校准曲线（Calibration Curve）
    5. 期望校准误差（ECE）
    6. 贝叶斯更新热力图
    7. 置信度分层分析
    8. 后验概率分布（小提琴图）
    9. 不确定性分析（方差vs预测）
    10. 决策阈值最优化
    11. ROC曲线（基于置信度）
    12. 累积增益图
    """
    
    def __init__(self, output_dir='./bayesian_viz'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.colors = {
            'good': '#2ecc71',      # 绿
            'bad': '#e74c3c',       # 红
            'neutral': '#3498db',   # 蓝
            'prior': '#f39c12',     # 橙
            'posterior': '#9b59b6'  # 紫
        }
    
    # ======================== 图表1：Beta分布后验 ========================
    
    def plot_beta_posterior_distributions(self, results_df, sample_indices=None, 
                                         figsize=(16, 10)):
        """
        为选定样本绘制Beta分布后验
        
        参数：
            results_df: 包含posterior_alpha, posterior_beta的DataFrame
            sample_indices: 要绘制的样本索引列表
        """
        
        if sample_indices is None:
            # 自动选择不同置信度的样本
            sample_indices = [
                results_df['confidence'].idxmin(),  # 最低置信度
                int(len(results_df) * 0.25),        # 25分位
                int(len(results_df) * 0.5),         # 中位
                int(len(results_df) * 0.75),        # 75分位
                results_df['confidence'].idxmax()   # 最高置信度
            ]
        
        sample_indices = [i for i in sample_indices if i < len(results_df)]
        
        n_samples = len(sample_indices)
        fig, axes = plt.subplots(n_samples, 1, figsize=figsize)
        
        if n_samples == 1:
            axes = [axes]
        
        for ax_idx, sample_idx in enumerate(sample_indices):
            ax = axes[ax_idx]
            
            row = results_df.iloc[sample_idx]
            alpha = row['posterior_alpha']
            beta = row['posterior_beta']
            pred = row['prediction']
            ci_lower = row['ci_lower']
            ci_upper = row['ci_upper']
            conf = row['confidence']
            true_label = row.get('true_label', None)
            
            # 生成x轴
            x = np.linspace(0, 1, 1000)
            
            # Beta分布PDF
            y = beta_dist.pdf(x, alpha, beta)
            
            # 绘制分布
            ax.fill_between(x, y, alpha=0.3, color=self.colors['posterior'], 
                           label='Beta postprior')
            ax.plot(x, y, linewidth=2.5, color=self.colors['posterior'])
            
            # 标记预测值
            ax.axvline(pred, color='red', linestyle='--', linewidth=2.5, 
                      label=f'predict value ({pred:.3f})')
            
            # 标记置信区间
            ax.axvline(ci_lower, color='gray', linestyle=':', linewidth=2, alpha=0.7)
            ax.axvline(ci_upper, color='gray', linestyle=':', linewidth=2, alpha=0.7)
            ax.fill_between([ci_lower, ci_upper], 0, max(y), alpha=0.1, 
                           color='gray', label=f'95% CI')
            
            # 标记真实标签
            if true_label is not None:
                ax.axvline(true_label, color='green' if true_label == 0 else 'red',
                          linestyle='-', linewidth=2.5, alpha=0.7,
                          label=f'True value ({true_label})')
            
            # 标记决策阈值
            ax.axvline(0.5, color='black', linestyle='--', linewidth=1.5, 
                      alpha=0.5, label='decision threshold (0.5)')
            
            # 标题和标签
            title = f'smaple#{sample_idx}: α={alpha:.1f}, β={beta:.1f}, 置信度={conf:.1%}'
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_xlabel('probability', fontsize=11, fontweight='bold')
            ax.set_ylabel('probability density', fontsize=11, fontweight='bold')
            ax.legend(fontsize=9, loc='upper right')
            ax.set_xlim(0, 1)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/01_beta_posterior_distributions.png',
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ Beta分布后验图已生成")
        return fig
    
    # ======================== 图表2：先验 vs 后验对比 ========================
    
    def plot_prior_vs_posterior_comparison(self, results_df, figsize=(14, 8)):
        """
        比较先验和后验概率
        """
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # 子图1：散点图
        ax = axes[0, 0]
        
        colors = [self.colors['good'] if y == 0 else self.colors['bad'] 
                 for y in results_df['true_label']]
        
        ax.scatter(results_df['prior_prob'], results_df['prediction'],
                  c=colors, alpha=0.5, s=50, edgecolor='k', linewidth=0.3)
        
        # 对角线（完美校准）
        ax.plot([0, 1], [0, 1], 'k--', linewidth=2.5, alpha=0.7, label='perfect')
        ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
        ax.axvline(0.5, color='gray', linestyle=':', alpha=0.5)
        
        ax.set_xlabel('prior probability P(y=1|X)', fontsize=11, fontweight='bold')
        ax.set_ylabel('posterior probabilty P(y=1|X,mask)', fontsize=11, fontweight='bold')
        ax.set_title('prior vs posterior probabilty', fontsize=12, fontweight='bold')
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # 子图2：更新幅度分布
        ax = axes[0, 1]
        
        shift = results_df['prediction'] - results_df['prior_prob']
        
        ax.hist(shift, bins=30, color=self.colors['neutral'], alpha=0.7, 
               edgecolor='black', linewidth=1)
        ax.axvline(0, color='red', linestyle='--', linewidth=2.5, 
                  label=f'average update: {shift.mean():+.4f}')
        ax.axvline(shift.mean(), color='orange', linestyle='-', linewidth=2.5)
        
        ax.set_xlabel('Update magnitude (posterior - prior)', fontsize=11, fontweight='bold')
        ax.set_ylabel('frequency', fontsize=11, fontweight='bold')
        ax.set_title('Bayesian update magnitude distribution', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        # 子图3：更新幅度 vs 缺失率
        ax = axes[1, 0]
        
        missing_ratio = results_df['missing_ratio']
        
        ax.scatter(missing_ratio, shift, alpha=0.5, s=50, 
                  color=self.colors['neutral'], edgecolor='k', linewidth=0.3)
        
        # 趋势线
        if len(missing_ratio) > 2:
            z = np.polyfit(missing_ratio, shift, 2)
            p = np.poly1d(z)
            x_trend = np.linspace(0, 1, 100)
            ax.plot(x_trend, p(x_trend), 'r--', linewidth=2.5, label='二阶趋势')
        
        ax.axhline(0, color='black', linestyle='-', alpha=0.3)
        ax.set_xlabel('Missing Rate', fontsize=11, fontweight='bold')
        ax.set_ylabel('Update Frequency', fontsize=11, fontweight='bold')
        ax.set_title('Update Frequency vs. Missing Rate', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # 子图4：概率分布对比
        ax = axes[1, 1]
        
        ax.hist(results_df['prior_prob'], bins=20, alpha=0.5, 
               label='prior', color=self.colors['prior'], edgecolor='black')
        ax.hist(results_df['prediction'], bins=20, alpha=0.5, 
               label='posterior', color=self.colors['posterior'], edgecolor='black')
        
        ax.set_xlabel('probability', fontsize=11, fontweight='bold')
        ax.set_ylabel('frequency', fontsize=11, fontweight='bold')
        ax.set_title('Prior vs. Posterior Distribution Comparison', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/02_prior_vs_posterior_comparison.png',
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ 先验vs后验对比图已生成")
        return fig
    
    # ======================== 图表3：置信区间散点图 ========================
    
    def plot_confidence_intervals_scatter(self, results_df, figsize=(16, 8)):
        """
        绘制所有样本的置信区间
        """
        
        # 按置信度排序
        results_sorted = results_df.sort_values('confidence').reset_index(drop=True)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        for i in range(len(results_sorted)):
            row = results_sorted.iloc[i]
            
            pred = row['prediction']
            ci_lower = row['ci_lower']
            ci_upper = row['ci_upper']
            conf = row['confidence']
            true_label = row.get('true_label', None)
            
            # 颜色：根据真实标签
            if true_label is not None:
                color = self.colors['good'] if true_label == 0 else self.colors['bad']
            else:
                color = self.colors['neutral']
            
            # 绘制CI线
            ax.plot([i, i], [ci_lower, ci_upper], color=color, linewidth=2, alpha=0.6)
            
            # 绘制预测点
            ax.scatter(i, pred, color=color, s=50, zorder=3, edgecolor='k', linewidth=0.5)
        
        # 决策阈值
        ax.axhline(0.5, color='black', linestyle='--', linewidth=2.5, 
                  label='decision threshold (0.5)', alpha=0.7)
        
        ax.set_ylabel('predict probability', fontsize=12, fontweight='bold')
        ax.set_xlabel('Prior vs. Posterior Distribution Comparison by Sample Index (Sorted by Confidence)', fontsize=12, fontweight='bold')
        ax.set_title('95% confidence interval for all samples', fontsize=13, fontweight='bold')
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        
        # 添加置信度色条
        sm = plt.cm.ScalarMappable(
            cmap=plt.cm.RdYlGn,
            norm=plt.Normalize(vmin=results_sorted['confidence'].min(),
                             vmax=results_sorted['confidence'].max())
        )
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('confidence', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/03_confidence_intervals_scatter.png',
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ 置信区间散点图已生成")
        return fig
    
    # ======================== 图表4：校准曲线 ========================
    
    def plot_calibration_curve(self, results_df, n_bins=10, figsize=(12, 8)):
        """
        绘制校准曲线：预测概率 vs 实际频率
        
        完美校准：所有点应该在y=x线上
        """
        
        if 'true_label' not in results_df.columns:
            print("⚠️ 缺少true_label列，跳过校准曲线")
            return None
        
        # 预测标签（使用0.5阈值）
        pred_label = (results_df['prediction'] >= 0.5).astype(int)
        
        # 是否正确预测
        is_correct = (pred_label == results_df['true_label']).astype(int)
        
        # 分箱
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        calibration = []
        counts = []
        
        for i in range(n_bins):
            mask = (results_df['prediction'] >= bin_edges[i]) & \
                   (results_df['prediction'] < bin_edges[i+1])
            
            if mask.sum() > 0:
                # 在该区间内，真实为正类的比例
                actual_pos_rate = (results_df[mask]['true_label'] == 1).mean()
                calibration.append(actual_pos_rate)
                counts.append(mask.sum())
            else:
                calibration.append(np.nan)
                counts.append(0)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # 完美校准线
        ax.plot([0, 1], [0, 1], 'k--', linewidth=2.5, alpha=0.7, label='perfect')
        
        # 实际校准曲线
        valid_idx = [i for i, c in enumerate(counts) if c > 0]
        
        if valid_idx:
            valid_centers = [bin_centers[i] for i in valid_idx]
            valid_calib = [calibration[i] for i in valid_idx]
            valid_counts = [counts[i] for i in valid_idx]
            
            # 散点（大小与样本数成正比）
            scatter = ax.scatter(valid_centers, valid_calib, 
                               s=[c*2 for c in valid_counts],
                               alpha=0.6, color=self.colors['neutral'],
                               edgecolor='black', linewidth=1.5,
                               label='observed data')
            
            # 连接线
            ax.plot(valid_centers, valid_calib, 'b-', linewidth=2, alpha=0.5)
        
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel('Average predicted probability', fontsize=12, fontweight='bold')
        ax.set_ylabel('Actual positive class frequency', fontsize=12, fontweight='bold')
        ax.set_title('Calibration curve: predicted probability vs. actual frequency', fontsize=13, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # 计算ECE (Expected Calibration Error)
        ece = 0
        for i, c in enumerate(counts):
            if c > 0:
                ece += abs(bin_centers[i] - calibration[i]) * c / results_df.shape[0]
        
        # 添加ECE注释
        ax.text(0.05, 0.95, f'ECE = {ece:.4f}',
               transform=ax.transAxes, fontsize=12, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
               verticalalignment='top')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/04_calibration_curve.png',
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ 校准曲线已生成 (ECE={ece:.4f})")
        return fig
    
    # ======================== 图表5：期望校准误差热力图 ========================
    
    def plot_calibration_error_heatmap(self, results_df, n_bins=10, figsize=(12, 8)):
        """
        绘制ECE热力图：不同区间的校准误差
        """
        
        if 'true_label' not in results_df.columns:
            print("⚠️ 缺少true_label列")
            return None
        
        # 构建矩阵
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # 计算每个区间的指标
        metrics_matrix = np.zeros((4, n_bins))
        
        for i in range(n_bins):
            mask = (results_df['prediction'] >= bin_edges[i]) & \
                   (results_df['prediction'] < bin_edges[i+1])
            
            if mask.sum() > 0:
                subset = results_df[mask]
                
                # Row 0: 预测概率（均值）
                metrics_matrix[0, i] = subset['prediction'].mean()
                
                # Row 1: 实际正类频率
                metrics_matrix[1, i] = (subset['true_label'] == 1).mean()
                
                # Row 2: 校准误差
                metrics_matrix[2, i] = abs(metrics_matrix[0, i] - metrics_matrix[1, i])
                
                # Row 3: 样本数
                metrics_matrix[3, i] = mask.sum() / len(results_df)
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # 子图1：预测概率
        ax = axes[0, 0]
        sns.heatmap(metrics_matrix[[0]], cmap='RdYlGn', annot=True, fmt='.3f',
                   cbar_kws={'label': 'predict probabilty'}, ax=ax, vmin=0, vmax=1)
        ax.set_title('Predicted probability (by bin)', fontsize=12, fontweight='bold')
        ax.set_xticklabels([f'{c:.2f}' for c in bin_centers], rotation=45)
        
        # 子图2：实际频率
        ax = axes[0, 1]
        sns.heatmap(metrics_matrix[[1]], cmap='RdYlGn', annot=True, fmt='.3f',
                   cbar_kws={'label': 'actual frequency'}, ax=ax, vmin=0, vmax=1)
        ax.set_title('Actual positive class frequency (by bin)', fontsize=12, fontweight='bold')
        ax.set_xticklabels([f'{c:.2f}' for c in bin_centers], rotation=45)
        
        # 子图3：校准误差
        ax = axes[1, 0]
        sns.heatmap(metrics_matrix[[2]], cmap='Reds', annot=True, fmt='.4f',
                   cbar_kws={'label': 'Calibration error'}, ax=ax, vmin=0, vmax=0.2)
        ax.set_title('Calibration error (per box)', fontsize=12, fontweight='bold')
        ax.set_xticklabels([f'{c:.2f}' for c in bin_centers], rotation=45)
        
        # 子图4：样本数
        ax = axes[1, 1]
        sns.heatmap(metrics_matrix[[3]], cmap='Blues', annot=True, fmt='.3f',
                   cbar_kws={'label': 'sample ratio'}, ax=ax, vmin=0, vmax=metrics_matrix[3].max())
        ax.set_title('sample ratio(by bin)', fontsize=12, fontweight='bold')
        ax.set_xticklabels([f'{c:.2f}' for c in bin_centers], rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/05_calibration_error_heatmap.png',
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ 校准误差热力图已生成")
        return fig
    
    # ======================== 图表6：贝叶斯更新热力图 ========================
    
    def plot_bayesian_update_heatmap(self, results_df, figsize=(14, 8)):
        """
        展示先验、似然、后验的关系
        """
        
        # 排序
        results_sorted = results_df.sort_values('prediction').reset_index(drop=True)
        
        # 采样（如果太多样本）
        sample_size = min(100, len(results_sorted))
        sample_indices = np.linspace(0, len(results_sorted)-1, sample_size, dtype=int)
        results_plot = results_sorted.iloc[sample_indices]
        
        # 构建矩阵
        heatmap_data = np.vstack([
            results_plot['prior_prob'].values,
            results_plot['prediction'].values,
            (results_plot['prediction'] - results_plot['prior_prob']).values,
            results_plot['confidence'].values,
            results_plot['ci_width'].values
        ])
        
        fig, ax = plt.subplots(figsize=figsize)
        
        sns.heatmap(heatmap_data, cmap='RdYlGn', annot=False,
                   cbar_kws={'label': '值'},
                   yticklabels=['prior probability', 'posterior probability', 'update magnitud', 'confidence level', 'CI width'],
                   ax=ax)
        
        ax.set_xlabel('Sample Index (Sorted by Posterior Probability)', fontsize=12, fontweight='bold')
        ax.set_title('Bayesian Update Heatmap', fontsize=13, fontweight='bold')
        # ===== 修复：手动设置少量x轴刻度，避免标签数量不匹配 =====
        n_cols = heatmap_data.shape[1]
        max_xticks = 20  # 最多显示20个横轴标签，避免太密
        tick_step = max(1, int(np.ceil(n_cols / max_xticks)))

        tick_positions = np.arange(0, n_cols, tick_step)
        tick_labels = [str(sample_indices[i]) for i in tick_positions]

        # heatmap的格子中心在 i + 0.5
        ax.set_xticks(tick_positions + 0.5)
        ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/06_bayesian_update_heatmap.png',
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ 贝叶斯更新热力图已生成")
        return fig
    
    # ======================== 图表7：置信度分层分析 ========================
    
    def plot_confidence_stratification(self, results_df, figsize=(14, 10)):
        """
        按置信度分层，分析每层的性能
        """
        
        # 分层
        conf_bins = [0, 0.6, 0.7, 0.8, 0.9, 1.0]
        conf_labels = ['0.0-0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9', '0.9-1.0']
        
        results_df['conf_layer'] = pd.cut(results_df['confidence'], 
                                         bins=conf_bins, labels=conf_labels)
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()
        
        # 子图1：准确率
        ax = axes[0]
        if 'true_label' in results_df.columns:
            pred_label = (results_df['prediction'] >= 0.5).astype(int)
            accuracy = (pred_label == results_df['true_label']).groupby(
                results_df['conf_layer']).mean()
            
            accuracy.plot(kind='bar', ax=ax, color=self.colors['neutral'], 
                         edgecolor='black', linewidth=1.5, alpha=0.7)
            ax.set_ylabel('accuracy', fontsize=11, fontweight='bold')
            ax.set_title('By confidence level: accuracy', fontsize=12, fontweight='bold')
            ax.set_ylim(0, 1.1)
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        
        # 子图2：样本数
        ax = axes[1]
        counts = results_df['conf_layer'].value_counts().sort_index()
        counts.plot(kind='bar', ax=ax, color=self.colors['good'], 
                   edgecolor='black', linewidth=1.5, alpha=0.7)
        ax.set_ylabel('sample size', fontsize=11, fontweight='bold')
        ax.set_title('Stratified by confidence: sample distribution', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        
        # 添加样本数标签
        for i, v in enumerate(counts.values):
            ax.text(i, v + 5, str(v), ha='center', fontweight='bold')
        
        # 子图3：预测概率分布
        ax = axes[2]
        for label in conf_labels:
            data = results_df[results_df['conf_layer'] == label]['prediction']
            ax.hist(data, bins=15, alpha=0.5, label=label, edgecolor='black')
        ax.set_xlabel('predict probability', fontsize=11, fontweight='bold')
        ax.set_ylabel('frequency', fontsize=11, fontweight='bold')
        ax.set_title('Stratified by confidence: prediction distribution', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        
        # 子图4：CI宽度
        ax = axes[3]
        ci_width_mean = results_df.groupby('conf_layer')['ci_width'].mean()
        ci_width_std = results_df.groupby('conf_layer')['ci_width'].std()
        
        ci_width_mean.plot(kind='bar', ax=ax, color=self.colors['bad'], 
                          edgecolor='black', linewidth=1.5, alpha=0.7,
                          yerr=ci_width_std, capsize=5)
        ax.set_ylabel('CI width', fontsize=11, fontweight='bold')
        ax.set_title('Stratified by confidence level: width of the confidence interval', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        
        # 子图5：更新幅度
        ax = axes[4]
        shift = results_df['prediction'] - results_df['prior_prob']
        shift_mean = shift.groupby(results_df['conf_layer']).mean()
        
        shift_mean.plot(kind='bar', ax=ax, color=self.colors['prior'], 
                       edgecolor='black', linewidth=1.5, alpha=0.7)
        ax.axhline(0, color='black', linestyle='-', linewidth=1)
        ax.set_ylabel('Average update magnitude', fontsize=11, fontweight='bold')
        ax.set_title('Stratified by confidence: Bayesian update magnitude', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        
        # 子图6：概率密度
        ax = axes[5]
        for label in conf_labels:
            data = results_df[results_df['conf_layer'] == label]['prediction']
            if len(data) > 2:
                density = gaussian_kde(data)
                x = np.linspace(0, 1, 200)
                ax.plot(x, density(x), linewidth=2, label=label)
        ax.set_xlabel('predict probability', fontsize=11, fontweight='bold')
        ax.set_ylabel('probability density', fontsize=11, fontweight='bold')
        ax.set_title('Stratified by confidence level: Probability density curve', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/07_confidence_stratification.png',
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ 置信度分层分析已生成")
        return fig
    
    # ======================== 图表8：后验概率小提琴图 ========================
    
    def plot_posterior_violin_plots(self, results_df, figsize=(14, 8)):
        """
        使用小提琴图展示不同组群的后验分布
        """
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # 子图1：按真实标签
        ax = axes[0, 0]
        if 'true_label' in results_df.columns:
            data_to_plot = [
                results_df[results_df['true_label'] == 0]['prediction'].values,
                results_df[results_df['true_label'] == 1]['prediction'].values
            ]
            
            parts = ax.violinplot(data_to_plot, positions=[0, 1], 
                                 widths=0.7, showmeans=True, showmedians=True)
            
            ax.set_ylabel('Posterior probability', fontsize=11, fontweight='bold')
            ax.set_title('Posterior probability distribution: according to true labels', fontsize=12, fontweight='bold')
            ax.set_xticks([0, 1])
            ax.set_xticklabels(['好样本(0)', '坏样本(1)'])
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, alpha=0.3, axis='y')
        
        # 子图2：按缺失等级
        ax = axes[0, 1]
        if 'missing_level' in results_df.columns:
            missing_levels = results_df['missing_level'].unique()
            data_to_plot = [
                results_df[results_df['missing_level'] == level]['prediction'].values
                for level in sorted(missing_levels)
            ]
            
            positions = list(range(len(data_to_plot)))
            parts = ax.violinplot(data_to_plot, positions=positions, 
                                 widths=0.7, showmeans=True, showmedians=True)
            
            ax.set_ylabel('Posterior probability', fontsize=11, fontweight='bold')
            ax.set_title('Posterior probability distribution: by missing level', fontsize=12, fontweight='bold')
            ax.set_xticks(positions)
            ax.set_xticklabels(sorted(missing_levels))
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, alpha=0.3, axis='y')
        
        # 子图3：按置信度分层
        ax = axes[1, 0]
        conf_bins = [0, 0.6, 0.7, 0.8, 0.9, 1.0]
        conf_labels = ['0.0-0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9', '0.9-1.0']
        
        results_df['conf_layer'] = pd.cut(results_df['confidence'], 
                                         bins=conf_bins, labels=conf_labels)
        
        data_to_plot = [
            results_df[results_df['conf_layer'] == label]['prediction'].values
            for label in conf_labels
        ]
        
        positions = list(range(len(data_to_plot)))
        parts = ax.violinplot([d for d in data_to_plot if len(d) > 0], 
                             positions=[i for i, d in enumerate(data_to_plot) if len(d) > 0],
                             widths=0.7, showmeans=True, showmedians=True)
        
        ax.set_ylabel('Posterior probability', fontsize=11, fontweight='bold')
        ax.set_title('Posterior probability distribution: by confidence level', fontsize=12, fontweight='bold')
        ax.set_xticks([i for i, d in enumerate(data_to_plot) if len(d) > 0])
        ax.set_xticklabels([l for l, d in zip(conf_labels, data_to_plot) if len(d) > 0], 
                          rotation=45)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3, axis='y')
        
        # 子图4：箱线图汇总
        ax = axes[1, 1]
        
        data_for_boxplot = []
        labels_for_boxplot = []
        
        if 'true_label' in results_df.columns:
            data_for_boxplot.append(results_df[results_df['true_label'] == 0]['prediction'])
            data_for_boxplot.append(results_df[results_df['true_label'] == 1]['prediction'])
            labels_for_boxplot.extend(['good sample', 'bad sample'])
        
        bp = ax.boxplot(data_for_boxplot, labels=labels_for_boxplot, patch_artist=True)
        
        for patch, color in zip(bp['boxes'], [self.colors['good'], self.colors['bad']]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_ylabel('Posterior probability', fontsize=11, fontweight='bold')
        ax.set_title('Posterior probability distribution: Box plot', fontsize=12, fontweight='bold')
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/08_posterior_violin_plots.png',
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ 后验分布小提琴图已生成")
        return fig
    
    # ======================== 图表9：不确定性分析 ========================
    
    def plot_uncertainty_analysis(self, results_df, figsize=(14, 10)):
        """
        分析模型不确定性来源
        """
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # 子图1：置信度 vs CI宽度
        ax = axes[0, 0]
        ax.scatter(results_df['confidence'], results_df['ci_width'],
                  alpha=0.5, s=30, color=self.colors['neutral'], 
                  edgecolor='k', linewidth=0.3)
        
        # 拟合曲线
        z = np.polyfit(results_df['confidence'], results_df['ci_width'], 2)
        p = np.poly1d(z)
        x_fit = np.linspace(results_df['confidence'].min(), 
                           results_df['confidence'].max(), 100)
        ax.plot(x_fit, p(x_fit), 'r--', linewidth=2.5, label='拟合曲线')
        
        ax.set_xlabel('confidence level', fontsize=11, fontweight='bold')
        ax.set_ylabel('width of confidence interval', fontsize=11, fontweight='bold')
        ax.set_title('Confidence level vs. interval width', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # 子图2：缺失率 vs 不确定性
        ax = axes[0, 1]
        ax.scatter(results_df['missing_ratio'], results_df['ci_width'],
                  alpha=0.5, s=30, color=self.colors['bad'],
                  edgecolor='k', linewidth=0.3)
        
        # 拟合曲线
        z = np.polyfit(results_df['missing_ratio'], results_df['ci_width'], 2)
        p = np.poly1d(z)
        x_fit = np.linspace(0, 1, 100)
        ax.plot(x_fit, p(x_fit), 'r--', linewidth=2.5, label='fitted curve')
        
        ax.set_xlabel('Missing Rate', fontsize=11, fontweight='bold')
        ax.set_ylabel('width of confidence interval', fontsize=11, fontweight='bold')
        ax.set_title('Missing Rate vs. Uncertainty', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # 子图3：后验参数(α,β)
        ax = axes[1, 0]
        ax.scatter(results_df['posterior_alpha'], results_df['posterior_beta'],
                  c=results_df['confidence'], cmap='RdYlGn', s=50, alpha=0.6,
                  edgecolor='k', linewidth=0.3)
        
        cbar = plt.colorbar(ax.collections[0], ax=ax)
        cbar.set_label('confidence level', fontsize=10, fontweight='bold')
        
        ax.set_xlabel('Posterior α parameter', fontsize=11, fontweight='bold')
        ax.set_ylabel('Posterior β parameter', fontsize=11, fontweight='bold')
        ax.set_title('Beta posterior parameter distribution', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 子图4：方差vs预测
        ax = axes[1, 1]
        
        # 计算后验方差
        post_var = (results_df['posterior_alpha'] * results_df['posterior_beta']) / (
            (results_df['posterior_alpha'] + results_df['posterior_beta']) ** 2 * 
            (results_df['posterior_alpha'] + results_df['posterior_beta'] + 1)
        )
        
        ax.scatter(results_df['prediction'], post_var,
                  alpha=0.5, s=30, color=self.colors['neutral'],
                  edgecolor='k', linewidth=0.3)
        
        ax.set_xlabel('Predictive probability', fontsize=11, fontweight='bold')
        ax.set_ylabel('posterior variance', fontsize=11, fontweight='bold')
        ax.set_title('Predictive probability vs. posterior variance', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/09_uncertainty_analysis.png',
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ 不确定性分析图已生成")
        return fig
    
    # ======================== 图表10：决策阈值最优化 ========================
    
    def plot_threshold_optimization(self, results_df, figsize=(14, 8)):
        """
        展示不同阈值下的性能曲线
        """
        
        if 'true_label' not in results_df.columns:
            print("⚠️ 缺少true_label列")
            return None
        
        thresholds = np.linspace(0.1, 0.9, 20)
        
        metrics_by_threshold = {
            'accuracy': [],
            'precision': [],
            'recall_0': [],
            'recall_1': [],
            'f1': [],
            'gmean': []
        }
        
        for th in thresholds:
            pred_label = (results_df['prediction'] >= th).astype(int)
            true_label = results_df['true_label'].values
            
            # 计算各项指标
            tp = ((pred_label == 1) & (true_label == 1)).sum()
            tn = ((pred_label == 0) & (true_label == 0)).sum()
            fp = ((pred_label == 1) & (true_label == 0)).sum()
            fn = ((pred_label == 0) & (true_label == 1)).sum()
            
            accuracy = (tp + tn) / len(true_label)
            precision = tp / (tp + fp + 1e-8)
            recall_1 = tp / (tp + fn + 1e-8)
            recall_0 = tn / (tn + fp + 1e-8)
            f1 = 2 * precision * recall_1 / (precision + recall_1 + 1e-8)
            gmean = np.sqrt(recall_0 * recall_1)
            
            metrics_by_threshold['accuracy'].append(accuracy)
            metrics_by_threshold['precision'].append(precision)
            metrics_by_threshold['recall_0'].append(recall_0)
            metrics_by_threshold['recall_1'].append(recall_1)
            metrics_by_threshold['f1'].append(f1)
            metrics_by_threshold['gmean'].append(gmean)
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # 子图1：各项指标随阈值变化
        ax = axes[0]
        
        ax.plot(thresholds, metrics_by_threshold['accuracy'], 
               label='准确率', linewidth=2.5, marker='o')
        ax.plot(thresholds, metrics_by_threshold['precision'],
               label='精确率', linewidth=2.5, marker='s')
        ax.plot(thresholds, metrics_by_threshold['recall_1'],
               label='负类recall', linewidth=2.5, marker='^')
        ax.plot(thresholds, metrics_by_threshold['recall_0'],
               label='正类recall', linewidth=2.5, marker='v')
        ax.plot(thresholds, metrics_by_threshold['f1'],
               label='F1-Score', linewidth=2.5, marker='d')
        ax.plot(thresholds, metrics_by_threshold['gmean'],
               label='G-Mean', linewidth=2.5, marker='*', markersize=12)
        
        # 标记最优值
        best_gmean_idx = np.argmax(metrics_by_threshold['gmean'])
        best_gmean_th = thresholds[best_gmean_idx]
        best_gmean_val = metrics_by_threshold['gmean'][best_gmean_idx]
        
        ax.axvline(best_gmean_th, color='red', linestyle='--', linewidth=2,
                  alpha=0.7, label=f'最优G-Mean阈值 ({best_gmean_th:.2f})')
        
        ax.set_xlabel('决策阈值', fontsize=12, fontweight='bold')
        ax.set_ylabel('指标值', fontsize=12, fontweight='bold')
        ax.set_title('性能指标 vs 决策阈值', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10, loc='lower left')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0.1, 0.9)
        ax.set_ylim(0, 1.1)
        
        # 子图2：Recall权衡曲线
        ax = axes[1]
        
        ax.plot(metrics_by_threshold['recall_0'], metrics_by_threshold['recall_1'],
               linewidth=2.5, marker='o', label='Recall权衡曲线')
        
        # 标记最优点
        ax.scatter([metrics_by_threshold['recall_0'][best_gmean_idx]],
                  [metrics_by_threshold['recall_1'][best_gmean_idx]],
                  s=200, color='red', marker='*', zorder=5,
                  label=f'最优点 (阈值={best_gmean_th:.2f})')
        
        # 完美点
        ax.scatter([1], [1], s=200, color='green', marker='X', zorder=5,
                  label='完美点 (1, 1)')
        
        # 对角线（平衡线）
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1, label='平衡线')
        
        ax.set_xlabel('负类Recall (正确识别好样本)', fontsize=12, fontweight='bold')
        ax.set_ylabel('正类Recall (正确识别坏样本)', fontsize=12, fontweight='bold')
        ax.set_title('两类Recall权衡分析', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/10_threshold_optimization.png',
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ 决策阈值最优化已生成 (最优G-Mean阈值={best_gmean_th:.3f})")
        return fig, best_gmean_th
    
    # ======================== 图表11：ROC曲线（基于置信度） ========================
    
    def plot_roc_curves_by_confidence(self, results_df, figsize=(12, 8)):
        """
        绘制不同置信度级别的ROC曲线对比
        """
        
        if 'true_label' not in results_df.columns:
            print("⚠️ 缺少true_label列")
            return None
        
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # 全体样本ROC
        fpr_all, tpr_all, _ = roc_curve(results_df['true_label'], 
                                        results_df['prediction'])
        auc_all = auc(fpr_all, tpr_all)
        
        ax.plot(fpr_all, tpr_all, linewidth=2.5, label=f'全体 (AUC={auc_all:.4f})',
               color='black')
        
        # 按置信度分层ROC
        conf_bins = [0, 0.6, 0.7, 0.8, 0.9, 1.0]
        conf_labels = ['0.0-0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9', '0.9-1.0']
        colors_conf = plt.cm.RdYlGn(np.linspace(0, 1, len(conf_labels)))
        
        results_df['conf_layer'] = pd.cut(results_df['confidence'],
                                         bins=conf_bins, labels=conf_labels)
        
        for label, color in zip(conf_labels, colors_conf):
            subset = results_df[results_df['conf_layer'] == label]
            
            if len(subset) > 10 and (subset['true_label'].nunique() > 1):
                fpr, tpr, _ = roc_curve(subset['true_label'], subset['prediction'])
                auc_score = auc(fpr, tpr)
                
                ax.plot(fpr, tpr, linewidth=2, label=f'{label} (AUC={auc_score:.4f})',
                       color=color, alpha=0.7)
        
        # 参考线
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.5, label='随机分类')
        
        ax.set_xlabel('假正率 (FPR)', fontsize=12, fontweight='bold')
        ax.set_ylabel('真正率 (TPR)', fontsize=12, fontweight='bold')
        ax.set_title('ROC曲线：按置信度分层', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10, loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/11_roc_curves_by_confidence.png',
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ 分层ROC曲线已生成")
        return fig
    
    # ======================== 图表12：累积增益曲线 ========================
    
    def plot_cumulative_gain_chart(self, results_df, figsize=(12, 8)):
        """
        绘制累积增益图：评估排序能力
        """
        
        if 'true_label' not in results_df.columns:
            print("⚠️ 缺少true_label列")
            return None
        
        # 按预测概率降序排列
        results_sorted = results_df.sort_values('prediction', ascending=False).reset_index(drop=True)
        
        # 计算累积增益
        total_positives = (results_sorted['true_label'] == 1).sum()
        
        cumsum_positives = np.cumsum(results_sorted['true_label'] == 1)
        cumsum_percentages = 100 * cumsum_positives / total_positives
        percentiles = 100 * np.arange(len(results_sorted)) / len(results_sorted)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # 子图1：累积增益曲线
        # ax = axes[0]
        
        # # 实际曲线
        # ax.plot(percentiles, cumsum_percentages, linewidth=2.5, 
        #        label='Actual cumulative gain', color=self.colors['good'])
        
        # # 基线（随机）
        # ax.plot([0, 100], [0, 100], 'k--', linewidth=2, alpha=0.7, 
        #        label='Random gain baseline')
        
        # # 完美曲线
        # ax.plot([0, total_positives/len(results_sorted)*100, 100], 
        #        [0, 100, 100], 'g--', linewidth=2, alpha=0.7,
        #        label='Perfect Gain')
        
        # # 填充面积
        # ax.fill_between(percentiles, cumsum_percentages, percentiles,
        #                alpha=0.2, color=self.colors['good'])
        
        # ax.set_xlabel('sample percentag(%)', fontsize=12, fontweight='bold')
        # ax.set_ylabel('Cumulative percentage of true positives(%)', fontsize=12, fontweight='bold')
        # ax.set_title('Cumulative gain curve', fontsize=13, fontweight='bold')
        # ax.legend(fontsize=11)
        # ax.grid(True, alpha=0.3)
        # ax.set_xlim(0, 100)
        # ax.set_ylim(0, 105)
        
        # # 子图2：增益指数
        # ax = axes[1]
        
        gains = cumsum_percentages / percentiles * 100  # 相对于随机的增益倍数
        gains[0] = 100  # 避免除以零
        
        ax.plot(percentiles, gains, linewidth=2.5, color='blue')
        ax.axhline(100, color='k', linestyle='--', linewidth=2, alpha=0.7, 
                  label='baseline (1.0×)')
        
        # 标记关键百分比点
        key_percentiles = [10, 20, 30, 40, 50, 100]
        for kp in key_percentiles:
            idx = int(len(gains) * kp / 100)
            if idx < len(gains):
                ax.scatter(kp, gains[idx], s=100, color='blue', zorder=5)
                ax.text(kp, gains[idx] + 10, f'{gains[idx]:.1f}%', 
                       ha='center', fontsize=9, fontweight='bold')
        
        ax.set_xlabel('sample percentage(%)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Gain factor(%)', fontsize=12, fontweight='bold')
        ax.set_title('Gain Index', fontsize=13, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 100)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/12_cumulative_gain_chart.svg',
                   dpi=500, bbox_inches='tight')
        plt.close()
        
        print("✓ 累积增益曲线已生成")
        return fig
    
    # ======================== 生成所有图表的统一接口 ========================
    
    def generate_all_visualizations(self, results_df):
        """
        一键生成所有12个图表
        """
        
        print("\n" + "="*80)
        print("【贝叶斯结果完整可视化】开始生成所有图表...")
        print("="*80 + "\n")
        
        self.plot_beta_posterior_distributions(results_df)
        self.plot_prior_vs_posterior_comparison(results_df)
        self.plot_confidence_intervals_scatter(results_df)
        self.plot_calibration_curve(results_df)
        self.plot_calibration_error_heatmap(results_df)
        self.plot_bayesian_update_heatmap(results_df)
        self.plot_confidence_stratification(results_df)
        self.plot_posterior_violin_plots(results_df)
        self.plot_uncertainty_analysis(results_df)
        fig_threshold, best_th = self.plot_threshold_optimization(results_df)
        self.plot_roc_curves_by_confidence(results_df)
        self.plot_cumulative_gain_chart(results_df)
        
        print("\n" + "="*80)
        print("【完成】所有12个图表已生成！")
        print("="*80)
        print(f"\n📊 最优决策阈值: {best_th:.3f}")
        print(f"📁 输出目录: {self.output_dir}")
        print(f"\n生成的图表：")
        print(f"  01_beta_posterior_distributions.png - Beta分布后验")
        print(f"  02_prior_vs_posterior_comparison.png - 先验vs后验对比")
        print(f"  03_confidence_intervals_scatter.png - 置信区间散点图")
        print(f"  04_calibration_curve.png - 校准曲线")
        print(f"  05_calibration_error_heatmap.png - 校准误差热力图")
        print(f"  06_bayesian_update_heatmap.png - 贝叶斯更新热力图")
        print(f"  07_confidence_stratification.png - 置信度分层分析")
        print(f"  08_posterior_violin_plots.png - 后验概率小提琴图")
        print(f"  09_uncertainty_analysis.png - 不确定性分析")
        print(f"  10_threshold_optimization.png - 决策阈值优化")
        print(f"  11_roc_curves_by_confidence.png - 分层ROC曲线")
        print(f"  12_cumulative_gain_chart.png - 累积增益曲线")
        
        return best_th
# ======================== 使用示例 ========================



if __name__ == "__main__":
    
    print("""
    ╔════════════════════════════════════════════════════════════════╗
    ║  GL-MA-RAG v3.0: 类别平衡优化版本                               ║
    ║  融合三大核心模块：                                             ║
    ║  1. 代价敏感贝叶斯预测 (Cost-Sensitive Bayesian)               ║
    ║  2. 分层检索策略 (Stratified Retrieval)                       ║
    ║  3. G-Mean优化阈值 (G-Mean Threshold Calibration)             ║
    ╚════════════════════════════════════════════════════════════════╝
    
    选择运行模式:
    
    1️⃣  完整多轮实验（推荐） - 包含所有模型对比
    2️⃣  单次GL-MA-RAG v3.0演示 - 快速验证新功能
    3️⃣  融合预测可视化 - 置信区间、模型权重等
    4️⃣  降维与数据分析 - PCA/UMAP/t-SNE对比
    5️⃣  消融实验 - 验证各组件贡献度
    6️⃣  退出
    """)
    
    choice = input("\n请输入选择 (1/2/3/4/5/6): ").strip()
    
    if choice == '1':
        # ★ 完整多轮实验（v3.0）
        print("\n【完整多轮GL-MA-RAG v3.0实验】")
        summary_df, all_results = run_complete_experiment_with_multiple_runs()
        
        print("\n" + "="*80)
        print("【汇总表】")
        print("="*80)
        print(summary_df.to_string())
        print("="*80)
        
        # 生成对比可视化
        viz = ComparisonVisualizer()
        
        fig_comp = viz.plot_all_models_comparison(
            {m: summary_df.loc[m].to_dict() for m in summary_df.index}
        )
        fig_comp.savefig('model_comparison_all.png', dpi=300, bbox_inches='tight')
        plt.close(fig_comp)
        print("✓ 模型对比图已保存: model_comparison_all.png")
        
        # 统计检验
        # perform_statistical_tests(
        #     {m: summary_df.loc[m].to_dict() for m in summary_df.index}
        # )
        

        def visualize_bayesian_results(results_df, output_dir='./bayesian_visualization'):
            """
            可视化贝叶斯结果的完整工作流
            """
            
            print("\n" + "="*80)
            print("【启动贝叶斯结果可视化模块】")
            print("="*80)
            
            # Step 1: 初始化可视化套件
            viz_suite = BayesianVisualizationSuite(output_dir=output_dir)
            
            # Step 2: 生成所有图表
            best_threshold = viz_suite.generate_all_visualizations(results_df)
            
            # Step 3: 返回最优阈值
            return best_threshold, output_dir
            # ★ 新增：贝叶斯可视化
        print("\n" + "="*80)
        print("【第三阶段】贝叶斯结果可视化")
        print("="*80)
        
        best_threshold, viz_dir = visualize_bayesian_results(
            results_df,
            output_dir=f'./results/bayesian_viz_seed_{seed}'
        )
        
        print(f"\n✅ 所有可视化完成！")
        print(f"   最优决策阈值: {best_threshold:.3f}")
        print(f"   可视化输出目录: {viz_dir}")
        
        # 保存最优阈值供后续使用
        with open(f'{viz_dir}/optimal_threshold.txt', 'w') as f:
            f.write(f"Optimal Decision Threshold: {best_threshold:.4f}\n")
            f.write(f"Generation Time: {pd.Timestamp.now()}\n")
    elif choice == '3':
        # 融合预测可视化
        print("\n【融合预测可视化】")
        print("此功能需要单次实验后的预测结果...")
        
        seed = RANDOM_SEED_BASE + 1
        (X_train, X_val, X_test, y_train, y_val, y_test,
        mask_train, mask_val, mask_test,
        local_train_info, local_val_info, local_test_info,
        global_missing_stats, pos_weight,
        X_train_missing, X_val_missing, X_test_missing,
        X_train_woe, X_val_woe, X_test_woe,
        adaptive_thresholds, adaptive_params, mech_prob) = \
            load_credit_data_optimized(DATA_PATH, seed, MISSING_RATES[0], MECHS[0])
        
        # 生成综合仪表板
        # （需要从之前的实验获取results_df和weights_list）
        print("✓ 可视化模块已准备就绪")
        
    elif choice == '4':
        # 降维分析
        print("\n【数据降维与分析】")
        
        seed = RANDOM_SEED_BASE + 1
        (X_train, X_val, X_test, y_train, y_val, y_test,
        mask_train, mask_val, mask_test,
        local_train_info, local_val_info, local_test_info,
        global_missing_stats, pos_weight,
        X_train_missing, X_val_missing, X_test_missing,
        X_train_woe, X_val_woe, X_test_woe,
        adaptive_thresholds, adaptive_params, mech_prob) = \
            load_credit_data_optimized(DATA_PATH, seed, MISSING_RATES[0], MECHS[0])
        
        # 合并数据
        X_combined = np.vstack([X_train, X_val])
        y_combined = np.hstack([y_train, y_val])
        mask_combined = np.vstack([mask_train, mask_val])
        
        # 执行降维分析
        dr_comp = DimensionalityReductionComparison(X_combined, y_combined, mask_combined)
        dr_comp.generate_comparison_report(output_dir='./results/dr_analysis')
        
        print("✓ 降维分析完成，结果保存至 ./results/dr_analysis")
        
    elif choice == '5':
        # 消融实验
        print("\n【v3.0消融实验】")
        print("（验证各模块的贡献度）")
        # run_ablation_experiment()
