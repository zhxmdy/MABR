# -*- coding: utf-8 -*-
from .config import *

# =====================================================================
#      第21部分：修改主程序入口（集成所有功能）
# =====================================================================

# =====================================================================
#          ★★★ 第X部分：高级可视化模块（PCA/UMAP/t-SNE）★★★
# =====================================================================

# ====================== UMAP安装与导入处理 ======================
try:
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("⚠️ UMAP未安装，请运行: pip install umap-learn")

try:
    TSNE_AVAILABLE = True
except ImportError:
    TSNE_AVAILABLE = False
    print("⚠️ t-SNE已包含在scikit-learn中")

# ====================== PCA可视化模块 ======================
class PCAVisualizer:
    """
    PCA降维与可视化
    """
    
    def __init__(self, n_components=2, random_state=42):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components, random_state=random_state)
        self.explained_variance_ratio_ = None
        self.cumsum_variance_ = None
    
    def fit_transform(self, X):
        """拟合并转换数据"""
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        X = np.nan_to_num(X, nan=0.0).astype(np.float32)
        X_pca = self.pca.fit_transform(X)
        
        self.explained_variance_ratio_ = self.pca.explained_variance_ratio_
        self.cumsum_variance_ = np.cumsum(self.explained_variance_ratio_)
        
        return X_pca
    
    def plot_variance_explained(self, figsize=(12, 4)):
        """绘制方差解释比例"""
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # 子图1：各成分方差
        axes[0].bar(range(1, len(self.explained_variance_ratio_) + 1), 
                   self.explained_variance_ratio_, 
                   alpha=0.7, color='steelblue', edgecolor='black')
        axes[0].set_xlabel('PC Number', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Explained Variance Ratio', fontsize=12, fontweight='bold')
        axes[0].set_title('Variance Explained by Each PC', fontsize=13, fontweight='bold')
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # 子图2：累积方差
        axes[1].plot(range(1, len(self.cumsum_variance_) + 1), 
                    self.cumsum_variance_, 
                    'o-', linewidth=2.5, markersize=8, color='darkblue')
        axes[1].axhline(y=0.95, color='red', linestyle='--', linewidth=2, label='95% threshold')
        axes[1].set_xlabel('Number of Components', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Cumulative Explained Variance', fontsize=12, fontweight='bold')
        axes[1].set_title('Cumulative Variance Explained', fontsize=13, fontweight='bold')
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim([0, 1.05])
        
        plt.tight_layout()
        return fig
    
    def plot_2d_projection(self, X_pca, y, title='PCA 2D Projection', figsize=(10, 8)):
        """绘制2D投影"""
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = ['green' if label == 0 else 'red' for label in y]
        sizes = [50 if label == 0 else 100 for label in y]
        
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], 
                            c=colors, s=sizes, alpha=0.6, edgecolor='black', linewidth=1)
        
        ax.set_xlabel(f'PC1 ({self.explained_variance_ratio_[0]:.1%} variance)', 
                     fontsize=12, fontweight='bold')
        ax.set_ylabel(f'PC2 ({self.explained_variance_ratio_[1]:.1%} variance)', 
                     fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 添加图例
        low_risk = mpatches.Patch(color='green', label='Low Risk (0)', alpha=0.6)
        high_risk = mpatches.Patch(color='red', label='High Risk (1)', alpha=0.6)
        ax.legend(handles=[low_risk, high_risk], fontsize=11)
        
        plt.tight_layout()
        return fig
    
    def plot_3d_projection(self, X_pca, y, title='PCA 3D Projection', figsize=(12, 9)):
        """绘制3D投影"""
        
        if X_pca.shape[1] < 3:
            print("⚠️ 3D投影需要至少3个主成分，跳过")
            return None
        
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        colors = ['green' if label == 0 else 'red' for label in y]
        
        ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], 
                  c=colors, s=50, alpha=0.6, edgecolor='black', linewidth=1)
        
        ax.set_xlabel(f'PC1 ({self.explained_variance_ratio_[0]:.1%})', fontsize=11, fontweight='bold')
        ax.set_ylabel(f'PC2 ({self.explained_variance_ratio_[1]:.1%})', fontsize=11, fontweight='bold')
        ax.set_zlabel(f'PC3 ({self.explained_variance_ratio_[2]:.1%})', fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=13, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def plot_pca_loadings(self, feature_names=None, figsize=(12, 8)):
        """绘制主成分载荷"""
        if feature_names is None:
            feature_names = [f'Feature {i}' for i in range(len(self.pca.components_[0]))]
        
        loadings = self.pca.components_.T * np.sqrt(self.pca.explained_variance_)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        loadings_df = pd.DataFrame(
            loadings[:, :2],
            columns=['PC1', 'PC2'],
            index=feature_names
        )
        
        # 绘制箭头
        for i, feature in enumerate(feature_names):
            ax.arrow(0, 0, loadings_df.loc[feature, 'PC1'], 
                    loadings_df.loc[feature, 'PC2'],
                    head_width=0.05, head_length=0.05, fc='steelblue', ec='black', linewidth=2)
            ax.text(loadings_df.loc[feature, 'PC1']*1.15, 
                   loadings_df.loc[feature, 'PC2']*1.15,
                   feature, fontsize=10, fontweight='bold', ha='center')
        
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_xlabel(f'PC1 ({self.explained_variance_ratio_[0]:.1%})', 
                     fontsize=12, fontweight='bold')
        ax.set_ylabel(f'PC2 ({self.explained_variance_ratio_[1]:.1%})', 
                     fontsize=12, fontweight='bold')
        ax.set_title('PCA Feature Loadings', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linewidth=0.5)
        ax.axvline(x=0, color='k', linewidth=0.5)
        
        plt.tight_layout()
        return fig

# ====================== UMAP可视化模块 ======================
class UMAPVisualizer:
    """
    UMAP非线性降维与可视化
    """
    
    def __init__(self, n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42):
        if not UMAP_AVAILABLE:
            raise ImportError("UMAP not installed. Run: pip install umap-learn")
        
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        self.random_state = random_state
        self.reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=random_state,
            verbose=0
        )
    
    def fit_transform(self, X):
        """拟合并转换数据"""
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        X = np.nan_to_num(X, nan=0.0).astype(np.float32)
        X_umap = self.reducer.fit_transform(X)
        
        return X_umap
    
    def plot_2d_projection(self, X_umap, y, title='UMAP 2D Projection', figsize=(10, 8)):
        """绘制2D投影"""
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = ['green' if label == 0 else 'red' for label in y]
        sizes = [50 if label == 0 else 100 for label in y]
        
        ax.scatter(X_umap[:, 0], X_umap[:, 1], 
                  c=colors, s=sizes, alpha=0.6, edgecolor='black', linewidth=1)
        
        ax.set_xlabel('UMAP Dimension 1', fontsize=12, fontweight='bold')
        ax.set_ylabel('UMAP Dimension 2', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        low_risk = mpatches.Patch(color='green', label='Low Risk (0)', alpha=0.6)
        high_risk = mpatches.Patch(color='red', label='High Risk (1)', alpha=0.6)
        ax.legend(handles=[low_risk, high_risk], fontsize=11)
        
        plt.tight_layout()
        return fig
    
    def plot_with_missing_level(self, X_umap, missing_levels, title='UMAP with Missing Levels', 
                               figsize=(10, 8)):
        """按缺失等级着色"""
        fig, ax = plt.subplots(figsize=figsize)
        
        level_colors = {
            'low': 'blue',
            'mid': 'yellow',
            'high': 'red'
        }
        
        for level, color in level_colors.items():
            mask = missing_levels == level
            ax.scatter(X_umap[mask, 0], X_umap[mask, 1], 
                      c=color, label=f'{level} missing', 
                      s=80, alpha=0.6, edgecolor='black', linewidth=1)
        
        ax.set_xlabel('UMAP Dimension 1', fontsize=12, fontweight='bold')
        ax.set_ylabel('UMAP Dimension 2', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig

# ====================== t-SNE可视化模块 ======================
class TSNEVisualizer:
    """
    t-SNE非线性降维与可视化
    """
    
    def __init__(self, n_components=2, perplexity=30, n_iter=1000, random_state=42):
        self.n_components = n_components
        self.perplexity = perplexity
        self.n_iter = n_iter
        self.random_state = random_state
        self.tsne = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            n_iter=n_iter,
            random_state=random_state,
            verbose=1
        )
    
    def fit_transform(self, X):
        """拟合并转换数据"""
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        X = np.nan_to_num(X, nan=0.0).astype(np.float32)
        
        # t-SNE可能比较慢，给出进度提示
        print("t-SNE运行中（可能需要1-5分钟）...")
        X_tsne = self.tsne.fit_transform(X)
        print("t-SNE完成！")
        
        return X_tsne
    
    def plot_2d_projection(self, X_tsne, y, title='t-SNE 2D Projection', figsize=(10, 8)):
        """绘制2D投影"""
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = ['green' if label == 0 else 'red' for label in y]
        sizes = [50 if label == 0 else 100 for label in y]
        
        ax.scatter(X_tsne[:, 0], X_tsne[:, 1], 
                  c=colors, s=sizes, alpha=0.6, edgecolor='black', linewidth=1)
        
        ax.set_xlabel('t-SNE Dimension 1', fontsize=12, fontweight='bold')
        ax.set_ylabel('t-SNE Dimension 2', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        low_risk = mpatches.Patch(color='green', label='Low Risk (0)', alpha=0.6)
        high_risk = mpatches.Patch(color='red', label='High Risk (1)', alpha=0.6)
        ax.legend(handles=[low_risk, high_risk], fontsize=11)
        
        plt.tight_layout()
        return fig

# ====================== 缺失模式聚类可视化 ======================
class MissingPatternVisualizer:
    """
    可视化缺失模式与样本关系
    """
    
    @staticmethod
    def plot_missing_pattern_heatmap(X, missing_mask, title='Missing Pattern Heatmap', 
                                    figsize=(14, 10)):
        """绘制缺失模式热力图"""
        # 选择缺失率最高的特征
        feature_missing_rate = missing_mask.sum(axis=0) / len(missing_mask)
        top_features_idx = np.argsort(feature_missing_rate)[-20:][::-1]  # Top 20
        
        # 按样本缺失率排序
        sample_missing_rate = missing_mask.sum(axis=1) / missing_mask.shape[1]
        sample_order = np.argsort(sample_missing_rate)[::-1]
        
        # 提取子矩阵
        missing_subset = missing_mask[sample_order][:, top_features_idx]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        sns.heatmap(missing_subset, cmap='RdYlGn_r', cbar_kws={'label': 'Missing (1) vs Present (0)'},
                   ax=ax, xticklabels=True, yticklabels=False)
        
        ax.set_xlabel('Feature Index (Top 20 by Missing Rate)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Sample Index (Sorted by Missing Rate)', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=13, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_missing_correlation_network(missing_mask, figsize=(12, 10)):
        """绘制特征缺失关联网络"""
        # 计算缺失相关性
        missing_corr = np.corrcoef(missing_mask.T)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # 绘制热力图
        sns.heatmap(missing_corr, cmap='coolwarm', center=0, square=True,
                   cbar_kws={'label': 'Correlation'}, ax=ax)
        
        ax.set_title('Missing Feature Correlation Matrix', fontsize=13, fontweight='bold')
        ax.set_xlabel('Feature Index', fontsize=12, fontweight='bold')
        ax.set_ylabel('Feature Index', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_sample_missing_distribution(missing_mask, y=None, figsize=(12, 5)):
        """绘制样本缺失分布"""
        sample_missing_ratio = missing_mask.sum(axis=1) / missing_mask.shape[1]
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # 子图1：整体分布
        axes[0].hist(sample_missing_ratio, bins=30, color='skyblue', 
                    edgecolor='black', alpha=0.7)
        axes[0].axvline(np.mean(sample_missing_ratio), color='red', 
                       linestyle='--', linewidth=2.5, label=f'Mean: {np.mean(sample_missing_ratio):.2%}')
        axes[0].axvline(np.median(sample_missing_ratio), color='green', 
                       linestyle='--', linewidth=2.5, label=f'Median: {np.median(sample_missing_ratio):.2%}')
        axes[0].set_xlabel('Sample Missing Ratio', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
        axes[0].set_title('Overall Missing Distribution', fontsize=13, fontweight='bold')
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # 子图2：按风险等级分布（如果提供了y）
        if y is not None:
            for label, color in [(0, 'green'), (1, 'red')]:
                mask = y == label
                axes[1].hist(sample_missing_ratio[mask], bins=20, 
                            label=f'{"Low Risk" if label == 0 else "High Risk"}',
                            color=color, alpha=0.6, edgecolor='black')
            
            axes[1].set_xlabel('Sample Missing Ratio', fontsize=12, fontweight='bold')
            axes[1].set_ylabel('Frequency', fontsize=12, fontweight='bold')
            axes[1].set_title('Missing Distribution by Risk Level', fontsize=13, fontweight='bold')
            axes[1].legend(fontsize=11)
            axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        return fig

# ====================== 完整降维对比可视化 ======================
class DimensionalityReductionComparison:
    """
    对比PCA/UMAP/t-SNE的降维效果
    """
    
    def __init__(self, X, y, missing_mask=None, random_state=42):
        self.X = X if isinstance(X, np.ndarray) else X.values
        self.y = y
        self.missing_mask = missing_mask
        self.random_state = random_state
        
        # 数据标准化
        self.X_scaled = StandardScaler().fit_transform(self.X)
    
    def generate_comparison_report(self, output_dir='./results', figsize=(18, 12)):
        """生成完整的对比报告"""
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n" + "="*60)
        print("【降维方法对比】")
        print("="*60)
        
        # 1. PCA
        print("\n1. PCA降维中...")
        pca_viz = PCAVisualizer(n_components=3)
        X_pca = pca_viz.fit_transform(self.X)
        
        # PCA方差图
        fig_pca_var = pca_viz.plot_variance_explained()
        fig_pca_var.savefig(os.path.join(output_dir, 'pca_variance_explained.png'), dpi=300, bbox_inches='tight')
        plt.close(fig_pca_var)
        print("  ✓ PCA 2D投影")
        
        fig_pca_2d = pca_viz.plot_2d_projection(X_pca, self.y)
        fig_pca_2d.savefig(os.path.join(output_dir, 'pca_2d_projection.png'), dpi=300, bbox_inches='tight')
        plt.close(fig_pca_2d)
        
        # PCA 3D投影
        if X_pca.shape[1] >= 3:
            print("  ✓ PCA 3D投影")
            fig_pca_3d = pca_viz.plot_3d_projection(X_pca, self.y)
            fig_pca_3d.savefig(os.path.join(output_dir, 'pca_3d_projection.png'), dpi=300, bbox_inches='tight')
            plt.close(fig_pca_3d)
        
        # PCA载荷
        print("  ✓ PCA特征载荷")
        fig_pca_load = pca_viz.plot_pca_loadings()
        fig_pca_load.savefig(os.path.join(output_dir, 'pca_loadings.png'), dpi=300, bbox_inches='tight')
        plt.close(fig_pca_load)
        
        # 2. UMAP
        if UMAP_AVAILABLE:
            print("\n2. UMAP降维中...")
            umap_viz = UMAPVisualizer(n_neighbors=15, min_dist=0.1)
            X_umap = umap_viz.fit_transform(self.X)
            
            print("  ✓ UMAP 2D投影")
            fig_umap_2d = umap_viz.plot_2d_projection(X_umap, self.y)
            fig_umap_2d.savefig(os.path.join(output_dir, 'umap_2d_projection.png'), dpi=300, bbox_inches='tight')
            plt.close(fig_umap_2d)
            
            if self.missing_mask is not None:
                print("  ✓ UMAP按缺失等级着色")
                missing_levels = self._get_missing_levels()
                fig_umap_missing = umap_viz.plot_with_missing_level(X_umap, missing_levels)
                fig_umap_missing.savefig(os.path.join(output_dir, 'umap_missing_levels.png'), 
                                        dpi=300, bbox_inches='tight')
                plt.close(fig_umap_missing)
        else:
            print("\n⚠️ UMAP不可用，跳过")
        
        # 3. t-SNE
        if TSNE_AVAILABLE:
            print("\n3. t-SNE降维中（请稍候）...")
            tsne_viz = TSNEVisualizer(perplexity=30, n_iter=1000)
            X_tsne = tsne_viz.fit_transform(self.X)
            
            print("  ✓ t-SNE 2D投影")
            fig_tsne_2d = tsne_viz.plot_2d_projection(X_tsne, self.y)
            fig_tsne_2d.savefig(os.path.join(output_dir, 'tsne_2d_projection.png'), dpi=300, bbox_inches='tight')
            plt.close(fig_tsne_2d)
        else:
            print("\n⚠️ t-SNE不可用")
        
        # 4. 缺失模式分析
        if self.missing_mask is not None:
            print("\n4. 缺失模式可视化中...")
            miss_viz = MissingPatternVisualizer()
            
            print("  ✓ 缺失模式热力图")
            fig_missing_heat = miss_viz.plot_missing_pattern_heatmap(
                self.X, self.missing_mask
            )
            fig_missing_heat.savefig(os.path.join(output_dir, 'missing_pattern_heatmap.png'), 
                                    dpi=300, bbox_inches='tight')
            plt.close(fig_missing_heat)
            
            print("  ✓ 缺失关联网络")
            fig_missing_corr = miss_viz.plot_missing_correlation_network(self.missing_mask)
            fig_missing_corr.savefig(os.path.join(output_dir, 'missing_correlation_network.png'), 
                                    dpi=300, bbox_inches='tight')
            plt.close(fig_missing_corr)
            
            print("  ✓ 样本缺失分布")
            fig_missing_dist = miss_viz.plot_sample_missing_distribution(self.missing_mask, self.y)
            fig_missing_dist.savefig(os.path.join(output_dir, 'sample_missing_distribution.png'), 
                                    dpi=300, bbox_inches='tight')
            plt.close(fig_missing_dist)
        
        print("\n✓ 所有可视化完成！")
        print(f"  结果保存至：{output_dir}")
        
        return {
            'pca': X_pca,
            'umap': X_umap if UMAP_AVAILABLE else None,
            'tsne': X_tsne if TSNE_AVAILABLE else None
        }
    
    def _get_missing_levels(self, low_threshold=0.1, mid_threshold=0.3):
        """获取缺失等级"""
        sample_missing_ratio = self.missing_mask.sum(axis=1) / self.missing_mask.shape[1]
        
        missing_levels = np.array([
            'low' if r < low_threshold else
            'mid' if r < mid_threshold else
            'high'
            for r in sample_missing_ratio
        ])
        
        return missing_levels
    
    def generate_summary_figure(self, output_dir='./results', figsize=(20, 12)):
        """生成对比总结图"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 准备降维数据
        pca_viz = PCAVisualizer(n_components=2)
        X_pca = pca_viz.fit_transform(self.X)
        
        results = {
            'PCA': X_pca,
            'UMAP': None,
            'tSNE': None
        }
        
        if UMAP_AVAILABLE:
            umap_viz = UMAPVisualizer()
            results['UMAP'] = umap_viz.fit_transform(self.X)
        
        if TSNE_AVAILABLE:
            tsne_viz = TSNEVisualizer(n_iter=500)  # 减少迭代次数加快速度
            results['tSNE'] = tsne_viz.fit_transform(self.X)
        
        # 绘制对比图
        n_plots = sum(1 for v in results.values() if v is not None)
        fig, axes = plt.subplots(1, n_plots, figsize=(7*n_plots, 6))
        
        if n_plots == 1:
            axes = [axes]
        
        plot_idx = 0
        for method_name, X_transformed in results.items():
            if X_transformed is None:
                continue
            
            ax = axes[plot_idx]
            
            colors = ['green' if label == 0 else 'red' for label in self.y]
            ax.scatter(X_transformed[:, 0], X_transformed[:, 1], 
                      c=colors, s=50, alpha=0.6, edgecolor='black', linewidth=1)
            
            ax.set_xlabel('Dimension 1', fontsize=11, fontweight='bold')
            ax.set_ylabel('Dimension 2', fontsize=11, fontweight='bold')
            ax.set_title(f'{method_name} 2D Projection', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            plot_idx += 1
        
        plt.suptitle('Dimensionality Reduction Methods Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        fig.savefig(os.path.join(output_dir, 'comparison_summary.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"\n✓ 对比总结图已保存至：{output_dir}/comparison_summary.png")

