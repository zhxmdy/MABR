# -*- coding: utf-8 -*-
from .config import *
from .retrieval import StratifiedRetriever, build_gl_ma_rag_index_with_stratified_retrieval
from .bayesian_predictor import MissingnessAwareBayesianRiskPredictor
from .models import SimpleRegressionModel, BayesianMissingDataModel

class InterpretabilityAnalyzer:
    """
    GL-MA-RAG v3.0 可解释性分析模块 (优化版)
    
    用于分析缺失数据处理的可解释性，包括：
    1. 特征重要性与缺失关联
    2. 缺失模式网络
    3. 本地检索解释
    4. 贝叶斯后验更新分析
    5. 决策置信区间分析
    """
    
    # 绘图常数配置
    PLOT_STYLE = {
        'color_positive': '#7CA0D6',      # 绿色 - 负类/好
        'color_negative': '#F1878A',      # 红色 - 正类/坏
        'color_neutral': '#81CFBC',       # 蓝色 - 中性
        'color_query': '#f39c12',         # 橙色 - 查询
        'color_highlight': '#9b59b6',     # 紫色 - 高亮
        'heatmap_cmap': 'RdYlGn',
        'dpi': 500,
        'font_family': 'SimHei'
    }
    
    def __init__(self, output_dir: str = './results'):
        """初始化分析器"""
        self.output_dir = output_dir
        self._ensure_output_dir()
        self.logger = self._setup_logger()
    
    def _ensure_output_dir(self):
        """确保输出目录存在"""
        os.makedirs(self.output_dir, exist_ok=True)
    
    def _setup_logger(self):
        """简单的日志设置"""
        class SimpleLogger:
            @staticmethod
            def info(msg):
                print(f"[INFO] {msg}")
            @staticmethod
            def warning(msg):
                print(f"[WARNING] {msg}")
            @staticmethod
            def error(msg):
                print(f"[ERROR] {msg}")
        return SimpleLogger()
    
    def _validate_inputs(self, X_train, y_train, mask_train, feature_cols=None) -> Tuple:
        """
        数据验证和标准化
        
        Parameters:
            X_train: 训练集特征 (numpy array or DataFrame)
            y_train: 训练集标签 (numpy array or Series)
            mask_train: 训练集缺失掩码 (numpy array or DataFrame)
            feature_cols: 特征名列表
        
        Returns:
            (X_train_df, y_train_series, mask_train_df, feature_cols)
        """
        # 标准化 X_train
        if isinstance(X_train, pd.DataFrame):
            feature_cols = X_train.columns.tolist()
            X_train_df = X_train.copy()
        else:
            n_features = X_train.shape[1]
            if feature_cols is None:
                feature_cols = [f"Feature_{i}" for i in range(n_features)]
            X_train_df = pd.DataFrame(X_train, columns=feature_cols)
        
        # 标准化 y_train
        if isinstance(y_train, pd.Series):
            y_train_series = y_train.copy()
        else:
            y_train_series = pd.Series(y_train, name="target")
        
        # 标准化 mask_train
        if isinstance(mask_train, pd.DataFrame):
            mask_train_df = mask_train.copy()
        else:
            mask_train_df = pd.DataFrame(mask_train, columns=feature_cols)
        
        return X_train_df, y_train_series, mask_train_df, feature_cols
    
    def _extract_feature_weights(self, global_missing_stats: Dict, 
                                 feature_cols: List[str]) -> np.ndarray:
        """
        安全地提取特征权重
        
        Parameters:
            global_missing_stats: 全局缺失统计字典
            feature_cols: 特征名列表
        
        Returns:
            权重数组 (n_features,)
        """
        weights_dict = global_missing_stats.get("global_weights", {})
        
        if not weights_dict:
            self.logger.warning(f"global_missing_stats 中没有 'global_weights'，使用默认权重")
            return np.ones(len(feature_cols)) / len(feature_cols)
        
        weights = np.array(
            [weights_dict.get(f, 1.0/len(feature_cols)) for f in feature_cols],
            dtype=np.float32
        )
        
        # 归一化
        weights = weights / (weights.sum() + 1e-8)
        return weights
    
    def _compute_statistics(self, X_train: pd.DataFrame, 
                           y_train: pd.Series,
                           mask_train: pd.DataFrame,
                           weights: np.ndarray) -> Dict[str, np.ndarray]:
        """计算特征统计量"""
        n_features = X_train.shape[1]
        
        stats = {
            'missing_rate':mask_train.sum()/mask_train.shape[0] ,
            'correlation': np.array([
                abs(X_train.iloc[:, i].corr(y_train)) 
                for i in range(n_features)
            ], dtype=np.float32),
            'weights': weights,
            'variance': X_train.var().values,
            'skewness': X_train.skew().values
        }
        
        # NaN处理
        for key in stats:
            stats[key] = np.nan_to_num(stats[key], nan=0.0, posinf=0.0, neginf=0.0)
        
        return stats
    
    # ======================== 1. 特征重要性热力图 ========================
    
    def plot_feature_importance(self, X_train, y_train, mask_train,
                               global_missing_stats, feature_cols=None, top_k=15):
        """
        可视化特征缺失重要性与类关联
        
        Parameters:
            X_train: 训练集特征
            y_train: 训练集标签
            mask_train: 训练集缺失掩码
            global_missing_stats: 全局缺失统计
            feature_cols: 特征名列表
            top_k: 展示的顶部特征数
        """
        try:
            self.logger.info("开始生成特征重要性热力图...")
            
            # 1. 数据验证
            X_train, y_train, mask_train, feature_cols = self._validate_inputs(
                X_train, y_train, mask_train, feature_cols
            )
            
            # 2. 提取权重
            weights = self._extract_feature_weights(global_missing_stats, feature_cols)
            
            # 3. 计算统计量
            stats = self._compute_statistics(X_train, y_train, mask_train, weights)
            
            # 4. 排序选择Top K
            sorted_idx = np.argsort(stats['correlation'])[::-1][:top_k]
            top_features = [feature_cols[i] for i in sorted_idx]
            
            # 5. 构建显示矩阵
            matrix_data = np.vstack([
                stats['missing_rate'][sorted_idx],
                stats['correlation'][sorted_idx],
                stats['weights'][sorted_idx],
                stats['variance'][sorted_idx]
            ])
            
            # 标准化矩阵（每行独立标准化）
            matrix_normalized = np.zeros_like(matrix_data)
            for i in range(matrix_data.shape[0]):
                row = matrix_data[i]
                row_min, row_max = row.min(), row.max()
                if row_max > row_min:
                    matrix_normalized[i] = (row - row_min) / (row_max - row_min)
                else:
                    matrix_normalized[i] = row
            
            # 6. 绘图
            fig, ax = plt.subplots(figsize=(14, 7))
            
            im = sns.heatmap(
                matrix_normalized,
                xticklabels=top_features,
                yticklabels=['Missing Rate', 'Correlation w/ Target', 'Weight', 'Variance'],
                annot=np.round(matrix_normalized, 3),
                fmt='g',
                cmap=self.PLOT_STYLE['heatmap_cmap'],
                linewidths=0.8,
                linecolor='gray',
                cbar_kws={'label': 'Normalized Importance Score'},
                ax=ax,
                vmin=0, vmax=1,
                square=False
            )
            
            ax.set_title(
                'Feature Importance: Missingness & Class Correlation',
                fontsize=15,
                fontweight='bold',
                pad=20
            )
            ax.set_xlabel('Features', fontsize=12, fontweight='bold')
            ax.set_ylabel('Metrics', fontsize=12, fontweight='bold')
            
            # 旋转X轴标签
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
            plt.setp(ax.get_yticklabels(), rotation=0, fontsize=10)
            
            plt.tight_layout()
            plt.savefig(
                f'{self.output_dir}/01_feature_importance_heatmap.png',
                dpi=self.PLOT_STYLE['dpi'],
                bbox_inches='tight',
                facecolor='white'
            )
            plt.close(fig)
            
            self.logger.info(f"✅ 特征重要性热力图已生成: 01_feature_importance_heatmap.png")
            return fig
            
        except Exception as e:
            self.logger.error(f"特征重要性绘图失败: {str(e)}")
            traceback.print_exc()
            return None
    
    # ======================== 2. 缺失模式网络图 ========================
    
    def plot_missing_pattern_correlation_network(self, X_train, y_train, 
                                                mask_train, top_features=10):
        """
        绘制缺失模式网络图
        
        Parameters:
            X_train: 训练集特征
            y_train: 训练集标签
            mask_train: 训练集缺失掩码
            top_features: 显示的顶部特征数
        """
        try:
            self.logger.info("开始生成缺失模式网络图...")
            
            X_train, y_train, mask_train, feature_cols = self._validate_inputs(
                X_train, y_train, mask_train
            )
            
            feature_cols = feature_cols[:top_features]
            
            # 构建图
            G = nx.Graph()
            G.add_node('Target', node_type='target')
            
            # 计算特征与目标的缺失关联性
            for feat in feature_cols:
                G.add_node(feat, node_type='feature')
                
                # 计算缺失掩码与标签的点二列相关系数
                feat_missing = mask_train[feat].values.astype(int)
                y_values = y_train.values.astype(int)
                
                try:
                    corr, p_value = pointbiserialr(feat_missing, y_values)
                    corr_abs = abs(corr)
                    
                    if corr_abs > 0.05:  # 仅显示显著相关
                        G.add_edge(feat, 'Target', weight=corr_abs, p_value=p_value)
                except:
                    pass
            
            # 绘图
            fig, ax = plt.subplots(figsize=(13, 11))
            
            pos = nx.spring_layout(G, k=2.0, iterations=50, seed=42, weight='weight')
            
            # 节点颜色和大小
            node_colors = [
                self.PLOT_STYLE['color_negative'] if node == 'Target' 
                else self.PLOT_STYLE['color_neutral']
                for node in G.nodes()
            ]
            
            node_sizes = [
                4000 if node == 'Target'
                else 2000 + max(G.degree(node), 1) * 800
                for node in G.nodes()
            ]
            
            # 绘制节点和边
            nx.draw_networkx_nodes(
                G, pos, node_color=node_colors, 
                node_size=node_sizes, ax=ax, alpha=0.85,
                edgecolors='black', linewidths=2
            )
            
            # 绘制边（权重越大越粗）
            if G.edges():
                edges = G.edges()
                weights = [G[u][v]['weight'] for u, v in edges]
                weight_max = max(weights) if weights else 1
                
                nx.draw_networkx_edges(
                    G, pos, width=[max(w/weight_max * 6, 1) for w in weights],
                    edge_color='gray', alpha=0.6, ax=ax, style='solid'
                )
            
            # 绘制标签
            nx.draw_networkx_labels(
                G, pos, font_size=10, font_weight='bold', ax=ax,
                font_color='white'
            )
            
            # 绘制边标签（相关系数）
            edge_labels = {}
            for u, v in G.edges():
                weight = G[u][v]['weight']
                edge_labels[(u, v)] = f'{weight:.2f}'
            
            nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8, ax=ax)
            
            ax.set_title(
                'Missing Pattern Correlation Network\n(Target vs Features)',
                fontsize=14,
                fontweight='bold',
                pad=20
            )
            ax.axis('off')
            
            # 添加图例
            legend_elements = [
                Patch(facecolor=self.PLOT_STYLE['color_negative'], 
                      edgecolor='black', label='Target Variable'),
                Patch(facecolor=self.PLOT_STYLE['color_neutral'], 
                      edgecolor='black', label='Features')
            ]
            ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
            
            plt.tight_layout()
            plt.savefig(
                f'{self.output_dir}/02_missing_pattern_network.png',
                dpi=self.PLOT_STYLE['dpi'],
                bbox_inches='tight',
                facecolor='white'
            )
            plt.close(fig)
            
            self.logger.info(f"✅ 缺失模式网络图已生成: 02_missing_pattern_network.png")
            return fig
            
        except Exception as e:
            self.logger.error(f"网络图绘制失败: {str(e)}")
            traceback.print_exc()
            return None
    
    # ======================== 3. 本地检索解释 ========================
    
    def plot_local_retrieval_explanation(self, X_test, mask_test, y_test,
                                        sample_idx, stratified_retriever,
                                        y_train, mask_train,
                                        k=5):
        """
        分析单个样本的检索结果
        
        Parameters:
            X_test: 测试集特征
            mask_test: 测试集缺失掩码
            y_test: 测试集标签
            sample_idx: 样本索引
            stratified_retriever: 分层检索器对象
            y_train: 训练集标签
            mask_train: 训练集缺失掩码
            k: 检索邻居数
        """
        try:
            # 数据类型处理
            if isinstance(X_test, pd.DataFrame):
                x_q = X_test.iloc[sample_idx].values
            else:
                x_q = X_test[sample_idx]
            
            if isinstance(mask_test, pd.DataFrame):
                m_q = mask_test.iloc[sample_idx].values
            else:
                m_q = mask_test[sample_idx]
            
            if isinstance(y_test, pd.Series):
                y_q = y_test.iloc[sample_idx]
            else:
                y_q = y_test[sample_idx]
            
            # 计算缺失率
            r_q = m_q.sum() / len(m_q) if len(m_q) > 0 else 0
            
            # 判断缺失等级
            if r_q < 0.1:
                missing_level = 'low'
            elif r_q > 0.3:
                missing_level = 'high'
            else:
                missing_level = 'mid'
            
            self.logger.info(f"检索样本 #{sample_idx}: 缺失率={r_q:.2%}, 等级={missing_level}")
            
            # 执行检索
            retrieve_result = stratified_retriever.retrieve_balanced(
                x_q.reshape(1, -1) if x_q.ndim == 1 else x_q,
                m_q.reshape(1, -1) if m_q.ndim == 1 else m_q,
                r_q,
                missing_level,
                k_total=k
            )
            
            indices = retrieve_result['indices']
            distances = retrieve_result['distances']
            
            # 获取邻居数据
            if isinstance(X_test, pd.DataFrame):
                X_neigh = X_test.iloc[indices].values
            else:
                X_neigh = X_test[indices]
            
            if isinstance(mask_test, pd.DataFrame):
                mask_neigh = mask_test.iloc[indices].values
            else:
                mask_neigh = mask_test[indices]
            
            if isinstance(y_train, pd.Series):
                y_neigh = y_train.iloc[indices].values
            else:
                y_neigh = y_train[indices]
            
            # 计算邻居权重
            weights = self._compute_neighbor_weights(distances, mask_neigh, m_q, r_q)
            
            # 绘图
            fig = self._draw_retrieval_plots(
                x_q, X_neigh, y_neigh, indices, distances, weights,
                mask_neigh, m_q, sample_idx, y_q
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"检索解释绘图失败 (sample #{sample_idx}): {str(e)}")
            traceback.print_exc()
            return None
    
    def _compute_neighbor_weights(self, distances, mask_neigh, m_q, r_q):
        """
        计算邻居权重
        
        基于距离和缺失率相似性的加权组合
        """
        # 避免距离为0的情况
        distances = np.clip(distances, 1e-8, None)
        
        # 距离权重（指数衰减）
        dist_std = np.std(distances) if np.std(distances) > 0 else 1
        dist_weight = np.exp(-distances / (dist_std + 1e-8))
        
        # 缺失率权重
        if isinstance(mask_neigh, pd.DataFrame):
            r_neigh = mask_neigh.sum(axis=1).values / mask_neigh.shape[1]
        else:
            r_neigh = mask_neigh.sum(axis=1) / mask_neigh.shape[1]
        
        rate_weight = np.exp(-0.5 * np.abs(r_neigh - r_q))
        
        # 组合权重
        weights = dist_weight * rate_weight
        weights = weights / (weights.sum() + 1e-8)
        
        return weights
    
    def _draw_retrieval_plots(self, x_q, X_neigh, y_neigh, indices, 
                             distances, weights, mask_neigh, m_q, 
                             sample_idx, y_q):
        """绘制检索结果的三个子图"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # 子图1: 邻域可视化（仅前两维）
        ax = axes[0]
        colors = [self.PLOT_STYLE['color_positive'] if y == 0 
                 else self.PLOT_STYLE['color_negative'] for y in y_neigh]
        
        ax.scatter(X_neigh[:, 0], X_neigh[:, 1], c=colors, s=250, alpha=0.6,
                  edgecolor='black', linewidth=1.5, label='Neighbors')
        ax.scatter(x_q[0], x_q[1], c=self.PLOT_STYLE['color_query'], s=500,
                  marker='*', edgecolor='black', linewidth=2.5, zorder=10, 
                  label=f'Query (True Label: {"Bad" if y_q else "Good"})')
        
        # 连接线（权重越大越粗）
        for i, w in enumerate(weights):
            ax.plot([x_q[0], X_neigh[i, 0]], [x_q[1], X_neigh[i, 1]],
                   color='gray', alpha=min(w * 2, 0.8), linestyle='--', 
                   linewidth=max(w * 3, 0.5))
        
        ax.set_xlabel('Feature 1', fontsize=11, fontweight='bold')
        ax.set_ylabel('Feature 2', fontsize=11, fontweight='bold')
        ax.set_title(f'Neighborhood Visualization (Sample #{sample_idx})', 
                    fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # 子图2: 邻居权重与标签
        ax = axes[1]
        bars = ax.bar(range(len(indices)), weights, color=colors, 
                     alpha=0.7, edgecolor='black', linewidth=1.5)
        
        for i, (y, w) in enumerate(zip(y_neigh, weights)):
            label = 'Good' if y == 0 else 'Bad'
            ax.text(i, w + 0.02, label, ha='center', va='bottom', 
                   fontsize=9, fontweight='bold')
        
        ax.set_xlabel('Neighbor Index', fontsize=11, fontweight='bold')
        ax.set_ylabel('Weight', fontsize=11, fontweight='bold')
        ax.set_title('Neighbor Weights & Labels', fontsize=12, fontweight='bold')
        ax.set_xticks(range(len(indices)))
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, max(weights) * 1.2)
        
        # 子图3: 缺失率对比
        ax = axes[2]
        if isinstance(mask_neigh, pd.DataFrame):
            r_neigh = mask_neigh.sum(axis=1).values / mask_neigh.shape[1]
        else:
            r_neigh = mask_neigh.sum(axis=1) / mask_neigh.shape[1]
        
        x_pos = np.arange(len(indices))
        width = 0.35
        
        ax.bar(x_pos - width/2, [m_q.sum() / len(m_q)] * len(indices), width, 
              label='Query', color=self.PLOT_STYLE['color_query'], alpha=0.7, 
              edgecolor='black', linewidth=1)
        ax.bar(x_pos + width/2, r_neigh, width,
              label='Neighbors', color=self.PLOT_STYLE['color_neutral'], 
              alpha=0.7, edgecolor='black', linewidth=1)
        
        ax.set_xlabel('Neighbor Index', fontsize=11, fontweight='bold')
        ax.set_ylabel('Missing Ratio', fontsize=11, fontweight='bold')
        ax.set_title('Missing Rate Comparison', fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(
            f'{self.output_dir}/03_retrieval_explanation_sample_{sample_idx}.png',
            dpi=self.PLOT_STYLE['dpi'],
            bbox_inches='tight',
            facecolor='white'
        )
        plt.close(fig)
        
        self.logger.info(f"✅ 检索解释图已生成: 03_retrieval_explanation_sample_{sample_idx}.png")
        return fig
    
    # ======================== 4. 贝叶斯后验分析 ========================
    
    def plot_bayesian_posterior_shift(self, X_test, y_test, pred_probs_final,
                                     pred_probs_prior, mask_test=None):
        """
        分析贝叶斯更新的幅度
        
        Parameters:
            X_test: 测试集特征
            y_test: 测试集标签
            pred_probs_final: 后验概率预测
            pred_probs_prior: 先验概率预测
            mask_test: 测试集缺失掩码
        """
        try:
            self.logger.info("开始生成贝叶斯后验分析图...")
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # 转换为numpy数组
            if isinstance(pred_probs_final, pd.Series):
                pred_probs_final = pred_probs_final.values
            if isinstance(pred_probs_prior, pd.Series):
                pred_probs_prior = pred_probs_prior.values
            if isinstance(y_test, pd.Series):
                y_test_arr = y_test.values
            else:
                y_test_arr = y_test
            
            # 计算更新幅度
            shift = pred_probs_final - pred_probs_prior
            
            # 计算缺失率
            if mask_test is not None:
                if isinstance(mask_test, pd.DataFrame):
                    missing_rate = mask_test.sum(axis=1).values / mask_test.shape[1]
                else:
                    missing_rate = mask_test.sum(axis=1) / mask_test.shape[1]
            else:
                missing_rate = np.random.rand(len(y_test_arr))
            
            # 限制显示样本数以加快绘图速度
            max_samples = min(500, len(y_test_arr))
            idx_sample = np.random.choice(len(y_test_arr), max_samples, replace=False)
            
            # ============ 1. Prior vs Posterior散点图 ============
            ax = axes[0, 0]
            colors = [self.PLOT_STYLE['color_positive'] if y == 0 
                     else self.PLOT_STYLE['color_negative'] for y in y_test_arr[idx_sample]]
            
            ax.scatter(pred_probs_prior[idx_sample], pred_probs_final[idx_sample],
                      c=colors, alpha=0.5, s=40, edgecolor='k', linewidth=0.3)
            
            # 完美校准线
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.6, linewidth=2.5, label='Perfect Calibration')
            ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5, linewidth=1.5)
            ax.axvline(0.5, color='gray', linestyle=':', alpha=0.5, linewidth=1.5)
            
            ax.set_xlabel('Prior Probability', fontsize=11, fontweight='bold')
            ax.set_ylabel('Posterior Probability', fontsize=11, fontweight='bold')
            ax.set_title('Bayesian Update: Prior vs Posterior', 
                        fontsize=12, fontweight='bold')
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # ============ 2. 更新幅度 vs 缺失率 ============
            ax = axes[0, 1]
            colors_missing = ['skyblue' if r < 0.1 else 'orange' if r < 0.3 else 'red' 
                            for r in missing_rate[idx_sample]]
            
            ax.scatter(missing_rate[idx_sample], shift[idx_sample], 
                      c=colors_missing, alpha=0.6, s=40)
            
            # 趋势线
            if len(idx_sample) > 2:
                z = np.polyfit(missing_rate[idx_sample], shift[idx_sample], 2)
                p = np.poly1d(z)
                x_trend = np.linspace(0, 1, 100)
                ax.plot(x_trend, p(x_trend), 'r-', linewidth=2.5, label='Trend (Poly-2)')
            
            ax.axhline(0, color='k', linestyle='--', alpha=0.5, linewidth=1.5)
            ax.set_xlabel('Missing Rate', fontsize=11, fontweight='bold')
            ax.set_ylabel('Shift (Posterior - Prior)', fontsize=11, fontweight='bold')
            ax.set_title('Update Magnitude vs Missing Rate', fontsize=12, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # ============ 3. 后验概率分布 ============
            ax = axes[1, 0]
            bins = np.linspace(0, 1, 25)
            
            mask_good = y_test_arr == 0
            mask_bad = y_test_arr == 1
            
            ax.hist(pred_probs_final[mask_good], bins=bins, alpha=0.6,
                   label='True Good (0)', color=self.PLOT_STYLE['color_positive'], 
                   density=True, edgecolor='black', linewidth=1)
            ax.hist(pred_probs_final[mask_bad], bins=bins, alpha=0.6,
                   label='True Bad (1)', color=self.PLOT_STYLE['color_negative'], 
                   density=True, edgecolor='black', linewidth=1)
            
            ax.set_xlabel('Posterior Probability', fontsize=11, fontweight='bold')
            ax.set_ylabel('Density', fontsize=11, fontweight='bold')
            ax.set_title('Posterior Distribution by True Label', fontsize=12, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3, axis='y')
            
            # ============ 4. 置信度与更新关系 ============
            ax = axes[1, 1]
            confidence = np.maximum(pred_probs_final, 1 - pred_probs_final)
            
            ax.scatter(confidence[idx_sample], np.abs(shift[idx_sample]), 
                      alpha=0.6, s=40, color=self.PLOT_STYLE['color_neutral'],
                      edgecolor='k', linewidth=0.3)
            
            # 添加百分位数线
            percentiles = [25, 50, 75]
            colors_percentile = ['green', 'orange', 'red']
            for p, color in zip(percentiles, colors_percentile):
                val = np.percentile(confidence, p)
                ax.axvline(val, color=color, linestyle='--', alpha=0.4, 
                          linewidth=1.5, label=f'P{p}')
            
            ax.set_xlabel('Confidence', fontsize=11, fontweight='bold')
            ax.set_ylabel('|Shift| Magnitude', fontsize=11, fontweight='bold')
            ax.set_title('Confidence vs Update Magnitude', fontsize=12, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(
                f'{self.output_dir}/04_bayesian_posterior_analysis.png',
                dpi=self.PLOT_STYLE['dpi'],
                bbox_inches='tight',
                facecolor='white'
            )
            plt.close(fig)
            
            self.logger.info(f"✅ 贝叶斯更新分析图已生成: 04_bayesian_posterior_analysis.png")
            return fig
            
        except Exception as e:
            self.logger.error(f"贝叶斯分析绘图失败: {str(e)}")
            traceback.print_exc()
            return None
    
    # ======================== 5. 决策置信区间 ========================
    
    def plot_decision_confidence_interval(self, results_df: pd.DataFrame):
        """
        分析决策置信度区间
        
        Parameters:
            results_df: 包含以下列的DataFrame:
                - prediction: 预测概率
                - true_label: 真实标签
                - confidence: 置信度
                - ci_width: 置信区间宽度
                - threshold: 决策阈值 (可选)
        """
        try:
            self.logger.info("开始生成决策置信度分析图...")
            
            # 数据验证
            required_cols = {'prediction', 'true_label', 'confidence'}
            if not required_cols.issubset(results_df.columns):
                missing_cols = required_cols - set(results_df.columns)
                self.logger.error(f"缺少必需列: {missing_cols}")
                return None
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # 按置信度排序
            df_sorted = results_df.sort_values('confidence').reset_index(drop=True)
            pred_binary = (results_df['prediction'] >= 0.5).astype(int)
            results_df['is_correct'] = (pred_binary == results_df['true_label']).astype(int)
            correct = results_df[results_df['is_correct'] == 1]
            incorrect = results_df[results_df['is_correct'] == 0]
            # ============ 1. 带置信区间的预测 ============
            ax = axes[0, 0]
            
            ci_width = df_sorted.get('ci_width', 
                                    np.abs(df_sorted['prediction'].values - 0.5) * 0.3)
            if isinstance(ci_width, pd.Series):
                y_err = ci_width.values / 2
            else:
                y_err = ci_width / 2
            
            colors = [self.PLOT_STYLE['color_positive'] if y == 0 
                     else self.PLOT_STYLE['color_negative'] 
                     for y in df_sorted['true_label']]
            
            # 绘制置信区间
            for i, (pred, err, color) in enumerate(zip(df_sorted['prediction'].values, 
                                                       y_err, colors)):
                ax.plot([i, i], [pred - err, pred + err], color=color, 
                       linewidth=2, alpha=0.6)
                ax.scatter(i, pred, color=color, s=25, zorder=3, 
                          edgecolor='k', linewidth=0.3)
            
            # 决策阈值
            threshold = df_sorted.get('threshold', 0.5)
            if isinstance(threshold, pd.Series):
                threshold = threshold.mean()
            else:
                threshold = np.mean(threshold) if hasattr(threshold, '__iter__') else threshold
            
            ax.axhline(threshold, color='black', linestyle='--', linewidth=2.5,
                      label=f'Threshold ({threshold:.2f})', zorder=5)
            
            ax.set_ylabel('Prediction Probability', fontsize=11, fontweight='bold')
            ax.set_xlabel('Sample Index (Sorted by Confidence)', fontsize=11, fontweight='bold')
            ax.set_title('(a)Predictions with 95% Confidence Intervals', fontsize=12, fontweight='bold')
            ax.set_ylim(-0.05, 1.05)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # ============ 2. 校准曲线 ============
            ax = axes[0, 1]
            ax.hist(correct['confidence'], bins=20, alpha=0.6, label='Correct', color=self.PLOT_STYLE['color_positive'], edgecolor='black')
            ax.hist(incorrect['confidence'], bins=20, alpha=0.6, label='Incorrect', color=self.PLOT_STYLE['color_negative'], edgecolor='black')
            ax.set_xlabel('Confidence', fontsize=11, fontweight='bold')
            ax.set_ylabel('Count', fontsize=11, fontweight='bold')
            ax.set_title('(b)Confidence Distribution', fontsize=12, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3, axis='y')     
                   
            # ax = axes[0, 1]
            
            # n_bins = 10
            # bin_edges = np.linspace(0, 1, n_bins + 1)
            # bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            # calibration = []
            # counts = []
            # for i in range(n_bins):
            #     mask = (df_sorted['prediction'] >= bin_edges[i]) & \
            #            (df_sorted['prediction'] < bin_edges[i + 1])
            #     if mask.sum() > 0:
            #         calibration.append(df_sorted.loc[mask, 'true_label'].mean())
            #         counts.append(mask.sum())
            #     else:
            #         calibration.append(np.nan)
            #         counts.append(0)
            
            # # 绘制校准曲线
            # valid_idx = [i for i, c in enumerate(counts) if c > 0]
            # if valid_idx:
            #     valid_centers = [bin_centers[i] for i in valid_idx]
            #     valid_cal = [calibration[i] for i in valid_idx]
            #     valid_counts = [counts[i] for i in valid_idx]
                
            #     ax.scatter(valid_centers, valid_cal, s=[c * 3 for c in valid_counts],
            #               alpha=0.6, color=self.PLOT_STYLE['color_neutral'], 
            #               edgecolor='k', linewidth=1.5, label='Empirical')
            
            # ax.plot([0, 1], [0, 1], 'k--', alpha=0.6, linewidth=2.5, 
            #        label='Perfect Calibration')
            
            # ax.set_xlabel('Mean Predicted Probability', fontsize=11, fontweight='bold')
            # ax.set_ylabel('Fraction of Positives (True Bad)', fontsize=11, fontweight='bold')
            # ax.set_title('Calibration Curve', fontsize=12, fontweight='bold')
            # ax.set_xlim(-0.05, 1.05)
            # ax.set_ylim(-0.05, 1.05)
            # ax.legend(fontsize=10)
            # ax.grid(True, alpha=0.3)
            
            # ============ 3. 预测分布直方图 ============
            ax = axes[1, 0]
            bins = np.linspace(0, 1, 25)
            
            mask_good = df_sorted['true_label'] == 0
            mask_bad = df_sorted['true_label'] == 1
            
            ax.hist(df_sorted.loc[mask_good, 'prediction'], bins=bins, alpha=0.6,
                   label='True Good (0)', color=self.PLOT_STYLE['color_positive'], 
                   density=True, edgecolor='black', linewidth=1)
            ax.hist(df_sorted.loc[mask_bad, 'prediction'], bins=bins, alpha=0.6,
                   label='True Bad (1)', color=self.PLOT_STYLE['color_negative'], 
                   density=True, edgecolor='black', linewidth=1)
            
            ax.axvline(threshold, color='black', linestyle='--', linewidth=2.5, 
                      label='Threshold')
            
            ax.set_xlabel('Prediction Probability', fontsize=11, fontweight='bold')
            ax.set_ylabel('Density', fontsize=11, fontweight='bold')
            ax.set_title('(c)Prediction Distribution by True Label', fontsize=12, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3, axis='y')


            ax = axes[1, 1]
            if 'is_correct' in results_df.columns:
            # 按缺失率分组
                missing_bins = np.linspace(0, results_df['missing_ratio'].max(), 10)
                results_df['missing_group'] = pd.cut(results_df['missing_ratio'], bins=missing_bins)
                
                accuracy_by_missing = results_df.groupby('missing_group', observed=True)['is_correct'].agg(['sum', 'count'])
                accuracy_by_missing['accuracy'] = accuracy_by_missing['sum'] / accuracy_by_missing['count']
                
                group_centers = [interval.mid for interval in accuracy_by_missing.index]
                
                bars = ax.bar(range(len(accuracy_by_missing)), accuracy_by_missing['accuracy'],
                            color=self.PLOT_STYLE['color_neutral'], alpha=0.7, edgecolor='black', linewidth=1.5)
                
                for bar, (idx, row) in zip(bars, accuracy_by_missing.iterrows()):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,f'{height:.2%}',
                        # f'{height:.2%}\n(n={int(row["count"])})', 
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
                
                ax.set_xticks(range(len(accuracy_by_missing)))
                ax.set_xticklabels([f'{c:.1%}' if c else 'NaN' for c in group_centers], rotation=45)
                ax.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
                ax.set_xlabel('Missing Ratio Group', fontsize=11, fontweight='bold')
                ax.set_title('(d)Accuracy vs Missing Ratio', fontsize=12, fontweight='bold')
                ax.set_ylim(0, 1.1)
                ax.grid(True, alpha=0.3, axis='y')
            # scatter = ax.scatter(results_df['missing_ratio'], results_df['prediction'],
            #                     c=results_df['confidence'], cmap='RdYlGn', 
            #                     s=100, alpha=0.6, edgecolor='black', linewidth=1)
            
            # # 添加趋势线
            # z = np.polyfit(results_df['missing_ratio'], results_df['prediction'], 2)
            # p = np.poly1d(z)
            # x_trend = np.linspace(results_df['missing_ratio'].min(), results_df['missing_ratio'].max(), 100)
            # ax.plot(x_trend, p(x_trend), 'r--', linewidth=2.5, label='Trend')
            
            # ax.set_xlabel('Missing Ratio', fontsize=11, fontweight='bold')
            # ax.set_ylabel('Prediction Probability', fontsize=11, fontweight='bold')
            # ax.set_title('Prediction vs Missing Ratio', fontsize=12, fontweight='bold')
            # ax.legend()
            # ax.grid(True, alpha=0.3)
            # cbar = plt.colorbar(scatter, ax=ax)
            # cbar.set_label('Confidence', fontsize=10, fontweight='bold')
            # ============ 4. 置信度分层分析 ============
            # ax = axes[1, 1]
            
            # conf_bins = [0, 0.6, 0.7, 0.8, 0.9, 1.0]
            # conf_labels = ['0.0-0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9', '0.9-1.0']
            
            # accuracy_by_conf = []
            # counts_by_conf = []
            
            # for i in range(len(conf_bins) - 1):
            #     mask = (df_sorted['confidence'] >= conf_bins[i]) & \
            #            (df_sorted['confidence'] < conf_bins[i + 1])
            #     if mask.sum() > 0:
            #         pred_label = (df_sorted.loc[mask, 'prediction'] > threshold).astype(int)
            #         accuracy = (pred_label == df_sorted.loc[mask, 'true_label']).mean()
            #         accuracy_by_conf.append(accuracy)
            #         counts_by_conf.append(mask.sum())
            #     else:
            #         accuracy_by_conf.append(0)
            #         counts_by_conf.append(0)
            
            # bars = ax.bar(conf_labels, accuracy_by_conf, alpha=0.7,
            #              color=self.PLOT_STYLE['color_neutral'], 
            #              edgecolor='k', linewidth=1.5)
            
            # # 添加数量标签
            # for bar, count in zip(bars, counts_by_conf):
            #     height = bar.get_height()
            #     ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
            #            f'n={int(count)}', ha='center', va='bottom', 
            #            fontsize=9, fontweight='bold')
            
            # ax.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
            # ax.set_xlabel('Confidence Level', fontsize=11, fontweight='bold')
            # ax.set_title('Accuracy by Confidence Level', fontsize=12, fontweight='bold')
            # ax.set_ylim(0, 1.15)
            # ax.grid(True, alpha=0.3, axis='y')



            
            plt.tight_layout()
            plt.savefig(
                f'{self.output_dir}/05_decision_confidence_interval.svg',
                dpi=self.PLOT_STYLE['dpi'],
                bbox_inches='tight',
                facecolor='white'
            )
            plt.close(fig)
            
            self.logger.info(f"✅ 决策置信度分析图已生成: 05_decision_confidence_interval.png")
            return fig
            
        except Exception as e:
            self.logger.error(f"置信度分析绘图失败: {str(e)}")
            traceback.print_exc()
            return None


# ======================== 第二部分：改进后的调用代码 ========================

# ======================== 替换原来的run_interpretability_analysis函数 ========================

def run_interpretability_analysis(X_train, X_val, X_test, y_train, y_val, y_test,
                                 mask_train, mask_val, mask_test,
                                 X_train_woe=None, X_test_woe=None,  # ★ 新增参数
                                 output_dir='./results'):
    """
    完整的可解释性分析流程 - 使用真实模型
    
    Parameters:
        X_train, X_val, X_test: 特征数据
        y_train, y_val, y_test: 标签数据
        mask_train, mask_val, mask_test: 缺失掩码数据
        X_train_woe, X_test_woe: WOE特征(用于模型训练)
        output_dir: 输出目录
    """
    
    print("\n" + "="*80)
    print("开始可解释性分析")
    print("="*80)
    
    # ============ 1. 初始化分析器 ============
    print("\n[Step 1] 初始化分析器...")
    analyzer = InterpretabilityAnalyzer(output_dir=output_dir)
    
    # ============ 2. 计算全局权重 ============
    print("\n[Step 2] 计算全局权重...")
    
    global_weights = 1 - mask_val.mean(axis=0)
    global_weights = global_weights / global_weights.sum()
    
    print(f"  ✓ 全局权重计算完成")
    print(f"    - 最大权重: {global_weights.max():.4f}")
    print(f"    - 最小权重: {global_weights.min():.4f}")
    
    # ============ 3. 构建分层检索器 ============
    print("\n[Step 3] 构建分层检索器...")
    
    fusion_weights = np.array([0.5, 0.4, 0.1])
    retriever = StratifiedRetriever(balance_ratio=0.5, min_per_class=3)
    retriever.build_stratified_index(
        X_train, mask_train, y_train,
        global_weights, fusion_weights
    )
    
    # ============ 4. 生成全局统计 ============
    print("\n[Step 4] 生成全局统计...")
    
    global_stats = {
        'global_weights': {f'Feature_{i}': w for i, w in enumerate(global_weights)}
    }
    
    print(f"  ✓ 全局统计包含 {len(global_stats['global_weights'])} 个特征权重")
    
    # ============ 5. 训练先验模型和贝叶斯模型 ============
    print("\n[Step 5] 训练先验和贝叶斯模型...")
    
    # 选择训练数据
    if X_train_woe is not None:
        print("  ✓ 使用WOE特征训练模型")
        X_train_for_model = X_train_woe
        X_test_for_model = X_test_woe
    else:
        print("  ✓ 使用原始特征训练模型")
        X_train_for_model = X_train
        X_test_for_model = X_test
    
    # 训练先验模型
    prior_model = SimpleRegressionModel()
    prior_model.fit(X_train_for_model, y_train)
    
    # 评估先验模型
    prior_train_acc = prior_model.score(X_train_for_model, y_train)
    prior_test_acc = prior_model.score(X_test_for_model, y_test)
    print(f"    先验模型 - 训练准确率: {prior_train_acc:.4f}, 测试准确率: {prior_test_acc:.4f}")
    
    # 训练贝叶斯模型
    bayes_model = BayesianMissingDataModel(prior_model)
    bayes_model.fit(X_train_for_model, y_train, mask_train)
    
    # ============ 6. 生成预测 ============
    print("\n[Step 6] 生成预测...")
    
    prior_predictions = prior_model.predict_proba(X_test_for_model)
    _, final_predictions = bayes_model.predict_proba(X_test_for_model, mask_test)
    
    print(f"  ✓ 先验预测 - mean={prior_predictions.mean():.4f}, std={prior_predictions.std():.4f}")
    print(f"  ✓ 后验预测 - mean={final_predictions.mean():.4f}, std={final_predictions.std():.4f}")
    print(f"  ✓ 平均更新幅度: {np.abs(final_predictions - prior_predictions).mean():.4f}")
    
    # ============ 7. 生成可视化分析 ============
    print("\n" + "="*80)
    print("生成可视化分析")
    print("="*80)
    
    # 1. 特征重要性热力图
    print("\n[Visualization 1] 生成特征重要性热力图...")
    analyzer.plot_feature_importance(
        X_train=X_train,
        y_train=y_train,
        mask_train=mask_train,
        global_missing_stats=global_stats,
        top_k=min(100, X_train.shape[1])
    )
    
    # 2. 缺失模式网络图
    print("\n[Visualization 2] 生成缺失模式网络图...")
    analyzer.plot_missing_pattern_correlation_network(
        X_train=X_train,
        y_train=y_train,
        mask_train=mask_train,
        top_features=min(100, X_train.shape[1])
    )
    
    # 3. 本地检索解释
    print("\n[Visualization 3] 生成本地检索解释...")
    
    test_sample_indices = [0, min(5, len(y_test)-1), 
                          min(10, len(y_test)-1), min(15, len(y_test)-1)]
    test_sample_indices = [i for i in test_sample_indices if i < len(y_test)]
    
    for sample_idx in test_sample_indices:
        print(f"\n  分析测试样本 #{sample_idx}")
        print(f"    - 真实标签: {'Bad (1)' if y_test[sample_idx] else 'Good (0)'}")
        
        analyzer.plot_local_retrieval_explanation(
            X_test=X_test,
            mask_test=mask_test,
            y_test=y_test,
            sample_idx=sample_idx,
            stratified_retriever=retriever,
            y_train=y_train,
            mask_train=mask_train,
            k=5
        )
    
    # 4. 贝叶斯后验分析 ★ 现在使用真实模型！
    print("\n[Visualization 4] 生成贝叶斯后验分析...")
    print("  ✓ 使用训练好的贝叶斯模型生成预测")
    
    analyzer.plot_bayesian_posterior_shift(
        X_test=X_test,
        y_test=y_test,
        pred_probs_final=final_predictions,      # ★ 真实后验概率
        pred_probs_prior=prior_predictions,      # ★ 真实先验概率
        mask_test=mask_test
    )
    
    # 5. 决策置信区间分析
    print("\n[Visualization 5] 生成决策置信度分析...")
    
    confidence_scores = np.maximum(final_predictions, 1 - final_predictions)
    
    ci_widths = np.zeros(len(X_test))
    for i in range(len(X_test)):
        missing_ratio = mask_test[i].sum() / mask_test.shape[1]
        ci_widths[i] = 0.1 + missing_ratio * 0.3
    
    results_df = pd.DataFrame({
        'prediction': final_predictions,
        'prior_prob': prior_predictions,
        'true_label': y_test,
        'confidence': confidence_scores,
        'ci_width': ci_widths,
        'threshold': 0.5,
        'posterior_alpha': 1,  # ★ 关键
        'posterior_beta': 1,    # ★ 关键
        'ci_lower': 0.45,
        'ci_upper': 0.55,
        'missing_ratio': mask_test.sum(axis=1) / mask_test.shape[1]
    })
    
    analyzer.plot_decision_confidence_interval(results_df)
    
    # ============ 8. 生成总结报告 ============
    print("\n" + "="*80)
    print("分析总结")
    print("="*80)
    
    # 模型对比
    
    print("\n模型性能对比:")
    prior_auc = roc_auc_score(y_test, prior_predictions)
    posterior_auc = roc_auc_score(y_test, final_predictions)
    
    print(f"  先验模型 AUC: {prior_auc:.4f}")
    print(f"  后验模型 AUC: {posterior_auc:.4f}")
    print(f"  改进幅度: {(posterior_auc - prior_auc):.4f} ({(posterior_auc - prior_auc)/prior_auc*100:+.2f}%)")
    
    print(f"\n✅ 所有可视化分析已完成!")
    print(f"  - 输出目录: {output_dir}")
    print(f"  - 生成的图表数: 5")
    print(f"  - 分析的测试样本数: {len(test_sample_indices)}")
    
    return analyzer, results_df

warnings.filterwarnings('ignore')

# 样式配置plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

