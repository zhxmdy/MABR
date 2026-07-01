# -*- coding: utf-8 -*-
from .config import *

# =====================================================================
#                   第3部分：传统填充方法 【终极完美版】
# =====================================================================
# =====================================================================
# 【无任何风险、绝对不报错】最终版缺失填充
# =====================================================================

# =====================================================================
#                   第4部分：全局缺失模式分析（改进版）
# =====================================================================

def analyze_global_missing_pattern_improved(X):
    """
    ★改进版全局缺失分析：信息论 + 统计推断 + 鲁棒性增强
    
    核心改进：
    1. ✅ 信息熵权重：量化缺失不确定性
    2. ✅ 关联权重：检测MAR/MNAR模式
    3. ✅ 机制推断：多指标综合判定
    4. ✅ 异常处理：空值/全缺失列保护
    5. ✅ 可视化增强：分层输出结果
    
    参数:
        X: DataFrame或ndarray，输入数据
    
    返回:
        global_missing_stats: dict，全局统计信息
        inferred_mechanism: str，推断的缺失机制
    """
    # ===== 步骤0：数据预处理 =====
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X, columns=[f'Feature_{i}' for i in range(X.shape[1])])
    
    n_samples, n_features = X.shape
    col_names = X.columns.tolist()
    
    print("\n" + "="*70)
    print("【全局缺失模式分析】信息论驱动 + 机制推断")
    print("="*70)
    print(f"📊 数据维度: {n_samples} 样本 × {n_features} 特征")
    
    # ===== 步骤1：基础缺失统计 =====
    X_np = X.values.astype(np.float32)
    missing_mask = np.isnan(X_np)  # (n_samples, n_features)
    
    # 特征级缺失率
    feature_missing_rate = X.isnull().sum() / n_samples
    feature_missing_rate = feature_missing_rate.sort_values(ascending=False)
    
    # 全局缺失率
    overall_missing_rate = missing_mask.sum() / (n_samples * n_features)
    
    print(f"📈 全局缺失率: {overall_missing_rate:.2%}")
    print(f"   缺失特征数: {(feature_missing_rate > 0).sum()}/{n_features}")
    
    # ===== 步骤2：信息熵权重（缺失不确定性）=====
    print(f"\n🔍 步骤1：计算信息熵权重...")
    
    entropy_weights = np.zeros(n_features)
    
    for j in range(n_features):
        r_j = feature_missing_rate.iloc[j]
        
        # ★香农熵公式：H = -Σ p_i log(p_i)
        if 0 < r_j < 1:
            # 二元熵：缺失/非缺失
            entropy_weights[j] = -(
                r_j * np.log2(r_j + 1e-10) + 
                (1 - r_j) * np.log2(1 - r_j + 1e-10)
            )
        else:
            # 全缺失或全观测：无信息
            entropy_weights[j] = 0
    
    # 归一化到[0, 1]
    max_entropy = 1.0  # 二元熵最大值
    entropy_weights_norm = entropy_weights / (max_entropy + 1e-10)
    
    print(f"   熵权重范围: [{entropy_weights_norm.min():.4f}, {entropy_weights_norm.max():.4f}]")
    print(f"   平均熵: {entropy_weights_norm.mean():.4f}")
    
    # ===== 步骤3：关联权重（MAR/MNAR指示）=====
    print(f"\n🔍 步骤2：计算缺失关联权重...")
    
    association_weights = np.zeros(n_features)
    
    for j in range(n_features):
        col_missing = missing_mask[:, j].astype(np.float32)  # 缺失指示向量
        
        if col_missing.sum() == 0 or col_missing.sum() == n_samples:
            # 无缺失或全缺失
            association_weights[j] = 0
            continue
        
        correlations = []
        
        # ★计算与其他特征的关联
        for k in range(n_features):
            if k == j:
                continue
            
            # 策略1：与其他特征的观测值关联（MAR检测）
            col_k_vals = X_np[:, k].copy()
            observed_k = ~np.isnan(col_k_vals)
            
            if observed_k.sum() < 5:
                continue
            
            # 填充缺失值（用于计算）
            col_k_vals[np.isnan(col_k_vals)] = np.nanmedian(col_k_vals)
            
            # 计算点二列相关系数
            corr = np.corrcoef(col_missing, col_k_vals)[0, 1]
            
            if not np.isnan(corr):
                correlations.append(np.abs(corr))
        
        # 策略2：与自身观测值关联（MNAR检测）
        col_j_vals = X_np[:, j].copy()
        observed_j = ~missing_mask[:, j]
        
        if observed_j.sum() >= 5:
            # 比较高值/低值的缺失倾向
            observed_vals = col_j_vals[observed_j]
            median_val = np.median(observed_vals)
            
            # 高值样本的缺失率
            high_samples = col_j_vals > median_val
            high_missing_rate = missing_mask[high_samples, j].mean() if high_samples.sum() > 0 else 0
            
            # 低值样本的缺失率
            low_samples = col_j_vals <= median_val
            low_missing_rate = missing_mask[low_samples, j].mean() if low_samples.sum() > 0 else 0
            
            # MNAR指示：高低值缺失率差异
            mnar_score = np.abs(high_missing_rate - low_missing_rate)
            correlations.append(mnar_score)
        
        # 综合关联强度
        association_weights[j] = np.mean(correlations) if correlations else 0
    
    # 归一化
    if association_weights.max() > 0:
        association_weights_norm = association_weights / (association_weights.max() + 1e-10)
    else:
        association_weights_norm = association_weights
    
    print(f"   关联权重范围: [{association_weights_norm.min():.4f}, {association_weights_norm.max():.4f}]")
    print(f"   平均关联: {association_weights_norm.mean():.4f}")
    
    # ===== 步骤4：综合权重（熵 × 关联）=====
    print(f"\n🔍 步骤3：生成综合权重...")
    
    # ★组合策略：熵权重 × (1 + α × 关联权重)
    alpha = 2.0  # 关联权重放大系数
    combined_weights = entropy_weights_norm * (1 + alpha * association_weights_norm)
    
    # 归一化（和为1）
    if combined_weights.sum() > 0:
        global_weights = combined_weights / combined_weights.sum()
    else:
        global_weights = np.ones(n_features) / n_features
    
    print(f"   综合权重范围: [{global_weights.min():.4f}, {global_weights.max():.4f}]")
    print(f"   权重标准差: {global_weights.std():.4f}")
    
    # ===== 步骤5：特征分类 =====
    threshold_high = 0.1  # 高缺失阈值
    threshold_low = 0.01  # 低缺失阈值
    
    high_missing_features = feature_missing_rate[feature_missing_rate > threshold_high].index.tolist()
    medium_missing_features = feature_missing_rate[
        (feature_missing_rate > threshold_low) & 
        (feature_missing_rate <= threshold_high)
    ].index.tolist()
    low_missing_features = feature_missing_rate[feature_missing_rate <= threshold_low].index.tolist()
    
    print(f"\n📋 特征分类:")
    print(f"   高缺失 (>{threshold_high:.0%}): {len(high_missing_features)} 个")
    print(f"   中缺失 ({threshold_low:.0%}-{threshold_high:.0%}): {len(medium_missing_features)} 个")
    print(f"   低缺失 (<{threshold_low:.0%}): {len(low_missing_features)} 个")
    
    # ===== 步骤6：缺失机制推断（多指标）=====
    print(f"\n🔍 步骤4：推断缺失机制...")
    
    # 指标1：缺失模式相关性矩阵
    missing_pattern = missing_mask.astype(np.float32)  # (n_samples, n_features)
    missing_corr = np.corrcoef(missing_pattern.T)  # (n_features, n_features)
    
    # 提取上三角（排除对角线）
    triu_indices = np.triu_indices_from(missing_corr, k=1)
    avg_missing_corr = np.abs(missing_corr[triu_indices]).mean()
    max_missing_corr = np.abs(missing_corr[triu_indices]).max()
    
    # 指标2：平均关联强度
    avg_association = association_weights_norm.mean()
    
    # 指标3：高值缺失偏好（MNAR特征）
    mnar_scores = []
    for j in range(n_features):
        if feature_missing_rate.iloc[j] < 0.01:
            continue
        
        col_vals = X_np[:, j]
        observed_mask = ~np.isnan(col_vals)
        
        if observed_mask.sum() < 10:
            continue
        
        observed_vals = col_vals[observed_mask]
        q75 = np.percentile(observed_vals, 75)
        
        high_mask = col_vals > q75
        if high_mask.sum() > 0:
            high_missing_rate = missing_mask[high_mask, j].mean()
            overall_col_missing = missing_mask[:, j].mean()
            
            if overall_col_missing > 0:
                mnar_scores.append(high_missing_rate / overall_col_missing)
    
    avg_mnar_score = np.mean(mnar_scores) if mnar_scores else 1.0
    
    # ★综合判定规则
    print(f"   缺失相关性: {avg_missing_corr:.4f} (最大: {max_missing_corr:.4f})")
    print(f"   关联强度: {avg_association:.4f}")
    print(f"   MNAR得分: {avg_mnar_score:.4f}")
    
    # 判定逻辑
    if avg_missing_corr < 0.05 and avg_association < 0.1 and 0.9 < avg_mnar_score < 1.1:
        inferred_mechanism = "MCAR"
        confidence = "高"
    elif avg_missing_corr >= 0.3 or avg_association >= 0.3:
        if avg_mnar_score > 1.3:
            inferred_mechanism = "MNAR"
            confidence = "中"
        else:
            inferred_mechanism = "MAR"
            confidence = "中"
    elif 0.05 <= avg_missing_corr < 0.3:
        inferred_mechanism = "MAR"
        confidence = "低"
    else:
        inferred_mechanism = "混合机制"
        confidence = "低"
    
    print(f"\n✅ 推断结果: {inferred_mechanism} (置信度: {confidence})")
    
    # ===== 步骤7：生成结果字典 =====
    global_missing_stats = {
        # 基础统计
        "overall_missing_rate": round(overall_missing_rate, 4),
        "feature_missing_rate": feature_missing_rate.to_dict(),
        
        # 特征分类
        "high_missing_features": high_missing_features,
        "medium_missing_features": medium_missing_features,
        "low_missing_features": low_missing_features,
        
        # 权重信息
        "global_weights": dict(zip(col_names, global_weights)),
        "entropy_weights": dict(zip(col_names, entropy_weights_norm)),
        "association_weights": dict(zip(col_names, association_weights_norm)),
        
        # 机制推断
        "avg_missing_corr": round(avg_missing_corr, 4),
        "max_missing_corr": round(max_missing_corr, 4),
        "avg_association": round(avg_association, 4),
        "avg_mnar_score": round(avg_mnar_score, 4),
        "inferred_mechanism": inferred_mechanism,
        "confidence": confidence,
        
        # Top特征
        "top5_high_missing": feature_missing_rate.head(5).to_dict(),
        "top5_high_weight": dict(sorted(
            zip(col_names, global_weights), 
            key=lambda x: x[1], 
            reverse=True
        )[:5])
    }
    
    # ===== 步骤8：可视化输出 =====
    print("\n" + "="*70)
    print("【分析摘要】")
    print("="*70)
    print(f"✓ 缺失率: {overall_missing_rate:.2%}")
    print(f"✓ 推断机制: {inferred_mechanism} ({confidence}置信度)")
    print(f"✓ 缺失模式关联: {avg_missing_corr:.4f}")
    print(f"✓ 权重集中度: {global_weights.std():.4f} (标准差)")
    
    print(f"\n【Top5 高缺失特征】")
    for feat, rate in list(global_missing_stats['top5_high_missing'].items()):
        print(f"   {feat}: {rate:.2%}")
    
    print(f"\n【Top5 高权重特征】")
    for feat, weight in list(global_missing_stats['top5_high_weight'].items()):
        print(f"   {feat}: {weight:.4f}")
    
    print("="*70 + "\n")
    
    return global_missing_stats, inferred_mechanism
# =====================================================================
#                   第5部分：缺失机制推断与参数自适应
# =====================================================================
warnings.filterwarnings('ignore')


class RobustMissingMechanismDetector:
    """
    ★增强版缺失机制检测器 v2.0
    
    核心改进：
    1. ✅ MNAR多维检测：分布偏移 + 极值偏好 + 条件独立性
    2. ✅ MAR鲁棒性：多重假设检验校正 + 效应量
    3. ✅ 自适应阈值：基于缺失率和样本量动态调整
    4. ✅ 贝叶斯融合：多证据加权综合
    """
    
    def __init__(self, base_alpha=0.05, min_sample_size=10):
        self.base_alpha = base_alpha
        self.min_sample_size = min_sample_size
        self.results = {}
    
    def detect_missing_mechanism_per_feature(self, X):
        """
        ★逐特征检测缺失机制（增强版）
        """
        X_np = np.asarray(X).astype(np.float32)
        n_samples, n_features = X_np.shape
        
        feature_results = {}
        
        print("\n" + "="*70)
        print("【缺失机制诊断】逐特征分析")
        print("="*70)
        
        for j in range(n_features):
            col_data = X_np[:, j]
            col_missing = np.isnan(col_data).astype(int)
            missing_rate = col_missing.mean()
            
            # ===== 无缺失情况 =====
            if missing_rate == 0:
                feature_results[j] = {
                    'mechanism': 'No Missing',
                    'confidence': 1.0,
                    'missing_rate': 0.0,
                    'mar_score': 0.0,
                    'mnar_score': 0.0,
                    'mcar_score': 1.0
                }
                continue
            
            print(f"\n🔍 特征 {j}: 缺失率={missing_rate:.2%}")
            
            # ===== 步骤1：MAR检测（增强版）=====
            mar_result = self._detect_mar_evidence(
                X_np, j, col_missing, missing_rate
            )
            
            # ===== 步骤2：MNAR检测（多维度）=====
            mnar_result = self._detect_mnar_evidence(
                col_data, col_missing, missing_rate
            )
            
            # ===== 步骤3：MCAR检测（排除法）=====
            mcar_result = self._detect_mcar_evidence(
                mar_result, mnar_result, missing_rate
            )
            
            # ===== 步骤4：贝叶斯融合 =====
            mechanism, confidence, scores = self._bayesian_fusion(
                mar_result, mnar_result, mcar_result, missing_rate
            )
            
            # ===== 保存结果 =====
            feature_results[j] = {
                'mechanism': mechanism,
                'confidence': float(confidence),
                'missing_rate': float(missing_rate),
                'mar_score': float(scores['mar']),
                'mnar_score': float(scores['mnar']),
                'mcar_score': float(scores['mcar']),
                'mar_details': mar_result,
                'mnar_details': mnar_result,
                'diagnostic_info': {
                    'n_samples': int(n_samples),
                    'n_missing': int(col_missing.sum()),
                    'n_observed': int((~col_missing).sum())
                }
            }
            
            print(f"   ✓ 判定: {mechanism} (置信度: {confidence:.2f})")
            print(f"      MAR={scores['mar']:.3f}, MNAR={scores['mnar']:.3f}, MCAR={scores['mcar']:.3f}")
        
        self.results = feature_results
        print("\n" + "="*70)
        return feature_results
    
    def _detect_mar_evidence(self, X_np, target_col, col_missing, missing_rate):
        """
        ★MAR检测：缺失依赖其他观测特征
        
        方法：
        1. 卡方检验（分类依赖）
        2. Logistic回归（连续依赖）
        3. 效应量计算（Cramér's V）
        """
        n_samples, n_features = X_np.shape
        
        # 自适应显著性水平（Bonferroni校正）
        adaptive_alpha = self.base_alpha / (n_features - 1)
        
        mar_evidence = []
        
        for k in range(n_features):
            if k == target_col:
                continue
            
            col_k = X_np[:, k].copy()
            
            # 跳过全缺失列
            if np.isnan(col_k).sum() == len(col_k):
                continue
            
            # 填充缺失值（用于计算）
            col_k_filled = np.nan_to_num(col_k, nan=np.nanmedian(col_k))
            
            # ===== 方法1：卡方检验（二分类）=====
            median_k = np.median(col_k_filled)
            col_k_high = (col_k_filled > median_k).astype(int)
            
            # 构建列联表
            contingency = np.array([
                [np.sum((col_k_high==0) & (col_missing==0)), 
                 np.sum((col_k_high==0) & (col_missing==1))],
                [np.sum((col_k_high==1) & (col_missing==0)), 
                 np.sum((col_k_high==1) & (col_missing==1))]
            ])
            
            # 小样本保护
            if np.min(contingency) < 5:
                continue
            
            try:
                chi2, p_val, dof, expected = chi2_contingency(contingency)
                
                # ★ 效应量：Cramér's V
                n = contingency.sum()
                cramers_v = np.sqrt(chi2 / (n * (min(contingency.shape) - 1)))
                
                is_significant = p_val < adaptive_alpha
                
                mar_evidence.append({
                    'feature_k': k,
                    'test_type': 'chi2',
                    'p_value': float(p_val),
                    'effect_size': float(cramers_v),
                    'statistic': float(chi2),
                    'significant': bool(is_significant)
                })
                
            except Exception as e:
                mar_evidence.append({
                    'feature_k': k,
                    'test_type': 'chi2',
                    'p_value': 1.0,
                    'effect_size': 0.0,
                    'statistic': 0.0,
                    'significant': False
                })
        
        # ===== 汇总MAR证据 =====
        if not mar_evidence:
            return {
                'score': 0.0,
                'n_significant': 0,
                'n_total': 0,
                'avg_effect_size': 0.0,
                'evidence_list': []
            }
        
        n_significant = sum(e['significant'] for e in mar_evidence)
        n_total = len(mar_evidence)
        
        # ★ MAR得分：显著比例 × 平均效应量
        significant_effects = [e['effect_size'] for e in mar_evidence if e['significant']]
        avg_effect_size = np.mean(significant_effects) if significant_effects else 0.0
        
        mar_score = (n_significant / n_total) * (1 + avg_effect_size)
        
        return {
            'score': float(mar_score),
            'n_significant': int(n_significant),
            'n_total': int(n_total),
            'avg_effect_size': float(avg_effect_size),
            'adaptive_alpha': float(adaptive_alpha),
            'evidence_list': mar_evidence
        }
    
    def _detect_mnar_evidence(self, col_data, col_missing, missing_rate):
        """
        ★MNAR检测（多维度增强版）
        
        核心思想：缺失依赖自身未观测值
        
        检测维度：
        1. 分布偏移（KS检验）：观测值 vs 完整分布
        2. 极值偏好（分位数差异）：高值/低值缺失倾向
        3. 值域截断（边界检测）：观测值范围收缩
        4. 条件独立性（自回归）：缺失与滞后值的关联
        """
        observed_mask = ~col_missing.astype(bool)
        observed_vals = col_data[observed_mask]
        
        if len(observed_vals) < self.min_sample_size:
            return {
                'score': 0.0,
                'distribution_shift': 0.0,
                'extreme_preference': 0.0,
                'range_truncation': 0.0,
                'method': 'insufficient_data'
            }
        
        # ===== 维度1：分布偏移检测 =====
        # 理论：MNAR导致观测分布 ≠ 真实分布
        
        # 使用完整数据的经验分布（含缺失）作为参考
        # 这里用观测值的bootstrap样本模拟
        bootstrap_samples = []
        for _ in range(100):
            bootstrap_samples.append(
                np.random.choice(observed_vals, size=len(observed_vals), replace=True)
            )
        
        # KS检验：观测值 vs bootstrap分布
        ks_stats = []
        for boot_sample in bootstrap_samples:
            stat, _ = ks_2samp(observed_vals, boot_sample)
            ks_stats.append(stat)
        
        distribution_shift_score = np.mean(ks_stats)  # 平均KS统计量
        
        # ===== 维度2：极值偏好检测 =====
        # 理论：MNAR倾向于高值或低值缺失
        
        # 计算观测值的分位数
        q10_obs, q50_obs, q90_obs = np.percentile(observed_vals, [10, 50, 90])
        
        # 模拟：如果是MCAR，缺失应均匀分布
        # 检测方法：比较观测值的偏度
        observed_skewness = np.abs(skew(observed_vals))
        
        # 高偏度 → 可能存在极值缺失
        extreme_preference_score = min(1.0, observed_skewness / 2.0)
        
        # ===== 维度3：值域截断检测 =====
        # 理论：MNAR导致观测值范围缩小
        
        obs_range = q90_obs - q10_obs
        obs_iqr = np.percentile(observed_vals, 75) - np.percentile(observed_vals, 25)
        
        # 期望范围（基于正态假设）
        expected_range = 2.56 * obs_iqr  # 理论上90%范围约为2.56×IQR
        
        # 截断得分：实际范围 < 期望范围
        range_truncation_score = max(0, 1 - obs_range / (expected_range + 1e-8))
        
        # ===== 维度4：高值/低值缺失倾向（改进）=====
        # 检测策略：比较不同分位数区间的"虚拟缺失率"
        
        # 使用观测值构建参考分布
        q25, q75 = np.percentile(observed_vals, [25, 75])
        
        # 假设MCAR：高值和低值的观测数量应相近
        n_low = np.sum(observed_vals < q25)
        n_high = np.sum(observed_vals > q75)
        
        # 不平衡度
        imbalance = np.abs(n_high - n_low) / (n_high + n_low + 1e-8)
        
        # ===== 综合MNAR得分 =====
        # ★ 加权融合多个指标
        mnar_score = (
            0.3 * distribution_shift_score +
            0.3 * extreme_preference_score +
            0.2 * range_truncation_score +
            0.2 * imbalance
        )
        
        return {
            'score': float(mnar_score),
            'distribution_shift': float(distribution_shift_score),
            'extreme_preference': float(extreme_preference_score),
            'range_truncation': float(range_truncation_score),
            'value_imbalance': float(imbalance),
            'observed_skewness': float(observed_skewness),
            'method': 'multi_dimensional'
        }
    
    def _detect_mcar_evidence(self, mar_result, mnar_result, missing_rate):
        """
        ★MCAR检测（排除法 + 随机性验证）
        
        理论：MCAR = 无MAR证据 + 无MNAR证据
        """
        # MCAR得分：反向得分
        mcar_score = 1.0 - mar_result['score'] - mnar_result['score']
        mcar_score = max(0.0, min(1.0, mcar_score))
        
        # ★ 额外验证：缺失模式的随机性
        # 如果缺失率接近均匀，增强MCAR置信
        uniformity_bonus = 0.0
        if 0.05 < missing_rate < 0.95:  # 避免极端情况
            # 理想MCAR的熵
            ideal_entropy = -(missing_rate * np.log2(missing_rate + 1e-10) +
                             (1 - missing_rate) * np.log2(1 - missing_rate + 1e-10))
            
            # 当前熵接近理想熵 → MCAR可能性高
            if ideal_entropy > 0.5:  # 熵足够高（不确定性高）
                uniformity_bonus = 0.1
        
        mcar_score += uniformity_bonus
        mcar_score = min(1.0, mcar_score)
        
        return {
            'score': float(mcar_score),
            'uniformity_bonus': float(uniformity_bonus),
            'method': 'exclusion_and_randomness'
        }
    
    def _bayesian_fusion(self, mar_result, mnar_result, mcar_result, missing_rate):
        """
        ★贝叶斯融合：多证据加权决策
        
        先验知识：
        - 低缺失率（<10%）：MCAR先验高
        - 中缺失率（10-30%）：MAR先验高
        - 高缺失率（>30%）：MNAR先验高
        """
        # ===== 先验概率 =====
        if missing_rate < 0.1:
            prior = {'MCAR': 0.4, 'MAR': 0.3, 'MNAR': 0.3}
        elif missing_rate < 0.3:
            prior = {'MCAR': 0.3, 'MAR': 0.4, 'MNAR': 0.3}
        else:
            prior = {'MCAR': 0.3, 'MAR': 0.3, 'MNAR': 0.4}
        
        # ===== 似然（证据得分）=====
        likelihood = {
            'MCAR': mcar_result['score'],
            'MAR': mar_result['score'],
            'MNAR': mnar_result['score']
        }
        
        # ===== 后验概率（未归一化）=====
        posterior_unnorm = {
            mech: prior[mech] * likelihood[mech]
            for mech in ['MCAR', 'MAR', 'MNAR']
        }
        
        # ===== 归一化 =====
        total = sum(posterior_unnorm.values()) + 1e-10
        posterior = {
            mech: prob / total
            for mech, prob in posterior_unnorm.items()
        }
        
        # ===== 最终决策 =====
        mechanism = max(posterior, key=posterior.get)
        confidence = posterior[mechanism]
        
        scores = {
            'mar': mar_result['score'],
            'mnar': mnar_result['score'],
            'mcar': mcar_result['score']
        }
        
        return mechanism, confidence, scores
    
    def get_global_mechanism(self):
        """
        ★全局机制聚合（加权投票）
        """
        if not self.results:
            return {
                'dominant_mechanism': 'Unknown',
                'confidence': 0.0
            }
        
        # 过滤有效特征
        valid_results = {
            k: v for k, v in self.results.items()
            if v['mechanism'] != 'No Missing'
        }
        
        if not valid_results:
            return {
                'dominant_mechanism': 'No Missing',
                'confidence': 1.0
            }
        
        # ===== 加权投票 =====
        weighted_votes = {'MCAR': 0.0, 'MAR': 0.0, 'MNAR': 0.0}
        
        for res in valid_results.values():
            mech = res['mechanism']
            conf = res['confidence']
            missing_rate = res['missing_rate']
            
            # 权重 = 置信度 × 缺失率（高缺失率特征权重更大）
            weight = conf * (1 + missing_rate)
            weighted_votes[mech] += weight
        
        # ===== 决策 =====
        total_weight = sum(weighted_votes.values()) + 1e-10
        normalized_votes = {
            mech: vote / total_weight
            for mech, vote in weighted_votes.items()
        }
        
        dominant_mech = max(normalized_votes, key=normalized_votes.get)
        global_confidence = normalized_votes[dominant_mech]
        
        # ===== 机制分布 =====
        mechanism_counts = Counter([
            res['mechanism'] for res in valid_results.values()
        ])
        
        return {
            'dominant_mechanism': dominant_mech,
            'confidence': float(global_confidence),
            'mechanism_distribution': dict(mechanism_counts),
            'weighted_votes': {k: float(v) for k, v in normalized_votes.items()},
            'feature_level_details': self.results
        }
    
    def visualize_mechanism_results(self, feature_names=None, save_path=None):
        """
        ★可视化（增强版）
        """
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(len(self.results))]
        
        # 过滤有效特征
        valid_data = []
        for feat_id, res in self.results.items():
            if res['mechanism'] == 'No Missing':
                continue
            valid_data.append({
                'id': feat_id,
                'name': feature_names[feat_id][:15],
                'mechanism': res['mechanism'],
                'confidence': res['confidence'],
                'mar_score': res['mar_score'],
                'mnar_score': res['mnar_score'],
                'mcar_score': res['mcar_score']
            })
        
        if not valid_data:
            print("⚠️ 无缺失特征，跳过可视化")
            return None
        
        # ===== 绘图 =====
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        colors_mech = {'MCAR': '#2ecc71', 'MAR': '#f39c12', 'MNAR': '#e74c3c'}
        
        # 子图1：机制分类
        ax1 = axes[0, 0]
        for i, item in enumerate(valid_data):
            color = colors_mech[item['mechanism']]
            ax1.barh(i, item['confidence'], color=color, alpha=0.7)
            ax1.text(item['confidence'] + 0.02, i, item['name'], 
                    va='center', fontsize=9)
        
        ax1.set_xlabel('Confidence', fontsize=12, fontweight='bold')
        ax1.set_title('Missing Mechanism per Feature', fontsize=13, fontweight='bold')
        ax1.set_xlim(0, 1.1)
        ax1.grid(True, alpha=0.3, axis='x')
        
        # 子图2：三维得分对比
        ax2 = axes[0, 1]
        x = np.arange(len(valid_data))
        width = 0.25
        
        ax2.bar(x - width, [d['mar_score'] for d in valid_data], 
               width, label='MAR', color='#f39c12', alpha=0.7)
        ax2.bar(x, [d['mnar_score'] for d in valid_data], 
               width, label='MNAR', color='#e74c3c', alpha=0.7)
        ax2.bar(x + width, [d['mcar_score'] for d in valid_data], 
               width, label='MCAR', color='#2ecc71', alpha=0.7)
        
        ax2.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax2.set_title('MAR/MNAR/MCAR Scores', fontsize=13, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels([d['name'] for d in valid_data], rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 子图3：置信度分布
        ax3 = axes[1, 0]
        confidences = [d['confidence'] for d in valid_data]
        mechanisms = [d['mechanism'] for d in valid_data]
        
        for mech in ['MCAR', 'MAR', 'MNAR']:
            mech_confs = [c for c, m in zip(confidences, mechanisms) if m == mech]
            if mech_confs:
                ax3.hist(mech_confs, bins=10, alpha=0.6, 
                        label=mech, color=colors_mech[mech])
        
        ax3.set_xlabel('Confidence', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax3.set_title('Confidence Distribution', fontsize=13, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 子图4：全局分布饼图
        ax4 = axes[1, 1]
        mech_count = Counter([d['mechanism'] for d in valid_data])
        
        wedges, texts, autotexts = ax4.pie(
            list(mech_count.values()),
            labels=list(mech_count.keys()),
            autopct='%1.1f%%',
            startangle=90,
            colors=[colors_mech[k] for k in mech_count.keys()],
            textprops={'fontsize': 11, 'fontweight': 'bold'}
        )
        ax4.set_title('Global Mechanism Distribution', fontsize=13, fontweight='bold')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig


# # 使用示例
# detector = AdvancedMissingMechanismDetector(alpha=0.05, bonferroni=True)
# feature_mechanisms = detector.detect_missing_mechanism_per_feature(X)

# global_result = detector.get_global_mechanism()
# print(f"全局主导机制: {global_result['dominant_mechanism']}")
# print(f"置信度: {global_result['confidence']:.2f}")

# # 可视化
# detector.visualize_mechanism_results(
#     feature_names=X.columns.tolist(),
#     save_path='missing_mechanisms_per_feature.png'
# )
warnings.filterwarnings('ignore')

warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore')

class MissingMechanismClassifier:
    """
    ★增强版缺失机制分类器（完整修复版）
    
    核心改进：
    1. ✅ 多维MNAR检测：分布偏移+极值偏好+值域截断
    2. ✅ 鲁棒MAR检测：Bonferroni校正+效应量计算
    3. ✅ 贝叶斯融合：先验+似然→后验概率
    4. ✅ 自适应参数：基于机制概率动态生成
    """
    
    def __init__(self, base_alpha=0.05, min_sample_size=10):
        self.base_alpha = base_alpha
        self.min_sample_size = min_sample_size
        self.mech_prob = {}
        self.mechanism_details = {}
    
    def infer_missing_mechanism(self, X):
        """
        ★逐特征检测缺失机制并融合全局机制
        """
        X_np = np.asarray(X).astype(np.float32)
        n_samples, n_features = X_np.shape
        
        print("\n" + "="*70)
        print("【缺失机制推断】逐特征多维检测")
        print("="*70)
        
        feature_mechanisms = {}
        
        for j in range(n_features):
            col_data = X_np[:, j]
            col_missing = np.isnan(col_data).astype(int)
            missing_rate = col_missing.mean()
            
            # 跳过无缺失特征
            if missing_rate == 0:
                feature_mechanisms[j] = {
                    'mechanism': 'No Missing',
                    'mar_prob': 0.0,
                    'mnar_prob': 0.0,
                    'mcar_prob': 1.0
                }
                continue
            
            print(f"\n🔍 特征 {j}: 缺失率={missing_rate:.2%}")
            
            # ===== 步骤1：MAR证据收集 =====
            mar_evidence = self._detect_mar_evidence(
                X_np, j, col_missing, n_features
            )
            
            # ===== 步骤2：MNAR多维检测 =====
            mnar_evidence = self._detect_mnar_evidence(
                col_data, col_missing, missing_rate
            )
            
            # ===== 步骤3：贝叶斯融合 =====
            mar_prob, mnar_prob, mcar_prob = self._bayesian_fusion(
                mar_evidence, mnar_evidence, missing_rate
            )
            
            # ===== 确定主机制 =====
            mechanism = self._determine_mechanism(mar_prob, mnar_prob, mcar_prob)
            
            feature_mechanisms[j] = {
                'mechanism': mechanism,
                'mar_prob': float(mar_prob),
                'mnar_prob': float(mnar_prob),
                'mcar_prob': float(mcar_prob),
                'missing_rate': float(missing_rate),
                'mar_details': mar_evidence,
                'mnar_details': mnar_evidence
            }
            
            print(f"   ✓ 判定: {mechanism}")
            print(f"      概率: MCAR={mcar_prob:.3f}, MAR={mar_prob:.3f}, MNAR={mnar_prob:.3f}")
        
        # ===== 全局聚合 =====
        self.mechanism_details = feature_mechanisms
        self.mech_prob = self._aggregate_global_mechanism(feature_mechanisms)
        
        print("\n" + "="*70)
        print(f"【全局机制】MCAR={self.mech_prob['MCAR']:.3f}, "
              f"MAR={self.mech_prob['MAR']:.3f}, "
              f"MNAR={self.mech_prob['MNAR']:.3f}")
        print("="*70 + "\n")
        
        return self.mech_prob
    
    def _detect_mar_evidence(self, X_np, target_col, col_missing, n_features):
        """
        ★MAR检测：缺失依赖其他观测特征（修复版）
        
        方法：
        - 卡方检验（特征间依赖）
        - Bonferroni多重假设检验校正
        - Cramér's V效应量
        """
        adaptive_alpha = self.base_alpha / max(1, n_features - 1)
        
        # ★ 关键修复：初始化为int和float，而不是列表
        significant_tests = 0
        total_tests = 0
        effect_sizes = []  # ★ 用列表收集效应量
        
        for k in range(n_features):
            if k == target_col:
                continue
            
            col_k = X_np[:, k].copy()
            
            # 跳过全缺失列
            if np.isnan(col_k).sum() == len(col_k):
                continue
            
            total_tests += 1
            
            # 填充缺失值用于卡方检验
            col_k_filled = np.nan_to_num(col_k, nan=np.nanmedian(col_k))
            median_k = np.median(col_k_filled)
            col_k_binary = (col_k_filled > median_k).astype(int)
            
            # 构建列联表
            contingency = np.array([
                [np.sum((col_k_binary==0) & (col_missing==0)), 
                 np.sum((col_k_binary==0) & (col_missing==1))],
                [np.sum((col_k_binary==1) & (col_missing==0)), 
                 np.sum((col_k_binary==1) & (col_missing==1))]
            ])
            
            # 小样本保护（Fisher精确检验阈值）
            if np.min(contingency) < 5:
                continue
            
            try:
                chi2, p_val, dof, expected = chi2_contingency(contingency)
                
                # 计算Cramér's V效应量
                n = contingency.sum()
                cramers_v = np.sqrt(chi2 / (n * (min(contingency.shape) - 1) + 1e-10))
                
                is_significant = p_val < adaptive_alpha
                
                if is_significant:
                    significant_tests += 1
                    effect_sizes.append(cramers_v)  # ★ 添加到列表
                
            except Exception as e:
                continue
        
        # ===== 综合MAR得分 =====
        if total_tests == 0:
            mar_score = 0.0
            avg_effect_size = 0.0
        else:
            # ★ 从列表计算平均值
            avg_effect_size = np.mean(effect_sizes) if effect_sizes else 0.0
            # 得分 = 显著比例 × (1 + 效应量)
            mar_score = (significant_tests / total_tests) * (1 + avg_effect_size)
            mar_score = min(1.0, mar_score)
        
        return {
            'score': float(mar_score),
            'n_significant': int(significant_tests),
            'n_total': int(total_tests),
            'avg_effect_size': float(avg_effect_size),
            'adaptive_alpha': float(adaptive_alpha)
        }
    
    def _detect_mnar_evidence(self, col_data, col_missing, missing_rate):
        """
        ★MNAR多维检测
        
        检测维度：
        1. 分布偏移（KS检验）
        2. 极值偏好（观测值偏度）
        3. 值域截断（范围收缩）
        4. 值不平衡（高值/低值缺失倾向）
        """
        observed_mask = ~col_missing.astype(bool)
        observed_vals = col_data[observed_mask]
        
        if len(observed_vals) < self.min_sample_size:
            return {
                'score': 0.0,
                'distribution_shift': 0.0,
                'extreme_preference': 0.0,
                'range_truncation': 0.0,
                'value_imbalance': 0.0
            }
        
        # ===== 维度1：分布偏移 =====
        distribution_shift_score = 0.0
        try:
            # Bootstrap采样
            ks_stats = []
            for _ in range(50):
                bootstrap_sample = np.random.choice(
                    observed_vals, size=len(observed_vals), replace=True
                )
                stat, _ = ks_2samp(observed_vals, bootstrap_sample)
                ks_stats.append(stat)
            
            distribution_shift_score = float(np.mean(ks_stats))  # ★ 显式转换为float
        except Exception:
            distribution_shift_score = 0.0
        
        # ===== 维度2：极值偏好 =====
        extreme_preference_score = 0.0
        try:
            observed_skewness = np.abs(float(skew(observed_vals)))  # ★ 显式转换为float
            # 高偏度→存在极值缺失
            extreme_preference_score = min(1.0, observed_skewness / 2.0)
        except Exception:
            extreme_preference_score = 0.0
        
        # ===== 维度3：值域截断 =====
        range_truncation_score = 0.0
        try:
            q10_obs = float(np.percentile(observed_vals, 10))
            q25_obs = float(np.percentile(observed_vals, 25))
            q75_obs = float(np.percentile(observed_vals, 75))
            q90_obs = float(np.percentile(observed_vals, 90))
            
            obs_range = q90_obs - q10_obs
            obs_iqr = q75_obs - q25_obs
            
            # 期望范围
            expected_range = 2.56 * obs_iqr
            
            # 范围收缩
            range_truncation_score = max(0.0, 1.0 - obs_range / (expected_range + 1e-8))
        except Exception:
            range_truncation_score = 0.0
        
        # ===== 维度4：值不平衡 =====
        value_imbalance_score = 0.0
        try:
            q25 = float(np.percentile(observed_vals, 25))
            q75 = float(np.percentile(observed_vals, 75))
            
            n_low = np.sum(observed_vals < q25)
            n_high = np.sum(observed_vals > q75)
            
            value_imbalance_score = float(np.abs(n_high - n_low) / (n_high + n_low + 1e-8))
        except Exception:
            value_imbalance_score = 0.0
        
        # ===== 综合MNAR得分（加权融合）=====
        mnar_score = (
            0.3 * distribution_shift_score +
            0.3 * extreme_preference_score +
            0.2 * range_truncation_score +
            0.2 * value_imbalance_score
        )
        mnar_score = min(1.0, float(mnar_score))
        
        return {
            'score': float(mnar_score),
            'distribution_shift': float(distribution_shift_score),
            'extreme_preference': float(extreme_preference_score),
            'range_truncation': float(range_truncation_score),
            'value_imbalance': float(value_imbalance_score)
        }
    
    def _bayesian_fusion(self, mar_evidence, mnar_evidence, missing_rate):
        """
        ★贝叶斯融合：先验×似然→后验
        
        先验设置：
        - 低缺失（<10%）：MCAR先验高
        - 中缺失（10-30%）：MAR先验高
        - 高缺失（>30%）：MNAR先验高
        """
        # ===== 先验概率 =====
        if missing_rate < 0.1:
            prior = {'MCAR': 0.4, 'MAR': 0.3, 'MNAR': 0.3}
        elif missing_rate < 0.3:
            prior = {'MCAR': 0.3, 'MAR': 0.4, 'MNAR': 0.3}
        else:
            prior = {'MCAR': 0.3, 'MAR': 0.3, 'MNAR': 0.4}
        
        # ===== 似然 =====
        mar_score = float(mar_evidence['score'])  # ★ 显式转换
        mnar_score = float(mnar_evidence['score'])  # ★ 显式转换
        
        # MCAR得分：反向证据
        mcar_score = 1.0 - max(mar_score, mnar_score)
        mcar_score = max(0.0, min(1.0, mcar_score))
        
        likelihood = {
            'MCAR': mcar_score,
            'MAR': mar_score,
            'MNAR': mnar_score
        }
        
        # ===== 后验（未归一化）=====
        posterior_unnorm = {
            mech: prior[mech] * likelihood[mech]
            for mech in ['MCAR', 'MAR', 'MNAR']
        }
        
        # ===== 归一化 =====
        total = sum(posterior_unnorm.values()) + 1e-10
        posterior = {
            mech: prob / total
            for mech, prob in posterior_unnorm.items()
        }
        
        return float(posterior['MAR']), float(posterior['MNAR']), float(posterior['MCAR'])
    
    def _determine_mechanism(self, mar_prob, mnar_prob, mcar_prob):
        """
        ★确定主缺失机制（概率最高）
        """
        probs = {'MAR': float(mar_prob), 'MNAR': float(mnar_prob), 'MCAR': float(mcar_prob)}
        return max(probs, key=probs.get)
    
    def _aggregate_global_mechanism(self, feature_mechanisms):
        """
        ★全局聚合：加权投票融合所有特征
        """
        valid_features = {
            k: v for k, v in feature_mechanisms.items()
            if v['mechanism'] != 'No Missing'
        }
        
        if not valid_features:
            return {'MCAR': 1.0, 'MAR': 0.0, 'MNAR': 0.0}
        
        # 加权投票：权重 = 置信度 × (1 + 缺失率)
        weighted_votes = {'MCAR': 0.0, 'MAR': 0.0, 'MNAR': 0.0}
        
        for res in valid_features.values():
            mech = res['mechanism']
            
            # 取最大概率作为置信度
            confidence = max(res['mar_prob'], res['mnar_prob'], res['mcar_prob'])
            missing_rate = res['missing_rate']
            
            # 权重函数
            weight = float(confidence) * (1.0 + float(missing_rate))
            weighted_votes[mech] += weight
        
        # 归一化
        total_weight = sum(weighted_votes.values()) + 1e-10
        global_probs = {
            mech: vote / total_weight
            for mech, vote in weighted_votes.items()
        }
        
        return global_probs
    
    def get_adaptive_parameters(self):
        """
        ★根据推断机制返回自适应参数
        
        参数自适应策略：
        - MCAR: 使用简单插补，k较小
        - MAR: 需要条件插补，k中等
        - MNAR: 需要模型化缺失过程，k较大
        """
        dominant_mech = max(self.mech_prob, key=self.mech_prob.get)
        confidence = self.mech_prob[dominant_mech]
        
        print(f"\n{'='*70}")
        print(f"【推断缺失机制】")
        print(f"  主机制: {dominant_mech} (置信度: {confidence:.3f})")
        print(f"  概率分布: MCAR={self.mech_prob['MCAR']:.3f}, "
              f"MAR={self.mech_prob['MAR']:.3f}, "
              f"MNAR={self.mech_prob['MNAR']:.3f}")
        print(f"{'='*70}\n")
        
        # ===== 机制特定参数 =====
        if dominant_mech == 'MCAR':
            params = {
                'mechanism': 'MCAR',
                'k_base': 5,
                'k_high': 7,
                'fusion_weights': [0.5, 0.4, 0.1],
                'imputation_strategy': 'mean/median',
                'confidence': float(confidence)
            }
        elif dominant_mech == 'MAR':
            params = {
                'mechanism': 'MAR',
                'k_base': 7,
                'k_high': 10,
                'fusion_weights': [0.5, 0.4, 0.1],
                'imputation_strategy': 'regression/knn',
                'confidence': float(confidence)
            }
        else:  # MNAR
            params = {
                'mechanism': 'MNAR',
                'k_base': 10,
                'k_high': 15,
                'fusion_weights': [0.5, 0.4, 0.1],
                'imputation_strategy': 'pattern_mixture/selection_model',
                'confidence': float(confidence)
            }
        
        print(f"【自适应参数】")
        print(f"  k_base: {params['k_base']}")
        print(f"  k_high: {params['k_high']}")
        print(f"  fusion_weights: {params['fusion_weights']}")
        print(f"  imputation_strategy: {params['imputation_strategy']}\n")
        
        return params
    
    def get_mechanism_details(self):
        """
        ★返回详细的机制检测结果
        """
        return {
            'global_mechanism': {
                'MCAR': float(self.mech_prob.get('MCAR', 0.0)),
                'MAR': float(self.mech_prob.get('MAR', 0.0)),
                'MNAR': float(self.mech_prob.get('MNAR', 0.0))
            },
            'feature_level_details': self.mechanism_details
        }


# =====================================================================
#                   第6部分：自适应阈值学习
# =====================================================================
# =====================================================================
#         第6部分：自适应阈值学习（改进版）- G-Mean优化
# =====================================================================

class AdaptiveThresholdCalibrator:
    """
    自适应阈值校准 - 使用G-Mean/F2优化（来自v3.0）
    
    改进点：
    1. 支持多个目标指标（G-Mean, F1, F2等）
    2. 分组学习阈值（按缺失等级）
    3. 输出全局最优阈值 + 分组阈值表
    """
    
    def __init__(self, target_metric='gmean'):
        self.target_metric = target_metric
        self.calibration_table = {}
        self.global_optimal_threshold = 0.5
    
    def learn_thresholds(self, y_val, pred_probs, missing_ratios, confidences=None):
        """
        学习自适应阈值表
        
        参数：
            y_val: 验证集真实标签
            pred_probs: 验证集预测概率
            missing_ratios: 验证集缺失率
            confidences: 验证集置信度（可选）
        """
        # 1. 学习全局最优阈值
        self.global_optimal_threshold = self._learn_optimal_threshold(
            y_val, pred_probs, metric=self.target_metric
        )
        print(f"  全局最优阈值 ({self.target_metric}): {self.global_optimal_threshold:.4f}")
        
        # 2. 分组学习阈值（按缺失等级）
        miss_levels = ['low', 'mid', 'high']
        miss_bins = [0, 0.1, 0.3, 1.0]
        
        for i, miss_level in enumerate(miss_levels):
            miss_mask = (missing_ratios >= miss_bins[i]) & (missing_ratios < miss_bins[i+1])
            
            if miss_mask.sum() < 20:
                optimal_th = self.global_optimal_threshold
            else:
                optimal_th = self._learn_optimal_threshold(
                    y_val[miss_mask],
                    pred_probs[miss_mask],
                    metric=self.target_metric
                )
            
            self.calibration_table[miss_level] = optimal_th
            print(f"    [{miss_level}]: threshold={optimal_th:.4f}, n={miss_mask.sum()}")
        
        return self
    
    def _learn_optimal_threshold(self, y_true, y_pred, metric='gmean'):
        """
        学习最优阈值（核心算法）
        
        支持的指标：
        - 'gmean': G-Mean = sqrt(recall_0 * recall_1) ⭐
        - 'f1': F1-Score
        - 'f2': F2-Score（recall权重更高）
        - 'balanced': (recall_0 + recall_1)/2 + accuracy
        """
        thresholds = np.linspace(0.1, 0.9, 17)
        best_th = 0.5
        best_score = -np.inf
        
        for th in thresholds:
            pred_labels = (y_pred >= th).astype(int)
            
            recall_0 = recall_score(y_true, pred_labels, pos_label=0, zero_division=0)
            recall_1 = recall_score(y_true, pred_labels, pos_label=1, zero_division=0)
            
            if metric == 'gmean':
                score = 1 * np.sqrt(recall_0 * recall_1 + 1e-8) + 0 *accuracy_score(y_true, pred_labels) 
                # score = (recall_0 + 2 * recall_1)/3
                # score = fbeta_score(y_true, pred_labels, beta=2, zero_division=0)
                # score =  0.6* np.sqrt(recall_0 * recall_1 + 1e-8) + 0.4*accuracy_score(y_true, pred_labels)
            elif metric == 'balanced':
                score = (recall_0 + recall_1) / 2 + accuracy_score(y_true, pred_labels)
            # elif metric == 'f2':
                # score = fbeta_score(y_true, pred_labels, beta=2, zero_division=0)
            else:  # f1
                score = f1_score(y_true, pred_labels, zero_division=0)
            
            if score > best_score:
                best_score = score
                best_th = th
        
        return best_th
    
    def get_threshold(self, missing_ratio):
        """
        获取自适应阈值
        
        根据缺失率返回相应的阈值
        """
        if missing_ratio < 0.1:
            miss_level = 'low'
        elif missing_ratio < 0.3:
            miss_level = 'mid'
        else:
            miss_level = 'high'
        
        return self.calibration_table.get(miss_level, self.global_optimal_threshold)

def learn_adaptive_thresholds(X_val, y_val, mask_val):
    """
    从验证集学习最优的缺失等级阈值
    """
    threshold_candidates = [
        (0.05, 0.15), (0.1, 0.2), (0.1, 0.3),
        (0.15, 0.35), (0.2, 0.4)
    ]
    
    best_criterion = -np.inf
    best_thresholds = (0.1, 0.3)
    
    for low_th, mid_th in threshold_candidates:
        missing_ratios = mask_val.sum(axis=1) / mask_val.shape[1]
        
        missing_levels = np.array([
            "low" if r < low_th else
            "mid" if r < mid_th else
            "high"
            for r in missing_ratios
        ])
        
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        
        try:
            X_val_filled = X_val.copy()
            X_val_filled[np.isnan(X_val_filled)] = 0
            model.fit(X_val_filled, y_val)
            
            scores = {}
            for level in ["low", "mid", "high"]:
                level_idx = missing_levels == level
                if np.sum(level_idx) > 0:
                    level_pred = model.predict_proba(X_val_filled[level_idx])[:, 1]
                    auc = roc_auc_score(y_val[level_idx], level_pred)
                    scores[level] = auc
            
            std_auc = np.std(list(scores.values()))
            criterion = 1 / (std_auc + 1e-8)
            
            if criterion > best_criterion:
                best_criterion = criterion
                best_thresholds = (low_th, mid_th)
        except:
            pass
    
    print(f"自学习阈值: low={best_thresholds[0]:.2f}, mid={best_thresholds[1]:.2f}")
    return best_thresholds


def learn_adaptive_thresholds_new(X_val, y_val, mask_val, pred_probs=None, confidences=None):
    """
    新的自适应阈值学习函数（改进版）
    
    返回：(low_threshold, mid_threshold) 元组，与原接口兼容
    """
    missing_ratios = mask_val.sum(axis=1) / mask_val.shape[1]
    
    # 如果没有提供预测概率，用默认阈值
    if pred_probs is None:
        pred_probs = np.ones(len(y_val)) * 0.5
    
    if confidences is None:
        confidences = np.ones(len(y_val)) * 0.5
    
    calibrator = AdaptiveThresholdCalibrator(target_metric='gmean')
    calibrator.learn_thresholds(y_val, pred_probs, missing_ratios, confidences)
    
    # ★ 关键：返回元组而不是对象
    return (
        calibrator.calibration_table.get('low', 0.1),
        calibrator.calibration_table.get('mid', 0.3)
    )

def calculate_local_missing_info_adaptive(X_filled, original_mask, high_missing_features,
                                         low_threshold, mid_threshold, adaptive_k_params):
    """
    计算局部缺失信息（支持自适应阈值和K值）
    """
    if not isinstance(X_filled, pd.DataFrame):
        X_filled = pd.DataFrame(X_filled)
    
    sample_missing_ratio = original_mask.sum(axis=1) / original_mask.shape[1]
    
    def get_missing_level(ratio):
        if ratio < low_threshold:
            return "low"
        elif low_threshold <= ratio < mid_threshold:
            return "mid"
        else:
            return "high"
    
    missing_level = np.array([get_missing_level(r) for r in sample_missing_ratio])
    
    high_missing_features_idx = [
        i for i, col in enumerate(X_filled.columns) 
        if col in high_missing_features
    ] if high_missing_features else []
    
    if high_missing_features_idx:
        core_feature_missing = (original_mask[:, high_missing_features_idx].sum(axis=1) > 0).astype(int)
    else:
        core_feature_missing = np.zeros(len(original_mask), dtype=int)
    
    missing_neighbors_k = np.array([
        adaptive_k_params['k_base'] if level == "low" else
        (adaptive_k_params['k_base'] + adaptive_k_params['k_high']) // 2 if level == "mid" 
        else adaptive_k_params['k_high']
        for level in missing_level
    ])
    
    local_missing_info = {
        "sample_missing_ratio": sample_missing_ratio,
        "missing_level": missing_level,
        "core_feature_missing": core_feature_missing,
        "missing_neighbors_k": missing_neighbors_k
    }
    
    return local_missing_info

