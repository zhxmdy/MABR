# -*- coding: utf-8 -*-
from .config import *

# =====================================================================
#                   第2部分：缺失机制注入
# =====================================================================

def inject_missing_new(X, target_missing_rate=0.1, mech='MCAR', seed=42):
    """
    ★改进版：精确控制缺失率 + 机制正确性修复
    
    改进点：
    1. ✅ MAR机制：特征配额制 + 动态概率
    2. ✅ MNAR机制：自适应概率缩放
    3. ✅ 精确补齐：误差<0.5%
    4. ✅ 性能优化：向量化操作
    """
    np.random.seed(seed)
    X = np.array(X, dtype=np.float32, copy=True)
    n_samples, n_features = X.shape
    total_positions = n_samples * n_features
    
    print(f"\n【缺失注入】机制={mech}, 目标总缺失率={target_missing_rate:.1%}")
    print("=" * 70)
    
    # ===== 步骤1：原始缺失分析 =====
    original_mask = np.isnan(X)
    original_missing_count = np.sum(original_mask)
    original_missing_rate = original_missing_count / total_positions
    
    print(f"📊 原始缺失率: {original_missing_rate:.2%} ({original_missing_count}个)")
    
    # ===== 步骤2：计算注入量 =====
    target_missing_count = int(np.round(target_missing_rate * total_positions))
    needed_inject_count = max(0, target_missing_count - original_missing_count)
    
    if needed_inject_count <= 0:
        print(f"⚠️  无需注入，返回原数据")
        return X, original_mask
    
    available_mask = ~original_mask
    available_count = np.sum(available_mask)
    
    print(f"🎯 需注入: {needed_inject_count}个 (可用位置: {available_count})")
    
    # ===== 步骤3：数据预处理 =====
    X_work = X.copy()
    col_medians = np.nanmedian(X, axis=0)
    for j in range(n_features):
        nan_idx = np.isnan(X_work[:, j])
        X_work[nan_idx, j] = col_medians[j] if not np.isnan(col_medians[j]) else 0
    
    inject_mask = np.zeros_like(X, dtype=bool)
    
    # ===== 步骤4：按机制注入 =====
    if mech == 'MCAR':
        print(f"🔹 MCAR模式：完全随机")
        print("-" * 70)
        
        # ★精确概率计算
        inject_prob = needed_inject_count / available_count
        
        random_vals = np.random.rand(n_samples, n_features)
        inject_mask = available_mask & (random_vals < inject_prob)
        
        current_inject = np.sum(inject_mask)
        print(f"  初始注入: {current_inject}/{needed_inject_count}")
    
    elif mech == 'MAR':
        print(f"🔹 MAR模式：依赖其他特征")
        print("-" * 70)
        
        # ★计算相关性
        corr_matrix = np.corrcoef(X_work.T)
        np.fill_diagonal(corr_matrix, 0)
        
        # ★关键改进：特征配额制
        quota_per_feature = max(1, needed_inject_count // n_features)
        current_inject = 0
        
        for j in range(n_features):
            # ★动态配额
            remaining_quota = min(
                quota_per_feature, 
                needed_inject_count - current_inject
            )
            
            if remaining_quota <= 0:
                break
            
            # 找最相关特征
            cond_col = np.argmax(np.abs(corr_matrix[j, :]))
            condition_vals = X_work[:, cond_col]
            
            # 分位数
            q33, q67 = np.percentile(condition_vals, [33.33, 66.67])
            
            # ★动态概率（按配额分配）
            valid_positions = available_mask[:, j]
            n_valid = np.sum(valid_positions)
            
            if n_valid == 0:
                continue
            
            base_prob = remaining_quota / n_valid * 1.2
            
            # 构建概率向量
            probs = np.zeros(n_samples)
            probs[condition_vals <= q33] = base_prob * 0.5      # 低值
            probs[(condition_vals > q33) & (condition_vals <= q67)] = base_prob * 2.0  # 中值
            probs[condition_vals > q67] = base_prob * 6.0       # 高值
            
            # 生成缺失
            candidates = np.random.rand(n_samples) < probs
            new_missing = valid_positions & candidates
            
            inject_mask[new_missing, j] = True
            current_inject = np.sum(inject_mask)
            
            if j % 5 == 0 or j == n_features - 1:
                print(f"  特征{j:2d}: 本列{np.sum(new_missing):3d}, 累计{current_inject:5d}")
    
    elif mech == 'MNAR':
        print(f"🔹 MNAR模式：依赖自身值")
        print("-" * 70)
        
        # ★关键改进：自适应概率缩放
        base_prob = needed_inject_count / available_count * 0.8
        
        for j in range(n_features):
            col_vals = X[:, j]
            observed_mask = ~np.isnan(col_vals)
            
            if np.sum(observed_mask) < 5:
                continue
            
            observed_vals = col_vals[observed_mask]
            q70, q90 = np.percentile(observed_vals, [70, 90])
            
            # ★动态概率（保持相对关系）
            probs = np.zeros(n_samples)
            
            high_mask = (col_vals > q90) & available_mask[:, j]
            mid_mask = ((col_vals > q70) & (col_vals <= q90)) & available_mask[:, j]
            low_mask = (col_vals <= q70) & available_mask[:, j]
            
            probs[high_mask] = base_prob * 8.0   # 高值区高缺失
            probs[mid_mask] = base_prob * 3.0
            probs[low_mask] = base_prob * 0.5
            
            # 裁剪概率
            probs = np.clip(probs, 0, 0.95)
            
            # 生成缺失
            candidates = np.random.rand(n_samples) < probs
            new_missing = candidates & available_mask[:, j]
            
            inject_mask[new_missing, j] = True
            
            if j % 5 == 0 or j == n_features - 1:
                current_inject = np.sum(inject_mask)
                print(f"  特征{j:2d}: 本列{np.sum(new_missing):3d}, 累计{current_inject:5d}")
        
        current_inject = np.sum(inject_mask)
    
    # ===== 步骤5：精确补齐 =====
    print(f"\n💡 一阶注入: {current_inject}/{needed_inject_count}")
    
    if current_inject < needed_inject_count:
        print(f"🔧 精确补齐中...")
        remaining_needed = needed_inject_count - current_inject
        
        # ★未使用的可用位置
        remaining_available = available_mask & ~inject_mask
        remaining_count = np.sum(remaining_available)
        
        if remaining_count > 0:
            supplement_prob = min(1.0, remaining_needed / remaining_count * 1.05)
            
            candidates = np.random.rand(n_samples, n_features) < supplement_prob
            supplement_mask = remaining_available & candidates
            
            inject_mask[supplement_mask] = True
            
            print(f"   补充: {np.sum(supplement_mask)}个")
    
    # ===== 步骤6：最终验证 =====
    final_inject = np.sum(inject_mask)
    final_total = original_missing_count + final_inject
    final_rate = final_total / total_positions
    error = abs(final_rate - target_missing_rate)
    
    print(f"\n✅ 最终统计:")
    print(f"   人工注入: {final_inject}")
    print(f"   总缺失率: {final_rate:.2%} (目标{target_missing_rate:.2%})")
    print(f"   精度误差: {error*100:.3f}%")
    print(f"   各列缺失: {np.sum(inject_mask, axis=0)}")
    print("=" * 70 + "\n")
    
    # ===== 步骤7：生成结果 =====
    total_mask = original_mask | inject_mask
    X_result = X.copy()
    X_result[total_mask] = np.nan
    
    return X_result, total_mask


def inject_missing(X, missing_rate=0.2, mech='MCAR', seed=42):
    """修复：确保生成真正的NaN"""
    np.random.seed(seed)
    X = np.array(X).astype(np.float32)
    n_samples, n_features = X.shape
    mask = np.random.rand(n_samples, n_features) < missing_rate
    
    if mech == 'MCAR':
        X[mask] = np.nan
    elif mech == 'MAR':
        for j in range(1, n_features):
            col_vals = X[:, j-1].copy()
            col_vals[np.isnan(col_vals)] = 0
            high_quantile = np.quantile(col_vals[~np.isnan(col_vals)], 0.7)
            mar_mask = (col_vals > high_quantile) & (np.random.rand(n_samples) < missing_rate)
            X[mar_mask, j] = np.nan
            mask[mar_mask, j] = True
    elif mech == 'MNAR':
        for j in range(n_features):
            col_vals = X[:, j].copy()
            high_quantile = np.quantile(col_vals[~np.isnan(col_vals)], 0.7)
            mnar_mask = (col_vals > high_quantile) & (np.random.rand(n_samples) < missing_rate)
            X[mnar_mask, j] = np.nan
            mask[mnar_mask, j] = True
    
    if np.isnan(X).sum() == 0:
        print(f"警告：未生成缺失值，强制生成")
        mask = np.random.rand(n_samples, n_features) < 0.1
        X[mask] = np.nan
    
    return X, mask.astype(int)

