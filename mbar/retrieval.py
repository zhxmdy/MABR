# -*- coding: utf-8 -*-
from .config import *

# =====================================================================
#                   第9部分：改进的FAISS索引构建
# =====================================================================

# =====================================================================
#        第9部分增强版：分层检索策略（Stratified Retrieval）
# =====================================================================

class StratifiedRetriever:
    """
    分层检索：保证检索到的邻居中两类样本均衡
    
    改进点：
    1. 分别建立正负样本索引
    2. 自适应K值调整
    3. 多模态融合向量构建
    """
    
    def __init__(self, balance_ratio=0.5, min_per_class=3):
        self.balance_ratio = balance_ratio
        self.min_per_class = min_per_class
        
        self.index_pos = None
        self.index_neg = None
        self.pos_indices = None
        self.neg_indices = None
        self.scaler_X = None
        self.global_weights = None
        self.fusion_weights = None
    
    def build_stratified_index(self, X_val, mask_val, y_val, 
                               global_weights, fusion_weights):
        """
        构建分层索引
        
        参数：
            X_val: 验证集特征 (n_samples, n_features)
            mask_val: 验证集缺失掩码
            y_val: 验证集标签
            global_weights: 全局特征权重
            fusion_weights: 多模态融合权重 [feat_w, mask_w, ratio_w]
        """
        self.global_weights = global_weights
        self.fusion_weights = fusion_weights
        
        # 标准化特征
        self.scaler_X = StandardScaler()
        X_val_scaled = self.scaler_X.fit_transform(X_val)
        
        # 分离正负类
        pos_mask = (y_val == 1)
        neg_mask = (y_val == 0)
        
        self.pos_indices = np.where(pos_mask)[0]
        self.neg_indices = np.where(neg_mask)[0]
        
        print(f"  正类样本: {len(self.pos_indices)}, 负类样本: {len(self.neg_indices)}")
        
        # 构建多模态向量
        V_pos = self._build_multimodal_vectors(
            X_val_scaled[pos_mask], mask_val[pos_mask]
        )
        
        V_neg = self._build_multimodal_vectors(
            X_val_scaled[neg_mask], mask_val[neg_mask]
        )
        
        # 构建两个FAISS索引
        self.index_pos = faiss.IndexFlatL2(V_pos.shape[1])
        self.index_pos.add(V_pos.astype(np.float32))
        
        self.index_neg = faiss.IndexFlatL2(V_neg.shape[1])
        self.index_neg.add(V_neg.astype(np.float32))
        
        print(f"  分层索引构建完成: 正类={self.index_pos.ntotal}, 负类={self.index_neg.ntotal}")
        
        return self
    
    def _build_multimodal_vectors(self, X_scaled, mask):
        """
        构建多模态向量（融合特征+缺失+比率）
        """
        X_weighted = X_scaled * self.global_weights.reshape(1, -1)
        
        # 缺失掩码归一化
        mask_normalized = mask / (mask.sum(axis=1, keepdims=True) + 1e-8)
        mask_normalized = np.nan_to_num(mask_normalized, 0)
        
        # 缺失率
        missing_ratio = mask.sum(axis=1) / mask.shape[1]
        missing_ratio_scaled = (missing_ratio - missing_ratio.min()) / \
                              (missing_ratio.max() - missing_ratio.min() + 1e-8)
        missing_ratio_scaled = missing_ratio_scaled.reshape(-1, 1)
        
        # 融合
        multimodal = np.hstack([
            X_weighted * self.fusion_weights[0],
            mask_normalized * self.fusion_weights[1],
            missing_ratio_scaled * self.fusion_weights[2]
        ])
        
        return multimodal
    
    def retrieve_balanced(self, x, mask, missing_ratio, missing_level, k_total):
        """
        平衡检索（核心方法）
        
        返回：正负样本均衡的邻居集合
        """
        # 构建查询向量
        x_scaled = self.scaler_X.transform(x.reshape(1, -1))
        query_multimodal = self._build_multimodal_vectors(x_scaled, mask.reshape(1, -1))
        
        # 根据缺失等级调整K值
        k_adjust = {'low': 0.8, 'mid': 1.0, 'high': 1.2}
        k_total = int(k_total * k_adjust.get(missing_level, 1.0))
        
        # 计算每类检索数量
        k_pos = max(self.min_per_class, int(k_total * self.balance_ratio))
        k_neg = max(self.min_per_class, k_total - k_pos)
        
        k_pos = min(k_pos, self.index_pos.ntotal)
        k_neg = min(k_neg, self.index_neg.ntotal)
        
        # 分别检索
        query = query_multimodal.astype(np.float32)
        
        dist_pos, idx_pos = self.index_pos.search(query, k_pos)
        dist_neg, idx_neg = self.index_neg.search(query, k_neg)
        
        # 转换回原始索引
        original_idx_pos = self.pos_indices[idx_pos[0]]
        original_idx_neg = self.neg_indices[idx_neg[0]]
        
        # 合并并按距离排序
        combined_indices = np.concatenate([original_idx_pos, original_idx_neg])
        combined_distances = np.concatenate([dist_pos[0], dist_neg[0]])
        
        sort_idx = np.argsort(combined_distances)
        combined_indices = combined_indices[sort_idx]
        combined_distances = combined_distances[sort_idx]
        
        return {
            'indices': combined_indices,
            'distances': combined_distances,
            'k': len(combined_indices)
        }


# 更新索引构建函数以支持分层检索
def build_gl_ma_rag_index_with_stratified_retrieval(X_val, val_mask, y_val, 
                                                    global_missing_stats, 
                                                    local_val_info,
                                                    fusion_weights=None):
    """
    改进的索引构建 - 融合分层检索
    """
    
    if fusion_weights is None:
        fusion_weights = [0.6, 0.3, 0.1]
    
    # 1. 提取全局权重
    feature_cols = list(global_missing_stats["global_weights"].keys())
    global_weights = np.array([global_missing_stats["global_weights"][col] 
                               for col in feature_cols])
    
    # 2. 创建分层检索器
    retriever = StratifiedRetriever(balance_ratio=0.5, min_per_class=3)
    retriever.build_stratified_index(
        X_val, val_mask, y_val,
        global_weights, fusion_weights
    )
    
    print(f"✓ 分层检索索引构建完成")
    
    return retriever, global_weights

def build_gl_ma_rag_index_improved(X_val, val_mask, global_missing_stats, 
                                   local_val_info, multimodal_fusion_net=None,
                                   fusion_weights=None, metric='cosine'):
    """
    改进的GL-MA-RAG索引构建
    """
    
    if isinstance(X_val, pd.DataFrame):
        X_val = X_val.values
    
    if fusion_weights is None:
        fusion_weights = [0.6, 0.3, 0.1]
    
    scaler_X = StandardScaler()
    X_val_scaled = scaler_X.fit_transform(X_val)
    
    # 缺失掩码归一化
    mask_normalized = val_mask / (val_mask.sum(axis=1, keepdims=True) + 1e-8)
    mask_normalized = np.nan_to_num(mask_normalized, 0)
    
    # 缺失率标准化
    missing_ratio = local_val_info["sample_missing_ratio"]
    missing_ratio_scaled = (missing_ratio - missing_ratio.min()) / \
                          (missing_ratio.max() - missing_ratio.min() + 1e-8)
    missing_ratio_scaled = missing_ratio_scaled.reshape(-1, 1)
    
    # 全局权重应用
    feature_cols = list(global_missing_stats["global_weights"].keys())
    global_weights = np.array([global_missing_stats["global_weights"][col] 
                               for col in feature_cols])
    X_val_weighted = X_val_scaled * global_weights.reshape(1, -1)
    
    # 构建多模态向量
    multimodal_val = np.hstack([
        X_val_weighted * fusion_weights[0],
        mask_normalized * fusion_weights[1],
        missing_ratio_scaled * fusion_weights[2]
    ]).astype(np.float32)
    
    # 创建FAISS索引
    if metric == 'euclidean':
        index = faiss.IndexFlatL2(multimodal_val.shape[1])
        index.add(multimodal_val)
    elif metric == 'cosine':
        multimodal_norm = multimodal_val / (
            np.linalg.norm(multimodal_val, axis=1, keepdims=True) + 1e-8
        )
        index = faiss.IndexFlatIP(multimodal_norm.shape[1])
        index.add(multimodal_norm)
    else:
        index = faiss.IndexFlatL2(multimodal_val.shape[1])
        index.add(multimodal_val)
    
    print(f"✓ GL-MA-RAG索引构建完成")
    print(f"  距离度量: {metric}")
    print(f"  融合权重: {fusion_weights}")
    print(f"  多模态维度: {multimodal_val.shape[1]}")
    
    return index, multimodal_val, scaler_X, global_weights
# =====================================================================
#                   第10部分：检索与融合预测
# =====================================================================

def gl_ma_retrieve(query, query_mask, query_ratio, query_level, 
                   index, local_val_info, global_weights, scaler_X,
                   fusion_weights, adaptive_k_params):
    """修复：避免数据转换重复"""
    
    k_dict = {
        "low": adaptive_k_params.get('k_base', 25),
        "mid": (adaptive_k_params.get('k_base', 25) + 
                adaptive_k_params.get('k_high', 50)) // 2,
        "high": adaptive_k_params.get('k_high', 50)
    }
    k = k_dict.get(query_level, 100)
    
    # ★ 修复：确保维度匹配
    query = np.asarray(query, dtype=np.float32).reshape(1, -1)
    query_scaled = scaler_X.transform(query)
    query_weighted = query_scaled * global_weights.reshape(1, -1)
    
    query_mask = np.asarray(query_mask, dtype=np.float32).reshape(1, -1)
    mask_sum = query_mask.sum() + 1e-8
    query_mask_norm = query_mask / mask_sum
    
    query_ratio_norm = np.clip(query_ratio, 0, 1).reshape(1, 1)
    
    query_multimodal = np.hstack([
        query_weighted * fusion_weights[0],
        query_mask_norm * fusion_weights[1],
        query_ratio_norm * fusion_weights[2]
    ]).astype(np.float32)
    
    # FAISS检索
    distances, indices = index.search(query_multimodal, min(k, index.ntotal))
    
    return {
        'indices': indices[0],
        'distances': distances[0],
        'k': k
    }


# =====================================================================
#        第10部分扩展：分层检索预测函数
# =====================================================================

def gl_ma_retrieve_stratified(query, query_mask, query_ratio, query_level,
                              stratified_retriever, local_train_info, 
                              k_base=15):
    """
    使用分层检索器进行预测
    
    参数：
        stratified_retriever: StratifiedRetriever实例
        k_base: 基础K值
    """
    return stratified_retriever.retrieve_balanced(
        query, query_mask, query_ratio, query_level, k_base
    )


