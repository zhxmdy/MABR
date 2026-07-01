# -*- coding: utf-8 -*-
from .config import *

# =====================================================================
#                   第7部分：多模态融合网络（深度学习）
# =====================================================================

class MultimodalFusionNet(nn.Module):
    """
    深度多模态融合网络：替代简单np.hstack
    融合特征、缺失掩码、缺失程度三个模态
    """
    
    def __init__(self, feat_dim, hidden_dim=64, dropout_rate=0.2):
        super().__init__()
        
        # 模态1：特征编码器
        self.feat_encoder = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # 模态2：缺失掩码编码器
        self.mask_encoder = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # 模态3：缺失率编码器（非线性）
        self.ratio_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, hidden_dim),
            nn.BatchNorm1d(hidden_dim)
        )
        
        # 跨模态注意力机制
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            batch_first=True,
            dropout=dropout_rate
        )
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 可学习的模态权重
        self.modality_weights = nn.Parameter(
            torch.ones(3) / 3,
            requires_grad=True
        )
        
        self.hidden_dim = hidden_dim
    
    def forward(self, feat, mask, ratio):
        """
        Input:
            feat: (batch, feat_dim)
            mask: (batch, feat_dim)
            ratio: (batch, 1)
        Output:
            fused: (batch, hidden_dim)
        """
        feat_emb = self.feat_encoder(feat)
        mask_emb = self.mask_encoder(mask)
        ratio_emb = self.ratio_encoder(ratio)
        
        multimodal_seq = torch.stack(
            [feat_emb, mask_emb, ratio_emb], 
            dim=1
        )
        
        attn_output, _ = self.attention(
            multimodal_seq, 
            multimodal_seq, 
            multimodal_seq
        )
        
        weights = torch.softmax(self.modality_weights, dim=0)
        weighted_embs = attn_output * weights.unsqueeze(0).unsqueeze(2)
        
        fused = self.fusion(torch.cat([feat_emb, mask_emb, ratio_emb], dim=1))
        
        return fused

# =====================================================================
#                   第8部分：贝叶斯不确定性量化
# =====================================================================

class BayesianRetrievalModule:
    """
    量化检索的不确定性 - MC-Dropout方法
    """
    
    def __init__(self, model, n_mc_samples=30, device='cuda'):
        self.model = model
        self.n_mc_samples = n_mc_samples
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        for m in model.modules():
            if isinstance(m, nn.Dropout):
                m.train()
    
    def retrieve_with_uncertainty(self, query, query_mask, query_ratio, k=5):
        """
        返回：(indices, distances, uncertainty, confidence)
        """
        query = torch.tensor(query, dtype=torch.float32).unsqueeze(0).to(self.device)
        query_mask = torch.tensor(query_mask, dtype=torch.float32).unsqueeze(0).to(self.device)
        query_ratio = torch.tensor(query_ratio, dtype=torch.float32).reshape(1, 1).to(self.device)
        
        # MC-Dropout采样
        embeddings_samples = []
        for _ in range(self.n_mc_samples):
            with torch.no_grad():
                emb = self.model(query, query_mask, query_ratio)
                embeddings_samples.append(emb.cpu().numpy())
        
        embeddings_samples = np.array(embeddings_samples)
        
        # 统计
        emb_mean = embeddings_samples.mean(axis=0)
        emb_std = embeddings_samples.std(axis=0)
        
        # 不确定性
        uncertainty = emb_std.sum() / emb_mean.shape[-1]
        confidence = 1.0 / (1.0 + uncertainty)
        
        return {
            'embedding': emb_mean[0],
            'uncertainty': uncertainty[0],
            'confidence': float(confidence[0]),
            'std_embedding': emb_std[0]
        }

