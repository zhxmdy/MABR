# -*- coding: utf-8 -*-
from .config import *

class ImbalanceAwareBayesianPredictionResult:
    """贝叶斯预测结果（含类别决策）- 修复版"""
    
    def __init__(self):
        self.predictions = []
        self.pred_labels = []
        self.confidences = []
        self.ci_lowers = []
        self.ci_uppers = []
        self.ci_widths = []
        self.prior_probs = []
        self.posterior_alpha = []
        self.posterior_beta = []
        self.neighbor_weights = []
        self.true_labels = []
        self.missing_ratios = []
        self.missing_levels = []
        self.sample_ids = []

    def add_prediction(self, pred_result, true_label=None, missing_ratio=None,
                       missing_level=None, sample_id=None):
        """添加单个预测结果"""
        self.predictions.append(pred_result["prediction"])
        self.pred_labels.append(pred_result["pred_label"])
        self.confidences.append(pred_result["confidence"])
        self.ci_lowers.append(pred_result["ci_lower"])
        self.ci_uppers.append(pred_result["ci_upper"])
        self.ci_widths.append(pred_result["ci_width"])
        self.prior_probs.append(pred_result["prior_prob"])
        self.posterior_alpha.append(pred_result["posterior_alpha"])
        self.posterior_beta.append(pred_result["posterior_beta"])
        self.neighbor_weights.append(pred_result["neighbor_weights"])
        self.true_labels.append(true_label)
        self.missing_ratios.append(missing_ratio)
        self.missing_levels.append(missing_level)
        self.sample_ids.append(sample_id)

    # ★ 新增方法1：支持 len() 函数
    def __len__(self):
        """返回预测结果的数量"""
        return len(self.predictions)
    
    # ★ 新增方法2：支持布尔判断
    def __bool__(self):
        """支持 if bayes_results: 的判断"""
        return len(self.predictions) > 0
    
    # ★ 新增方法3：支持索引访问
    def __getitem__(self, idx):
        """支持索引访问"""
        return {
            'prediction': self.predictions[idx],
            'pred_label': self.pred_labels[idx],
            'confidence': self.confidences[idx],
            'ci_lower': self.ci_lowers[idx],
            'ci_upper': self.ci_uppers[idx],
            'ci_width': self.ci_widths[idx],
            'prior_prob': self.prior_probs[idx],
            'posterior_alpha': self.posterior_alpha[idx],
            'posterior_beta': self.posterior_beta[idx],
            'true_label': self.true_labels[idx],
            'missing_ratio': self.missing_ratios[idx],
            'missing_level': self.missing_levels[idx],
            'sample_id': self.sample_ids[idx]
        }

    def to_dataframe(self):
        """转换为DataFrame便于分析"""
        return pd.DataFrame({
            "prediction": self.predictions,
            "pred_label": self.pred_labels,
            "confidence": self.confidences,
            "ci_lower": self.ci_lowers,
            "ci_upper": self.ci_uppers,
            "ci_width": self.ci_widths,
            "prior_prob": self.prior_probs,
            "posterior_alpha": self.posterior_alpha,
            "posterior_beta": self.posterior_beta,
            "true_label": self.true_labels,
            "missing_ratio": self.missing_ratios,
            "missing_level": self.missing_levels,
            "sample_id": self.sample_ids
        })

    def __repr__(self):
        """字符串表示"""
        return f"ImbalanceAwareBayesianPredictionResult(n_samples={len(self.predictions)})"


class MissingnessAwareBayesianRiskPredictor:
    """
    ... (保留原有注释)
    
    ★ v3.0增强：
    - 动态代价调整（自适应FP/FN代价）
    - 缺失感知的先验强度衰减
    - Beta后验置信区间
    """
    
    def __init__(self,
                 kappa0=5.0,
                 lambda_miss=5.0,
                 tau=0.5,
                 conf_scale=12.0,
                 min_kappa=2.0,
                 fp_cost=1.0,          # ★ 新增
                 fn_cost=5.0,          # ★ 新增（风控应用中FN代价更高）
                 use_adaptive_cost=True,  # ★ 新增
                 use_mask=True,
                 use_missing_ratio=True):
        self.kappa0 = kappa0
        self.lambda_miss = lambda_miss
        self.tau = tau
        self.conf_scale = conf_scale
        self.min_kappa = min_kappa
        
        # ★ 代价敏感参数
        self.fp_cost = fp_cost
        self.fn_cost = fn_cost
        self.use_adaptive_cost = use_adaptive_cost
        self.decision_threshold = fp_cost / (fp_cost + fn_cost)
        
        self.use_mask = use_mask
        self.use_missing_ratio = use_missing_ratio
        self.prior_model = None
    def _compose_prior_features(self, X, mask=None):
        X = np.asarray(X, dtype=np.float32)
        X = np.nan_to_num(X, nan=0.0)
        feats = [X]

        if self.use_mask and mask is not None:
            feats.append(np.asarray(mask, dtype=np.float32))

        if self.use_missing_ratio and mask is not None:
            miss_ratio = mask.sum(axis=1, keepdims=True) / (mask.shape[1] + 1e-8)
            feats.append(miss_ratio.astype(np.float32))

        return np.concatenate(feats, axis=1)

    def fit_prior_model(self, X_train, y_train, mask_train=None, prior_model_type="logr", seed=42):
        """
        训练全局先验模型，并做概率校准
        """
        Xp = self._compose_prior_features(X_train, mask_train)

        pos = np.sum(y_train)
        neg = len(y_train) - pos
        scale_pos_weight = neg / (pos + 1e-8)
        pos = np.sum(y_train)
        neg = len(y_train) - pos
        pos_weight = (len(y_train) - y_train.sum()) / y_train.sum() if y_train.sum() > 0 else 1.0
        class_priors = np.array([neg / len(y_train), pos / len(y_train)], dtype=np.float32)
        if prior_model_type.lower() == "logr":
          
          
          
        #     base_model = 
        #   base_model = lgb.LGBMClassifier(
        #     n_estimators=100,
        #     max_depth=4,            # 浅树，风控核心
        #     # learning_rate=0.03,
        #     # subsample=0.7,
        #     # colsample_bytree=0.7,
        #     class_weight="balanced", # 不平衡样本
        #     objective='binary',
        #     metric='auc',
        #     use_missing=True,       # 处理缺失
        #     # min_child_samples=30,   # 叶子最小样本数，增大更稳定
        #     # reg_alpha=1,
        #     # reg_lambda=3,
        #     random_state=seed,
        #     # feature_name=X_train_woe.columns.tolist(),
        #     verbose=-1
        # )
            # base_model =xgb.XGBClassifier(
            #             n_estimators=100,        # 风控适当增加树数量
            #             max_depth=4,            # 风控必须浅！防止过拟合（3~5最佳）
            #             # learning_rate=0.03,     # 稍大一点，配合早停
            #             # subsample=0.7,          # 行采样，增强泛化
            #             # colsample_bytree=0.7,   # 列采样，风控必加
            #             scale_pos_weight=pos_weight,  # 处理样本不平衡
            #             missing=np.nan,         # 容忍缺失值
            #             eval_metric='auc',
            #             random_state=seed,
            #             use_label_encoder=False,
            #             # reg_alpha=1,             # L1 正则，风控防过拟合
            #             # reg_lambda=3,            # L2 正则，风控防过拟合
            #             # min_child_weight=5,      # 叶子节点最小权重，风控必加
            #             verbose=0
            #         )

            base_model =cb.CatBoostClassifier(
            n_estimators=100,
            max_depth=5,            # 浅树
            # learning_rate=0.1,
            scale_pos_weight=pos_weight,
            # eval_metric='AUC',
            random_state=seed,
            # l2_leaf_reg=5,          # 正则化
            # min_data_in_leaf=20,    # 叶子最小样本
            verbose=0,
            allow_writing_files=False  # 不生成中间文件，提速
        )
        #     base_model =  GaussianNB(
        #     priors=class_priors,
        #     var_smoothing=1e-8
        # )

        # ),
            # base_model =  xgb.XGBClassifier(
            #     n_estimators=100,        # 风控适当增加树数量
            #     max_depth=4,            # 风控必须浅！防止过拟合（3~5最佳）
            #     learning_rate=0.03,     # 稍大一点，配合早停
            #     subsample=0.7,          # 行采样，增强泛化
            #     colsample_bytree=0.7,   # 列采样，风控必加
            #     scale_pos_weight=pos_weight,  # 处理样本不平衡
            #     missing=np.nan,         # 容忍缺失值
            #     eval_metric='auc',
            #     random_state=seed,
            #     use_label_encoder=False,
            #     reg_alpha=1,             # L1 正则，风控防过拟合
            #     reg_lambda=3,            # L2 正则，风控防过拟合
            #     min_child_weight=5,      # 叶子节点最小权重，风控必加
            #     verbose=0
            # )
            # base_model =  LogisticRegression(
            #     class_weight="balanced",
            #     max_iter=3000,
            #     random_state=seed
            # )
        else:
             base_model = cb.CatBoostClassifier(
            n_estimators=100,
            max_depth=5,            # 浅树
            # learning_rate=0.1,
            scale_pos_weight=pos_weight,
            # eval_metric='AUC',
            random_state=seed,
            # l2_leaf_reg=5,          # 正则化
            # min_data_in_leaf=20,    # 叶子最小样本
            verbose=0,
            allow_writing_files=False  # 不生成中间文件，提速
        )
        #     base_model = xgb.XGBClassifier(
        #     n_estimators=100,        # 风控适当增加树数量
        #     max_depth=4,            # 风控必须浅！防止过拟合（3~5最佳）
        #     learning_rate=0.03,     # 稍大一点，配合早停
        #     subsample=0.7,          # 行采样，增强泛化
        #     colsample_bytree=0.7,   # 列采样，风控必加
        #     scale_pos_weight=pos_weight,  # 处理样本不平衡
        #     missing=np.nan,         # 容忍缺失值
        #     eval_metric='auc',
        #     random_state=seed,
        #     use_label_encoder=False,
        #     reg_alpha=1,             # L1 正则，风控防过拟合
        #     reg_lambda=3,            # L2 正则，风控防过拟合
        #     min_child_weight=5,      # 叶子节点最小权重，风控必加
        #     verbose=0
        # )


        # try:
        #     self.prior_model = CalibratedClassifierCV(
        #         estimator=base_model,
        #         method="sigmoid",
        #         cv=3
        #     )
        # except TypeError:
        #     self.prior_model = CalibratedClassifierCV(
        #         base_estimator=base_model,
        #         method="sigmoid",
        #         cv=3
        #     )
        self.prior_model = base_model
        self.prior_model.fit(Xp, y_train)
        return self

    def prior_prob(self, x, mask=None):
        x = np.asarray(x, dtype=np.float32).reshape(1, -1)
        x = np.nan_to_num(x, nan=0.0)

        if self.use_mask and mask is not None:
            mask = np.asarray(mask, dtype=np.float32).reshape(1, -1)
        else:
            mask = None

        xp = self._compose_prior_features(x, mask)
        p0 = self.prior_model.predict_proba(xp)[:, 1][0]
        return float(np.clip(p0, 1e-6, 1 - 1e-6))

    def _neighbor_weights(self, distances, query_missing_ratio=0.0, neighbor_missing_ratios=None):
        distances = np.asarray(distances, dtype=np.float32)
        w = np.exp(-distances / (self.tau + 1e-8))

        if neighbor_missing_ratios is not None:
            neighbor_missing_ratios = np.asarray(neighbor_missing_ratios, dtype=np.float32)
            w *= np.exp(-0.5 * np.abs(neighbor_missing_ratios - query_missing_ratio))

        w = w / (w.sum() + 1e-8)
        return w    
    def compute_adaptive_cost(self, mask, missing_ratio):
        """
        ★ 新增方法：动态代价调整
        
        缺失越多 → FP/FN代价越高（不确定性增加）
        """
        if not self.use_adaptive_cost:
            return self.fp_cost, self.fn_cost
        
        miss_factor = 1 + missing_ratio
        
        # 高风险特征缺失则风险更高
        high_risk_missing = mask[:5].sum() / 5 if len(mask) >= 5 else 0
        risk_factor = 1 + 0.5 * high_risk_missing
        
        fp_cost = self.fp_cost * miss_factor
        fn_cost = self.fn_cost * miss_factor * risk_factor
        
        return fp_cost, fn_cost
    
    # 代码位置：第1部分，MissingnessAwareBayesianRiskPredictor 类中

    def predict_single(self, x, mask, retrieve_result, y_ref, ref_missing_ratios,
                    X_ref=None):  # ← 只加这一个参数
        """改进版：充分利用邻居特征"""
        
        x = np.asarray(x, dtype=np.float32)
        mask = np.asarray(mask, dtype=np.float32)
        
        query_missing_ratio = float(mask.mean())
        prior_p = self.prior_prob(x, mask)
        
        idx = np.asarray(retrieve_result["indices"], dtype=int)
        distances = np.asarray(retrieve_result["distances"], dtype=np.float32)
        
        y_local = np.asarray(y_ref, dtype=np.float32)[idx]
        neighbor_missing_ratios = np.asarray(ref_missing_ratios, dtype=np.float32)[idx]
        
        # ========== 原有权重计算 ==========
        weights = self._neighbor_weights(
            distances,
            query_missing_ratio=query_missing_ratio,
            neighbor_missing_ratios=neighbor_missing_ratios
        )
        
        # ========== NEW：加入邻居特征权重 ==========
        if X_ref is not None:
            X_ref = np.asarray(X_ref, dtype=np.float32)
            X_neighbors = X_ref[idx]
            x_query = x.reshape(1, -1)
            
            # 计算特征距离
            feature_dist = np.sqrt(np.sum((X_neighbors - x_query) ** 2, axis=1))
            
            # 转为权重（用相同的τ）
            w_feat = np.exp(-feature_dist / (self.tau + 1e-8))
            w_feat = w_feat / (w_feat.sum() + 1e-8)
            
            neighbor_missing_ratios = np.asarray(ref_missing_ratios, dtype=np.float32)[idx]
            w = np.exp(-0.5 * np.abs(neighbor_missing_ratios - query_missing_ratio))
            # 融合权重：简单相乘后重新归一化
            weights =  w * w_feat
            weights = weights / (weights.sum() + 1e-8)
        
        # ========== 后续步骤完全不变 ==========
        cost_weight_pos = self.fn_cost / (self.fp_cost + self.fn_cost)
        cost_weight_neg = self.fp_cost / (self.fp_cost + self.fn_cost)
        
        local_pos = float(np.sum(weights * y_local * cost_weight_pos))
        local_neg = float(np.sum(weights * (1.0 - y_local) * cost_weight_neg))
        
        kappa_x = self.kappa0 / (1.0 + self.lambda_miss * query_missing_ratio)
        kappa_x = max(self.min_kappa, float(kappa_x))
        '''此处可以修改为是否采用哪种集成的方式，目前来说是先验结合近邻的方式，我们也可以仅仅是先验，也可以仅仅是近邻，也可以进行加权融合'''
        # w_f = np.exp(-distances / self.tau)
        # neighbor_missing_ratios = np.asarray(ref_missing_ratios, dtype=np.float32)[idx]
        # ww = np.exp(-0.5 * np.abs(neighbor_missing_ratios - query_missing_ratio))
        # wwe =w_f*ww
        # wwe = wwe / (wwe.sum() + 1e-8)
        # neighbor_consensu = np.sum(wwe * y_local)  # 加权平均
        # prior_p = self.prior_prob(x, mask)  # e.g., 0.65
        # # # 邻居共识先验
        # neighbor_prior = neighbor_consensu  # e.g., 0.57 
        # # # 融合先验（加权平均）
        # # # 给予邻居一定的信心，但不完全替代模型先验
        # alpha_fuse = 0  # 邻居权重系数
        # fused_prior = (1 - alpha_fuse) * prior_p + alpha_fuse * neighbor_prior
        # # ========== 步骤4：使用融合先验 ==========        
        # prior_p = fused_prior


        alpha0 = kappa_x * prior_p
        beta0 = kappa_x * (1.0 - prior_p)

        alpha_post = max(alpha0 + local_pos, 1e-6)
        beta_post = max(beta0 + local_neg, 1e-6)





        pred = alpha_post / (alpha_post + beta_post)
        
        ci_lower, ci_upper = beta_dist.ppf([0.025, 0.975], alpha_post, beta_post)
        ci_lower = float(np.clip(ci_lower, 0.0, 1.0))
        ci_upper = float(np.clip(ci_upper, 0.0, 1.0))
        ci_width = float(ci_upper - ci_lower)
        
        post_var = (alpha_post * beta_post) / (
            (alpha_post + beta_post) ** 2 * (alpha_post + beta_post + 1.0)
        )
        confidence = float(np.clip(
            1.0 / (1.0 + self.conf_scale * post_var), 0.0, 1.0
        ))
        
        return {
            "prediction": float(pred),
            "pred_label": int(pred >= self.decision_threshold),
            "confidence": confidence,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "ci_width": ci_width,
            "prior_prob": float(prior_p),
            "posterior_alpha": float(alpha_post),
            "posterior_beta": float(beta_post),
            "neighbor_weights": weights
        }


    # def predict_single(self, x, mask, retrieve_result, y_ref, ref_missing_ratios=None):
    #     """
    #     修改predict_single方法以支持代价敏感阈值
    #     """
    #     x = np.asarray(x, dtype=np.float32)
    #     mask = np.asarray(mask, dtype=np.float32)
        
    #     query_missing_ratio = float(mask.mean())
    #     prior_p = self.prior_prob(x, mask)
        
    #     # ★ 计算动态代价
    #     fp_cost, fn_cost = self.compute_adaptive_cost(mask, query_missing_ratio)
    #     cost_threshold = fp_cost / (fp_cost + fn_cost)
        
    #     # ... 其余代码保持不变 ...
        
    #     idx = np.asarray(retrieve_result["indices"], dtype=int)
    #     distances = np.asarray(retrieve_result["distances"], dtype=np.float32)
        
    #     if ref_missing_ratios is not None:
    #         neighbor_missing_ratios = np.asarray(ref_missing_ratios, dtype=np.float32)[idx]
    #     else:
    #         neighbor_missing_ratios = None
        
    #     weights = self._neighbor_weights(
    #         distances,
    #         query_missing_ratio=query_missing_ratio,
    #         neighbor_missing_ratios=neighbor_missing_ratios
    #     )
        
    #     y_local = np.asarray(y_ref, dtype=np.float32)[idx]
        
    #     # ★ 代价加权的证据
    #     cost_weight_pos = fn_cost / (fp_cost + fn_cost)
    #     cost_weight_neg = fp_cost / (fp_cost + fn_cost)
        
    #     local_pos = float(np.sum(weights * y_local * cost_weight_pos))
    #     local_neg = float(np.sum(weights * (1.0 - y_local) * cost_weight_neg))
        
    #     # Beta后验
    #     kappa_x = self.kappa0 / (1.0 + self.lambda_miss * query_missing_ratio)
    #     kappa_x = max(self.min_kappa, float(kappa_x))
        
    #     alpha0 = kappa_x * prior_p
    #     beta0 = kappa_x * (1.0 - prior_p)
        
    #     alpha_post = max(alpha0 + local_pos, 1e-6)
    #     beta_post = max(beta0 + local_neg, 1e-6)
        
    #     pred = alpha_post / (alpha_post + beta_post)
        
    #     # 置信区间
    #     ci_lower, ci_upper = beta_dist.ppf([0.025, 0.975], alpha_post, beta_post)
    #     ci_lower = float(np.clip(ci_lower, 0.0, 1.0))
    #     ci_upper = float(np.clip(ci_upper, 0.0, 1.0))
    #     ci_width = float(ci_upper - ci_lower)
        
    #     post_var = (alpha_post * beta_post) / (
    #         (alpha_post + beta_post) ** 2 * (alpha_post + beta_post + 1.0)
    #     )
    #     confidence = float(np.clip(1.0 / (1.0 + self.conf_scale * post_var), 0.0, 1.0))
        
    #     # ★ 使用代价敏感阈值决策
    #     pred_label = int(pred >= cost_threshold)
        
    #     return {
    #         "prediction": float(pred),
    #         "pred_label": pred_label,
    #         "confidence": confidence,
    #         "ci_lower": ci_lower,
    #         "ci_upper": ci_upper,
    #         "ci_width": ci_width,
    #         "prior_prob": float(prior_p),
    #         "cost_threshold": cost_threshold,
    #         "posterior_alpha": float(alpha_post),
    #         "posterior_beta": float(beta_post),
    #         "neighbor_weights": weights
    #     }
def bayesian_predict_with_retrieval(x, mask, retrieve_result,
                                    bayes_model, y_ref, ref_missing_ratios):
    return bayes_model.predict_single(
        x=x,
        mask=mask,
        retrieve_result=retrieve_result,
        y_ref=y_ref,
        ref_missing_ratios=ref_missing_ratios
    )


# =====================================================================
#                   第11部分：PyTorch自定义模块
# =====================================================================

class TabularDataset(Dataset):
    def __init__(self, X, y=None, missing_value=np.nan):
        # ★ 修复：正确处理缺失值
        X_array = np.asarray(X, dtype=np.float32)
        
        # 记录原始缺失位置
        self.mask = np.isnan(X_array).astype(np.float32)
        
        # 用0填充缺失值（用于输入）
        self.X = np.nan_to_num(X_array, nan=0.0)
        
        self.y = y if y is not None else np.zeros(len(X))
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.mask = torch.tensor(self.mask, dtype=torch.float32)
        self.y = torch.tensor(np.array(self.y), dtype=torch.float32).unsqueeze(1)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.mask[idx], self.y[idx]

# =====================================================================
#                   第12部分：可视化系统
# =====================================================================

