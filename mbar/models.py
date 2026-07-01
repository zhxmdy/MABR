# -*- coding: utf-8 -*-
from .config import *

class SimpleRegressionModel:
    """
    先验模型：用原始数据和填补数据训练的逻辑回归模型
    """
    
    def __init__(self):
        # self.model =   xgb.XGBClassifier(
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
        self.model = cb.CatBoostClassifier(
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
        # LogisticRegression(max_iter=1000, random_state=42)
        self.scaler = StandardScaler()
    
    def fit(self, X_train, y_train):
        """拟合模型"""
        print("    正在拟合先验模型...")
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled, y_train)
        print(f"    ✓ 先验模型已拟合")
        return self
    
    def predict_proba(self, X_test):
        """预测概率"""
        X_scaled = self.scaler.transform(X_test)
        proba = self.model.predict_proba(X_scaled)
        return proba[:, 1]  # 返回正类概率
    
    def predict(self, X_test):
        """预测标签"""
        X_scaled = self.scaler.transform(X_test)
        return self.model.predict(X_scaled)
    
    def score(self, X_test, y_test):
        """计算准确率"""
        y_pred = self.predict(X_test)
        return np.mean(y_pred == y_test)


class BayesianMissingDataModel:
    """
    贝叶斯缺失数据模型：
    在先验模型基础上，结合缺失掩码信息进行贝叶斯更新
    """
    
    def __init__(self, prior_model):
        self.prior_model = prior_model
        self.mask_impact = None
        self.feature_impact = None
        self.missing_rate_bins = None
        self.bad_rate_by_missing = None
    
    def fit(self, X_train, y_train, mask_train):
        """
        训练贝叶斯模型：学习缺失掩码与标签的关联
        """
        
        print("    正在训练贝叶斯缺失模型...")
        
        # 转换数据
        if isinstance(X_train, pd.DataFrame):
            X_train_array = X_train.values
        else:
            X_train_array = X_train
        
        if isinstance(mask_train, pd.DataFrame):
            mask_train_array = mask_train.values
        else:
            mask_train_array = mask_train
        
        if isinstance(y_train, pd.Series):
            y_train_array = y_train.values
        else:
            y_train_array = y_train
        
        n_features = X_train_array.shape[1]
        
        # ============ 计算特征缺失的影响 ============
        self.mask_impact = np.zeros(n_features)
        
        for i in range(n_features):
            feature_missing = mask_train_array[:, i].astype(int)
            
            if feature_missing.sum() > 0:
                try:
                    correlation, _ = pointbiserialr(feature_missing, y_train_array)
                    self.mask_impact[i] = np.nan_to_num(correlation, nan=0.0)
                except:
                    bad_rate_missing = y_train_array[feature_missing == 1].mean()
                    bad_rate_not_missing = y_train_array[feature_missing == 0].mean()
                    self.mask_impact[i] = bad_rate_missing - bad_rate_not_missing
        
        # ============ 计算特征重要性 ============
        self.feature_impact = np.zeros(n_features)
        for i in range(n_features):
            try:
                corr = np.abs(np.corrcoef(X_train_array[:, i], y_train_array)[0, 1])
                self.feature_impact[i] = np.nan_to_num(corr, nan=0.0)
            except:
                self.feature_impact[i] = 0.0
        
        # 标准化
        if self.feature_impact.sum() > 0:
            self.feature_impact = self.feature_impact / self.feature_impact.sum()
        else:
            self.feature_impact = np.ones(n_features) / n_features
        
        # ============ 学习缺失率与坏样本的关系 ============
        mask_train_rate = mask_train_array.sum(axis=1) / mask_train_array.shape[1]
        
        self.missing_rate_bins = np.linspace(0, 1, 11)
        self.bad_rate_by_missing = np.zeros(10)
        
        for i in range(10):
            mask = (mask_train_rate >= self.missing_rate_bins[i]) & \
                   (mask_train_rate < self.missing_rate_bins[i+1])
            if mask.sum() > 0:
                self.bad_rate_by_missing[i] = y_train_array[mask].mean()
            else:
                self.bad_rate_by_missing[i] = y_train_array.mean()
        
        print(f"    ✓ 贝叶斯模型已拟合")
        
        return self
    
    def predict_proba(self, X_test, mask_test):
        """
        贝叶斯更新：结合先验和缺失掩码信息
        """
        
        # 获取先验概率
        prior_proba = self.prior_model.predict_proba(X_test)
        
        # 转换数据
        if isinstance(X_test, pd.DataFrame):
            X_test_array = X_test.values
        else:
            X_test_array = X_test
        
        if isinstance(mask_test, pd.DataFrame):
            mask_test_array = mask_test.values
        else:
            mask_test_array = mask_test
        
        # ============ 贝叶斯更新 ============
        
        # 方法1: 基于特征级缺失影响
        feature_level_adjustment = np.zeros(len(X_test_array))
        for i in range(len(X_test_array)):
            sample_mask = mask_test_array[i]
            adjustment = (sample_mask * self.mask_impact * self.feature_impact).sum()
            feature_level_adjustment[i] = adjustment
        
        # 标准化
        if feature_level_adjustment.max() - feature_level_adjustment.min() > 0:
            feature_level_adjustment = (feature_level_adjustment - feature_level_adjustment.min()) / \
                                       (feature_level_adjustment.max() - feature_level_adjustment.min())
        
        # 方法2: 基于总体缺失率
        missing_rate = mask_test_array.sum(axis=1) / mask_test_array.shape[1]
        
        missing_rate_adjustment = np.zeros(len(missing_rate))
        for i, rate in enumerate(missing_rate):
            bin_idx = np.digitize(rate, self.missing_rate_bins) - 1
            bin_idx = np.clip(bin_idx, 0, 9)
            missing_rate_adjustment[i] = self.bad_rate_by_missing[bin_idx]
        
        # 标准化
        if missing_rate_adjustment.std() > 0:
            missing_rate_adjustment = (missing_rate_adjustment - missing_rate_adjustment.mean()) / \
                                      missing_rate_adjustment.std()
        
        if missing_rate_adjustment.max() - missing_rate_adjustment.min() > 0:
            missing_rate_adjustment = (missing_rate_adjustment - missing_rate_adjustment.min()) / \
                                      (missing_rate_adjustment.max() - missing_rate_adjustment.min() + 1e-8)
        
        # ============ 结合两个调整项 ============
        combined_adjustment = 0.6 * feature_level_adjustment + 0.4 * missing_rate_adjustment
        
        # 应用贝叶斯更新
        likelihood_ratio = 1 + combined_adjustment * 2
        
        posterior_proba = prior_proba * likelihood_ratio
        posterior_proba = np.clip(posterior_proba, 0, 1)
        
        return prior_proba, posterior_proba
