# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
"""
GL-MA-RAG: Global-Local Missingness-Aware Retrieval Framework
Optimized Version v2.0 - Pattern Recognition Submission

Core Improvements:
1. Information-Theoretic Weight Design
2. Multimodal Deep Fusion Network
3. Bayesian Uncertainty Quantification (MC-Dropout plus CI)
4. Missing Mechanism Inference (MCAR/MAR/MNAR)
5. Complete Visualization Suite
6. Comparison Model Ensemble
7. SOTA Missing Learning Models (MissForest/MIWAE/GAIN)
"""

# from scipy.stats import beta as beta_dist, fbeta_score
from sklearn.metrics import matthews_corrcoef, cohen_kappa_score, brier_score_loss
import numpy as np
import pandas as pd
import faiss
import warnings
import os
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
from openpyxl import Workbook
from scipy.stats import ttest_ind, chi2_contingency, spearmanr
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from scipy.stats import pointbiserialr
from typing import Optional, Dict, List, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import f1_score, accuracy_score, fbeta_score, roc_auc_score, recall_score, precision_score, roc_curve,precision_recall_curve,auc
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from pytorch_tabnet.tab_model import TabNetClassifier
import lightgbm as lgb
import catboost as cb
import xgboost as xgb
from sklearn.base import BaseEstimator, ClassifierMixin
from matplotlib.font_manager import FontManager
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef, cohen_kappa_score
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import chi2_contingency, ks_2samp, mannwhitneyu
from collections import Counter
from scipy.stats import skew
from scipy.stats import chi2_contingency, ks_2samp, skew
from scipy.stats import beta as beta_dist
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTENC
import traceback
from matplotlib.patches import Rectangle, Patch
from matplotlib.collections import PatchCollection
from scipy.stats import gaussian_kde
from scipy import interpolate
import umap
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from datetime import datetime
from scipy import stats
from matplotlib.patches import Patch
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc

matplotlib.use("TkAgg")

matplotlib.use('Agg')
sns.set_style("whitegrid")
warnings.filterwarnings('ignore')
# ======================== 添加在InterpretabilityAnalyzer类之前 ========================



# ====================== 警告过滤 ======================
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# 备选字体列表：SimHei (黑体), Microsoft YaHei (微软雅黑), STHeiti (华文黑体)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题

# 检查当前系统支持哪些中文字体（调试用）

def check_chinese_fonts():
    fm = FontManager()
    mat_fonts = set(f.name for f in fm.ttflist)
    print("当前系统可用字体部分列表：", list(mat_fonts)[:10])
    chinese_fonts = ['SimHei', 'Microsoft YaHei', 'SimSun', 'STSong', 'KaiTi']
    for font in chinese_fonts:
        if font in mat_fonts:
            print(f"✅ 检测到中文字体: {font}")
        else:
            print(f"❌ 未检测到字体: {font}")

check_chinese_fonts()
# ====================== 核心配置 ======================
DATA_PATH = "2信用噪声数据处理-属性噪声标签噪声--论文撰写--准备投稿/model_sample.csv"  #hmeq   model_sample 银联   ；  lendingclub
# DATA_PATH = "2信用噪声数据处理-属性噪声标签噪声/your_credit_dataset_GER.csv"
EXPERIMENT_TIMES =1
RANDOM_SEED_BASE = 42
MISSING_RATES = [0.0] 
MECHS = ['MNAR'] # ['MCAR', 'MAR', 'MNAR']
SAVE_EXCEL_PATH = "credit_risk_results_optimized.xlsx"

# =====================================================================
#          ★★★ 第X部分：SOTA缺失学习模型 (MissForest/miwae/GAIN) ★★★
# =====================================================================

# ====================== MissForest 实现 ======================
"""
    MissForest: 迭代随机森林缺失学习
    论文: MissForest – nonparametric missing value imputation for mixed-type data
    
    核心思想：
    1. 对每列缺失值用其他特征的随机森林预测
    2. 迭代更新直到收敛
    3. 分别处理数值和分类特征
    
    参数：
        X: 含缺失值的数据 (numpy array or DataFrame)
        n_estimators: 每列RF的树数量
        max_iter: 最大迭代次数
        seed: 随机种子
"""

# =====================================================================
#                   第1部分：统一指标计算函数
# =====================================================================
