# -*- coding: utf-8 -*-
from .config import *

def calculate_metrics(y_true, y_pred_prob, threshold=0.5):
    """完整版指标计算（包含 G-Mean）"""
    y_true = np.array(y_true, dtype=np.int32)
    y_pred_prob = np.array(y_pred_prob, dtype=np.float32)
    y_pred = (y_pred_prob >= threshold).astype(int)

    # 基础指标
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    # accuracy = (        tp + tn    ) / (        tp + tn + 1 * fp + 5 * fn    )
    accuracy = accuracy_score(y_true, y_pred)
    aucc = roc_auc_score(y_true, y_pred_prob)

    precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)
    auprc = auc(recall, precision)

    precision = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    recall_0 = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    recall_1 = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    ks = np.max(np.abs(tpr - fpr))

    # ★ 新增：G-Mean 计算
    gmean = np.sqrt(recall_0 * recall_1 + 1e-8)  # 避免数值问题
    balanced_acc = (recall_0 + recall_1) / 2
    recall_imbalance = abs(recall_0 - recall_1)

    # ★ 新增：MCC 和 Kappa（如果需要）
    try:
        mcc = matthews_corrcoef(y_true, y_pred)
        kappa = cohen_kappa_score(y_true, y_pred)
    except:
        mcc, kappa = 0.0, 0.0

    return {
        "Accuracy": round(accuracy, 4),
        "AUC": round(aucc, 4),
        "KS": round(ks, 4),
        "AUPRC": round(auprc, 4),
        "Recall_0": round(recall_0, 4),
        "Recall_1": round(recall_1, 4),
        "Recall_Imbalance": round(recall_imbalance, 4),
        "G-Mean": round(gmean, 4),  # ★ 新增
        "Balanced_Acc": round(balanced_acc, 4),  # ★ 新增
        "MCC": round(mcc, 4),  # ★ 新增
        "Kappa": round(kappa, 4),  # ★ 新增
        "Precision": round(precision, 4),
        "F1": round(f1, 4),

    }
# def calculate_metrics(y_true, y_pred_prob, threshold=0.5):
#     """整合版指标计算"""
#     y_true = np.array(y_true, dtype=np.int32)
#     y_pred_prob = np.array(y_pred_prob, dtype=np.float32)
#     y_pred = (y_pred_prob >= threshold).astype(int)
#     # print('ytrue',y_true)
#     # print('y_pred_prob ',y_pred_prob )
#     # print('y_pred',y_pred)
#     accuracy = accuracy_score(y_true, y_pred)
#     aucc = roc_auc_score(y_true, y_pred_prob)
    
#     precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)
#     # print('auprc', precision)
#     auprc = auc(recall, precision)
    
#     precision = precision_score(y_true, y_pred, zero_division=0)
#     f1 = f1_score(y_true, y_pred, zero_division=0)
#     recall_0 = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
#     recall_1 = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
#     fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
#     ks = np.max(np.abs(tpr - fpr))
    
#     return {
#         "Accuracy": round(accuracy, 4),
#         "AUC": round(aucc, 4),
#         "KS": round(ks, 4),
#         "AUPRC": round(auprc, 4),
#         "Recall_0": round(recall_0, 4),
#         "Recall_1": round(recall_1, 4),
#         "Precision": round(precision, 4),
#         "F1": round(f1, 4)
#     }
def learn_cost_sensitive_threshold(y_true, y_prob, metric='f1'):
    thresholds = np.linspace(0.05, 0.95, 19)
    best_th = 0.5
    best_score = -np.inf

    for th in thresholds:
        y_pred = (y_prob >= th).astype(int)

        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)

        if metric == 'f2':
            beta = 2
            score = (1 + beta**2) * precision * recall / (beta**2 * precision + recall + 1e-8)
        elif metric == 'recall':
            score = recall
        else:
            score = 2 * precision * recall / (precision + recall + 1e-8)

        if score > best_score:
            best_score = score
            best_th = th

    return best_th, best_score

