# -*- coding: utf-8 -*-
from .config import *
from .bayesian_predictor import MissingnessAwareBayesianRiskPredictor
from .retrieval import StratifiedRetriever, build_gl_ma_rag_index_improved, gl_ma_retrieve
from .metrics import calculate_metrics
from .experiment import load_credit_data_optimized

# =====================================================================
#      消融实验模块（Ablation Study with Multiple Runs）
# =====================================================================
# ✅ 正确的消融实验设计

class AblationStudyV3_Revised:
    """
    正确的消融实验框架（遵循ML最佳实践）
    """
    
    def run_ablation(self, X_train, X_val, X_test, y_train, y_val, y_test,
                    mask_train, mask_val, mask_test,
                    local_train_info, local_val_info, local_test_info,
                    global_missing_stats, pos_weight, seed):
        """
        ★ 原则：每个消融变体必须在完全相同的超参数和数据下运行
        ★ 目标：量化每个模块对性能的贡献
        """
        
        print("\n" + "="*80)
        print("【GL-MA-RAG v3.0 消融实验（修正版）】")
        print("="*80)
        
        ablation_results = {}
        
        # ========== 变体0：完整版本 ==========
        print("\n【变体0】Full - GL-MA-RAG v3.0 (完整版本)")
        print("-"*60)
        
        bayes_full = MissingnessAwareBayesianRiskPredictor(
            kappa0=5.0, lambda_miss=5.0, tau=0.5,
            fp_cost=1.0, fn_cost=5.0, use_adaptive_cost=True
        )
        bayes_full.fit_prior_model(X_train, y_train, mask_train,prior_model_type="logr", seed=seed)
        
        # 构建分层索引
        stratified_retriever_full = StratifiedRetriever(balance_ratio=0.5, min_per_class=3)
        feature_cols = list(global_missing_stats["global_weights"].keys())
        global_weights = np.array([global_missing_stats["global_weights"][col] 
                                   for col in feature_cols])
        stratified_retriever_full.build_stratified_index(
            X_val, mask_val, y_val,
            global_weights, [0.5, 0.4, 0.1]
        )
        
        # 学习G-Mean最优阈值
        val_pred_probs_full = []
        for i in range(len(X_val)):
            retrieve_result = stratified_retriever_full.retrieve_balanced(
                X_val[i], mask_val[i],
                local_val_info["sample_missing_ratio"][i],
                local_val_info["missing_level"][i],
                k_total=15
            )
            pred_result = bayes_full.predict_single(
                X_val[i], mask_val[i], retrieve_result,
                y_train, local_train_info["sample_missing_ratio"],X_train
            )
            val_pred_probs_full.append(pred_result["prediction"])
        
        calibrator_full = AdaptiveThresholdCalibrator(target_metric='gmean')
        calibrator_full.learn_thresholds(y_val, np.array(val_pred_probs_full),
                                        local_val_info["sample_missing_ratio"])
        bayes_full.decision_threshold = calibrator_full.global_optimal_threshold
        
        # 测试集评估
        test_pred_probs_full = []
        for i in range(len(X_test)):
            retrieve_result = stratified_retriever_full.retrieve_balanced(
                X_test[i], mask_test[i],
                local_test_info["sample_missing_ratio"][i],
                local_test_info["missing_level"][i],
                k_total=15
            )
            pred_result = bayes_full.predict_single(
                X_test[i], mask_test[i], retrieve_result,
                y_val, local_val_info["sample_missing_ratio"],X_val
            )
            test_pred_probs_full.append(pred_result["prediction"])
        
        metrics_full = calculate_metrics(y_test, np.array(test_pred_probs_full),
                                        threshold=calibrator_full.global_optimal_threshold)
        ablation_results['Full'] = metrics_full
        
        print(f"  AUC: {metrics_full['AUC']:.4f}, G-Mean: {metrics_full.get('G-Mean', 0):.4f}")
        print(f"  Recall_0: {metrics_full['Recall_0']:.4f}, Recall_1: {metrics_full['Recall_1']:.4f}")
        
        # ========== 变体1：移除类感知缺失分析 ==========
        print("\n【变体1】w/o Class-Aware - 移除类感知分析")
        print("-"*60)
        
        # ★ 关键：只改变这一个地方，其他完全相同
        global_weights_uniform = np.ones(X_train.shape[1]) / X_train.shape[1]  # 均匀权重
        
        stratified_retriever_v1 = StratifiedRetriever(balance_ratio=0.5, min_per_class=3)
        stratified_retriever_v1.build_stratified_index(
            X_val, mask_val, y_val,
            global_weights_uniform,  # ← 只有这里不同
            [0.5, 0.4, 0.1]
        )
        
        # 其他流程完全相同...
        bayes_v1 = MissingnessAwareBayesianRiskPredictor(
            kappa0=5.0, lambda_miss=5.0, tau=0.5,
            fp_cost=1.0, fn_cost=5.0, use_adaptive_cost=True
        )
        bayes_v1.fit_prior_model(X_train, y_train, mask_train, seed=seed)
        
        # 预测流程（与完整版相同）
        val_pred_probs_v1 = []
        for i in range(len(X_val)):
            retrieve_result = stratified_retriever_v1.retrieve_balanced(
                X_val[i], mask_val[i],
                local_val_info["sample_missing_ratio"][i],
                local_val_info["missing_level"][i],
                k_total=15
            )
            pred_result = bayes_v1.predict_single(
                X_val[i], mask_val[i], retrieve_result,
                y_train, local_train_info["sample_missing_ratio"], X_train
            )
            val_pred_probs_v1.append(pred_result["prediction"])
        
        calibrator_v1 = AdaptiveThresholdCalibrator(target_metric='gmean')
        calibrator_v1.learn_thresholds(y_val, np.array(val_pred_probs_v1),
                                       local_val_info["sample_missing_ratio"])
        bayes_v1.decision_threshold = calibrator_v1.global_optimal_threshold
        
        test_pred_probs_v1 = []
        for i in range(len(X_test)):
            retrieve_result = stratified_retriever_v1.retrieve_balanced(
                X_test[i], mask_test[i],
                local_test_info["sample_missing_ratio"][i],
                local_test_info["missing_level"][i],
                k_total=15
            )
            pred_result = bayes_v1.predict_single(
                X_test[i], mask_test[i], retrieve_result,
                y_val, local_val_info["sample_missing_ratio"],X_val
            )
            test_pred_probs_v1.append(pred_result["prediction"])
        
        metrics_v1 = calculate_metrics(y_test, np.array(test_pred_probs_v1),
                                      threshold=calibrator_v1.global_optimal_threshold)
        ablation_results['w/o Class-Aware'] = metrics_v1
        
        print(f"  AUC: {metrics_v1['AUC']:.4f}, G-Mean: {metrics_v1.get('G-Mean', 0):.4f}")
        
        # ========== 变体2：移除分层检索（使用普通FAISS） ==========
        print("\n【变体2】w/o Stratified - 使用普通FAISS检索")
        print("-"*60)
        
        # 构建普通索引（不分层）
        index_v2, multimodal_val, scaler_X, global_weights = \
            build_gl_ma_rag_index_improved(X_val, mask_val, global_missing_stats,
                                          local_val_info, fusion_weights=[0.5, 0.4, 0.1])
        
        bayes_v2 = MissingnessAwareBayesianRiskPredictor(
            kappa0=5.0, lambda_miss=5.0, tau=0.5,
            fp_cost=1.0, fn_cost=5.0, use_adaptive_cost=True
        )
        bayes_v2.fit_prior_model(X_train, y_train, mask_train, seed=seed)
        
        # 使用普通检索预测
        val_pred_probs_v2 = []
        for i in range(len(X_val)):
            retrieve_result = gl_ma_retrieve(
                X_val[i], mask_val[i],
                local_val_info["sample_missing_ratio"][i],
                local_val_info["missing_level"][i],
                index_v2, local_val_info, global_weights, scaler_X,
                [0.5, 0.4, 0.1], {'k_base': 15}
            )
            pred_result = bayes_v2.predict_single(
                X_val[i], mask_val[i], retrieve_result,
                y_train, local_train_info["sample_missing_ratio"],X_train
            )
            val_pred_probs_v2.append(pred_result["prediction"])
        
        calibrator_v2 = AdaptiveThresholdCalibrator(target_metric='gmean')
        calibrator_v2.learn_thresholds(y_val, np.array(val_pred_probs_v2),
                                       local_val_info["sample_missing_ratio"])
        bayes_v2.decision_threshold = calibrator_v2.global_optimal_threshold
        
        test_pred_probs_v2 = []
        for i in range(len(X_test)):
            retrieve_result = gl_ma_retrieve(
                X_test[i], mask_test[i],
                local_test_info["sample_missing_ratio"][i],
                local_test_info["missing_level"][i],
                index_v2, local_test_info, global_weights, scaler_X,
                [0.5, 0.4, 0.1], {'k_base': 15}
            )
            pred_result = bayes_v2.predict_single(
                X_test[i], mask_test[i], retrieve_result,
                y_val, local_val_info["sample_missing_ratio"],X_val
            )
            test_pred_probs_v2.append(pred_result["prediction"])
        
        metrics_v2 = calculate_metrics(y_test, np.array(test_pred_probs_v2),
                                      threshold=calibrator_v2.global_optimal_threshold)
        ablation_results['w/o Stratified'] = metrics_v2
        
        print(f"  AUC: {metrics_v2['AUC']:.4f}, G-Mean: {metrics_v2.get('G-Mean', 0):.4f}")
        
        # ========== 变体3：移除代价敏感（使用固定0.5阈值） ==========
        print("\n【变体3】w/o Cost-Sensitive - 使用固定阈值0.5")
        print("-"*60)
        
        bayes_v3 = MissingnessAwareBayesianRiskPredictor(
            kappa0=5.0, lambda_miss=5.0, tau=0.5,
            fp_cost=1.0, fn_cost=1.0,  # ← FP和FN代价相同
            use_adaptive_cost=False     # ← 禁用动态代价
        )
        bayes_v3.fit_prior_model(X_train, y_train, mask_train, seed=seed)
        bayes_v3.decision_threshold = 0.5  # ← 固定阈值
        
        stratified_retriever_v3 = StratifiedRetriever(balance_ratio=0.5, min_per_class=3)
        stratified_retriever_v3.build_stratified_index(
            X_val, mask_val, y_val,
            global_weights, [0.5, 0.4, 0.1]
        )
        
        test_pred_probs_v3 = []
        for i in range(len(X_test)):
            retrieve_result = stratified_retriever_v3.retrieve_balanced(
                X_test[i], mask_test[i],
                local_test_info["sample_missing_ratio"][i],
                local_test_info["missing_level"][i],
                k_total=15
            )
            pred_result = bayes_v3.predict_single(
                X_test[i], mask_test[i], retrieve_result,
                y_val, local_val_info["sample_missing_ratio"],X_val
            )
            test_pred_probs_v3.append(pred_result["prediction"])
        
        metrics_v3 = calculate_metrics(y_test, np.array(test_pred_probs_v3),
                                      threshold=0.5)  # ← 使用固定阈值
        ablation_results['w/o Cost-Sensitive'] = metrics_v3
        
        print(f"  AUC: {metrics_v3['AUC']:.4f}, G-Mean: {metrics_v3.get('G-Mean', 0):.4f}")
        
        # ========== 变体4：移除G-Mean优化（使用F1优化） ==========
        print("\n【变体4】w/o G-Mean - 使用F1优化阈值")
        print("-"*60)
        
        bayes_v4 = MissingnessAwareBayesianRiskPredictor(
            kappa0=5.0, lambda_miss=5.0, tau=0.5,
            fp_cost=1.0, fn_cost=5.0, use_adaptive_cost=True
        )
        bayes_v4.fit_prior_model(X_train, y_train, mask_train, seed=seed)
        
        stratified_retriever_v4 = StratifiedRetriever(balance_ratio=0.5, min_per_class=3)
        stratified_retriever_v4.build_stratified_index(
            X_val, mask_val, y_val,
            global_weights, [0.5, 0.4, 0.1]
        )
        
        val_pred_probs_v4 = []
        for i in range(len(X_val)):
            retrieve_result = stratified_retriever_v4.retrieve_balanced(
                X_val[i], mask_val[i],
                local_val_info["sample_missing_ratio"][i],
                local_val_info["missing_level"][i],
                k_total=15
            )
            pred_result = bayes_v4.predict_single(
                X_val[i], mask_val[i], retrieve_result,
                y_train, local_train_info["sample_missing_ratio"],X_train
            )
            val_pred_probs_v4.append(pred_result["prediction"])
        
        # ★ 改用F1优化而非G-Mean
        calibrator_v4 = AdaptiveThresholdCalibrator(target_metric='f1')  # ← F1而非F2
        calibrator_v4.learn_thresholds(y_val, np.array(val_pred_probs_v4),
                                       local_val_info["sample_missing_ratio"])
        bayes_v4.decision_threshold = calibrator_v4.global_optimal_threshold
        
        test_pred_probs_v4 = []
        for i in range(len(X_test)):
            retrieve_result = stratified_retriever_v4.retrieve_balanced(
                X_test[i], mask_test[i],
                local_test_info["sample_missing_ratio"][i],
                local_test_info["missing_level"][i],
                k_total=15
            )
            pred_result = bayes_v4.predict_single(
                X_test[i], mask_test[i], retrieve_result,
                y_val, local_val_info["sample_missing_ratio"],X_val
            )
            test_pred_probs_v4.append(pred_result["prediction"])
        
        metrics_v4 = calculate_metrics(y_test, np.array(test_pred_probs_v4),
                                      threshold=calibrator_v4.global_optimal_threshold)
        ablation_results['w/o G-Mean'] = metrics_v4
        
        print(f"  AUC: {metrics_v4['AUC']:.4f}, G-Mean: {metrics_v4.get('G-Mean', 0):.4f}")
        
        # ========== 汇总 ==========
        self._print_ablation_summary(ablation_results)
        return ablation_results
    
    def _print_ablation_summary(self, ablation_results):
        """打印消融实验汇总"""
        print("\n" + "="*80)
        print("【消融实验汇总】")
        print("="*80)
        
        # 按完整版本的AUC排序
        full_auc = ablation_results['Full']['AUC']
        
        print(f"\n{'变体':<25} {'AUC':<10} {'降幅':<10} {'G-Mean':<10} {'贡献度分析'}")
        print("-"*80)
        
        for variant, metrics in sorted(ablation_results.items(), 
                                      key=lambda x: x[1]['AUC'], reverse=True):
            auc = metrics['AUC']
            auc_drop = full_auc - auc
            gmean = metrics.get('G-Mean', 0)
            
            if variant == 'Full':
                symbol = '⭐'
                contribution = "基准"
            else:
                symbol = '▼' if auc_drop > 0 else '▲'
                contribution = f"{auc_drop:+.4f}"
            
            print(f"{variant:<25} {auc:<10.4f} {auc_drop:<+10.4f} {gmean:<10.4f} {contribution}")
        
        print("\n【组件贡献度定量分析】")
        print("-"*80)
        
        full_auc = ablation_results['Full']['AUC']
        
        contribution_analysis = {
            '类感知缺失分析': full_auc - ablation_results['w/o Class-Aware']['AUC'],
            '分层检索策略': full_auc - ablation_results['w/o Stratified']['AUC'],
            '代价敏感决策': full_auc - ablation_results['w/o Cost-Sensitive']['AUC'],
            'G-Mean优化': full_auc - ablation_results['w/o G-Mean']['AUC']
        }
        
        total_contribution = sum(contribution_analysis.values())
        
        for component, contribution in sorted(contribution_analysis.items(), 
                                            key=lambda x: x[1], reverse=True):
            percentage = (contribution / (total_contribution + 1e-8)) * 100
            bar = "█" * int(percentage / 2)
            print(f"  {component:<20}: {contribution:+.4f} AUC ({percentage:>5.1f}%) {bar}")
        
        print(f"\n  总贡献度: {total_contribution:.4f} AUC")
def run_ablation_with_statistics(X_train, X_val, X_test, y_train, y_val, y_test,
                                 mask_train, mask_val, mask_test, num_runs=20):
    """
    正确的消融实验（包含统计分析）
    """
    
    print(f"\n【正确的消融实验：运行{num_runs}次以计算统计显著性】")
    
    all_results = {
        'Full': [],
        'w/o Class-Aware': [],
        'w/o Stratified': [],
        'w/o Cost-Sensitive': [],
        'w/o G-Mean': []
    }
    
    for run_id in range(num_runs):
        print(f"\n【第{run_id+1}次运行】")
        
        # 使用相同的超参和数据，但不同的初始化
        ablator = AblationStudyV3_Revised()
        results = ablator.run_ablation(
            X_train, X_val, X_test, y_train, y_val, y_test,
            mask_train, mask_val, mask_test,
            local_train_info, local_val_info, local_test_info,
            global_missing_stats, pos_weight,
            seed=42 + run_id  # 略微改变种子以获得变异
        )
        
        for variant, metrics in results.items():
            all_results[variant].append(metrics['AUC'])
    
    # 统计分析
    print("\n" + "="*80)
    print("【消融实验统计汇总 (5次运行)】")
    print("="*80)
    print(f"{'变体':<25} {'AUC (mean)':<15} {'AUC (std)':<15} {'贡献度'}")
    print("-"*80)
    
    full_auc_mean = np.mean(all_results['Full'])
    
    for variant in sorted(all_results.keys()):
        auc_values = all_results[variant]
        auc_mean = np.mean(auc_values)
        auc_std = np.std(auc_values)
        
        if variant == 'Full':
            contribution = "基准"
            symbol = "⭐"
        else:
            contribution = f"{full_auc_mean - auc_mean:+.4f}"
            symbol = "▼"
        
        print(f"{symbol} {variant:<22} {auc_mean:.4f}±{auc_std:.4f}   {auc_std:<14.4f} {contribution}")
    
    # t-test 检验每个变体与完整版本的差异
    print("\n【统计显著性检验 (t-test)】")
    print("-"*80)
    
    full_values = all_results['Full']
    for variant in all_results.keys():
        if variant == 'Full':
            continue
        
        variant_values = all_results[variant]
        t_stat, p_value = stats.ttest_rel(full_values, variant_values)
        
        significance = "✓ p<0.05" if p_value < 0.05 else "✗ p≥0.05"
        print(f"{variant:<25}: t={t_stat:>7.3f}, p={p_value:.4f} {significance}")

def run_ablation_experiment():
    """
    运行消融实验（修复版）
    """
    print("\n" + "="*80)
    print("【GL-MA-RAG v3.0 消融实验】")
    print("="*80)

    mr = MISSING_RATES[0]
    mech = MECHS[0]

    # ★ 初始化存储所有运行结果的字典
    all_runs_results = {
        'Full': [],
        'w/o Class-Aware': [],
        'w/o Stratified': [],
        'w/o Cost-Sensitive': [],
        'w/o G-Mean': []
    }

    # 运行5次
    for run_id in range(1, 21):
        print(f"\n【第{run_id}次运行】")
        print("-"*60)

        seed = RANDOM_SEED_BASE + run_id

        # 加载数据
        (X_train, X_val, X_test, y_train, y_val, y_test,
         mask_train, mask_val, mask_test,
         local_train_info, local_val_info, local_test_info,
         global_missing_stats, pos_weight,
         X_train_missing, X_val_missing, X_test_missing,
         X_train_woe, X_val_woe, X_test_woe,
         adaptive_thresholds, adaptive_params, mech_prob) = \
            load_credit_data_optimized(DATA_PATH, seed, mr, mech)

        # 运行消融实验
        ablator = AblationStudyV3_Revised()
        ablation_results = ablator.run_ablation(
            X_train, X_val, X_test, y_train, y_val, y_test,
            mask_train, mask_val, mask_test,
            local_train_info, local_val_info, local_test_info,
            global_missing_stats, pos_weight, seed
        )

        # 记录所有结果
        for variant, metrics in ablation_results.items():
            all_runs_results[variant].append(metrics)

        print(f"\n  第{run_id}次运行简要结果:")
        for variant, metrics in ablation_results.items():
            print(f"    {variant}: AUC={metrics['AUC']:.4f}, G-Mean={metrics['G-Mean']:.4f}")

    # 计算平均结果
    averaged_results = {}
    for variant in all_runs_results.keys():
        averaged_metrics = {}
        for metric in all_runs_results[variant][0].keys():
            values = [run[metric] for run in all_runs_results[variant]]
            averaged_metrics[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'values': values
            }
        averaged_results[variant] = averaged_metrics

    # ★ 修复：传递 all_runs_results 参数
    export_ablation_results_to_excel(averaged_results, all_runs_results)

    # 打印汇总
    print_ablation_summary(averaged_results)

    # 生成可视化
    plot_ablation_results(averaged_results)

    return averaged_results, all_runs_results



def export_ablation_results_to_excel(averaged_results, all_runs_results, 
                                    output_path='ablation_results.xlsx'):
    """
    将消融实验结果导出到 Excel（完全修复版）
    """

    wb = Workbook()
    ws = wb.active
    ws.title = "Ablation Study Results"

    # 样式定义
    header_font = Font(bold=True, size=12)
    header_fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
    border = Border(left=Side(style='thin'), right=Side(style='thin'),
                   top=Side(style='thin'), bottom=Side(style='thin'))
    center_align = Alignment(horizontal='center', vertical='center')

    # ========== Sheet 1: 主要结果 ==========
    ws['A1'] = "GL-MA-RAG v3.0 消融实验结果（5次运行平均）"
    ws['A1'].font = Font(bold=True, size=14)
    ws.merge_cells('A1:M1')

    ws['A2'] = "比较不同组件对模型性能的影响（均值±标准差）"
    ws['A2'].font = Font(size=11, italic=True)

    variants = list(averaged_results.keys())
    metrics_list = list(averaged_results[variants[0]].keys())

    # 写表头
    ws.append(['变体'] + metrics_list)
    for i in range(1, len(metrics_list) + 2):
        ws.cell(row=3, column=i).fill = header_fill
        ws.cell(row=3, column=i).font = header_font
        ws.cell(row=3, column=i).border = border
        ws.cell(row=3, column=i).alignment = center_align

    # 写数据
    row = 4
    for variant in variants:
        ws.cell(row=row, column=1, value=variant)
        ws.cell(row=row, column=1).font = Font(bold=True)
        
        for col_idx, metric in enumerate(metrics_list, 1):
            mean_val = averaged_results[variant][metric]['mean']
            std_val = averaged_results[variant][metric]['std']
            cell_value = f"{mean_val:.4f}±{std_val:.4f}"
            
            ws.cell(row=row, column=col_idx+1, value=cell_value)
            ws.cell(row=row, column=col_idx+1).border = border
            ws.cell(row=row, column=col_idx+1).alignment = center_align

        row += 1

    # ========== Sheet 2: 详细数据 ==========
    ws_detail = wb.create_sheet("Detailed Results")
    detail_headers = ['变体', '运行号'] + metrics_list
    ws_detail.append(detail_headers)

    for i in range(1, len(detail_headers) + 1):
        ws_detail.cell(row=1, column=i).fill = header_fill
        ws_detail.cell(row=1, column=i).font = header_font
        ws_detail.cell(row=1, column=i).border = border
        ws_detail.cell(row=1, column=i).alignment = center_align

    detail_row = 2
    for variant in variants:
        for run_id in range(len(all_runs_results[variant])):
            ws_detail.cell(row=detail_row, column=1, value=variant)
            ws_detail.cell(row=detail_row, column=2, value=run_id + 1)
            
            for col_idx, metric in enumerate(metrics_list, 1):
                metric_value = all_runs_results[variant][run_id][metric]
                ws_detail.cell(row=detail_row, column=col_idx+2, value=metric_value)
                ws_detail.cell(row=detail_row, column=col_idx+2).border = border
                ws_detail.cell(row=detail_row, column=col_idx+2).alignment = center_align

            detail_row += 1

    # ========== Sheet 3: 贡献度分析 ==========
    ws_contribution = wb.create_sheet("Contribution Analysis")
    
    full_auc = averaged_results['Full']['AUC']['mean']
    contributions = {}
    for variant in variants:
        if variant != 'Full':
            auc_drop = full_auc - averaged_results[variant]['AUC']['mean']
            contributions[variant] = auc_drop

    total_contribution = sum(contributions.values())

    ws_contribution.append(['组件', 'AUC降幅', '相对贡献(%)'])
    for i in range(1, 4):
        ws_contribution.cell(row=1, column=i).fill = header_fill
        ws_contribution.cell(row=1, column=i).font = header_font
        ws_contribution.cell(row=1, column=i).border = border
        ws_contribution.cell(row=1, column=i).alignment = center_align

    contrib_row = 2
    for component in sorted(contributions.keys(), key=lambda x: contributions[x], reverse=True):
        auc_drop = contributions[component]
        relative_contribution = (auc_drop / (total_contribution + 1e-8)) * 100
        
        ws_contribution.cell(row=contrib_row, column=1, value=component.replace('w/o ', ''))
        ws_contribution.cell(row=contrib_row, column=2, value=auc_drop)
        ws_contribution.cell(row=contrib_row, column=3, value=f"{relative_contribution:.1f}%")
        
        for col in range(1, 4):
            ws_contribution.cell(row=contrib_row, column=col).border = border
            ws_contribution.cell(row=contrib_row, column=col).alignment = center_align

        contrib_row += 1

    # ★ 修复：统一调整所有sheet的列宽（安全方式）
    for ws_obj in [ws, ws_detail, ws_contribution]:
        # 设置第A列宽度
        ws_obj.column_dimensions['A'].width = 25
        
        # 设置其他列宽度
        for col_idx in range(2, len(metrics_list) + 3):
            col_letter = get_column_letter(col_idx)
            ws_obj.column_dimensions[col_letter].width = 15

    # 保存
    wb.save(output_path)
    print(f"\n✅ 消融实验结果已导出到: {output_path}")

def print_ablation_summary(averaged_results):
    """打印消融实验汇总（修复版）"""
    print("\n" + "="*80)
    print("【消融实验统计汇总 (5次运行)】")
    print("="*80)

    variants = list(averaged_results.keys())
    full_auc = averaged_results['Full']['AUC']['mean']

    print(f"\n{'变体':<25} {'AUC (均值)':<15} {'AUC (std)':<10} {'G-Mean':<10} {'ΔAUC':<10}")
    print("-"*80)

    for variant in variants:
        auc_mean = averaged_results[variant]['AUC']['mean']
        auc_std = averaged_results[variant]['AUC']['std']
        gmean = averaged_results[variant]['G-Mean']['mean']
        auc_drop = full_auc - auc_mean

        if variant == 'Full':
            symbol = '⭐'
            contribution = "基准"
        else:
            symbol = '▼'
            contribution = f"{auc_drop:+.4f}"

        print(f"{symbol} {variant:<22} {auc_mean:<15.4f} ({auc_std:<10.4f}) {gmean:<10.4f} {contribution}")

    print("\n【组件贡献度定量分析】")
    print("-"*80)

    contributions = {}
    for variant in variants:
        if variant == 'Full':
            continue
        auc_drop = full_auc - averaged_results[variant]['AUC']['mean']
        contributions[variant] = auc_drop

    total_contribution = sum(contributions.values())

    for component, contribution in sorted(contributions.items(),
                                        key=lambda x: x[1], reverse=True):
        percentage = (contribution / (total_contribution + 1e-8)) * 100
        bar = "█" * int(percentage / 2)
        print(f"  {component.replace('w/o ', ''):<20}: {contribution:+.4f} AUC ({percentage:>5.1f}%) {bar}")

    print(f"\n  总贡献度: {total_contribution:.4f} AUC")
def plot_ablation_results(averaged_results, output_path='ablation_plot.png'):
    """
    生成消融实验的可视化图表（修复版）
    """

    # 准备数据
    variants = list(averaged_results.keys())
    metrics = ['AUC', 'G-Mean', 'Recall_0', 'Recall_1', 'MCC', 'Kappa']

    # 创建图表
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        ax = axes[i]
        means = [averaged_results[variant][metric]['mean'] for variant in variants]
        stds = [averaged_results[variant][metric]['std'] for variant in variants]

        colors = ['green' if v == 'Full' else 'steelblue' for v in variants]

        bars = ax.bar(variants, means, yerr=stds, capsize=5,
                     color=colors, alpha=0.7,
                     edgecolor='black', linewidth=1.5)

        # 添加数值标签
        for bar, mean, std in zip(bars, means, stds):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{mean:.3f}±{std:.3f}',
                   ha='center', va='bottom', fontsize=9)

        ax.set_title(metric, fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, axis='y')

        if i == 0 or i == 3:  # 第一列
            ax.set_ylabel('Score', fontsize=11, fontweight='bold')

    plt.suptitle('Ablation Study: Performance Comparison (Mean ± Std)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ 消融实验可视化已保存: {output_path}")



