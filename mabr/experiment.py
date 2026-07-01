# -*- coding: utf-8 -*-
from .config import *
from .missing_injection import inject_missing, inject_missing_new
from .missing_analysis import analyze_global_missing_pattern_improved, RobustMissingMechanismDetector, MissingMechanismClassifier, AdaptiveThresholdCalibrator, calculate_local_missing_info_adaptive, get_missing_level
from .retrieval import StratifiedRetriever, build_gl_ma_rag_index_with_stratified_retrieval, build_gl_ma_rag_index_improved
from .bayesian_predictor import MissingnessAwareBayesianRiskPredictor, bayesian_predict_with_retrieval
from .visualization import GLMARagVisualizer, ConfidenceIntervalVisualizer, MissingRateAnalysisVisualizer, PredictionAccuracyVisualizer, bayesian_predict_batch_with_results, export_results_to_excel_advanced
from .metrics import calculate_metrics, learn_cost_sensitive_threshold

# =====================================================================
#                   第13部分：数据加载与预处理
# =====================================================================

def load_credit_data_optimized(csv_path, seed, missing_rate=0.2, mech='MCAR'):
    """
    加载数据 + 缺失注入 + 优化的预处理（修复版）
    """
    print(f"正在加载数据集：{csv_path}")
    df = pd.read_csv(csv_path, encoding='utf-8')
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(how="all")
        # 直接删除缺失率 >70% 的列
    threshold = 0.4
    drop_cols = df.columns[df.isnull().sum() / len(df) >= threshold]
    data =  df.drop(columns=drop_cols)

    label_col = data.columns[-1]
    feature_names = data.columns[:-1].tolist()
    
    for col in data.columns:
        if col != label_col:
            data[col] = data[col].astype(np.float32)
    
    print(f"数据集：{len(data)}样本，{len(feature_names)}特征")
    print(f"数据集h含有：{np.sum(df.iloc[:,-1])}坏样本")
    
    # 分离特征和标签
    X_original = data.iloc[:, :-1].copy()
    y = data.iloc[:, -1].copy()
    
    # 注入缺失值
    # print(f"注入缺失：{mech}，缺失率：{missing_rate}")
    # X_inject, inject_mask = inject_missing(X_original.values, missing_rate, mech, seed)
    # X_inject = pd.DataFrame(X_inject, columns=feature_names)


    # 调用函数：自动处理原生缺失 + 注入缺失
    X_inject, inject_mask = inject_missing_new(X_original.values, missing_rate, mech, seed)
    X_inject = pd.DataFrame(X_inject, columns=feature_names)

    # 统计验证
    total_missing = np.isnan(X_inject).sum().sum()
    original_missing = np.isnan(X_original).sum().sum()
    inject_missing = total_missing - original_missing

    print(f"原始真实缺失值：{original_missing}")
    print(f"人工注入缺失值：{inject_missing}")
    print(f"总缺失值：{total_missing}，总缺失率：{total_missing / X_inject.size:.4f}")
    print(f"mask矩阵中标记为缺失(1)的数量：{inject_mask.sum()}")  # 一定等于总缺失值


    missing_count = np.isnan(X_inject).sum().sum()
    print(f"缺失值总数：{missing_count}，实际缺失率：{missing_count / X_inject.size:.4f}")
    
    # 填充为0（用于建模）
    # X_filled = X_inject.fillna(0)
    X_filled = X_inject.fillna(X_inject.mean())
    

    
    # 全局缺失分析（改进版）
    global_missing_stats, inferred_mech = analyze_global_missing_pattern_improved(X_inject)
    


    # 缺失机制推断
    # 在load_credit_data_optimized函数中：
    detector = RobustMissingMechanismDetector()  # ★使用修复版
    feature_mechanisms = detector.detect_missing_mechanism_per_feature(X_inject.values)
    
    global_mech = detector.get_global_mechanism()
    
    print(f"\n属性级机制检测结果：")
    for feat_id, res in feature_mechanisms.items():
        if res['mechanism'] != 'No Missing':
            print(f"  特征{feat_id}: {res['mechanism']} (置信度: {res['confidence']:.2f})")
    
    print(f"全局主导机制: {global_mech['dominant_mechanism']} (置信度: {global_mech['confidence']:.2f})")
    
    # ★ 可视化时传入正确的feature_names
    viz_dir = f"./visualization_results/{mech}_{missing_rate*100:.0f}%_exp{seed}"
    os.makedirs(viz_dir, exist_ok=True)
    
    try:
        detector.visualize_mechanism_results(
            feature_names=X_inject.columns.tolist(),  # ★确保传入列名
            save_path=f'{viz_dir}/feature_missing_mechanisms.png'
        )
        print(f"✓ 缺失机制可视化已保存: {viz_dir}/feature_missing_mechanisms.png")
    except Exception as e:
        print(f"⚠️ 可视化失败（不影响主流程）: {str(e)[:100]}")



    mech_classifier = MissingMechanismClassifier()
    mech_prob = mech_classifier.infer_missing_mechanism(X_inject.values)
    adaptive_params = mech_classifier.get_adaptive_parameters()
    
    high_missing_features = global_missing_stats["high_missing_features"]
    
    # 数据划分
    
    X_train_filled, X_remain_filled, y_train, y_remain, mask_train, mask_remain = train_test_split(
        X_filled, y, inject_mask, test_size=0.2, stratify=y, random_state=seed
    )
    
    X_val_filled, X_test_filled, y_val, y_test, mask_val, mask_test = train_test_split(
        X_remain_filled, y_remain, mask_remain, test_size=1/2, stratify=y_remain, random_state=seed
    )
    

        # 类别权重
    pos_weight = (len(y_train) - y_train.sum()) / y_train.sum() if y_train.sum() > 0 else 1.0


    # 提取原始含缺失数据
    X_train_missing = X_inject.iloc[X_train_filled.index].copy()
    X_val_missing = X_inject.iloc[X_val_filled.index].copy()
    X_test_missing = X_inject.iloc[X_test_filled.index].copy()
    
    # 自适应阈值学习
    # ★ 修复版本 - 同时支持新旧接口
    try:
        # 先尝试调用新的v3.0版本
        adaptive_thresholds = learn_adaptive_thresholds_new(
            X_val_filled.values, y_val.values, mask_val
        )
    except (NameError, TypeError):
        # 如果失败，使用原始版本
        adaptive_thresholds = learn_adaptive_thresholds(
            X_val_filled.values, y_val.values, mask_val
        )

    # 确保返回值是元组
    if not isinstance(adaptive_thresholds, tuple):
        print(f"⚠️ 警告：adaptive_thresholds 类型为 {type(adaptive_thresholds)}")
        # 尝试从对象中提取
        if hasattr(adaptive_thresholds, 'calibration_table'):
            adaptive_thresholds = (
                adaptive_thresholds.calibration_table.get('low', 0.1),
                adaptive_thresholds.calibration_table.get('mid', 0.3)
            )
        else:
            adaptive_thresholds = (0.1, 0.3)  # 默认值

    print(f"自学习阈值: low={adaptive_thresholds[0]:.3f}, mid={adaptive_thresholds[1]:.3f}")
    
    # 计算局部缺失信息（自适应）
    local_train_info = calculate_local_missing_info_adaptive(
        X_train_filled, mask_train, high_missing_features,
        adaptive_thresholds[0], adaptive_thresholds[1], adaptive_params
    )
    
    local_val_info = calculate_local_missing_info_adaptive(
        X_val_filled, mask_val, high_missing_features,
        adaptive_thresholds[0], adaptive_thresholds[1], adaptive_params
    )
    
    local_test_info = calculate_local_missing_info_adaptive(
        X_test_filled, mask_test, high_missing_features,
        adaptive_thresholds[0], adaptive_thresholds[1], adaptive_params
    )
    
    # WOE编码和标准化
    numeric_cols = X_train_filled.select_dtypes(include=[np.number]).columns.tolist()
    
    X_train_woe = X_train_filled.copy()
    X_val_woe = X_val_filled.copy()
    X_test_woe = X_test_filled.copy()
    
    scaler = StandardScaler()
    if numeric_cols:
        X_train_woe[numeric_cols] = scaler.fit_transform(X_train_woe[numeric_cols])
        X_val_woe[numeric_cols] = scaler.transform(X_val_woe[numeric_cols])
        X_test_woe[numeric_cols] = scaler.transform(X_test_woe[numeric_cols])
    
    # 转为numpy
    X_train = X_train_woe.values.astype(np.float32)
    X_val = X_val_woe.values.astype(np.float32)
    X_test = X_test_woe.values.astype(np.float32)
    y_train = y_train.values
    y_val = y_val.values
    y_test = y_test.values
    
    # 添加特征名
    X_train_woe.columns = [f"feat_{i}" for i in range(X_train_woe.shape[1])]
    X_val_woe.columns = X_train_woe.columns
    X_test_woe.columns = X_train_woe.columns
    
    print(f"\n数据预处理完成：")
    print(f"  训练集：{len(X_train)}, 验证集：{len(X_val)}, 测试集：{len(X_test)}")
    print(f"  自适应阈值：low={adaptive_thresholds[0]:.3f}, mid={adaptive_thresholds[1]:.3f}")
    print(f"  自适应K值：base={adaptive_params['k_base']}, high={adaptive_params['k_high']}")
    
    return (X_train, X_val, X_test, y_train, y_val, y_test,
            mask_train, mask_val, mask_test,
            local_train_info, local_val_info, local_test_info,
            global_missing_stats, pos_weight,
            X_train_missing, X_val_missing, X_test_missing,
            X_train_woe, X_val_woe, X_test_woe,
            adaptive_thresholds, adaptive_params, mech_prob)

# =====================================================================
#      第14部分：完整实验框架 (★★★ 包含SOTA缺失学习 ★★★)
# =====================================================================

def run_full_experiment_optimized(
    X_train, X_val, X_test, y_train, y_val, y_test,
    mask_val, mask_train, mask_test, local_train_info, local_val_info, local_test_info,
    global_missing_stats, seed, pos_weight,
    X_train_missing, X_val_missing, X_test_missing,
    X_train_woe, X_val_woe, X_test_woe,
    adaptive_thresholds, adaptive_params, mech_prob,
    output_dir='./results'
):
    """
    ★ v3.0完整实验框架（融合三大核心模块）
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("【GL-MA-RAG v3.0 完整实验】")
    print("="*70)
    
    # ========== 步骤1：SMOTE处理 ==========
    print("\n【步骤1】处理类别不平衡（SMOTE）...")
    print("-"*70)
    
    d = X_train.shape[1]
    X_aug = np.hstack([X_train, mask_train.astype(np.float32)])
    cat_idx = list(range(d, 2*d))
    smote = SMOTENC(categorical_features=cat_idx, k_neighbors=5, random_state=seed)
    X_aug_s, y_train_s = smote.fit_resample(X_aug, y_train)
    
    X_trains = X_aug_s[:, :d].astype(np.float32)
    mask_train_s = X_aug_s[:, d:].astype(np.float32)
    
    print(f"  原始样本: {len(X_train)}, SMOTE后: {len(X_trains)}")
    print(f"  类别分布: {np.bincount(y_train_s)}")
    
    # ========== 步骤2：训练代价敏感贝叶斯模型 ==========
    print("\n【步骤2】训练代价敏感贝叶斯先验模型...")
    print("-"*70)
    
    bayes_model = MissingnessAwareBayesianRiskPredictor(
        kappa0=5,
        lambda_miss=5,
        tau=0.5,
        conf_scale=12.0,
        min_kappa=2.0,
        fp_cost=1.0,
        fn_cost=5.0,           # ★ v3.0：风控应用中FN代价更高
        use_adaptive_cost=True  # ★ v3.0：启用动态代价调整
    )
    
    bayes_model.fit_prior_model(
        X_train=X_train,
        y_train=y_train,
        mask_train=mask_train,
        prior_model_type="logr",
        seed=seed
    )
    print("  ✓ 贝叶斯先验模型训练完成")
    
    # ========== 步骤3：构建分层检索索引 ==========
    print("\n【步骤3】构建分层GL-MA-RAG检索索引...")
    print("-"*70)
    
    # ★ 改进：使用分层检索策略替代普通FAISS
    feature_cols = list(global_missing_stats["global_weights"].keys())
    global_weights = np.array([global_missing_stats["global_weights"][col] 
                               for col in feature_cols])
    
    # 创建分层检索器
    stratified_retriever = StratifiedRetriever(balance_ratio=0.5, min_per_class=3)
    stratified_retriever.build_stratified_index(
        X_val, mask_val, y_val,
        global_weights, adaptive_params['fusion_weights']
    )
    
    print("  ✓ 分层索引构建完成")
    
    # ========== 步骤4：验证集预测 + G-Mean阈值学习 ==========
    print("\n【步骤4】验证集预测与G-Mean阈值学习...")
    print("-"*70)
    
    val_pred_probs = []
    val_confidences = []
    
    for i in range(len(X_val)):
        # ★ 使用分层检索
        retrieve_result = stratified_retriever.retrieve_balanced(
            X_val[i], mask_val[i],
            local_val_info["sample_missing_ratio"][i],
            local_val_info["missing_level"][i],
            k_total=adaptive_params['k_base']
        )
        
        pred_result = bayes_model.predict_single(
            X_val[i], mask_val[i], retrieve_result,
            y_val, local_train_info["sample_missing_ratio"], X_val
        )
        
        val_pred_probs.append(pred_result["prediction"])
        val_confidences.append(pred_result["confidence"])
    
    val_pred_probs = np.array(val_pred_probs)
    val_confidences = np.array(val_confidences)
    
    # ★ v3.0：使用AdaptiveThresholdCalibrator学习G-Mean最优阈值
    calibrator = AdaptiveThresholdCalibrator(target_metric='gmean')
    calibrator.learn_thresholds(
        y_val, val_pred_probs,
        local_val_info["sample_missing_ratio"],
        val_confidences
    )
    
    # 将最优阈值写回模型
    bayes_model.decision_threshold = calibrator.global_optimal_threshold
    print(f"  ✓ 最优阈值: {calibrator.global_optimal_threshold:.4f}")
    
    # ========== 步骤5：测试集预测 ==========
    print("\n【步骤5】测试集批量预测...")
    print("-"*70)
    
    test_pred_probs, test_pred_labels, bayes_results = bayesian_predict_batch_with_results(
        X_test, mask_test,
        local_test_info["sample_missing_ratio"],
        local_test_info["missing_level"],
        bayes_model,
        stratified_retriever,  # ★ 改为使用分层检索器
        y_val,
        local_val_info["sample_missing_ratio"],
        None, None, adaptive_params,  # scaler_X和global_weights在retriever中已有
        local_test_info=local_test_info,
        store_results=True
    )
    
    # ========== 步骤6：计算完整评估指标 ==========
    print("\n【步骤6】计算完整评估指标...")
    print("-"*70)
    
    rag_metrics = calculate_metrics(
        y_test, test_pred_probs, 
        threshold=calibrator.global_optimal_threshold
    )
    
    # 添加平衡性指标
    pred_labels_final = (test_pred_probs >= calibrator.global_optimal_threshold).astype(int)
    recall_0 = recall_score(y_test, pred_labels_final, pos_label=0, zero_division=0)
    recall_1 = recall_score(y_test, pred_labels_final, pos_label=1, zero_division=0)
    gmean = np.sqrt(recall_0 * recall_1 + 1e-8)
    recall_imbalance = abs(recall_0 - recall_1)
    
    rag_metrics['Recall_Imbalance'] = round(recall_imbalance, 4)
    rag_metrics['G-Mean'] = round(gmean, 4)
    rag_metrics['Balanced_Acc'] = round((recall_0 + recall_1) / 2, 4)
    
    # 添加v3.0的额外指标
    try:
        rag_metrics['MCC'] = round(matthews_corrcoef(y_test, pred_labels_final), 4)
        rag_metrics['Kappa'] = round(cohen_kappa_score(y_test, pred_labels_final), 4)
    except:
        pass
    
    print(f"\n  ★ GL-MA-RAG v3.0 性能指标:")
    print(f"    AUC: {rag_metrics['AUC']:.4f}")
    print(f"    G-Mean: {rag_metrics['G-Mean']:.4f}")
    print(f"    Recall_0: {rag_metrics['Recall_0']:.4f}, Recall_1: {rag_metrics['Recall_1']:.4f}")
    print(f"    Recall_Imbalance: {rag_metrics['Recall_Imbalance']:.4f}")
    print(f"    MCC: {rag_metrics.get('MCC', 0):.4f}")
    
    # ========== 步骤7：可视化 ==========
    print("\n【步骤7】生成可视化...")
    print("-"*70)
    
    # 7.1 缺失模式分析
    try:
        viz = GLMARagVisualizer()
        fig1 = viz.plot_missing_pattern_analysis(X_train, mask_train, mech_prob, global_missing_stats)
        fig1.savefig(os.path.join(output_dir, f'missing_pattern_seed{seed}.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close(fig1)
        print("  ✓ 缺失模式分析")
    except Exception as e:
        print(f"  ✗ 缺失模式分析失败: {e}")
    
    # 7.2 置信区间分析（修复版）
    try:
        # ★ 修复：使用 __len__() 而不是直接 len() 判断
        if bayes_results is not None and len(bayes_results) > 0:  # 现在支持 len()
            results_df = bayes_results.to_dataframe()
            results_df['true_label'] = y_test
            results_df['is_correct'] = (pred_labels_final == y_test).astype(int)
            
            # 置信区间宽度分布
            ci_viz = ConfidenceIntervalVisualizer()
            fig_ci = ci_viz.plot_ci_width_distribution(results_df)
            fig_ci.savefig(os.path.join(output_dir, f'ci_width_distribution_seed{seed}.png'),
                          dpi=300, bbox_inches='tight')
            plt.close(fig_ci)
            
            # 准确性vs置信度
            fig_acc = PredictionAccuracyVisualizer.plot_prediction_accuracy_vs_confidence(results_df)
            fig_acc.savefig(os.path.join(output_dir, f'accuracy_vs_confidence_seed{seed}.png'),
                           dpi=300, bbox_inches='tight')
            plt.close(fig_acc)
            
            print("  ✓ 置信区间与准确性分析")
        else:
            print("  ⚠️ 无贝叶斯结果，跳过置信区间分析")
    
    except Exception as e:
        print(f"  ✗ 置信区间分析失败: {e}")
        traceback.print_exc()
    
    # 7.3 缺失模式与性能关系（修复版）
    try:
        # ★ 修复：使用 is not None 和 __len__() 判断
        if bayes_results is not None and len(bayes_results) > 0:
            results_df = bayes_results.to_dataframe()
            results_df['true_label'] = y_test
            
            missing_viz = MissingRateAnalysisVisualizer()
            fig_missing_perf = missing_viz.plot_performance_vs_missing(results_df)
            fig_missing_perf.savefig(os.path.join(output_dir, f'missing_vs_performance_seed{seed}.png'),
                                    dpi=300, bbox_inches='tight')
            plt.close(fig_missing_perf)
            
            print("  ✓ 缺失率与性能分析")
        else:
            print("  ⚠️ 无贝叶斯结果，跳过缺失率与性能分析")
    
    except Exception as e:
        print(f"  ✗ 缺失率与性能分析失败: {e}")
        traceback.print_exc()
    
    # ========== 整合所有结果 ==========
    print("\n" + "="*70)
    print("【实验完成】")
    print("="*70)
    print(f"✓ GL-MA-RAG v3.0 完整框架执行完毕")
    print(f"  可视化输出: {output_dir}")
    
    all_results = {}
    all_results['GL-MA-RAG'] = rag_metrics
    
    return all_results

# =====================================================================
#                   第15部分：主程序入口
# =====================================================================


# =====================================================================
def run_complete_experiment_with_multiple_runs():
    """
    多轮实验执行（含v3.0融合）
    """
    
    all_experiment_results = {}
    
    print(f"\n【GL-MA-RAG v3.0 多轮实验】")
    print(f"  实验次数: {EXPERIMENT_TIMES}")
    print(f"  缺失机制: {MECHS}")
    print(f"  缺失率: {MISSING_RATES}")
    print("="*70)
    
    # ✓ 三层循环
    for mech in MECHS:
        for mr in MISSING_RATES:
            for exp_id in range(1, EXPERIMENT_TIMES + 1):
                
                exp_key = f"Exp{exp_id}_{mech}_{mr*100:.0f}%"
                
                print(f"\n【{exp_key}】")
                print("-"*70)
                
                seed = RANDOM_SEED_BASE + exp_id
                
                try:
                    # 步骤1：加载数据
                    (X_train, X_val, X_test, y_train, y_val, y_test,
                     mask_train, mask_val, mask_test,
                     local_train_info, local_val_info, local_test_info,
                     global_missing_stats, pos_weight,
                     X_train_missing, X_val_missing, X_test_missing,
                     X_train_woe, X_val_woe, X_test_woe,
                     adaptive_thresholds, adaptive_params, mech_prob) = \
                        load_credit_data_optimized(DATA_PATH, seed, mr, mech)
                    
                    # ★ 步骤2：运行GL-MA-RAG v3.0完整框架
                    all_results = run_full_experiment_optimized(
                        X_train, X_val, X_test, y_train, y_val, y_test,
                        mask_val, mask_train, mask_test,
                        local_train_info, local_val_info, local_test_info,
                        global_missing_stats, seed, pos_weight,
                        X_train_missing, X_val_missing, X_test_missing,
                        X_train_woe, X_val_woe, X_test_woe,
                        adaptive_thresholds, adaptive_params, mech_prob,
                        output_dir=f'./results/seed_{seed}'
                    )
                    
                    # 步骤3：保存结果
                    all_experiment_results[exp_key] = {
                        'seed': seed,
                        'mech': mech,
                        'missing_rate': mr,
                        'results': all_results,
                        'exp_id': exp_id
                    }
                    
                    print(f"✓ {exp_key} 完成")
                    if 'GL-MA-RAG' in all_results:
                        print(f"  AUC: {all_results['GL-MA-RAG']['AUC']:.4f}, "
                              f"G-Mean: {all_results['GL-MA-RAG'].get('G-Mean', 0):.4f}")
                    
                except Exception as e:
                    print(f"✗ {exp_key} 失败: {e}")
                    traceback.print_exc()
                    all_experiment_results[exp_key] = None
    
    # 步骤4：汇总结果
    print("\n【汇总所有实验结果】")
    print("="*70)
    
    summary_df = aggregate_all_results(all_experiment_results)
    
    return summary_df, all_experiment_results

def aggregate_all_results(all_results):
    """聚合所有实验结果"""
    
    # 统计每个模型的多次实验平均性能
    model_performances = {}
    
    for exp_key, exp_data in all_results.items():
        if exp_data is None:
            continue
        
        results = exp_data['results']
        
        for model_name, metrics in results.items():
            if model_name not in model_performances:
                model_performances[model_name] = {
                    'auc': [],
                    'ks': [],
                    'auprc':[],
                    'recall_0':[],
                    'recall_1':[],
                    'accuracy': [],
                    'f1': []
                }
            
            model_performances[model_name]['auc'].append(metrics.get('AUC', 0))
            model_performances[model_name]['ks'].append(metrics.get('KS', 0))
            model_performances[model_name]['auprc'].append(metrics.get('AUPRC', 0))
            model_performances[model_name]['recall_0'].append(metrics.get('Recall_0', 0))
            model_performances[model_name]['recall_1'].append(metrics.get('Recall_1', 0))
            model_performances[model_name]['accuracy'].append(metrics.get('Accuracy', 0))
            model_performances[model_name]['f1'].append(metrics.get('F1', 0))
    
    # 计算均值和标准差
    summary = {}
    for model_name, perf in model_performances.items():
        summary[model_name] = {
            'AUC_mean': np.mean(perf['auc']),
            'AUC_std': np.std(perf['auc']),
            'KS_mean': np.mean(perf['ks']),
            'KS_std': np.std(perf['ks']),
            'AUPRC_mean': np.mean(perf['auprc']),
            'AUPRC_std': np.std(perf['auprc']),
            'Recall_0_mean': np.mean(perf['recall_0']),
            'Recall_0_std': np.std(perf['recall_0']),
            'Recall_1_mean': np.mean(perf['recall_1']),
            'Recall_1_std': np.std(perf['recall_1']),
            'Accuracy_mean': np.mean(perf['accuracy']),
            'Accuracy_std': np.std(perf['accuracy']),
            'F1_mean': np.mean(perf['f1']),
            'F1_std': np.std(perf['f1']),
        }
    
    print("\n【多次实验的汇总结果】")
    print("="*100)
    
    summary_df = pd.DataFrame(summary).T
    summary_df = summary_df.sort_values('AUC_mean', ascending=False)
    
    print(summary_df.to_string())
    
    # 保存到Excel
    # export_results_to_excel_advanced
    export_multiple_experiment_results_to_excel(all_results, summary_df, "multi_experiment_results.xlsx")
    
    return summary_df

# =====================================================================
#      Excel导出函数：支持多次实验结果聚合
# =====================================================================


def export_multiple_experiment_results_to_excel(all_experiment_results, summary_input, 
                                               output_path="multi_experiment_results.xlsx"):
    """
    导出多次实验结果到Excel（多Sheet）
    
    参数：
        all_experiment_results: 所有实验结果字典
        summary_input: 汇总结果（可以是dict或DataFrame）
        output_path: 输出文件路径
    """
    
    
    # ✓ 修改：兼容dict和DataFrame
    if isinstance(summary_input, dict):
        summary_df = pd.DataFrame(summary_input).T
        summary_df = summary_df.sort_values('AUC_mean', ascending=False)
    elif isinstance(summary_input, pd.DataFrame):
        summary_df = summary_input.copy()
    else:
        raise TypeError(f"summary_input必须是dict或DataFrame，得到{type(summary_input)}")
    
    wb = Workbook()
    wb.remove(wb.active)
    
    # ===== 样式定义 =====
    header_fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
    header_font = Font(bold=True, color="FFFFFF", size=12)
    
    highlight_fill = PatternFill(start_color="FFC000", end_color="FFC000", fill_type="solid")
    highlight_font = Font(bold=True, size=11)
    
    subheader_fill = PatternFill(start_color="D9E1F2", end_color="D9E1F2", fill_type="solid")
    subheader_font = Font(bold=True, size=11)
    
    border = Border(
        left=Side(style='thin'),
        right=Side(style='thin'),
        top=Side(style='thin'),
        bottom=Side(style='thin')
    )
    
    center_align = Alignment(horizontal='center', vertical='center', wrap_text=True)
    
    # ===== Sheet 0: 汇总统计 =====
    print("  【Sheet 0】生成汇总统计...")
    ws_summary = wb.create_sheet("Summary", 0)
    
    # 标题
    ws_summary['A1'] = "多次实验汇总结果"
    ws_summary['A1'].font = Font(bold=True, size=14)
    ws_summary.merge_cells('A1:H1')
    
    # 导出时间
    ws_summary['A2'] = f"导出时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    ws_summary['A2'].font = Font(italic=True, size=10)
    
    # 实验配置
    ws_summary['A4'] = "实验配置:"
    ws_summary['A4'].font = Font(bold=True, size=11)
    
    ws_summary['A5'] = f"总实验次数: {len(all_experiment_results)}"
    ws_summary['A6'] = f"缺失机制: {', '.join(MECHS)}"
    ws_summary['A7'] = f"缺失率: {', '.join([f'{mr*100:.0f}%' for mr in MISSING_RATES])}"
    ws_summary['A8'] = f"每组实验次数: {EXPERIMENT_TIMES}"
    
    # 空一行
    ws_summary['A10'] = "模型性能汇总（均值 ± 标准差）"
    ws_summary['A10'].font = Font(bold=True, size=12)
    ws_summary['A10'].fill = subheader_fill
    
    # 写入汇总表
    headers = ['Model'] + list(summary_df.columns)
    ws_summary.append(headers)
    
    # 格式化标题
    for cell in ws_summary[11]:
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = center_align
        cell.border = border
    
    # 写入数据
    for idx, (model_name, row) in enumerate(summary_df.iterrows(), start=12):
        row_data = [model_name] + [row[col] for col in summary_df.columns]
        ws_summary.append(row_data)
        
        ws_row = ws_summary[idx]
        
        # 高亮GL-MA-RAG
        if model_name == 'GL-MA-RAG':
            for cell in ws_row:
                cell.fill = highlight_fill
                cell.font = highlight_font
        
        # 格式化
        for col_idx, cell in enumerate(ws_row):
            cell.border = border
            cell.alignment = center_align
            
            # 数字列：4位小数
            if col_idx > 0:
                cell.number_format = '0.0000'
    
    # ✓ 修复：调整列宽
    ws_summary.column_dimensions['A'].width = 20
    for col_idx in range(2, len(headers) + 1):  # ✓ 使用列号
        ws_summary.column_dimensions[get_column_letter(col_idx)].width = 14
    
    # ===== Sheet 1: 详细实验结果 =====
    print("  【Sheet 1】生成详细实验结果...")
    ws_detail = wb.create_sheet("Detailed Results", 1)
    
    # 标题
    headers_detail = ['Experiment ID', 'Missing Mechanism', 'Missing Rate', 'Seed', 
                     'Model', 'AUC', 'KS','AUPRC', 'Accuracy', 'Recall_0', 'Recall_1', 'Precision', 'F1']
    ws_detail.append(headers_detail)
    
    # 格式化标题
    for cell in ws_detail[1]:
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = center_align
        cell.border = border
    
    # 写入数据
    row_num = 2
    for exp_key, exp_data in all_experiment_results.items():
        if exp_data is None:
            continue
        
        exp_id = exp_data['exp_id']
        mech = exp_data['mech']
        missing_rate = exp_data['missing_rate']
        seed = exp_data['seed']
        results = exp_data['results']
        
        for model_name, metrics in results.items():
            row_data = [
                exp_id,
                mech,
                f"{missing_rate*100:.0f}%",
                seed,
                model_name,
                metrics.get('AUC', 0),
                metrics.get('KS', 0),
                metrics.get('AUPRC', 0),
                metrics.get('Accuracy', 0),
                metrics.get('Recall_0', 0),
                metrics.get('Recall_1', 0),
                metrics.get('Precision', 0),
                metrics.get('F1', 0)
            ]
            ws_detail.append(row_data)
            
            ws_row = ws_detail[row_num]
            
            # 高亮GL-MA-RAG
            if model_name == 'GL-MA-RAG':
                for cell in ws_row:
                    cell.fill = highlight_fill
                    cell.font = highlight_font
            
            # 格式化
            for col_idx, cell in enumerate(ws_row):
                cell.border = border
                cell.alignment = center_align
                
                # 数字列
                if col_idx > 3:
                    cell.number_format = '0.0000'
            
            row_num += 1
    
    # ✓ 修复：调整列宽
    ws_detail.column_dimensions['A'].width = 12
    ws_detail.column_dimensions['B'].width = 18
    ws_detail.column_dimensions['C'].width = 12
    ws_detail.column_dimensions['D'].width = 8
    ws_detail.column_dimensions['E'].width = 18
    for col_idx in range(6, 12):  # ✓ F到L列
        ws_detail.column_dimensions[get_column_letter(col_idx)].width = 12
    
    # ===== Sheet 2: 按机制分类 =====
    print("  【Sheet 2】生成按机制分类...")
    ws_by_mech = wb.create_sheet("By Mechanism", 2)
    
    ws_by_mech['A1'] = "按缺失机制分类统计"
    ws_by_mech['A1'].font = Font(bold=True, size=13)
    ws_by_mech['A1'].fill = subheader_fill
    
    row_num = 3
    for mech in MECHS:
        # 机制标题
        ws_by_mech[f'A{row_num}'] = f"缺失机制: {mech}"
        ws_by_mech[f'A{row_num}'].font = subheader_font
        ws_by_mech[f'A{row_num}'].fill = subheader_fill
        row_num += 1
        
        # 该机制下的所有实验
        mech_experiments = {k: v for k, v in all_experiment_results.items() if v and v['mech'] == mech}
        
        if not mech_experiments:
            ws_by_mech[f'A{row_num}'] = "暂无数据"
            row_num += 1
            continue
        
        # 小标题
        headers_mech = ['Experiment', 'Missing Rate', 'Model', 'AUC', 'KS','AUPRC','Recall_0','Recall_1', 'Accuracy', 'F1']
        row_num += 1
        
        for col_idx, header in enumerate(headers_mech):
            cell = ws_by_mech[f'{get_column_letter(col_idx+1)}{row_num}']
            cell.value = header
            cell.fill = header_fill
            cell.font = header_font
            cell.border = border
        
        row_num += 1
        
        # 数据
        for exp_key, exp_data in sorted(mech_experiments.items()):
            for model_name, metrics in exp_data['results'].items():
                row_data = [
                    exp_key,
                    f"{exp_data['missing_rate']*100:.0f}%",
                    model_name,
                    f"{metrics.get('AUC', 0):.4f}",
                    f"{metrics.get('KS', 0):.4f}",
                    f"{metrics.get('AUPRC', 0):.4f}",
                    f"{metrics.get('Recall_0', 0):.4f}",
                    f"{metrics.get('Recall_1', 0):.4f}",
                    f"{metrics.get('Accuracy', 0):.4f}",
                    f"{metrics.get('F1', 0):.4f}"
                ]
                
                for col_idx, value in enumerate(row_data):
                    cell = ws_by_mech[f'{get_column_letter(col_idx+1)}{row_num}']
                    cell.value = value
                    cell.border = border
                    cell.alignment = center_align
                
                row_num += 1
        
        row_num += 1  # 间隔
    
    # ✓ 修复：调整列宽
    for col_idx in range(1, 8):  # A到G列
        ws_by_mech.column_dimensions[get_column_letter(col_idx)].width = 14
    
    # ===== Sheet 3: 按缺失率分类 =====
    print("  【Sheet 3】生成按缺失率分类...")
    ws_by_rate = wb.create_sheet("By Missing Rate", 3)
    
    ws_by_rate['A1'] = "按缺失率分类统计"
    ws_by_rate['A1'].font = Font(bold=True, size=13)
    ws_by_rate['A1'].fill = subheader_fill
    
    row_num = 3
    for mr in MISSING_RATES:
        # 缺失率标题
        ws_by_rate[f'A{row_num}'] = f"缺失率: {mr*100:.0f}%"
        ws_by_rate[f'A{row_num}'].font = subheader_font
        ws_by_rate[f'A{row_num}'].fill = subheader_fill
        row_num += 1
        
        # 该缺失率下的所有实验
        rate_experiments = {k: v for k, v in all_experiment_results.items() 
                           if v and abs(v['missing_rate'] - mr) < 1e-6}
        
        if not rate_experiments:
            ws_by_rate[f'A{row_num}'] = "暂无数据"
            row_num += 1
            continue
        
        # 小标题
        headers_rate = ['Experiment', 'Mechanism', 'Model', 'AUC', 'KS','AUPRC','Recall_0','Recall_1', 'Accuracy', 'F1']
        row_num += 1
        
        for col_idx, header in enumerate(headers_rate):
            cell = ws_by_rate[f'{get_column_letter(col_idx+1)}{row_num}']
            cell.value = header
            cell.fill = header_fill
            cell.font = header_font
            cell.border = border
        
        row_num += 1
        
        # 数据
        for exp_key, exp_data in sorted(rate_experiments.items()):
            for model_name, metrics in exp_data['results'].items():
                row_data = [
                    exp_key,
                    exp_data['mech'],
                    model_name,
                    f"{metrics.get('AUC', 0):.4f}",
                    f"{metrics.get('KS', 0):.4f}",
                    f"{metrics.get('AUPRC', 0):.4f}",
                    f"{metrics.get('Recall_0', 0):.4f}",
                    f"{metrics.get('Recall_1', 0):.4f}",
                    f"{metrics.get('Accuracy', 0):.4f}",
                    f"{metrics.get('F1', 0):.4f}"
                ]
                
                for col_idx, value in enumerate(row_data):
                    cell = ws_by_rate[f'{get_column_letter(col_idx+1)}{row_num}']
                    cell.value = value
                    cell.border = border
                    cell.alignment = center_align
                
                row_num += 1
        
        row_num += 1
    
    # ✓ 修复：调整列宽
    for col_idx in range(1, 8):  # A到G列
        ws_by_rate.column_dimensions[get_column_letter(col_idx)].width = 14
    
    # ===== Sheet 4: 模型性能排名 =====
    print("  【Sheet 4】生成模型性能排名...")
    ws_ranking = wb.create_sheet("Model Ranking", 4)
    
    ws_ranking['A1'] = "模型性能综合排名"
    ws_ranking['A1'].font = Font(bold=True, size=13)
    ws_ranking['A1'].fill = subheader_fill
    
    # 按AUC_mean排序
    ranking_df = summary_df.sort_values('AUC_mean', ascending=False).copy()
    ranking_df.insert(0, 'Rank', range(1, len(ranking_df) + 1))
    
    headers_ranking = list(ranking_df.columns)
    ws_ranking.append(headers_ranking)
    
    for cell in ws_ranking[1]:
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = center_align
        cell.border = border
    
    for idx, (model_name, row) in enumerate(ranking_df.iterrows(), start=2):
        row_data = list(row.values)
        ws_ranking.append(row_data)
        
        ws_row = ws_ranking[idx]
        
        # 高亮Top 3
        if idx <= 3:
            for cell in ws_row:
                cell.fill = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
                cell.font = Font(bold=True)
        
        # 高亮GL-MA-RAG
        if model_name == 'GL-MA-RAG':
            for cell in ws_row:
                cell.fill = highlight_fill
                cell.font = highlight_font
        
        # 格式化
        for col_idx, cell in enumerate(ws_row):
            cell.border = border
            cell.alignment = center_align
            
            if col_idx > 0:
                cell.number_format = '0.0000'
    
    # ✓ 修复：使用Excel列号（A、B、C等）而不是DataFrame列名
    for col_idx in range(len(headers_ranking)):
        col_letter = get_column_letter(col_idx + 1)  # ✓ 转换为列号
        
        if col_idx == 0:  # 第一列
            ws_ranking.column_dimensions[col_letter].width = 8
        elif col_idx == 1:  # 第二列（模型名）
            ws_ranking.column_dimensions[col_letter].width = 20
        else:
            ws_ranking.column_dimensions[col_letter].width = 14
    
    # ===== Sheet 5: 性能统计分析 =====
    print("  【Sheet 5】生成性能统计分析...")
    ws_stats = wb.create_sheet("Statistical Analysis", 5)
    
    ws_stats['A1'] = "性能统计分析"
    ws_stats['A1'].font = Font(bold=True, size=13)
    ws_stats['A1'].fill = subheader_fill
    
    # 各指标的统计信息
    metrics_list = [col for col in summary_df.columns if '_mean' in col]
    
    row_num = 3
    for metric in metrics_list:
        if metric not in summary_df.columns:
            continue
        
        ws_stats[f'A{row_num}'] = f"{metric}的统计信息"
        ws_stats[f'A{row_num}'].font = subheader_font
        row_num += 1
        
        # 统计数据
        metric_values = summary_df[metric].values
        
        stats_data = [
            ['统计指标', '值'],
            ['样本数', len(metric_values)],
            ['均值', f"{np.mean(metric_values):.6f}"],
            ['中位数', f"{np.median(metric_values):.6f}"],
            ['标准差', f"{np.std(metric_values):.6f}"],
            ['最小值', f"{np.min(metric_values):.6f}"],
            ['最大值', f"{np.max(metric_values):.6f}"],
            ['四分位数(Q1)', f"{np.percentile(metric_values, 25):.6f}"],
            ['四分位数(Q3)', f"{np.percentile(metric_values, 75):.6f}"],
        ]
        
        for stat_row in stats_data:
            ws_stats.append(stat_row)
            ws_row = ws_stats[row_num]
            
            # 格式化标题行
            if stat_row == stats_data[0]:
                for cell in ws_row:
                    cell.fill = header_fill
                    cell.font = header_font
                    cell.border = border
            else:
                for cell in ws_row:
                    cell.border = border
            
            row_num += 1
        
        row_num += 1
    
    # ✓ 修复：使用列号
    ws_stats.column_dimensions['A'].width = 20
    ws_stats.column_dimensions['B'].width = 15
    
    # 保存文件
    try:
        wb.save(output_path)
        
        print(f"\n✓ 多次实验结果已导出！")
        print(f"  文件路径: {output_path}")
        print(f"  包含内容:")
        print(f"    ├─ Sheet 1: Summary (汇总统计)")
        print(f"    ├─ Sheet 2: Detailed Results (详细结果)")
        print(f"    ├─ Sheet 3: By Mechanism (按机制分类)")
        print(f"    ├─ Sheet 4: By Missing Rate (按缺失率分类)")
        print(f"    ├─ Sheet 5: Model Ranking (模型排名)")
        print(f"    └─ Sheet 6: Statistical Analysis (统计分析)")
        print("="*70)
        
    except Exception as e:
        print(f"\n✗ Excel导出失败: {e}")
        print("  尝试备用方案：导出为CSV文件...")
        
        # 备用方案：导出为CSV
        try:
            summary_df.to_csv(output_path.replace('.xlsx', '.csv'))
            print(f"✓ CSV导出成功: {output_path.replace('.xlsx', '.csv')}")
        except:
            print(f"✗ CSV导出也失败了")

