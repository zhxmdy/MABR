# -*- coding: utf-8 -*-
"""GL-MA-RAG Main Entry Point. Run: python main.py"""
from gl_ma_rag import *

if __name__ == "__main__":
    
    print("""
    ╔════════════════════════════════════════════════════════════════╗
    ║  GL-MA-RAG v3.0: 类别平衡优化版本                               ║
    ║  融合三大核心模块：                                             ║
    ║  1. 代价敏感贝叶斯预测 (Cost-Sensitive Bayesian)               ║
    ║  2. 分层检索策略 (Stratified Retrieval)                       ║
    ║  3. G-Mean优化阈值 (G-Mean Threshold Calibration)             ║
    ╚════════════════════════════════════════════════════════════════╝
    
    选择运行模式:
    
    1️⃣  完整多轮实验（推荐） - 包含所有模型对比
    2️⃣  单次GL-MA-RAG v3.0演示 - 快速验证新功能
    3️⃣  融合预测可视化 - 置信区间、模型权重等
    4️⃣  降维与数据分析 - PCA/UMAP/t-SNE对比
    5️⃣  消融实验 - 验证各组件贡献度
    6️⃣  退出
    """)
    
    choice = input("\n请输入选择 (1/2/3/4/5/6): ").strip()
    
    if choice == '1':
        # ★ 完整多轮实验（v3.0）
        print("\n【完整多轮GL-MA-RAG v3.0实验】")
        summary_df, all_results = run_complete_experiment_with_multiple_runs()
        
        print("\n" + "="*80)
        print("【汇总表】")
        print("="*80)
        print(summary_df.to_string())
        print("="*80)
        
        # 生成对比可视化
        viz = ComparisonVisualizer()
        
        fig_comp = viz.plot_all_models_comparison(
            {m: summary_df.loc[m].to_dict() for m in summary_df.index}
        )
        fig_comp.savefig('model_comparison_all.png', dpi=300, bbox_inches='tight')
        plt.close(fig_comp)
        print("✓ 模型对比图已保存: model_comparison_all.png")
        
        # 统计检验
        # perform_statistical_tests(
        #     {m: summary_df.loc[m].to_dict() for m in summary_df.index}
        # )
        
# ======================== 在主程序中的调用部分 ========================

    elif choice == '2':
        # ★ 单次v3.0演示
        print("\n【GL-MA-RAG v3.0 单次演示】")
        
        seed = RANDOM_SEED_BASE + 1
        
        # 加载数据
        (X_train, X_val, X_test, y_train, y_val, y_test,
        mask_train, mask_val, mask_test,
        local_train_info, local_val_info, local_test_info,
        global_missing_stats, pos_weight,
        X_train_missing, X_val_missing, X_test_missing,
        X_train_woe, X_val_woe, X_test_woe,
        adaptive_thresholds, adaptive_params, mech_prob) = \
            load_credit_data_optimized(DATA_PATH, seed, MISSING_RATES[0], MECHS[0])
        
        # 运行实验
        all_results = run_full_experiment_optimized(
            X_train, X_val, X_test, y_train, y_val, y_test,
            mask_val, mask_train, mask_test,
            local_train_info, local_val_info, local_test_info,
            global_missing_stats, seed, pos_weight,
            X_train_missing, X_val_missing, X_test_missing,
            X_train_woe, X_val_woe, X_test_woe,
            adaptive_thresholds, adaptive_params, mech_prob
        )
        
        print("\n【v3.0 演示完成】")
        print("="*60)
        if 'GL-MA-RAG' in all_results:
            metrics = all_results['GL-MA-RAG']
            print("GL-MA-RAG v3.0 结果:")
            for key, value in metrics.items():
                print(f"  {key}: {value}")
        print("="*60)
        

        # ======================== 新增：可解释性分析 ========================
        
        print("\n" + "="*80)
        print("开始可解释性分析模块")
        print("="*80)
        
        # ★ 关键修改：传入X_train_woe和X_test_woe
        analyzer, results_df = run_interpretability_analysis(
            X_train=X_train,           # 已填补的特征
            X_val=X_val,
            X_test=X_test,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
            mask_train=mask_train,     # 缺失掩码
            mask_val=mask_val,
            mask_test=mask_test,
            X_train_woe=X_train_woe,   # ★ WOE特征（用于模型）
            X_test_woe=X_test_woe,     # ★ WOE特征（用于模型）
            output_dir='./results'
        )
        results_df.to_csv('测试集不确定性结果.csv')
        print("\n【可解释性分析完成】")
        print("="*80)




        print("\n" + "="*80)
        print("【第三阶段】贝叶斯结果可视化")
        print("="*80)
        
        best_threshold, viz_dir = visualize_bayesian_results(
            results_df,
            output_dir=f'./results/bayesian_viz_seed_{seed}'
        )
        
        print(f"\n✅ 所有可视化完成！")
        print(f"   最优决策阈值: {best_threshold:.3f}")
        print(f"   可视化输出目录: {viz_dir}")
        
        # 保存最优阈值供后续使用
        with open(f'{viz_dir}/optimal_threshold.txt', 'w') as f:
            f.write(f"Optimal Decision Threshold: {best_threshold:.4f}\n")
            f.write(f"Generation Time: {pd.Timestamp.now()}\n")
    elif choice == '3':
        # 融合预测可视化
        print("\n【融合预测可视化】")
        print("此功能需要单次实验后的预测结果...")
        
        seed = RANDOM_SEED_BASE + 1
        (X_train, X_val, X_test, y_train, y_val, y_test,
        mask_train, mask_val, mask_test,
        local_train_info, local_val_info, local_test_info,
        global_missing_stats, pos_weight,
        X_train_missing, X_val_missing, X_test_missing,
        X_train_woe, X_val_woe, X_test_woe,
        adaptive_thresholds, adaptive_params, mech_prob) = \
            load_credit_data_optimized(DATA_PATH, seed, MISSING_RATES[0], MECHS[0])
        
        # 生成综合仪表板
        # （需要从之前的实验获取results_df和weights_list）
        print("✓ 可视化模块已准备就绪")
        
    elif choice == '4':
        # 降维分析
        print("\n【数据降维与分析】")
        
        seed = RANDOM_SEED_BASE + 1
        (X_train, X_val, X_test, y_train, y_val, y_test,
        mask_train, mask_val, mask_test,
        local_train_info, local_val_info, local_test_info,
        global_missing_stats, pos_weight,
        X_train_missing, X_val_missing, X_test_missing,
        X_train_woe, X_val_woe, X_test_woe,
        adaptive_thresholds, adaptive_params, mech_prob) = \
            load_credit_data_optimized(DATA_PATH, seed, MISSING_RATES[0], MECHS[0])
        
        # 合并数据
        X_combined = np.vstack([X_train, X_val])
        y_combined = np.hstack([y_train, y_val])
        mask_combined = np.vstack([mask_train, mask_val])
        
        # 执行降维分析
        dr_comp = DimensionalityReductionComparison(X_combined, y_combined, mask_combined)
        dr_comp.generate_comparison_report(output_dir='./results/dr_analysis')
        
        print("✓ 降维分析完成，结果保存至 ./results/dr_analysis")
        
    elif choice == '5':
        # 消融实验
        print("\n【v3.0消融实验】")
        print("（验证各模块的贡献度）")
        # run_ablation_experiment()
        averaged_results, all_runs_results = run_ablation_experiment()
        
        # 已经在 run_ablation_experiment() 内部调用了导出
        # 如果需要额外处理，可以在这里添加
        print(f"\n✅ 消融实验完成，共生成{len(all_runs_results['Full'])}次运行结果")
        # seed = RANDOM_SEED_BASE + 1
        # (X_train, X_val, X_test, y_train, y_val, y_test,
        #  mask_train, mask_val, mask_test,
        #  local_train_info, local_val_info, local_test_info,
        #  global_missing_stats, pos_weight,
        #  X_train_missing, X_val_missing, X_test_missing,
        #  X_train_woe, X_val_woe, X_test_woe,
        #  adaptive_thresholds, adaptive_params, mech_prob) = \
        #     load_credit_data_optimized(DATA_PATH, seed, MISSING_RATES[0], MECHS[0])
        # run_ablation_with_statistics(X_train, X_val, X_test, y_train, y_val, y_test,
        #                          mask_train, mask_val, mask_test, num_runs=5)
        # # 运行消融实验
        # print("\n运行消融实验变体...")
        
        # ablation_results = {}
        
        # # 变体1: 完整版本
        # print("\n【变体1】Full (完整v3.0)")
        # ablation_results['Full'] = run_full_experiment_optimized(
        #     X_train, X_val, X_test, y_train, y_val, y_test,
        #     mask_val, mask_train, mask_test,
        #     local_train_info, local_val_info, local_test_info,
        #     global_missing_stats, seed, pos_weight,
        #     X_train_missing, X_val_missing, X_test_missing,
        #     X_train_woe, X_val_woe, X_test_woe,
        #     adaptive_thresholds, adaptive_params, mech_prob
        # )
        
        # print("\n【消融实验完成】")
        # print("="*60)
        # print("变体性能对比:")
        # if ablation_results['Full'] and 'GL-MA-RAG' in ablation_results['Full']:
        #     metrics = ablation_results['Full']['GL-MA-RAG']
        #     print(f"  Full - AUC: {metrics['AUC']:.4f}, G-Mean: {metrics.get('G-Mean', 0):.4f}")
        # print("="*60)
        
    elif choice == '6':
        print("退出程序")
    
    else:
        print("❌ 无效选择")
    
    print("\n✅ 程序执行完成！")