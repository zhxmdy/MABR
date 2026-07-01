
A comprehensive credit risk prediction framework under missing data, featuring information-theoretic weight design, multimodal deep fusion, Bayesian uncertainty quantification, and stratified retrieval.

## Project Structure

```
gl-ma-rag/
├── main.py                       # Entry point (run: python main.py)
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── results/                      # Output directory
└── mbar/                    # Package directory
    ├── __init__.py               # Package init (imports all modules)
    ├── config.py                 # Global configuration + matplotlib setup
    ├── metrics.py                # Evaluation metrics (AUC, G-Mean, etc.)
    ├── missing_injection.py      # MCAR/MAR/MNAR missing data injection
    ├── missing_analysis.py       # Pattern analysis + mechanism detection + thresholds
    ├── fusion_network.py         # MultimodalFusionNet + BayesianRetrievalModule
    ├── retrieval.py              # StratifiedRetriever + FAISS index construction
    ├── bayesian_predictor.py     # Cost-sensitive Bayesian risk predictor
    ├── visualization.py          # All visualization classes (CI, weights, dashboard)
    ├── experiment.py             # Data loading + full experiment orchestration
    ├── dimensionality.py         # PCA / UMAP / t-SNE dimensionality reduction
    ├── ablation.py               # Ablation study framework
    ├── models.py                 # Baseline models (SimpleRegression, Bayesian)
    ├── interpretability.py       # InterpretabilityAnalyzer + SHAP-like analysis
    └── bayesian_viz.py           # Bayesian posterior visualization suite
```

## Module Overview

| Module | Lines | Contents |
|--------|-------|----------|
| `config.py` | 147 | Imports, matplotlib config, DATA_PATH, RANDOM_SEED_BASE, MISSING_RATES, MECHS |
| `metrics.py` | 111 | `calculate_metrics()`, `learn_cost_sensitive_threshold()` |
| `missing_injection.py` | 236 | `inject_missing_new()`, `inject_missing()` |
| `missing_analysis.py` | 1513 | `analyze_global_missing_pattern_improved()`, `RobustMissingMechanismDetector`, `MissingMechanismClassifier`, `AdaptiveThresholdCalibrator` |
| `fusion_network.py` | 143 | `MultimodalFusionNet(nn.Module)`, `BayesianRetrievalModule` |
| `retrieval.py` | 299 | `StratifiedRetriever`, `build_gl_ma_rag_index_*`, `gl_ma_retrieve*` |
| `bayesian_predictor.py` | 548 | `MissingnessAwareBayesianRiskPredictor`, `ImbalanceAwareBayesianPredictionResult` |
| `visualization.py` | 1484 | `ConfidenceIntervalVisualizer`, `PredictionAccuracyVisualizer`, `ModelWeightVisualizer`, `ComprehensiveDashboard`, `ComparisonVisualizer` |
| `experiment.py` | 1061 | `load_credit_data_optimized()`, `run_full_experiment_optimized()`, `run_complete_experiment_with_multiple_runs()` |
| `dimensionality.py` | 571 | `PCAVisualizer`, `UMAPVisualizer`, `TSNEVisualizer`, `DimensionalityReductionComparison` |
| `ablation.py` | 720 | `AblationStudyV3_Revised`, `run_ablation_experiment()` |
| `models.py` | 209 | `SimpleRegressionModel`, `BayesianMissingDataModel` |
| `interpretability.py` | 1166 | `InterpretabilityAnalyzer`, `run_interpretability_analysis()` |
| `bayesian_viz.py` | 1228 | `BayesianVisualizationSuite`, `visualize_bayesian_results()` |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the framework
python main.py
```

## Usage Modes

Interactive menu with 6 options:

1. **Complete Multi-Round Experiment** - All model comparisons with multiple runs
2. **Single mbar v3.0 Demo** - Quick verification with interpretability analysis
3. **Ensemble Prediction Visualization** - Confidence intervals, model weights
4. **Dimensionality Reduction Analysis** - PCA / UMAP / t-SNE comparison
5. **Ablation Study** - Validate component contributions
6. **Exit**

## Configuration

Edit `gl_ma_rag/config.py` to configure:

```python
DATA_PATH = "path/to/your/dataset.csv"  # Input dataset
EXPERIMENT_TIMES = 1                      # Number of runs
RANDOM_SEED_BASE = 42                     # Random seed
MISSING_RATES = [0.0]                     # Missing rates
MECHS = ['MNAR']                          # MCAR / MAR / MNAR
```

## Core Algorithm

The mbar framework integrates three key innovations:

1. **Cost-Sensitive Bayesian Prediction** - Beta-Binomial conjugate model with adaptive FP/FN costs
2. **Stratified Retrieval** - Class-balanced FAISS retrieval ensuring minority class representation
3. **G-Mean Threshold Calibration** - Optimized decision threshold for imbalanced credit data

See `algorithm.tex` for complete pseudocode in KBS journal format.

## Requirements

- Python 3.8+
- CUDA GPU recommended for deep learning (MultimodalFusionNet, TabNet)
- See `requirements.txt` for full dependency list

## Citation

If you use this framework, please cite accordingly.

## License

Research purposes only.
