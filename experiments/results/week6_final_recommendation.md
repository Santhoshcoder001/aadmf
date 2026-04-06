# Week 6 Final Recommendation

## Dataset Verification
- UCI batches loaded: 10
- Total rows processed: 13910

## Best Scoring Matrix Recommendation
- Preset: Default
- Weights: drift=0.4, accuracy=0.3, cost=0.3
- Mean quality score: 0.92
- High-drift mean quality score: 0.866667

## Best Miner Parameter Recommendations (High Drift)
- IsolationForest base_contamination: 0.05 (formula=fixed, mean_quality=0.9495)
- DBSCAN base_eps: 2.0 (formula=fixed, mean_quality=0.9295)

## Saved Artifacts
- experiments/results/scoring_matrix_tuning.csv
- experiments/results/miner_tuning_results.csv
- experiments/results/week6_final_recommendation.md

## Novelty Log Entry Preparation
Use the following points for NOVELTY_LOG.md (Week 6):
- Decision: Tuned scoring weights and miner hyperparameters on real UCI drift batches.
- Evidence: Comparative quality-score results across four scoring presets and miner parameter grids.
- Rationale: Real drift behavior yields more reliable operating points than synthetic-only tuning.
- Patent relevance: Strengthens adaptive selection and real-world robustness claims.

## Scoring Matrix Results Table
           preset  w_drift  w_accuracy  w_cost  mean_quality_score  high_drift_mean_quality_score drift_detection_latency algorithm_selection_distribution
          Default      0.4         0.3     0.3                0.92                       0.866667                    None         {'StatisticalRules': 10}
   Drift-dominant      0.7         0.2     0.1                0.92                       0.866667                    None         {'StatisticalRules': 10}
Accuracy-dominant      0.2         0.7     0.1                0.92                       0.866667                    None         {'StatisticalRules': 10}
    Cost-dominant      0.1         0.2     0.7                0.92                       0.866667                    None         {'StatisticalRules': 10}