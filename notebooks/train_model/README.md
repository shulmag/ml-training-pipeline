# Model Training and Evaluation Framework

This directory contains tools for training and evaluating yield spread models, with comprehensive backtesting capabilities added in July 2025.

## Overview

`train_model.py` has evolved from a simple training script to a comprehensive framework that can:
1. Train new models on recent data
2. Evaluate historical models on any date
3. Run production-style backtesting across all historical models
4. Upload predictions to BigQuery for analysis

## Usage

### Training a New Model
```bash
# Train on the most recent day
python train_model.py

# Or run in background
nohup python -u train_model.py >> output.txt 2>&1 &
```

### Listing Available Models
```bash
python train_model.py --mode list
```

### Evaluating Historical Models
```bash
# Evaluate a specific model on a specific date
python train_model.py --mode evaluate --model-dates 2025-04-03 --test-dates 2025-04-04

# Evaluate multiple combinations
python train_model.py --mode evaluate --model-dates 2025-03-01 2025-03-15 --test-dates 2025-03-02 2025-03-16
```

### Production Backtesting
```bash
# Run full production evaluation (tests each model on all days it would have been in production)
python train_model.py --mode production-eval

# Run with date range (for testing or specific analysis)
python train_model.py --mode production-eval --start-date 2025-03-01 --end-date 2025-03-31

# Run in background with monitoring
nohup python -u train_model.py --mode production-eval --start-date 2025-03-01 --end-date 2025-03-31 >> prod_eval_output.txt 2>&1 &
tail -f prod_eval_output.txt
```

## Key Features Added (July 2025)

1. **Historical Model Evaluation**: Load any pre-trained model from GCS and evaluate it on any date
2. **Production Backtesting**: Simulate production deployment by testing each model on all days until the next model was trained
3. **RMSE Metrics**: Added RMSE alongside MAE for all evaluations
4. **BigQuery Integration**: Upload all predictions to sandbox table with metadata (model_train_date, days_since_training)
5. **Comprehensive Reporting**: Generate CSV summaries of model performance over time

## Output Files

- `production_evaluation_summary_YYYYMMDD_HHMMSS.csv`: Detailed results for each model-date combination
- `model_performance_summary_YYYYMMDD_HHMMSS.csv`: Aggregated statistics by model
- `predictions_*.csv`: Sample predictions for debugging (when using evaluate mode)

## BigQuery Tables

Predictions are uploaded to:
- Production training: `eng-reactor-287421.historic_predictions.historical_predictions_similar_trades_v2`
- Historical evaluation: `eng-reactor-287421.historic_predictions.historical_predictions_similar_trades_v2_sandbox`

Sandbox table includes additional fields:
- `model_train_date`: Which model was used for prediction
- `evaluation_mode`: 'historical' for backtesting
- `days_since_training`: How old the model was when making the prediction

## Important Notes

1. Ensure your data pickle file contains dates you want to test
2. Models are stored in `gs://automated_training/similar-trades-v2-model-YYYY-MM-DD/`
3. Encoders are loaded from `gs://automated_training/encoders_similar_trades.pkl`
4. The script uses significant memory (~32GB) when loading the full dataset

## Development History

- **Original (2025-01-21)**: Basic training script
- **July 2025 Updates**: Added comprehensive evaluation framework for Goldman Sachs analysis
  - Historical model evaluation
  - Production backtesting
  - RMSE metrics
  - Enhanced BigQuery integration
```




