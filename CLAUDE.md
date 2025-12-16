# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is Ficc AI's internal Python package for bond pricing models and automated training systems. The repository contains:
- Machine learning models for yield spread and dollar price predictions  
- Data processing pipelines for BigQuery data
- Automated training infrastructure using both VMs and Vertex AI Pipelines
- Deployment scripts for model serving on Vertex AI

## Key Commands

### Installation
```bash
# Install the ficc package in development mode
pip install . --upgrade

# Install dependencies (Python 3.10)
pip install -r requirements_py310.txt
```

### Running Tests
```bash
# Run pipeline component tests
cd training-pipeline
python -m pytest tests/
```

### Building and Deploying Models

#### VM-based Automated Training (Production)
```bash
# Yield spread model deployment
sh yield_spread_deployment.sh

# Dollar price model deployment  
sh dollar_price_deployment.sh

# Yield spread with similar trades
sh yield_spread_with_similar_trades_deployment.sh
```

#### Vertex AI Pipeline (Development)
```bash
# Build base Docker images
sh build-data-processing-base.sh
sh build-training-base.sh

# Build component images
cd training-pipeline/src
sh rebuild-all-components.sh

# Deploy cloud function for pipeline scheduling
cd training-pipeline
sh deploy_gcf_function.sh
```

## Architecture

### Core Package Structure (`ficc/`)
- `ficc.data`: Data processing and BigQuery integration
  - Main driver: `ficc.data.process_data.process_data()`
- `ficc.pricing`: Pricing models and predictions
- `ficc.utils`: Utility functions for training and evaluation
- `ficc.models`: Model definitions and versioning

### Automated Training Components
- `automated_training_auxiliary_functions.py`: Shared training utilities
- `automated_training_auxiliary_variables.py`: Configuration constants
- `*_model.py` files: Model training implementations
- `*_deployment.sh` scripts: Deployment automation

### Vertex AI Pipeline (`training-pipeline/`)
The pipeline consists of 5 main components:
1. **Data Processing**: Fetches from BigQuery, processes features, saves to GCS
2. **Model Training**: Trains models, evaluates MAE, saves artifacts
3. **Model Deployment**: Deploys trained models to Vertex AI for inference
4. **Archiving**: Archives previous model versions
5. **Log Compilation**: Aggregates logs and sends notification emails

## Important Considerations

### Data Processing
- The `process_data` method requires:
  - BigQuery query and client
  - `num_trades_in_history` (max 32)
  - `num_features_for_each_trade_in_history`
  - File path for raw data storage
  - Yield curve type: "S&P" or "ficc"

### Model Training
- Models are automatically trained daily at 2:30am via cron jobs on dedicated VMs
- Training logs are saved to `/home/mitas/training_logs/`
- Trained model metadata is stored in `trained_models/` directory

### Vertex AI Pipeline Notes
- Components run as containerized services on Kubeflow
- Base images must be pushed to Google Artifact Registry before component builds
- All data exchange between components happens via GCS artifacts
- Use standard output for logging to enable Google Cloud Logging visibility

## Environment Details
- Python version: 3.10
- Main branch: `main`
- Pipeline development branch: `vertexai-pipeline-development`
- VMs: `yield_spread_model_training_vm`, `dollar_price_model_training_vm`
- GCP Project: `eng-reactor-287421`