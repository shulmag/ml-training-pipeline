# Vertex AI Pipeline Migration Plan

## Overview

This document outlines the migration plan to update the Vertex AI Pipeline (`vertexai-pipeline-development` branch) with all changes from the current production VM-based pipeline (`main` branch). The Vertex AI pipeline is approximately 2 years old and needs significant updates to match the current production system.

## Major Changes Required

### 1. Core Function Migration

#### Main Entry Point
**Current Production**: `automated_training.auxiliary_functions.train_save_evaluate_model()`
- **Key Feature**: Returns boolean for traffic switching decision
- **Exit Codes**: Uses `SWITCH_TRAFFIC_EXIT_CODE=10` and `DO_NOT_SWITCH_TRAFFIC_EXIT_CODE=11`
- **Decision Logic**: Compares MAE between newly trained and currently deployed models
- **Email Integration**: Sends detailed results with traffic switching decision

**Old Vertex AI**: `automated_training_auxiliary_functions.train_save_evaluate_model()`  
- Missing the traffic switching logic
- Simpler email functionality
- No exit code system

**Migration Required**: Update Vertex AI components to use new decision logic and exit code system.

#### Smart Traffic Switching Logic
**Current Production** (`auxiliary_functions.py:1222-1242`):
```python
newly_trained_model_mae = current_date_data_current_date_model_result_df.loc[ROW_NAME_DETERMINING_MODEL_SWITCH, 'Mean Absolute Error']
currently_deployed_model_mae = current_date_data_previous_business_date_model_result_df.loc[ROW_NAME_DETERMINING_MODEL_SWITCH, 'Mean Absolute Error']
switch_traffic = newly_trained_model_mae <= currently_deployed_model_mae
```

**Old Vertex AI**: No automated traffic switching decision logic

**Migration Required**: Implement the MAE comparison logic in model training component.

### 2. Model Architecture Updates

#### Yield Spread with Similar Trades Model
**Current Production**: `yield_spread_with_similar_trades` is now a primary model type
- New model implementation in `automated_training.yield_with_similar_trades_model.py`
- Additional features: `similar_trade_history`, `recent_5_year_mat`
- Separate BigQuery table: `trade_history_same_issue_5_yr_mat_bucket_1_materialized`


**Migration Required**: 
- Add similar trades model support throughout all pipeline components
- Update data processing to handle similar trades features
- Create separate component variant or extend existing ones

#### Model Naming Convention
**Current Production**: Unified model naming with zip files
- `model_dollar_price_v2.zip`
- `model_similar_trades_v2.zip`  
- Function: `get_model_zip_filename()` with proper suffix handling

**Old Vertex AI**: Older naming conventions

**Migration Required**: Update model saving/loading logic in model training and deployment components.

### 3. Configuration Management

#### New Variables in `auxiliary_variables.py`
**Current Production** - New Variables:
```python
EMAIL_RECIPIENTS_FOR_LOGS = ['jesse@ficc.ai', 'gil@ficc.ai', 'mitas@ficc.ai']
TRAINING_LOGS_DIRECTORY = 'training_logs'
MAX_NUM_WEEK_DAYS_IN_THE_PAST_TO_CHECK = 10  # was MAX_NUM_BUSINESS_DAYS_IN_THE_PAST_TO_CHECK
USE_END_OF_DAY_YIELD_CURVE_COEFFICIENTS = True
ROW_NAME_DETERMINING_MODEL_SWITCH = 'Entire set'  # Critical for traffic switching
```

**Migration Required**: Update all component variable imports and usage.

#### Model-to-Folder Mapping
**Current Production**:
```python
MODEL_NAME_TO_ARCHIVED_MODEL_FOLDER = {
    'yield_spread': 'yield_spread_model',
    'dollar_price': 'dollar_price_models',
    'yield_spread_with_similar_trades': 'yield_spread_with_similar_trades_model'
}
```

**Migration Required**: Update model deployment component with new folder structure.

### 4. Data Processing Enhancements

#### Query Feature Extensions  
**Current Production** - New Features:
```python
ADDITIONAL_QUERY_FEATURES_FOR_YIELD_SPREAD_WITH_SIMILAR_TRADES_MODEL = ['recent_5_year_mat']
```

**Current Production** - Enhanced Data Validation:
- `check_no_duplicate_rtrs_control_numbers()` function
- Improved error handling for data quality issues

**Migration Required**: Update data processing component with new features and validation.

#### Yield Curve Integration
**Current Production**: New yield curve utilities
- `ficc.utils.get_treasury_rate` - treasury rate integration
- `ficc.utils.yc_data.add_yield_curve` - enhanced yield curve processing

**Migration Required**: Update data processing component to use new yield curve methods.

### 5. Enhanced Email System

#### Multi-Table Email Reports
**Current Production** (`auxiliary_functions.py:1085-1107`):
```python
def send_results_email_multiple_tables(df_list: list, text_list: list, model_train_date: str, recipients: list, model: str, intro_text: str = '') -> str:
    # Sends detailed HTML emails with multiple result tables
    # Includes traffic switching decision explanation
    # Provides training transparency info
```

**Old Vertex AI**: Basic single-table email functionality

**Migration Required**: 
- Update log compilation component with enhanced email templates
- Add traffic switching decision explanations
- Include training metadata and transparency reporting

#### Email Recipient Management
**Current Production**: Differentiated recipient lists
- `EMAIL_RECIPIENTS_FOR_LOGS` - technical team
- `EMAIL_RECIPIENTS` - broader stakeholder list

**Migration Required**: Update email logic in all components.

### 6. Error Handling & Production Features

#### GPU Setup Enhancement
**Current Production** (`auxiliary_functions.py:101-124`):
- Automatic CUDA driver installation if GPU not detected
- Improved memory growth configuration
- Better error handling for GPU setup failures

**Old Vertex AI**: Basic GPU setup

**Migration Required**: Update base training image with enhanced GPU setup.

#### File Management
**Current Production**: 
- Enhanced pickle file management with `USE_PICKLED_DATA` flag
- Better temporary file cleanup
- Improved error handling for file operations

**Migration Required**: Update all components with better file handling.

## Component-Specific Migration Tasks

### Data Processing Component

**Priority**: High

**Tasks**:
1. Update `get_new_data()` to support `yield_spread_with_similar_trades` model
2. Add similar trades BigQuery table handling
3. Implement new data validation functions
4. Update feature processing for new fields
5. Add treasury rate integration
6. Implement enhanced yield curve processing

**Files to Update**:
- `src/data_processing_component/main.py`
- `src/data_processing_component/automated_training_auxiliary_functions.py`
- `src/data_processing_component/automated_training_auxiliary_variables.py`

### Model Training Component

**Priority**: Critical

**Tasks**:
1. **Implement traffic switching logic** - Core business requirement
2. Add support for `yield_spread_with_similar_trades` model
3. Update `train_save_evaluate_model()` to return boolean decision
4. Implement exit code system for component orchestration
5. Update model architecture definitions
6. Add enhanced error handling and GPU setup
7. Implement new email reporting system

**Files to Update**:
- `src/model_training_component/main.py`
- `src/model_training_component/automated_training_auxiliary_functions.py`
- Add `src/model_training_component/yield_with_similar_trades_model.py`
- Add `src/model_training_component/exit_codes.py`

### Model Deployment Component

**Priority**: High

**Tasks**:
1. **Update to conditionally deploy based on training component output**
2. Add support for archived model storage (when MAE is worse)
3. Update model naming conventions
4. Implement proper folder structure for different model types
5. Add traffic switching implementation for Vertex AI endpoints
6. Update deployment scripts for new model types

**Files to Update**:
- `src/model_deployment_component/main.py`

### Log Compilation Component  

**Priority**: Medium

**Tasks**:
1. Update email templates with traffic switching information
2. Add multiple table support for detailed reporting
3. Implement recipient management (logs vs general recipients)
4. Add training transparency reporting
5. Include model comparison metadata

**Files to Update**:
- `src/log_compilation_component/main.py`

### Archiving Component

**Priority**: Medium  

**Tasks**:
1. Update to handle conditional archiving based on model performance
2. Add proper folder management for different archive scenarios
3. Implement cleanup logic for successful vs archived models

**Files to Update**:
- `src/archiving_component/main.py`

## Infrastructure Requirements

### Base Images Updates

**Training Base Image** (`training-pipeline/base-images/training-base/`):
1. Update ficc_python package with latest code
2. Add enhanced GPU setup with CUDA auto-install
3. Include new Python dependencies
4. Update environment variables

**Data Processing Base Image** (`training-pipeline/base-images/data-processing-base/`):
1. Update ficc_python package 
2. Add new data processing dependencies
3. Include treasury rate integration libraries

### Pipeline Orchestration Updates

**Main Pipeline** (`training-pipeline/main.py` and `training-pipeline/workbench.ipynb`):
1. Update component definitions for new model types
2. Add conditional logic for deployment based on training results
3. Implement proper error handling and component dependencies
4. Add monitoring and logging for traffic switching decisions

## Migration Strategy

### Phase 1: Core Logic Migration (High Priority)
1. **Traffic switching logic** - Critical for production parity
2. Update `auxiliary_functions.py` in model training component
3. Add exit code system
4. Test traffic switching decisions against historical data

### Phase 2: Model Support (High Priority)  
1. Add `yield_spread_with_similar_trades` model support
2. Update data processing for similar trades features
3. Test new model training and evaluation

### Phase 3: Enhanced Features (Medium Priority)
1. Update email system with multi-table reports
2. Add enhanced error handling and GPU management
3. Implement treasury rate integration
4. Update deployment conditional logic

### Phase 4: Testing & Validation (High Priority)
1. End-to-end pipeline testing with all model types
2. Validation against historical production results  
3. Performance testing of containerized vs VM performance
4. Traffic switching decision validation

### Phase 5: Production Deployment
1. CI/CD pipeline setup for automatic container builds
2. Monitoring and alerting setup
3. Rollback procedures
4. Documentation updates

## Risk Mitigation

### Critical Path Items
1. **Traffic switching logic must be 100% accurate** - Any deviation affects production model deployment
2. **Model performance parity** - Containerized training must produce identical results to VM training
3. **Error handling completeness** - Pipeline must handle all edge cases the current VM system handles

### Testing Requirements
1. **Backtesting**: Run pipeline on historical data and validate identical decisions
2. **Shadow deployment**: Run pipeline in parallel with VM system for validation period
3. **Model accuracy verification**: Ensure containerized models produce identical predictions

### Rollback Plan
1. VM system remains operational during migration
2. Gradual cutover with monitoring
3. Immediate rollback capability if discrepancies detected

## Success Metrics

1. **Functional Parity**: Pipeline makes identical traffic switching decisions as VM system
2. **Performance Parity**: Model accuracy within 0.001 MAE of VM system
3. **Operational Improvement**: Reduced training time and resource usage
4. **Reliability Improvement**: Better error handling and monitoring vs VM system

## Timeline Estimate

- **Phase 1**: 1-2 weeks (traffic switching logic)  
- **Phase 2**: 1-2 weeks (model support)
- **Phase 3**: 1-2 weeks (enhanced features)
- **Phase 4**: 2-3 weeks (testing & validation)
- **Phase 5**: 1 week (production deployment)

**Total**: 6-10 weeks for complete migration

This migration will modernize the training infrastructure while maintaining full production compatibility and introducing improved scalability, monitoring, and resource efficiency.