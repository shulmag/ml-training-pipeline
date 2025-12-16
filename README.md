# ⚠️ LICENSE NOTICE

**This code is for viewing and portfolio demonstration purposes only.**

Copyright (c) 2025 Gil Shulman. All Rights Reserved.

You may NOT use, copy, modify, or distribute this code without explicit written permission.
See LICENSE file for full terms.

---

# ML Training Pipeline

### Automated ML training and deployment infrastructure for production pricing models.

This package contains:
- A means to access various models by name and version numbers. Largely under the ```ficc.models``` sub-module.
- A means to query/cache data from the server and process it for training and/or evaluation. Largely under the ```ficc.data``` sub-module.
- A variety of utility functions to aide in the training and evaluation process. Largely under the ```ficc.utils``` sub-module.
- Deployoment scripts used for automated training. Relevant files: ```automated_training*.py```, ```*_deployment.sh```, ```*_model.py```
    - The directory `trained_models/` contains each trained model metadata
    - The directory `training_logs/` print output during the procedures of training, and uploading the model to VertexAI

### Installation
```pip install . [-upgrade]```

### Requirements
To install the required Python packages for running the data package, use the following command:

``` pip install -r requirements_py310.txt ```

### API

#### Data package
The main driver for the ficc data processing package can be imported as folows ```from ficc.data.process_data import process_data``` 

The process data method takes 6 required and one optional parameter

- ```Query```  A query that will be used to fetch data from BigQuery.
- ```BigQuery Client```
- ```num_trades_in_history``` the sequence length of the trade history can take 32 as its maximum value. 
- ``` num_features_for_each_trade_in_history``` The number of features that the trade history contains.
- File path to save the raw data grabbed from BigQuery. 
- The yield curve to use acceptable options ```S&P``` or ```ficc```
- ```training_features``` A list containing the features that will be used for training. This is an optional parameter


### Example
An example of each API is available [here](https://github.com/Ficc-ai/ficc_python/blob/main/example.py).


### Automated Training
The yield spread model trains on the `yield_spread_model_training_vm` and the dollar price model trains on the `dollar_price_model_training_vm`. Both [VM instances](https://console.cloud.google.com/compute/instances?authuser=1&project=eng-reactor-287421&supportedpurview=project&tab=instances) are automatically switched on at 2:30am every day between Monday and Friday using the `automated-training-schedule` configured in the [Instance Schedules](https://console.cloud.google.com/compute/instances/instanceSchedules?authuser=1&project=eng-reactor-287421&supportedpurview=project&tab=instanceSchedules). The training and uploading to VertexAI is executed with a cron job. The cron job on the `yield_spread_model_training_vm` is

```45 10 * * 1-5 sh /home/user/ficc_python/yield_spread_deployment.sh >> /home/user/training_logs/yield_spread_training_$(TZ=America/New_York date +\%Y-\%m-\%d).log 2>&1```

and the cron job on the `dollar_price_model_training_vm` is

```45 10 * * 1-5 sh /home/user/ficc_python/dollar_price_deployment.sh >> /home/user/training_logs/dollar_price_training_$(TZ=America/New_York date +\%Y-\%m-\%d).log 2>&1```.

More details [here](https://www.notion.so/Daily-Model-Deployment-Process-d055c30e3c954d66b888015226cbd1a8).
