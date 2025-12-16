# @ Create date: 2024-03-22
# @ Modified date: 2024-06-07
echo "If there are errors, visit: https://www.notion.so/Daily-Model-Deployment-Process-d055c30e3c954d66b888015226cbd1a8"
echo "Search for warnings in the logs (even on a successful training procedure) and investigate"
echo "Set USE_PICKLED_DATA to True in auxiliary_functions.py to use saved data instead of re-running the query every time"

#!/bin/sh
who
HOME='/home/mitas'
TRAINED_MODELS_PATH="$HOME/trained_models/dollar_price_model"
MODEL="dollar_price"
TRAINING_LOG_PATH="$HOME/training_logs/retrain-dollar_price_training.log"

# Activate the virtual environment for Python3.10 (/usr/local/bin/python3.10) that contains all of the packages; to see all versions of Python use command `whereis python`
# If venv_py310 does not exist in `ficc_python/`, then in `ficc_python/` run `/usr/local/python3.10 -m venv venv_py310` and `source venv_py310/bin/activate` followed by `pip install -r requirements_py310.txt`
# NOTE: for sh script (which is different than bash script), we must use the '.' operator instead of 'source' to activate the virtual environment
. $HOME/ficc_python/venv_py310/bin/activate
python --version

DATE_STRINGS="2024-01-30 2024-01-31 2024-03-13 2024-03-15"

for DATE_STRING in $DATE_STRINGS; do
  DATE_WITH_YEAR=$(date -d "$DATE_STRING" +%Y-%m-%d)

  # Training the model
  python $HOME/ficc_python/automated_training_dollar_price_model.py $DATE_WITH_YEAR
  if [ $? -ne 0 ]; then
    echo "automated_training_dollar_price_model.py script failed with exit code $?"
    python $HOME/ficc_python/send_email_with_training_log.py $TRAINING_LOG_PATH $MODEL "Model training failed. See attached logs for more details."
    exit 1
  fi
  echo "Model trained"

  # Cleaning the logs to make more readable
  python $HOME/ficc_python/clean_training_log.py $TRAINING_LOG_PATH

  # Unzip model and uploading it to automated training bucket
  MODEL_NAME='dollar-model'-${DATE_WITH_YEAR}
  echo "Unzipping model $MODEL_NAME"
  gsutil cp -r gs://automated_training/model_dollar_price.zip $TRAINED_MODELS_PATH/model_dollar_price.zip
  unzip $TRAINED_MODELS_PATH/model_dollar_price.zip -d $TRAINED_MODELS_PATH/$MODEL_NAME
  if [ $? -ne 0 ]; then
    echo "Unzipping failed with exit code $?"
    python $HOME/ficc_python/send_email_with_training_log.py $TRAINING_LOG_PATH $MODEL "Unzipping model failed. See attached logs for more details."
    exit 1
  fi

  echo "Uploading model to bucket"
  gsutil cp -r $TRAINED_MODELS_PATH/$MODEL_NAME gs://automated_training
  if [ $? -ne 0 ]; then
    echo "Uploading model to bucket failed with exit code $?"
    python $HOME/ficc_python/send_email_with_training_log.py $TRAINING_LOG_PATH $MODEL "Uploading model to bucket failed. See attached logs for more details."
    exit 1
  fi
done

# Deactivate the virtual environment
deactivate
