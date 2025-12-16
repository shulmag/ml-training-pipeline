# Description: Use `$ bash model_deployment.sh <MODEL_NAME>` to call this script. `MODEL_NAME` must be either "yield_spread_with_similar_trades" or "dollar_price". 
#              The below are the cron jobs for the yield spread with similar trades and dollar price models set up on their respective automated training VMs.
#              45 10 * * 1-5 bash ~/ficc_python/automated_training/model_deployment.sh dollar_price >> ~/training_logs/dollar_price_training_$(TZ=America/New_York date +\%Y-\%m-\%d).log 2>&1
#              45 10 * * 1-5 bash ~/ficc_python/automated_training/model_deployment.sh yield_spread_with_similar_trades >> ~/training_logs/yield_spread_with_similar_trades_training_$(TZ=America/New_York date +\%Y-\%m-\%d).log 2>&1

#!/bin/bash

echo "If there are errors, visit: https://www.notion.so/Daily-Model-Deployment-Process-d055c30e3c954d66b888015226cbd1a8"
echo "Search for warnings in the logs (even on a successful training procedure) and investigate"

# Assert that an argument is provided
if [ -z "$1" ]; then
  echo "Error: No argument provided. Usage: bash ./model_deployment.sh <MODEL_NAME>"
  # sends the shutdown command to the VM, but this does not immediately stop the script (next few commands may run as the VM is shutting down) and so use `exit 1` to make sure no furhter commands are run
  sudo shutdown -h now
  exit 1
fi

# Check if the argument is one of the valid values
case "$1" in
  "yield_spread_with_similar_trades"|"dollar_price")
    echo "The argument is valid: $1"
    ;;
  *)
    echo "Error: Invalid argument. Allowed values are: yield_spread_with_similar_trades, dollar_price."
    # sends the shutdown command to the VM, but this does not immediately stop the script (next few commands may run as the VM is shutting down) and so use `exit 1` to make sure no furhter commands are run
    sudo shutdown -h now
    exit 1
    ;;
esac

who
HOME_DIRECTORY="$HOME"
AUTOMATED_TRAINING_DIRECTORY="$HOME_DIRECTORY/ficc_python/automated_training"
DATE_WITH_YEAR=$(TZ="America/New_York" date +%Y-%m-%d)    # Create date before training so that in case the training takes too long and goes into the next day, the date is correct
REGION='us-central1'

if [ "$1" == "yield_spread_with_similar_trades" ]; then
  TRAINED_MODELS_PATH="$HOME_DIRECTORY/trained_models/yield_spread_with_similar_trades_models"
  TRAINING_LOG_PATH="$HOME_DIRECTORY/training_logs/yield_spread_with_similar_trades_training_$DATE_WITH_YEAR.log"
  MODEL="yield_spread_with_similar_trades"
  TRAINING_SCRIPT="$AUTOMATED_TRAINING_DIRECTORY/automated_training_yield_spread_with_similar_trades_model.py"
  MODEL_NAME="similar-trades-v2-model-${DATE_WITH_YEAR}"
  MODEL_ZIP_NAME='model_similar_trades_v2'    # must match `auxiliary_functions.py::get_model_zip_filename(...)`
  ENDPOINT_ID=$(gcloud ai endpoints list --region=$REGION --format='value(ENDPOINT_ID)' --filter=display_name='yield_spread_with_similar_trades_model')
  ARCHIVED_DIRECTORY_IN_BUCKET='similar_trades_v2_model_inaccurate'
else
  TRAINED_MODELS_PATH="$HOME_DIRECTORY/trained_models/dollar_price_model"
  TRAINING_LOG_PATH="$HOME_DIRECTORY/training_logs/dollar_price_training_$DATE_WITH_YEAR.log"
  MODEL="dollar_price"
  TRAINING_SCRIPT="$AUTOMATED_TRAINING_DIRECTORY/automated_training_dollar_price_model.py"
  MODEL_NAME="dollar-v2-model-${DATE_WITH_YEAR}"
  MODEL_ZIP_NAME='model_dollar_price_v2'    # must match `auxiliary_functions.py::get_model_zip_filename(...)`
  ENDPOINT_ID=$(gcloud ai endpoints list --region=$REGION --format='value(ENDPOINT_ID)' --filter=display_name='dollar_price_model')
  ARCHIVED_DIRECTORY_IN_BUCKET='dollar_price_model_v2_inaccurate'
fi

# Activate the virtual environment for Python3.10 (/usr/local/bin/python3.10) that contains all of the packages; to see all versions of Python use command `whereis python`
# If venv_py310 does not exist in `ficc_python/`, then in `ficc_python/` run `/usr/local/python3.10 -m venv venv_py310` and `source venv_py310/bin/activate` followed by `pip install -r requirements_py310.txt`
source $HOME_DIRECTORY/ficc_python/venv_py310/bin/activate
python --version

# Training the model
python $TRAINING_SCRIPT
SWITCH_TRAFFIC_EXIT_CODE=$?
if [ $SWITCH_TRAFFIC_EXIT_CODE -ne 10 ] && [ $SWITCH_TRAFFIC_EXIT_CODE -ne 11 ]; then
  echo "$TRAINING_SCRIPT script failed with exit code $SWITCH_TRAFFIC_EXIT_CODE which should have been either 10 or 11"
  python $AUTOMATED_TRAINING_DIRECTORY/clean_training_log.py $TRAINING_LOG_PATH
  python $AUTOMATED_TRAINING_DIRECTORY/send_email_with_training_log.py $TRAINING_LOG_PATH $MODEL "Model training failed. See attached logs for more details. However, if there is not enough new trades on the previous business day, then this is the desired behavior."
  # sends the shutdown command to the VM, but this does not immediately stop the script (next few commands may run as the VM is shutting down) and so use `exit 1` to make sure no furhter commands are run
  sudo shutdown -h now
  exit 1
fi
echo "Model trained successfully (Exit Code: $SWITCH_TRAFFIC_EXIT_CODE)"

# Cleaning the logs to make more readable
python $AUTOMATED_TRAINING_DIRECTORY/clean_training_log.py $TRAINING_LOG_PATH

# Unzipping model and uploading it to automated training bucket
echo "Unzipping model $MODEL_NAME"
# Remove the ZIP file if it already exists
if [ -f "$TRAINED_MODELS_PATH/$MODEL_ZIP_NAME.zip" ]; then    # -f checks if the file exists
  echo "Removing existing ZIP file: $TRAINED_MODELS_PATH/$MODEL_ZIP_NAME.zip"
  rm "$TRAINED_MODELS_PATH/$MODEL_ZIP_NAME.zip"
fi
gsutil cp -r gs://automated_training/$MODEL_ZIP_NAME.zip $TRAINED_MODELS_PATH/$MODEL_ZIP_NAME.zip
# Remove the directory if it already exists
if [ -d "$TRAINED_MODELS_PATH/$MODEL_NAME" ]; then    # -d checks if the directory exists
  echo "Removing existing directory: $TRAINED_MODELS_PATH/$MODEL_NAME"
  rm -rf "$TRAINED_MODELS_PATH/$MODEL_NAME"
fi
unzip $TRAINED_MODELS_PATH/$MODEL_ZIP_NAME.zip -d $TRAINED_MODELS_PATH/$MODEL_NAME
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
  echo "Unzipping failed with exit code $EXIT_CODE"
  python $AUTOMATED_TRAINING_DIRECTORY/send_email_with_training_log.py $TRAINING_LOG_PATH $MODEL "Unzipping model failed. See attached logs for more details."
  # sends the shutdown command to the VM, but this does not immediately stop the script (next few commands may run as the VM is shutting down) and so use `exit 1` to make sure no furhter commands are run
  sudo shutdown -h now
  exit 1
fi

if [ $SWITCH_TRAFFIC_EXIT_CODE -eq 10 ]; then
  BUCKET='gs://automated_training'
else
  BUCKET="gs://automated_training/$ARCHIVED_DIRECTORY_IN_BUCKET/"
fi

echo "Uploading model to bucket: $BUCKET"
gsutil cp -r $TRAINED_MODELS_PATH/$MODEL_NAME $BUCKET
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
  echo "Uploading model to $BUCKET failed with exit code $EXIT_CODE"
  python $AUTOMATED_TRAINING_DIRECTORY/send_email_with_training_log.py $TRAINING_LOG_PATH $MODEL "Uploading model to $BUCKET failed. See attached logs for more details."
  # sends the shutdown command to the VM, but this does not immediately stop the script (next few commands may run as the VM is shutting down) and so use `exit 1` to make sure no furhter commands are run
  sudo shutdown -h now
  exit 1
fi

if [ $SWITCH_TRAFFIC_EXIT_CODE -eq 10 ]; then
  # Getting the endpoint ID we want to deploy the model on
  echo "ENDPOINT_ID $ENDPOINT_ID"
  echo "MODEL_NAME $MODEL_NAME"
  echo "Uploading model to Vertex AI Model Registry"
  gcloud ai models upload --region=$REGION --display-name=$MODEL_NAME --container-image-uri=us-docker.pkg.dev/vertex-ai/prediction/tf2-gpu.2-13:latest --artifact-uri=gs://automated_training/$MODEL_NAME
  EXIT_CODE=$?
  if [ $EXIT_CODE -ne 0 ]; then
    echo "Model upload to Vertex AI Model Registry failed with exit code $EXIT_CODE"
    python $AUTOMATED_TRAINING_DIRECTORY/send_email_with_training_log.py $TRAINING_LOG_PATH $MODEL "Model upload to Vertex AI Model Registry failed. See attached logs for more details."
    # sends the shutdown command to the VM, but this does not immediately stop the script (next few commands may run as the VM is shutting down) and so use `exit 1` to make sure no furhter commands are run
    sudo shutdown -h now
    exit 1
  fi

  NEW_MODEL_ID=$(gcloud ai models list --region=$REGION --format='value(name)' --filter='displayName'=$MODEL_NAME)
  echo "NEW_MODEL_ID $NEW_MODEL_ID"

  # getting the old deployed model ID here so for simplicity since there is only one model deployed to the endpoint
  OLD_DEPLOYED_MODEL_ID=$(gcloud ai endpoints describe $ENDPOINT_ID --region=$REGION --format='value(deployedModels[0].id)')
  echo "OLD_DEPLOYED_MODEL_ID $OLD_DEPLOYED_MODEL_ID"

  echo "Deploying model with model ID: $NEW_MODEL_ID to endpoint: $ENDPOINT_ID"

  if [ "$MODEL" == "yield_spread_with_similar_trades" ]; then
    gcloud ai endpoints deploy-model $ENDPOINT_ID --region=$REGION --display-name=$MODEL_NAME --model=$NEW_MODEL_ID --machine-type=n1-standard-2 --accelerator=type=nvidia-tesla-t4,count=1 --min-replica-count=1 --max-replica-count=1
  else
    gcloud ai endpoints deploy-model $ENDPOINT_ID --region=$REGION --display-name=$MODEL_NAME --model=$NEW_MODEL_ID --machine-type=n1-standard-2 --accelerator=type=nvidia-tesla-t4,count=1 --min-replica-count=1 --max-replica-count=1
  fi

  EXIT_CODE=$?
  if [ $EXIT_CODE -ne 0 ]; then
    echo "Model deployment to Vertex AI endpoint: $ENDPOINT_ID failed with exit code $EXIT_CODE"
    python $AUTOMATED_TRAINING_DIRECTORY/send_email_with_training_log.py $TRAINING_LOG_PATH $MODEL "Model deployment to Vertex AI endpoint: $ENDPOINT_ID failed. See attached logs for more details."
    # sends the shutdown command to the VM, but this does not immediately stop the script (next few commands may run as the VM is shutting down) and so use `exit 1` to make sure no furhter commands are run
    sudo shutdown -h now
    exit 1
  fi

  # Vertex AI assigns a new “deployed model ID” when you deploy a model to an endpoint (different from NEW_MODEL_ID which is the original model resource ID); when updating the traffic split in Vertex AI, you need to use the deployed model ID, not the original model ID
  NEW_DEPLOYED_MODEL_ID=$(gcloud ai endpoints describe $ENDPOINT_ID --region=$REGION --format=json | jq -r --arg MODEL_ID "$NEW_MODEL_ID" '.deployedModels[] | select(.model | endswith($MODEL_ID)) | .id')
  echo "Updating traffic split to 100% for new model (original model ID: $NEW_MODEL_ID) with deployed model ID: $NEW_DEPLOYED_MODEL_ID. When updating the traffic split in Vertex AI, the deployed model ID must be used, not the original model ID (the resource ID)."
  gcloud ai endpoints update $ENDPOINT_ID --region=$REGION --traffic-split=$NEW_DEPLOYED_MODEL_ID=100
  EXIT_CODE=$?
  if [ $EXIT_CODE -ne 0 ]; then
    echo "Traffic switch failed with exit code $EXIT_CODE"
    python $AUTOMATED_TRAINING_DIRECTORY/send_email_with_training_log.py $TRAINING_LOG_PATH $MODEL "Traffic switch to new model failed. See attached logs for more details."
    # sends the shutdown command to the VM, but this does not immediately stop the script (next few commands may run as the VM is shutting down) and so use `exit 1` to make sure no furhter commands are run
    sudo shutdown -h now
    exit 1
  fi

  # Undeploy the old model (if it exists); `[ -n "$OLD_DEPLOYED_MODEL_ID" ]` checks that the `OLD_DEPLOYED_MODEL_ID` is non-empty; `--quiet` suppresses the confirmation prompts allowing the command to run automatically
  if [ -n "$OLD_DEPLOYED_MODEL_ID" ]; then
    echo "Undeploying old model with deployed model ID: $OLD_DEPLOYED_MODEL_ID"
    gcloud ai endpoints undeploy-model $ENDPOINT_ID --region=$REGION --deployed-model-id=$OLD_DEPLOYED_MODEL_ID --quiet
  fi
fi

# Removing temporary files
echo "Removing local file: $HOME_DIRECTORY/trained_models/$MODEL_ZIP_NAME.zip"
rm $HOME_DIRECTORY/trained_models/$MODEL_ZIP_NAME.zip
echo "Removing file from Google Cloud Storage: gs://automated_training/$MODEL_ZIP_NAME.zip"
gsutil rm -r gs://automated_training/$MODEL_ZIP_NAME.zip

python $AUTOMATED_TRAINING_DIRECTORY/send_email_with_training_log.py $TRAINING_LOG_PATH $MODEL "No detected errors. Logs attached for reference."

sudo shutdown -h now