# @ Create date: 2024-12-09
# @ Modified date: 2025-05-20
#                necessary directories, (3) creating a virtual environment with all necessary packages, and (4) setting up the cron job. 
#                The user must set the `MODEL_NAME` correctly and put in their Github credentials. Additionally, the user must copy 
#                their GCP credentials into the VM. Note: since this script is in the `ficc_python` package, it should be copied to 
#                the VM and then run directly from the home directory. Use `$ bash training_setup.sh` to run this script.

#!/bin/bash

MODEL_NAME="yield_spread_with_similar_trades"    # "dollar_price"

# GitHub username and personal access token
GITHUB_USERNAME="your_username"    # Replace with your GitHub username (NOT email address)
GITHUB_TOKEN="your_personal_access_token"    # Replace with your GitHub personal access token
GITHUB_EMAIL="username@domain.com"    # Replace with your GitHub email address
GITHUB_NAME="first last"    # Replace with your GitHub name

FICC_REPO="https://${GITHUB_USERNAME}:${GITHUB_TOKEN}@github.com/Ficc-ai/ficc.git"
FICC_PYTHON_REPO="https://${GITHUB_USERNAME}:${GITHUB_TOKEN}@github.com/Ficc-ai/ficc_python.git"


# Configure Git to use the credential store
git config --global credential.helper store

# Configure Git identity
git config --global user.email $GITHUB_EMAIL
git config --global user.name $GITHUB_NAME

# Add the credentials to the store
echo "https://${GITHUB_USERNAME}:${GITHUB_TOKEN}@github.com" > ~/.git-credentials

# Confirm the credentials have been added
echo "GitHub credentials have been stored successfully in the credential store."

# Function to clone a GitHub repository
clone_repo() {
  REPO_URL=$1    # First argument: repository URL
  TARGET_DIR=$2    # Second argument: target directory

  # Check if the target directory already exists
  if [ -d "$TARGET_DIR" ]; then
    echo "The directory '$TARGET_DIR' already exists. Skipping clone."
  else
    # Clone the repository
    git clone "$REPO_URL" "$TARGET_DIR"
    
    if [ $? -eq 0 ]; then
      echo "Repository cloned successfully into '$TARGET_DIR'."
    else
      echo "Failed to clone the repository: $REPO_URL"
      exit 1
    fi
  fi
}

clone_repo $FICC_REPO "ficc"
clone_repo $FICC_PYTHON_REPO "ficc_python"


# Define directories to create
DIRS="$HOME/training_logs $HOME/trained_models"

# Loop through each directory and create it if it doesn't exist
for DIR in $DIRS; do
  if [ ! -d "$DIR" ]; then
    mkdir -p "$DIR" && echo "Created directory: $DIR" || echo "Failed to create directory: $DIR"
  else
    echo "Directory already exists: $DIR"
  fi
done


# Define the virtual environment directory and requirements file path
REQUIREMENTS_DIR="$HOME/ficc_python"
VENV_DIR="$REQUIREMENTS_DIR/venv_py310"
REQUIREMENTS_FILE="$REQUIREMENTS_DIR/requirements_py310.txt"

# Check if the virtual environment directory already exists
if [ -d "$VENV_DIR" ]; then
  echo "Virtual environment '$VENV_DIR' already exists."
else
  # Create the virtual environment
  python3 -m venv "$VENV_DIR"
  
  if [ $? -eq 0 ]; then
    echo "Virtual environment '$VENV_DIR' created successfully."
  else
    echo "Failed to create the virtual environment."
    exit 1
  fi
fi

# Activate the virtual environment
source "$VENV_DIR/bin/activate"

if [ $? -eq 0 ]; then
  echo "Virtual environment '$VENV_DIR' activated."
else
  echo "Failed to activate the virtual environment."
  exit 1
fi

# Check if the requirements file exists
if [ -f "$REQUIREMENTS_FILE" ]; then
  echo "Installing packages from $REQUIREMENTS_FILE..."
  pip install -r "$REQUIREMENTS_FILE"
  
  if [ $? -eq 0 ]; then
    echo "Packages installed successfully."
  else
    echo "Failed to install packages. Check $REQUIREMENTS_FILE for errors."
    exit 1
  fi
else
  echo "Requirements file '$REQUIREMENTS_FILE' not found in $REQUIREMENTS_DIR. Skipping package installation."
fi

echo "Setup complete. Use 'source $VENV_DIR/bin/activate' to activate the virtual environment."


# Define the cron job
CRON_JOB="45 10 * * 1-5 sh $HOME/ficc_python/automated_training/model_deployment.sh $MODEL_NAME >> $HOME/training_logs/${MODEL_NAME}_training_\$(TZ=America/New_York date +\\%Y-\\%m-\\%d).log 2>&1"

# Check if the cron job already exists
crontab -l 2>/dev/null | grep -F "$CRON_JOB" >/dev/null

if [ $? -eq 0 ]; then
  echo "Cron job already exists. No changes made."
else
  # Add the cron job
  (crontab -l 2>/dev/null; echo "$CRON_JOB") | crontab -
  echo "Cron job added successfully."
fi


TRAINING_SETUP_FILE_PATH="$HOME/training_setup.sh"

# Check if the file exists
if [ -f "$TRAINING_SETUP_FILE_PATH" ]; then
  # Remove the file
  rm "$TRAINING_SETUP_FILE_PATH"
  
  if [ $? -eq 0 ]; then
    echo "File '$TRAINING_SETUP_FILE_PATH' removed successfully."
  else
    echo "Failed to remove the file '$TRAINING_SETUP_FILE_PATH'."
    exit 1
  fi
else
  echo "File '$TRAINING_SETUP_FILE_PATH' does not exist."
fi
