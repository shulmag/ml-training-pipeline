# @ Create date: 2024-12-09
# @ Modified date: 2024-12-09

#!/bin/sh

# Set the path to your service account key JSON file
SERVICE_ACCOUNT_KEY="path/to/creds.json"

# Authenticate using the service account key
gcloud auth activate-service-account --key-file="$SERVICE_ACCOUNT_KEY"

# Verify authentication
echo "Authenticated as:"
gcloud auth list
