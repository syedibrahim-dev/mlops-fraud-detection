#!/bin/bash
# Download the Kaggle Credit Card Fraud dataset
# Requires: KAGGLE_USERNAME and KAGGLE_KEY env vars set, or ~/.kaggle/kaggle.json

set -e

echo "Downloading creditcard fraud dataset from Kaggle..."
pip install kaggle -q
kaggle datasets download -d mlg-ulb/creditcardfraud -p data/ --unzip
echo "Done. File saved to data/creditcard.csv"
