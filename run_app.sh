#!/bin/bash

# Activate virtual environment
source .venv/bin/activate

# Install/upgrade dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run company_data_comparison_ui.py 