# E-Commerce Transaction Fraud Detection

## Overview
Code and notebooks for training and testing a real-time credit-card fraud detector, created by **AI4ALL Ignite 2025 – Team 16B**.  
We use the public Kaggle dataset “Credit Card Fraud Detection” and a small Streamlit app for quick demos.

## How to get started
Follow these steps to run the project on your local machine:

1. **Clone the repository** (use: `git clone`)

        git clone https://github.com/AI4ALL-Group-16B-Finance-Business-2025/E-Commerce-Transaction-Fraud-Detection.git
        cd E-Commerce-Transaction-Fraud-Detection

2. **Install the required packages** (virtual env recommended)

        python -m venv .venv               # optional but cleaner
        source .venv/bin/activate          # Windows: .venv\Scripts\activate
        pip install -r requirements.txt
        pip install streamlit              # already in the file, but explicit here

3. **Download the dataset** (manual download is fine)

        • Visit https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
        • Download creditcard.csv
        • Create a folder named Data at the project root
        • Place creditcard.csv inside the Data folder

   *(Alternatively, use the Kaggle CLI and run  
   `kaggle datasets download -d mlg-ulb/creditcardfraud -p Data --unzip`)*

4. **Run the notebook** (EDA & model training)

        jupyter notebook Notebooks/dataOverview.ipynb

5. **Run the model demo** (Streamlit UI)

        streamlit run app.py

## What we learned
- Extreme class imbalance (≈ 0.17 % fraud) requires weighting or resampling to avoid misleading accuracy.  
- Weighted Random Forests combined with feature scaling deliver strong recall without excessive false positives.  
- A lightweight Streamlit front end lets non-technical users test the model quickly.  
- Clear directory structure and reproducible setup steps make onboarding straightforward for new contributors.
