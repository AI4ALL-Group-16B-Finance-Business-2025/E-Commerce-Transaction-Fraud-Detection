import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load your model and data
@st.cache_resource
def load_model():
    return joblib.load("WRS_model.sav")  # Replace with your model path

@st.cache_data
def load_data():
    return pd.read_csv("Data/creditcard.csv")  # Replace with your dataset path

model = load_model()
df = load_data()

# Sidebar Navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", [
    "1. Data Overview",
    "2. Preprocessing",
    "3. Performance",
])

# 1. Data Overview
if section == "1. Data Overview":
    st.title("Data Overview")
    st.markdown("""
    The dataset contains transactions made by credit cards in September 2013 by European cardholders. This dataset presents transactions that occured over a period of two days, where there were 492 frauds out of 284,807 transactions.""")
    st.subheader(f"Dataset Shape {df.shape}")
    st.write("Rows:", df.shape[0])
    st.write("Cols:", df.shape[1])
    
    st.subheader("Data Types:")
    st.write("Within the dataset the class is represented in signed 64-bit integer format, and all other features are in double precision floating point format")
    st.write(df.dtypes)
    
    st.subheader("Sample Data:")
    st.write("Features V1-V28 contain numerical values that are the result of principal component analysis (PCA) transformation. This means that the original data has been transformed into a new set of directions capturing the most variation while preserving essential patterns in the data.")
    st.write(" The features Time and Amount are the only features that have not been transformed. Time represents the seconds between the first transaction in the dataset and itself. Amount contains the transaction amount at that instance. The Class feature holds a value of 1 if fraud was detected, and a 0 otherwise.")
    st.dataframe(df.head())
    
    st.subheader("Summary Statistics:")
    st.write(df.describe())

    st.subheader("Missing Values:")
    st.write("Our dataset did not contain any missing values")
    st.write(df.isnull().sum())
    
    st.subheader("Class Distribution (Fraud vs Not Fraud):")
    st.write("The dataset is highly unbalanced, the positive class (frauds) account for 0.172 percent of all transactions.")
    st.write(df["Class"].value_counts())
    
    
    st.subheader("Correlation Heatmap:")
    st.write("This is a correlation heatmap to show the correlations between all of the features. The lighter a block is, the higher it is correlated.")
    plt.figure(figsize=(20,15))
    sns.heatmap(df.corr())
    st.pyplot(plt)

# 2. Preprocessing
elif section == "2. Preprocessing":
    st.title("Preprocessing")
    st.subheader("1. Feature Correlation (Greatest to Least)")
    st.write("The first step we took during our preprocessing stage was to organize our data by their correlation to the classification. We did this in order to identify the most influential features in a classification.")
    st.write("We evaluated our model by removing and dropping different features during our training phase.")
    st.write(df.corr()['Class'].sort_values(ascending=False))
    
    st.subheader("2. Scaling features (sklearn StandardScaler)")
    st.write("Next we decided to scale our amount feature using sklearn's StandardScaler due to its high variation and becasue the other features were PCA transformed.")
    st.markdown("Before Scaling")
    st.write(df['Amount'])
    
    st.markdown("After Scaling:")
    scaled_amount = StandardScaler().fit_transform(df[['Amount']])
    st.write(scaled_amount)
    
    st.subheader("3. Testing and Training")
    st.write("Finally we did a test training split of 80 to 20. 80 percent of the data went to training, and 20 percent went to testing.")
    
    scaler = StandardScaler()
    X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=['Class']), df['Class'], test_size=0.2, random_state=42)
    X_train['Amount_scaled'] = scaler.fit_transform(X_train[['Amount']])
    X_test['Amount_scaled'] = scaler.transform(X_test[['Amount']])
    
    st.write(f"Training set shape: {X_train.shape}")
    st.write(f"Testing set shape: {X_test.shape}")
    
# 3. Performance
elif section == "3. Performance":
    st.title("Model Performance & Predictions")
    st.write("During model training and testing, we prioritized optimizing model recall to ensure that the most fraudulent transactions as possible were correctly identified. This focus helped minimize false negatives, which is critical in fraud detection.")
    st.write("To identify model recall we used the sklearn.metrics x classification_report.")

    scaler = StandardScaler()
    X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=['Class']), df['Class'], test_size=0.2, random_state=42)
    X_train['Amount_scaled'] = scaler.fit_transform(X_train[['Amount']])
    X_test['Amount_scaled'] = scaler.transform(X_test[['Amount']])
    
    dropped = ['Amount', 'Time'] 
    # Drop from test and train set
    X_train_drop = X_train.drop(columns=dropped)
    X_test_drop = X_test.drop(columns=dropped)
    
    y_pred = model.predict(X_test_drop)

    st.subheader("Classification Report (Original)")
    report = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay(cm).plot(ax=ax)
    st.pyplot(fig)
