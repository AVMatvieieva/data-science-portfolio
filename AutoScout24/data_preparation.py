import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

pd.options.display.float_format = '{:.2f}'.format

def get_data(data: str):
    """The function takes a file name with the extension .csv and returns pd.DataFrame
    Args:
        data (str): File name 
    Returns:
        pd.DataFrame: DataFrame with loaded data 
    """
    df = pd.read_csv(data)
    return df
    
df = get_data("autoscout24.csv")    

# Feature Engineering (z. B. Alter berechnen)
df["age"] = datetime.now().year - df["year"]
df.drop(columns=["year"], inplace=True)
        
# Remove missing data
df = df.dropna() 
        
# Duplikate remote
df.drop_duplicates(inplace=True) 
        
# Remove extreme outliers "mileage", "price"
Q1 = df["mileage"].quantile(0.25)
Q3 = df["mileage"].quantile(0.75)
IQR = Q3 - Q1
df = df[(df["mileage"] >= Q1 - 1.5 * IQR) & (df["mileage"] <= Q3 + 1.5 * IQR)]

Q1 = df["price"].quantile(0.25)
Q3 = df["price"].quantile(0.75)
IQR = Q3 - Q1  # Interquartilsabstand
df = df[(df["price"] >= Q1 - 1.5 * IQR) & (df["price"] <= Q3 + 1.5 * IQR)]

# Label Encoding for Categorical Columns with LabelEncoder like in Research  
cat_features = ["make", "model", "fuel", "gear", "offerType"]
num_features = ["mileage", "hp", "age"]
label_encoders = {}

for col in cat_features:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])  # Converts text into numbers
    label_encoders[col] = le  # Saves the encoder
            
# Features & Labels
X = df[["mileage", "make", "model", "fuel", "gear", "offerType", "hp", "age"]]
y = df["price"]    
        
# Split date
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
# Scaling of numerical values
scaler = StandardScaler()
X_train[num_features] = scaler.fit_transform(X_train[num_features])
X_test[num_features] = scaler.transform(X_test[num_features])
        
# Save LabelEncoder & Skalierer
joblib.dump(label_encoders, "label_encoders.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(X_train, "X_train.pkl")
joblib.dump(y_train, "y_train.pkl")
joblib.dump(X_test, "X_test.pkl")
joblib.dump(y_test, "y_test.pkl")