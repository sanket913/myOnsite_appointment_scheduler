# data_preprocessing.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path) 
    # Clean the data
    df = df[df['Age'] > 0]
    df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay']).dt.dayofweek
    df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay']).dt.dayofweek
    le = LabelEncoder()
    df['Gender'] = le.fit_transform(df['Gender'])
    df = df.drop(['PatientId', 'AppointmentID', 'Neighbourhood'], axis=1)
    df['No-show'] = df['No-show'].map({'Yes': 1, 'No': 0})

    X = df[['Age', 'ScheduledDay', 'AppointmentDay', 'Gender']]
    y = df['No-show']
    
    return train_test_split(X, y, test_size=0.2, random_state=42)
