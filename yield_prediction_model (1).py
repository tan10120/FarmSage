import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

def yield_prediction(input_data):
    file_path = 'G:/pilask/models/crop_production.csv'
    df = pd.read_csv(file_path)
    df = df.drop(['Crop_Year'], axis=1)

    X = df.drop(['Production'], axis=1)
    y = df['Production']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    categorical_cols = ['State_Name','District_Name', 'Season', 'Crop']
    ohe = OneHotEncoder(handle_unknown='ignore')
    ohe.fit(X_train[categorical_cols])

    X_train_categorical = ohe.transform(X_train[categorical_cols])
    X_test_categorical = ohe.transform(X_test[categorical_cols])
    X_train_final = np.hstack((X_train_categorical.toarray(), X_train.drop(categorical_cols, axis=1)))
    X_test_final = np.hstack((X_test_categorical.toarray(), X_test.drop(categorical_cols, axis=1)))

    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train_final, y_train)

    # Transform the user input using the same feature names
    user_input = np.array([[Jdistrict, Jseason, Jcrops, Jarea]])
    user_input_categorical = ohe.transform(user_input[:, :4])
    user_input_final = np.hstack((user_input_categorical.toarray(), user_input[:, 4:].astype(float)))

    # Create a new feature, for example, a constant value of 0
    new_feature = 0
    # Append the new feature to the user_input_final array
    user_input_final = np.append(user_input_final, new_feature)
    # Reshape the user_input_final array to match the model's expected input
    user_input_final = user_input_final.reshape(1, -1)
    
    prediction = rf_model.predict(user_input_final)

    return prediction[0]