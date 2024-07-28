# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 00:28:50 2024

@author: jamiu
"""
#%%

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU, Add, Input
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)
#%%

file_path = 'c:/documents/dataset.csv'  
data = pd.read_csv(file_path)

#exploratory data analysis
print(data.head())
print(data.info())
print(data.describe())
print(data.isnull().sum())
#%%
#preprocessing
data.fillna(data.mean(), inplace=True)

#features extraction
features = ['feature1', 'feature2', 'feature3']  
X = data[features]


#target/output for classification
target_classification = 'target_classification'  # Binary target column
y_classification = data[target_classification]

#target/output for regression
target_regression = 'target_regression'        
y_regression = data[target_regression]
#%%




# Split the data into training and testing sets
X_train, X_test, y_train_class, y_test_class = train_test_split(
    X, y_classification, test_size=0.2, random_state=42
)

# For regression, 
#we use only the wet points (assuming non-zero regression target implies wet)
wet_points = y_train_class == 1
X_train_reg = X_train[wet_points]
y_train_reg = y_regression[wet_points]

wet_points_test = y_test_class == 1
X_test_reg = X_test[wet_points_test]
y_test_reg = y_regression[wet_points_test]


#%%
#standardization

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_reg_scaled = scaler.transform(X_train_reg)
X_test_reg_scaled = scaler.transform(X_test_reg)
#%%
#ANN MODELS

#classification
#first architecture
def our_classification_model_1(input_shape):
    model = Sequential([
        Dense(64, input_dim=input_shape, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')  
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

classification_model_1 = our_classification_model_1(X_train_scaled.shape[1])
classification_model_1.summary()


#second architecture
def our_classification_model_2(input_shape):
    model = Sequential([
        Dense(128, input_dim=input_shape, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

classification_model_2 = our_classification_model_2(X_train_scaled.shape[1])
classification_model_2.summary()


#third architecture
def our_classification_model_3(input_shape):
    model = Sequential([
        Dense(256, input_dim=input_shape, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.3),
        Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.3),
        Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

classification_model_3 = our_classification_model_3(X_train_scaled.shape[1])
classification_model_3.summary()
#%%
#regression
#first architecture
def our_regression_model_1(input_shape):
    model = Sequential([
        Dense(64, input_dim=input_shape, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1, activation='linear')  
    ])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model

regression_model_1 = our_regression_model_1(X_train_reg_scaled.shape[1])
regression_model_1.summary()


#second architecture
def our_regression_model_2(input_shape):
    model = Sequential([
        Dense(128, input_dim=input_shape),
        LeakyReLU(alpha=0.1),
        Dropout(0.3),
        Dense(64),
        LeakyReLU(alpha=0.1),
        Dropout(0.3),
        Dense(32),
        LeakyReLU(alpha=0.1),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model

regression_model_2 = our_regression_model_2(X_train_reg_scaled.shape[1])
regression_model_2.summary()



#third architecture
def our_regression_model_3(input_shape):
    inputs = Input(shape=(input_shape,))
    x = Dense(256, activation='relu')(inputs)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    residual = Dense(128)(x) 
    x = Add()([x, residual])
    x = Dense(64, activation='relu')(x)
    outputs = Dense(1, activation='linear')(x)
    
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model

regression_model_3 = our_regression_model_3(X_train_reg_scaled.shape[1])
regression_model_3.summary()

#%%
#training

#training classification
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)


print("\nTraining Classification Model 1")
classification_history_1 = classification_model_1.fit(
    X_train_scaled, y_train_class, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping]
)


print("\nTraining Classification Model 2")
classification_history_2 = classification_model_2.fit(
    X_train_scaled, y_train_class, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping]
)


print("\nTraining Classification Model 3")
classification_history_3 = classification_model_3.fit(
    X_train_scaled, y_train_class, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping]
)
#%%
#training regression

print("\nTraining Regression Model 1")
regression_history_1 = regression_model_1.fit(
    X_train_reg_scaled, y_train_reg, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping]
)


print("\nTraining Regression Model 2")
regression_history_2 = regression_model_2.fit(
    X_train_reg_scaled, y_train_reg, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping]
)


print("\nTraining Regression Model 3")
regression_history_3 = regression_model_3.fit(
    X_train_reg_scaled, y_train_reg, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping]
)
#%%
#Models Evaluation
#Classification Models
for i, model in enumerate([classification_model_1, classification_model_2, classification_model_3], start=1):
    y_pred_class = (model.predict(X_test_scaled) > 0.5).astype("int32")
    accuracy = accuracy_score(y_test_class, y_pred_class)
    cm = confusion_matrix(y_test_class, y_pred_class)
    
    print(f"\nClassification Model {i} Accuracy: {accuracy:.2f}")
    print(f"Classification Model {i} Confusion Matrix:\n{cm}")

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Classification Model {i} - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
#%%
#Regression Models
for i, model in enumerate([regression_model_1, regression_model_2, regression_model_3], start=1):
    y_pred_reg = model.predict(X_test_reg_scaled)
    mse = mean_squared_error(y_test_reg, y_pred_reg)
    mae = mean_absolute_error(y_test_reg, y_pred_reg)
    
    print(f"\nRegression Model {i} Mean Squared Error: {mse:.2f}")
    print(f"Regression Model {i} Mean Absolute Error: {mae:.2f}")

    plt.scatter(y_test_reg, y_pred_reg, alpha=0.5)
    plt.plot([y_test_reg.min(), y_test_reg.max()], [y_test_reg.min(), y_test_reg.max()], color='red')
    plt.title(f'Regression Model {i} - Predicted vs Actual')
    plt.xlabel('Actual Inundation Level')
    plt.ylabel('Predicted Inundation Level')
    plt.show()

#%%
#Results Visualization
#classification
def plot_training_history(history, title):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    axs[0].plot(history.history['accuracy'], label='Train Accuracy')
    axs[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axs[0].set_title(f'{title} - Accuracy')
    axs[0].legend()

    axs[1].plot(history.history['loss'], label='Train Loss')
    axs[1].plot(history.history['val_loss'], label='Validation Loss')
    axs[1].set_title(f'{title} - Loss')
    axs[1].legend()

    plt.show()


plot_training_history(classification_history_1, 'Classification Model 1')
plot_training_history(classification_history_2, 'Classification Model 2')
plot_training_history(classification_history_3, 'Classification Model 3')
#%%
#regression
def plot_regression_history(history, title):
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{title} - Loss')
    plt.legend()
    plt.show()

plot_regression_history(regression_history_1, 'Regression Model 1')
plot_regression_history(regression_history_2, 'Regression Model 2')
plot_regression_history(regression_history_3, 'Regression Model 3')
