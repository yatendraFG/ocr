import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import joblib
import os
import tensorflow as tf

# File paths
base_path = r'G:\My Drive\Project-10'
csv_path = os.path.join(base_path, 'extracted_text.csv')
model_save_path = os.path.join(base_path, 'best_model.keras')  # Folder (SavedModel format)
vectorizer_path = os.path.join(base_path, 'tfidf_vectorizer.pkl')

# Load data
df = pd.read_csv(csv_path)

# TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=1000)
text_features = vectorizer.fit_transform(df['extracted_text']).toarray()

# Reshape for LSTM (samples, timesteps=1, features)
text_features = text_features.reshape((text_features.shape[0], 1, text_features.shape[1]))

# Define LSTM model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(1, text_features.shape[2])))
model.add(Dense(text_features.shape[2]))  # Output layer

# Compile model
model.compile(optimizer='adam', loss='mse')

# Train
model.fit(text_features, text_features, epochs=10, batch_size=32, validation_split=0.2)

# Save model (SavedModel format)
model.save(model_save_path)

# Save TF-IDF vectorizer
joblib.dump(vectorizer, vectorizer_path)

print(f"Model saved to: {model_save_path}")
print(f"Vectorizer saved to: {vectorizer_path}")
