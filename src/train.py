import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, LSTM, TimeDistributed, Flatten, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Configurações de Diretórios
DATA_PATH = "data/processed"
MODEL_PATH = "models"
os.makedirs(MODEL_PATH, exist_ok=True)
IMAGE_SIZE = (224, 224, 3)
SEQUENCE_LENGTH = 20

print("Carregando os dados processados na memória...")
X_train = np.load(os.path.join(DATA_PATH, "X_train.npy"))
y_train = np.load(os.path.join(DATA_PATH, "y_train.npy"))
X_val = np.load(os.path.join(DATA_PATH, "X_val.npy"))
y_val = np.load(os.path.join(DATA_PATH, "y_val.npy"))

print(f"Formato dos dados de treino: {X_train.shape}")

# 1. Extrator de Características (Transfer Learning - Diferencial do Edital)
print("Baixando pesos da MobileNetV2...")
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=IMAGE_SIZE)
base_model.trainable = False  # Congelamos a base para um treinamento mais rápido e estável

# 2. Arquitetura CNN-LSTM
model = Sequential([
    TimeDistributed(base_model, input_shape=(SEQUENCE_LENGTH, *IMAGE_SIZE)),
    TimeDistributed(Flatten()),
    LSTM(64, return_sequences=False),
    Dropout(0.5),  # Dropout combate o Overfitting
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')  
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# 3. Callbacks para salvar o melhor modelo automaticamente
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint(filepath=os.path.join(MODEL_PATH, 'best_model.keras'), save_best_only=True)
]

# 4. Treinamento
print("\nIniciando o treinamento do modelo (isso pode levar algum tempo)...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=15,
    batch_size=4,
    class_weight={0: 3.0, 1: 1.0}, 
    callbacks=callbacks
)

print("\nTreinamento finalizado! O melhor modelo foi salvo na pasta 'models'.")