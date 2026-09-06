import numpy as np
import os
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = "data/processed"
MODEL_PATH = "models/best_model.keras"

print("Carregando dados de teste e o modelo treinado...")
X_test = np.load(os.path.join(DATA_PATH, "X_test.npy"))
y_test = np.load(os.path.join(DATA_PATH, "y_test.npy"))
model = tf.keras.models.load_model(MODEL_PATH)

print("\nExecutando predições na base de teste...")
y_pred_prob = model.predict(X_test)
# Como a saída é sigmoid (0 a 1), consideramos > 0.5 como classe 1 (Normal) e <= 0.5 como classe 0 (Agressão)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

# Cálculo das Métricas Obrigatórias
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("\n--- RESULTADOS DA AVALIAÇÃO ---")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")

# Geração da Matriz de Confusão Visual
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Assault (0)', 'Normal (1)'], 
            yticklabels=['Assault (0)', 'Normal (1)'])
plt.title('Matriz de Confusão')
plt.ylabel('Classe Real')
plt.xlabel('Classe Prevista')

# Salva a imagem na raiz do projeto
plt.savefig("matriz_confusao.png")
print("\nMatriz de confusão salva como 'matriz_confusao.png' na raiz do projeto.")