import tensorflow as tf
import os

print("Carregando o modelo original...")
model = tf.keras.models.load_model("models/best_model.keras")

print("Convertendo para TensorFlow Lite (Edge AI)...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Forçando o suporte a operações complexas (essencial para a camada LSTM)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS, 
    tf.lite.OpsSet.SELECT_TF_OPS
]
converter._experimental_lower_tensor_list_ops = False

tflite_model = converter.convert()

tflite_path = "models/modelo_edge.tflite"
with open(tflite_path, "wb") as f:
    f.write(tflite_model)

print(f"Modelo otimizado salvo em: {tflite_path}")