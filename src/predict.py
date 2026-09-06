import cv2
import numpy as np
import tensorflow as tf
import sys
import os

IMAGE_SIZE = (224, 224)
SEQUENCE_LENGTH = 20
MODEL_PATH = "models/best_model.keras"

def extract_frames(video_path):
    frames = []
    cap = cv2.VideoCapture(video_path)
    video_frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if video_frames_count == 0:
        print("Erro: Vídeo não encontrado ou vazio.")
        return None

    skip_frames_window = max(int(video_frames_count / SEQUENCE_LENGTH), 1)

    for frame_counter in range(SEQUENCE_LENGTH):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)
        success, frame = cap.read()
        if not success:
            break
            
        resized_frame = cv2.resize(frame, IMAGE_SIZE)
        rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        normalized_frame = rgb_frame / 255.0
        frames.append(normalized_frame)
        
    cap.release()
    
    while len(frames) < SEQUENCE_LENGTH:
        frames.append(np.zeros((IMAGE_SIZE[0], IMAGE_SIZE[1], 3)))
        
    # Retorna o array no formato esperado pelo modelo: (1 batch, 20 frames, 224, 224, 3)
    return np.array([frames])

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso correto: python src/predict.py caminho/do/video.mp4")
        sys.exit(1)
        
    video_path = sys.argv[1]
    
    if not os.path.exists(video_path):
        print(f"Arquivo não encontrado: {video_path}")
        sys.exit(1)
        
    print(f"Processando vídeo: {video_path}")
    X_new = extract_frames(video_path)
    
    if X_new is not None:
        print("\n" + "="*31)
        print("   Carregando rede neural...   ")
        print("="*31)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
        model = tf.keras.models.load_model(MODEL_PATH)
        
        print("\n" + "="*26)
        print("   Analisando frames...   ")
        print("="*26)
        prediction = model.predict(X_new)[0][0]
        
        # Mapeamento da saída (0 = Agressão, 1 = Normal)
        classe_prevista = "Normal" if prediction > 0.5 else "Agressão"
        confianca = prediction if classe_prevista == "Normal" else (1 - prediction)
        
        print("\n" + "="*29)
        print("   Resultado da Inferência    ")
        print("="*29)
        print(f"Classe Detectada: {classe_prevista.upper()}")
        print(f"Nível de Confiança: {confianca * 100:.2f}%")
        print("="*29 + "\n")