import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split

IMAGE_SIZE = (224, 224)
SEQUENCE_LENGTH = 20

def extract_frames(video_path):
    frames = []
    cap = cv2.VideoCapture(video_path)
    video_frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if video_frames_count == 0:
        cap.release()
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
        
    return np.array(frames)

def process_and_split_dataset(dataset_path, test_size=0.2, val_size=0.1):
    classes = os.listdir(dataset_path)
    video_paths = []
    labels = []
    
    class_mapping = {class_name: idx for idx, class_name in enumerate(classes)}
    print(f"Mapeamento de Classes encontrado: {class_mapping}")
    
    for class_name in classes:
        class_dir = os.path.join(dataset_path, class_name)
        if not os.path.isdir(class_dir):
            continue
            
        for video_name in os.listdir(class_dir):
            if video_name.endswith('.mp4'):
                video_paths.append(os.path.join(class_dir, video_name))
                labels.append(class_mapping[class_name])
            
    X_temp, X_test, y_temp, y_test = train_test_split(
        video_paths, labels, test_size=test_size, stratify=labels, random_state=42
    )
    
    val_ratio = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, stratify=y_temp, random_state=42
    )
    
    print(f"Total: {len(video_paths)} vídeos localizados.")
    print(f"Divisão -> Treino: {len(X_train)} | Validação: {len(X_val)} | Teste: {len(X_test)}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def create_dataset(video_paths, labels, dataset_type):
    X, y = [], []
    total = len(video_paths)
    
    for idx, (path, label) in enumerate(zip(video_paths, labels)):
        if idx % 10 == 0:
            print(f"Processando {dataset_type}: {idx}/{total} vídeos...")
            
        frames = extract_frames(path)
        if frames is not None:
            X.append(frames)
            y.append(label)
            
    return np.array(X), np.array(y)

if __name__ == "__main__":
    DATASET_PATH = "data/raw"
    SAVE_PATH = "data/processed"
    os.makedirs(SAVE_PATH, exist_ok=True)
    
    print("Iniciando varredura e separação...")
    X_train_paths, X_val_paths, X_test_paths, y_train, y_val, y_test = process_and_split_dataset(DATASET_PATH)
    
    print("\n--- Iniciando Extração de Frames ---")
    
    X_train_data, y_train_data = create_dataset(X_train_paths, y_train, "Treino")
    np.save(os.path.join(SAVE_PATH, "X_train.npy"), X_train_data)
    np.save(os.path.join(SAVE_PATH, "y_train.npy"), y_train_data)
    
    X_val_data, y_val_data = create_dataset(X_val_paths, y_val, "Validação")
    np.save(os.path.join(SAVE_PATH, "X_val.npy"), X_val_data)
    np.save(os.path.join(SAVE_PATH, "y_val.npy"), y_val_data)
    
    X_test_data, y_test_data = create_dataset(X_test_paths, y_test, "Teste")
    np.save(os.path.join(SAVE_PATH, "X_test.npy"), X_test_data)
    np.save(os.path.join(SAVE_PATH, "y_test.npy"), y_test_data)
    
    print("\nProcessamento concluído! Dados salvos na pasta 'data/processed'.")