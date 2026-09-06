# Projeto NeuAI - Detecção de Anomalias em Vídeo

Este projeto foi desenvolvido como parte do desafio técnico para a NeuAI. Trata-se de um pipeline de Machine Learning especializado na classificação binária de vídeos de segurança (CFTV) para detectar agressões físicas.

**Arquitetura Escolhida:**
Foi implementada uma rede CNN-LSTM. Utilizamos **Transfer Learning** com a rede MobileNetV2 (congelada) para a extração espacial de características (reduzindo o custo computacional) e uma camada LSTM (Long Short-Term Memory) para a análise temporal dos frames.

**Como Executar o Projeto**

1. **Instale as dependências:**
`pip install -r requirements.txt`

2. **Pré-processamento (Evitando Data Leakage):**
`python src/data_prep.py`
*(Extrai 20 frames por vídeo, redimensiona para 224x224 e salva tensores .npy na pasta processed)*

3. **Treinamento com Class Weights:**
`python src/train.py`
*(Aplica pesos para balancear as classes e utiliza Early Stopping para evitar overfitting)*

4. **Avaliação e Métricas:**
`python src/evaluate.py`
*(Gera Accuracy, Precision, Recall, F1-Score e a Matriz de Confusão)*

5. **Inferência (Teste Prático):**
`python src/predict.py caminho/do/video.mp4`

**Análise de Limitações e Melhorias**
Durante os testes de inferência, o modelo apresentou 100% de acerto na classe detectada, porém com uma margem de confiança oscilando entre 50% e 60%. 
* **Limitação:** Isso ocorre devido ao uso de um dataset reduzido para viabilizar o processamento local e ao congelamento total da MobileNetV2.
* **Soluções Propostas:** Para um ambiente de produção, as melhorias incluiriam a expansão do dataset, aplicação de Data Augmentation e o Fine-Tuning (descongelamento) das últimas camadas da rede base.
* **Otimização Edge AI:** O projeto inclui o script `export_tflite.py` que converte a rede gerada para o formato `.tflite`, tornando-a apta para implantação em dispositivos de borda (Edge Computing).
