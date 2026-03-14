# Ajuste Dinâmico de Tempos Semafóricos usando Previsão de Tráfego e Controle Fuzzy Sequencial

Este projeto implementa um pipeline para **previsão de tráfego** e **controle adaptativo de semáforo**:
- **LSTM (PyTorch)** para prever **velocidade** e **ocupação** (alvos suavizados).
- **Lógica Fuzzy Mamdani** para gerar tempo de verde.
- Controle **sequênial** com limitação de variação por ciclo (ΔG).
- **HPO (Hyperparameter Optimization)** usando **Algoritmo Genético (GA)**.

## 1) Estrutura de arquivos

###Treinamento / Modelo
- `train.py`
  Treina a LSTM com suporte a **DDP em 5 GPUs** (`torchrun`).
  Versão ajustada para aceitar `--weight_decay` e `--grad_clip`.

- `model.py`
