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
  Define a arquitetura `LSTMRegressor`.

- `data_module.py`
  Prepara o dataset: ordena por tempo, cria features, define targets, cria janelas (lookback),
  padroniza (z-score) e faz split temportal (train/val/test).

### Controle Fuzzy
- `fuzzy_controller.py`  
  Aplica o fuzzy usando a previsão da LSTM e calcula controle **sequencial**:
  - Geração do verde desejado pelo fuzzy: `G_fuzzy(t)`
  - Atualização sequencial:
    $$\Delta G(t) = \text{clip}(G_{fuzzy}(t) - G_{prev}(t-1), \pm \Delta G_{max})$$
    
    $$G_{prev}(t) = \text{clip}(G_{prev}(t-1) + \Delta G(t), [G_{min}, G_{max}])$$
  - Exporta `saida_fuzzy_seq.csv` com real/pred e sinais de controle.
 
### HPO (GA)
- `ga_hpo_lstm_v2.py`  
  Algoritmo Genético com:
  - seleção por torneio
  - crossover de 1 ponto
  - mutação por gene
  - elitismo
  - avaliação paralela em **5 GPUs** (1 indivíduo por GPU)
