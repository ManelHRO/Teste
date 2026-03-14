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


### Plot (opcional)
- `plot_pred_vs_real.py`  
  Gera gráfico Real vs Predito (targets y0/y1) em PDF/PNG.

---
## 2) Dataset

Coloque o arquivo na pasta do projeto:
- `smart_mobility_dataset.csv`

A amostragem é a cada **5 minutos**.

---

## 3) Targets suavizados (caminho 2)

Para aumentar previsibilidade, os alvos são definidos como **média móvel retroativa** (trailing):
\[
\overline{y}_t^{(W)}=\frac{1}{W}\sum_{i=0}^{W-1} y_{t-i}
\]

Com 5 min por passo:
- `W=12` → **60 minutos**

Targets:
- `Traffic_Speed_kmh_target`
- `Road_Occupancy_%_target`

---

## 4) Treinar a LSTM em 5 GPUs (DDP)

Exemplo com melhor configuração do GA:

```bash
torchrun --nproc_per_node=5 train.py \
  --csv_path smart_mobility_dataset.csv \
  --lookback 72 \
  --target_smooth_window 12 \
  --batch_size 64 \
  --lr 0.0003567106690649258 \
  --weight_decay 5.4254126473815405e-06 \
  --grad_clip 1.694140041744789 \
  --hidden_size 128 \
  --num_layers 2 \
  --dropout 0.0685181867837602 \
  --fc_size 32
