# Adaptive Traffic-Light Control with LSTM + Sequential Fuzzy + GA-HPO (ETC 2026)

Este projeto implementa um pipeline para *previsão de estado de tráfego* e *controle adaptativo de semáforo*:

- *LSTM (PyTorch)* para prever *velocidade* e *ocupação* da via usando *alvos suavizados*.
- *Controlador Fuzzy Mamdani* para mapear (velocidade, ocupação) → *tempo de verde desejado*.
- *Controle sequencial* (com estado G_prev) limitando a variação por ciclo (*ΔG*), aproximando um controle realista.
- *HPO (Hyperparameter Optimization)* com *Algoritmo Genético (GA)* rodando em *5 GPUs* (1 indivíduo por GPU).
- Exportação de *CSVs* e geração de *figuras* para o artigo.

---

## 1) Fundamentos do método

### 1.1 LSTM (Long Short-Term Memory)
A LSTM é uma variante de RNN projetada para modelar dependências temporais de curto e longo prazo, reduzindo problemas de gradiente que ocorrem em RNNs tradicionais. Em termos práticos, a LSTM recebe uma *sequência* de observações (janela temporal) e aprende padrões temporais para prever o próximo estado.

Neste projeto:
- Entrada: janelas com lookback = L passos (ex.: L=72 → 6 horas com dados a cada 5 min).
- Saída: regressão multivariada para *2 alvos*:
  - Traffic_Speed_kmh_target
  - Road_Occupancy_%_target

A previsão é feita no horizonte de *5 minutos à frente* (t+1), mas para reduzir ruído e facilitar aprendizado, os alvos são suavizados (ver seção 1.2).

### 1.2 Alvos suavizados (média móvel retroativa)
Com dados de tráfego, o valor instantâneo pode ter alta variância e baixa autocorrelação. Para aumentar previsibilidade, os alvos são definidos como *média móvel retroativa (trailing)*:

$$\overline{y}t^{(W)}=\frac{1}{W}\sum{i=0}^{W-1} y_{t-i}$$

Com amostragem de 5 min:
- W=12 → 60 minutos

Isso faz o modelo aprender o *nível médio do tráfego*, em vez de oscilações pontuais.

### 1.3 Lógica Fuzzy Mamdani (controle)
A lógica fuzzy permite modelar decisões usando regras linguísticas do tipo:

> SE (ocupação é ALTA) E (velocidade é BAIXA) ENTÃO (verde é MÁXIMO)

Etapas clássicas (Mamdani):
1. *Fuzzificação*: converte entradas numéricas (velocidade/ocupação) em graus de pertinência em conjuntos fuzzy (BAIXA/MÉDIA/ALTA).
2. *Inferência*: aplica regras fuzzy (AND = min, OR = max).
3. *Agregação*: combina as saídas das regras.
4. *Defuzzificação*: converte a saída fuzzy em um valor numérico (ex.: método do centróide).

Neste projeto, o fuzzy recebe:
- velocidade prevista (km/h)
- ocupação prevista (%)
e produz um verde desejado:
- G_fuzzy(t) em segundos

### 1.4 Controle sequencial (ΔG com histerese)
Para evitar mudanças abruptas no semáforo, o controle é aplicado de forma *sequencial*, usando o verde do ciclo anterior:

$$\Delta G(t)=clip(G_{fuzzy}(t)-G_{prev}(t-1), \pm \Delta G_{max})$$

$$G_{prev}(t)=clip(G_{prev}(t-1)+\Delta G(t), [G_{min}, G_{max}])$$

Isso:
- aumenta estabilidade,
- reduz saturação,
- aproxima um controle realista.

### 1.5 Algoritmo Genético (GA) para HPO
O Algoritmo Genético é um método de busca inspirado em evolução:
- cada *indivíduo* representa um conjunto de hiperparâmetros (genes),
- a *aptidão (fitness)* mede desempenho na validação.

Operadores usados:
- *Seleção por torneio*: escolhe pais com base no melhor fitness em um grupo.
- *Crossover (1 ponto)*: mistura genes dos pais para formar filho.
- *Mutação por gene*: altera alguns hiperparâmetros aleatoriamente.
- *Elitismo*: mantém os melhores indivíduos para a próxima geração.

Neste projeto:
- Fitness típico: val_scaled_mse (minimização) ou neg_avg_r2 (maximizar R² médio).
- Execução paralela em *5 GPUs*: 1 indivíduo por GPU.

---

## 2) Estrutura do repositório

### Treinamento / Modelo
- `train.py` — treino DDP em 5 GPUs via torchrun (*suporta* --weight_decay e --grad_clip)
- `model.py` — LSTMRegressor
- `data_module.py` — prepara dataset, targets, janelas, escalonamento, split temporal

### Controle
- `fuzzy_controller.py` — fuzzy Mamdani + controle sequencial (gera CSV e métricas)

### HPO
- `ga_hpo_lstm_v2.py` — GA-HPO com execução paralela em 5 GPUs

### Plot (opcional)
- `plot_pred_vs_real.py` — gráfico Real vs Predito (PDF/PNG)

---

## 3) Dataset
Coloque o arquivo na raiz do projeto:
- `smart_mobility_dataset.csv`

---

## 4) Como rodar

### 4.1 Treinar com 5 GPUs
Exemplo (melhor config do GA):

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
```

## Licença

Este projeto está licenciado sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes.
