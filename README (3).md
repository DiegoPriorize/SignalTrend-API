# Signals API — COMPRA/VENDA com Indicadores Técnicos

API HTTP (**FastAPI**) que recebe séries de **OHLCV** (listas de strings) e retorna:

- **Sinal agregado**: `COMPRA`, `VENDA` ou `NEUTRO`
- **Timestamp** do candle analisado
- **Precisão (%)** estimada (confiança do consenso)
- **Motivos** (explicação textual)
- **Snapshot de indicadores**
- **Previsões curtas** (até **5 timestamps** ≤ **30 min** no futuro) com operação e probabilidade

Pronto para:

- **Docker** (arquivo único `main.py`)
- **Render.com** (start command com uvicorn)
- Testes via **/docs** (Swagger) ou **cURL**

> **Compatibilidade:** aceita `L` (preferido) **ou** `I` (legado) para **low**.

---

## Sumário

- [Arquitetura & Estratégia](#arquitetura--estratégia)
  - [Indicadores Utilizados](#indicadores-utilizados)
  - [Lógica de Sinal](#lógica-de-sinal)
  - [Precisão (confiança)](#precisão-confiança)
  - [Previsões Futuras (≤ 30 min)](#previsões-futuras--30-min)
- [Instalação & Execução](#instalação--execução)
  - [Rodar Local (sem Docker)](#rodar-local-sem-docker)
  - [Docker](#docker)
  - [Render.com](#rendercom)
- [Endpoints](#endpoints)
  - [`POST /analyze`](#post-analyze)
  - [`GET /health`](#get-health)
- [Formato de Entrada (JSON)](#formato-de-entrada-json)
- [Formato de Saída](#formato-de-saída)
- [Configuração (`config`): parâmetros, efeitos e limites](#configuração-config-parâmetros-efeitos-e-limites)
- [Perfis prontos (Conservador, Balanceado, Agressivo)](#perfis-prontos-conservador-balanceado-agressivo)
- [Boas Práticas de Dados](#boas-práticas-de-dados)
- [FAQ](#faq)
- [Aviso Importante](#aviso-importante)
- [Anexos úteis](#anexos-úteis)

---

## Arquitetura & Estratégia

### Indicadores Utilizados

- **Médias Móveis (SMA/EMA/WMA)**
  - **SMA** e **EMA** principais para cruzamentos (curto vs longo).
  - **WMA** disponível no snapshot para referência.
- **RSI (14 padrão)**
  - Sobrevenda `< 30`, Sobrecompra `> 70` (ajustável).
- **MACD (12,26,9 padrão)**
  - Linha MACD vs Linha de Sinal e inclinação do histograma.
- **Bandas de Bollinger (20, 2.0 padrão)**
  - Toque na banda superior/inferior sugere reversão.
- **Estocástico (K=14, D=3 padrão)**
  - Cruzamentos de `%K` e `%D` em regiões extremas.
- **ADX (14 padrão) + DI+/DI-**
  - Reforça sinais quando tendência é forte (≥ 25 padrão).
- **Fibonacci (lookback 120 padrão)**
  - Níveis no snapshot (suporte/resistência).
- **Volume**
  - Confirmação via multiplicador (volume acima/abaixo da média recente).

### Lógica de Sinal

Calcula-se **scores de COMPRA e VENDA** a partir de regras:

- **Cruzamento de EMA (curto vs longo):** pró-compra se `EMA_curto > EMA_longo`, pró-venda se contrário.
- **MACD vs Sinal:** pró-compra se MACD acima, pró-venda se abaixo (+ leve bônus se histograma inclina pró).
- **RSI:** pró-compra se abaixo de `rsi_buy`; pró-venda se acima de `rsi_sell`.
- **Bollinger:** tocar **banda inferior** → pró-compra; tocar **banda superior** → pró-venda.
- **Estocástico:** cruzamento de `%K` sobre `%D` em região de **sobrevenda** → pró-compra; cruzamento de `%K` sob `%D` em **sobrecompra** → pró-venda.
- **ADX:** se forte (`≥ adx_trend_min`), reforça a direção **coerente** com EMA e MACD.
- **Volume:** aplica um **multiplicador** (ex.: `1.15×` se volume > 120% da média; `0.9×` se < 80%).

O **sinal final** é a classe com maior score (**COMPRA/VENDA**). Se empate/confluência fraca → **NEUTRO**.

### Precisão (confiança)

A confiança é uma **função do desequilíbrio** entre `buy_score` e `sell_score`, mapeada para **0–100%** e ancorada em **50%**:

- Se os scores forem iguais → ~**50%**.
- Quanto maior a confluência de indicadores numa direção, maior a **precisão (%)**.

### Previsões Futuras (≤ 30 min)

Gera até **5 pontos à frente** (respeitando `future_max_minutes`) usando:

- **Modo “trend”** quando **ADX** sugere tendência forte; **“mean_revert”** quando mercado está lateral.
- **Drift** derivado de **inclinação da EMA e MACD** (trend) ou da **distância até a média de Bollinger** (mean-revert).
- **Ruído pseudo-determinístico** (função seno do timestamp) para leve variabilidade **reprodutível** (sem aleatoriedade real).

Para cada ponto futuro, classifica **COMPRA/VENDA/NEUTRO** + **probabilidade** & **motivo**.

---

## Instalação & Execução

### Rodar Local (sem Docker)
```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 10000
```
Acesse **http://localhost:10000/docs**.

### Docker
```bash
docker build -t signals-api .
docker run --rm -p 10000:10000 signals-api
```

**Com `docker-compose.yml`:**
```bash
docker-compose up --build
```

### Render.com

**Opção A — Sem Dockerfile**

- Repo com `main.py` + `requirements.txt`.
- **Start command:**
  ```bash
  uvicorn main:app --host 0.0.0.0 --port $PORT
  ```

**Opção B — Com Dockerfile**

- Selecione **Docker** e use o `Dockerfile` incluso.
- **Porta:** o Render injeta `$PORT`, e o `CMD` já honra isso.

---

## Endpoints

### `POST /analyze`

Entrada: JSON com `T`, `O`, `H`, `L` (**ou** `I`), `C`, `V` como **listas de strings** do **mesmo tamanho**. `config` é opcional.

Saída: objeto com `signal`, `timestamp`, `precisao_pct`, `motivos`, `indicadores`, `futuros`.

**Exemplo (mínimo):**
```json
{
  "T": ["1622494800","1622495100","1622495400","1622495700","1622496000"],
  "O": ["100.50","101.00","100.80","101.20","101.50"],
  "H": ["101.50","101.30","101.20","101.40","101.60"],
  "L": ["100.40","100.60","100.50","100.70","100.90"],
  "C": ["101.00","100.90","101.10","101.30","101.00"],
  "V": ["1200","1500","1400","1300","1250"]
}
```

**Com `I` (compatibilidade):**
```json
{
  "T": ["1622494800","1622495100","1622495400","1622495700","1622496000"],
  "O": ["100.50","101.00","100.80","101.20","101.50"],
  "H": ["101.50","101.30","101.20","101.40","101.60"],
  "I": ["100.40","100.60","100.50","100.70","100.90"],
  "C": ["101.00","100.90","101.10","101.30","101.00"],
  "V": ["1200","1500","1400","1300","1250"]
}
```

**Com `config`:** veja seção [Configuração](#configuração-config-parâmetros-efeitos-e-limites).

### `GET /health`

Retorna:
```json
{ "status": "ok", "utc": "<timestamp>" }
```

---

## Formato de Entrada (JSON)

- Todos os arrays devem ter **mesmo tamanho** (mínimo recomendado: **50–100 candles** para indicadores ficarem estáveis).
- `T` (timestamp) em **epoch segundos**, como **strings**.
- `O`, `H`, `L` (ou `I`), `C`, `V` como **strings numéricas**.
- **Ordem cronológica crescente** (timestamps **crescentes**).

---

## Formato de Saída

```json
{
  "signal": "COMPRA|VENDA|NEUTRO",
  "timestamp": 1622496000,
  "precisao_pct": 63.8,
  "motivos": ["..."],
  "indicadores": {
    "price": 101.0,
    "sma_short": 100.98,
    "sma_long": 100.87,
    "ema_short": 100.95,
    "ema_long": 100.85,
    "wma": 100.92,
    "rsi": 48.4,
    "macd": 0.06,
    "macd_signal": 0.03,
    "macd_hist": 0.03,
    "bb_mid": 100.90,
    "bb_up": 101.45,
    "bb_lo": 100.35,
    "stoch_k": 55.2,
    "stoch_d": 52.7,
    "adx": 23.1,
    "plus_di": 17.9,
    "minus_di": 14.2,
    "volume": 1250.0,
    "volume_ma": 1336.0,
    "fib_levels": {
      "0.0": 101.6, "0.236": 101.3, "0.382": 101.1,
      "0.5": 101.0, "0.618": 100.9, "0.786": 100.7, "1.0": 100.4
    }
  },
  "futuros": [
    { "timestamp": 1622496300, "operacao": "COMPRA", "porcentagem": 61.4, "motivo": "..." }
  ]
}
```

---

## Configuração (`config`): parâmetros, efeitos e limites

A seguir, **cada parâmetro** com **Default**, **Efeito** e **Faixa recomendada**.

> **Dica geral:** períodos muito pequenos produzem sinais **ruidosos**; muito grandes geram sinais **lentos**.

### Períodos de tendência / osciladores

| Parâmetro      | Default | Aumentar → efeito                                  | Diminuir → efeito                                  | Faixa recomendada |
|----------------|---------|-----------------------------------------------------|-----------------------------------------------------|-------------------|
| `sma_short`    | 9       | Sinal **mais lento**, menos ruído                   | Sinal **mais rápido**, mais ruído                   | 5–20              |
| `sma_long`     | 21      | Tendência **mais estável**                          | Mais sensível a mudanças                            | 20–60             |
| `ema_short`    | 12      | EMA curto **mais lento**                            | EMA curto **mais responsivo**                       | 5–20              |
| `ema_long`     | 26      | EMA longo **mais estável**                          | Mais sensível                                       | 20–60             |
| `wma_period`   | 10      | WMA **mais suave**                                  | WMA **mais responsiva**                             | 5–20              |
| `rsi_period`   | 14      | RSI **suaviza**, menos falsos positivos             | RSI **mais volátil**                                | 7–21              |
| `macd_fast`    | 12      | MACD **menos sensível**                             | MACD **mais sensível**                              | 8–15              |
| `macd_slow`    | 26      | MACD **mais lento/estável**                         | MACD **mais rápido**                                | 20–35             |
| `macd_signal`  | 9       | Menos cruzamentos                                   | Mais cruzamentos                                    | 5–12              |
| `bb_period`    | 20      | Bandas **mais estáveis**                            | Bandas **mais rápidas**                             | 10–30             |
| `bb_mult`      | 2.0     | Bandas **mais largas** (menos toques)               | Bandas **mais estreitas** (mais toques)             | 1.5–3.0           |
| `stoch_k`      | 14      | Oscilador **mais suave**                            | **Mais rápido** (ruidoso)                           | 9–21              |
| `stoch_d`      | 3       | Menos sinais (mais filtrado)                        | Mais sinais (menos filtrado)                        | 2–5               |
| `adx_period`   | 14      | ADX **mais suave**                                  | ADX **mais rápido**                                 | 7–21              |
| `fib_lookback` | 120     | Níveis com janela **maior**                         | Níveis com janela **menor**                         | 60–300            |

### Pesos de votação (confluência)

| Parâmetro           | Default | Aumentar → efeito                                       | Diminuir → efeito                                      | Faixa recomendada |
|---------------------|---------|---------------------------------------------------------|--------------------------------------------------------|-------------------|
| `w_ma_cross`        | 1.2     | **Cruzamento de EMAs** pesa mais                        | Pesa menos                                             | 0.5–2.0           |
| `w_macd`            | 1.2     | **MACD** pesa mais                                      | Pesa menos                                             | 0.5–2.0           |
| `w_rsi`             | 1.0     | **RSI** pesa mais                                       | Pesa menos                                             | 0.5–2.0           |
| `w_bb`              | 0.8     | **Bollinger** pesa mais                                 | Pesa menos                                             | 0.3–1.5           |
| `w_stoch`           | 0.8     | **Estocástico** pesa mais                               | Pesa menos                                             | 0.3–1.5           |
| `w_adx_trend`       | 0.8     | **Reforço por ADX** mais forte (quando tendência clara) | Reforço menor                                          | 0.3–1.5           |
| `w_volume_confirm`  | 0.6     | *(Reservado)* — na v1, volume atua via **multiplicador**| —                                                      | —                 |

> **Nota sobre Volume:** nesta versão, o volume atua como **multiplicador** do score total (ex.: **+15%** se volume > 120% da média; **-10%** se < 80%). O campo `w_volume_confirm` permanece para futura granularidade, mas **não é usado diretamente** no cálculo v1.

### Thresholds de gatilho

| Parâmetro       | Default | Aumentar → efeito                                     | Diminuir → efeito                                    | Faixa recomendada |
|-----------------|---------|-------------------------------------------------------|------------------------------------------------------|-------------------|
| `rsi_buy`       | 30      | Compra só em **sobrevenda mais extrema** (mais raro)  | Compra em sobrevenda **menos extrema** (mais sinais) | 20–40             |
| `rsi_sell`      | 70      | Venda só em **sobrecompra mais extrema** (mais raro)  | Venda em sobrecompra **menos extrema** (mais sinais) | 60–80             |
| `stoch_buy`     | 20      | Compra só com estocástico **muito baixo**             | Compra com estocástico menos baixo                    | 10–30             |
| `stoch_sell`    | 80      | Venda só com estocástico **muito alto**               | Venda com estocástico menos alto                      | 70–90             |
| `adx_trend_min` | 25      | Exige **tendência mais forte** p/ reforço ADX         | Reforça mais cedo (mesmo com tendência fraca)         | 15–35             |

### Previsão (janela curta)

| Parâmetro            | Default | Aumentar → efeito                                        | Diminuir → efeito                               | Faixa/limite |
|----------------------|---------|----------------------------------------------------------|-------------------------------------------------|--------------|
| `future_points`      | 5       | Mais pontos (até **5**)                                  | Menos pontos                                    | **1–5**      |
| `future_max_minutes` | 30      | Permite prever intervalos maiores (limite **30 min**)    | Mais conservador                                 | **1–30**     |
| `forecast_bias`      | "auto"  | `"trend"` força seguidor de tendência; `"mean_revert"` força reversão; `"auto"` escolhe por ADX | — | `"auto"`, `"trend"`, `"mean_revert"` |

---

## Perfis prontos (Conservador, Balanceado, Agressivo)

Use um destes blocos **no campo `config`** do JSON de entrada.

### 1) Conservador (menos sinais, maior filtro)
```json
{
  "sma_short": 12,
  "sma_long": 34,
  "ema_short": 13,
  "ema_long": 34,
  "wma_period": 14,
  "rsi_period": 14,
  "macd_fast": 12,
  "macd_slow": 26,
  "macd_signal": 9,
  "bb_period": 20,
  "bb_mult": 2.2,
  "stoch_k": 14,
  "stoch_d": 3,
  "adx_period": 14,
  "fib_lookback": 180,

  "w_ma_cross": 1.3,
  "w_macd": 1.3,
  "w_rsi": 1.0,
  "w_bb": 0.7,
  "w_stoch": 0.7,
  "w_adx_trend": 1.0,
  "w_volume_confirm": 0.6,

  "rsi_buy": 28,
  "rsi_sell": 72,
  "stoch_buy": 18,
  "stoch_sell": 82,
  "adx_trend_min": 27,

  "future_points": 3,
  "future_max_minutes": 20,
  "forecast_bias": "auto"
}
```

### 2) Balanceado (defaults otimizados)
```json
{
  "sma_short": 9,
  "sma_long": 21,
  "ema_short": 12,
  "ema_long": 26,
  "wma_period": 10,
  "rsi_period": 14,
  "macd_fast": 12,
  "macd_slow": 26,
  "macd_signal": 9,
  "bb_period": 20,
  "bb_mult": 2.0,
  "stoch_k": 14,
  "stoch_d": 3,
  "adx_period": 14,
  "fib_lookback": 120,

  "w_ma_cross": 1.2,
  "w_macd": 1.2,
  "w_rsi": 1.0,
  "w_bb": 0.8,
  "w_stoch": 0.8,
  "w_adx_trend": 0.8,
  "w_volume_confirm": 0.6,

  "rsi_buy": 30,
  "rsi_sell": 70,
  "stoch_buy": 20,
  "stoch_sell": 80,
  "adx_trend_min": 25,

  "future_points": 5,
  "future_max_minutes": 30,
  "forecast_bias": "auto"
}
```

### 3) Agressivo (mais sinais, mais responsivo)
```json
{
  "sma_short": 7,
  "sma_long": 18,
  "ema_short": 9,
  "ema_long": 21,
  "wma_period": 8,
  "rsi_period": 12,
  "macd_fast": 10,
  "macd_slow": 22,
  "macd_signal": 7,
  "bb_period": 18,
  "bb_mult": 1.8,
  "stoch_k": 12,
  "stoch_d": 3,
  "adx_period": 12,
  "fib_lookback": 100,

  "w_ma_cross": 1.3,
  "w_macd": 1.3,
  "w_rsi": 1.1,
  "w_bb": 0.9,
  "w_stoch": 0.9,
  "w_adx_trend": 0.7,
  "w_volume_confirm": 0.6,

  "rsi_buy": 32,
  "rsi_sell": 68,
  "stoch_buy": 22,
  "stoch_sell": 78,
  "adx_trend_min": 22,

  "future_points": 5,
  "future_max_minutes": 30,
  "forecast_bias": "trend"
}
```

> **Regras práticas:**
> - **Conservador**: períodos e limiares **maiores**, menos trades.
> - **Agressivo**: períodos e limiares **menores/estreitos**, mais trades (maior ruído).
> - **Balanceado**: *defaults*.

---

## Boas Práticas de Dados

- **Timeframe consistente** (ex.: todos os candles de 1m, 5m, 15m…).  
- **Ordenação crescente** por `T`.  
- Envie **pelo menos ~100 candles** para cálculos estáveis (indicadores têm warm-up).  
- Evite dados com **buracos** (timestamps irregulares).  
- Para **FlutterFlow**: mantenha listas de strings e converta seus arrays para strings antes de enviar.  

---

## FAQ

**1) Posso enviar `L` e `I` juntos?**  
Pode, mas apenas um é usado. A API prioriza `L`; se ausente, usa `I`.

**2) O que é a “precisão (%)”?**  
É uma medida **interna de confiança** na direção escolhida com base na **confluência** dos indicadores ponderados; **não é acurácia futura garantida**.

**3) As previsões futuras usam IA treinada?**  
Não. São **heurísticas rápidas** (trend vs mean-revert) para **janela curtíssima** (≤ 30 min), com **ruído determinístico**. Servem como **pistas** e não como projeção de preço.

**4) Onde ajusto a sensibilidade?**  
- Diminua períodos (EMA/RSI/MACD/BB/Stoch) e afrouxe thresholds para ficar **mais sensível/agressivo**.  
- Aumente períodos e aperte thresholds para ficar **mais conservador**.

---

## Aviso Importante

Esta API é **educacional** e **não constitui recomendação financeira**. Operar em mercados financeiros envolve **risco elevado**. Use por sua conta e risco. Sempre valide com **backtests**, **paper trading** e **gestão de risco** adequada.

---

## Anexos úteis

**Exemplo de cURL**
```bash
curl -X POST "http://localhost:10000/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "T": ["1622494800","1622495100","1622495400","1622495700","1622496000"],
    "O": ["100.50","101.00","100.80","101.20","101.50"],
    "H": ["101.50","101.30","101.20","101.40","101.60"],
    "L": ["100.40","100.60","100.50","100.70","100.90"],
    "C": ["101.00","100.90","101.10","101.30","101.00"],
    "V": ["1200","1500","1400","1300","1250"],
    "config": { "future_points": 5, "future_max_minutes": 30, "forecast_bias": "auto" }
  }'
```

**Arquivos do projeto**
- `main.py` — API inteira  
- `requirements.txt` — dependências  
- `Dockerfile` — build/run em container  
- `docker-compose.yml` — *(opcional)*  
