# Signals API — COMPRA/VENDA com Indicadores Técnicos

API HTTP (FastAPI) que recebe séries de OHLCV (listas de strings) e retorna:
- **Sinal agregado**: `COMPRA`, `VENDA` ou `NEUTRO`
- **Timestamp** do candle analisado
- **Precisão (%)** estimada (confiança do consenso)
- **Motivos** (explicação textual)
- **Snapshot de indicadores**
- **Previsões curtas** (até 5 timestamps ≤ 30 min no futuro) com operação e probabilidade

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

---

## Arquitetura & Estratégia

### Indicadores Utilizados

1. **Médias Móveis (SMA/EMA/WMA)**
2. **RSI (Índice de Força Relativa)**
3. **MACD (Convergência/Divergência de Médias Móveis)**
4. **Bandas de Bollinger**
5. **Estocástico (%K e %D)**
6. **ADX (Average Directional Index)**
7. **Fibonacci (retracements)**
8. **Volume**

### Lógica de Sinal

- Calcula-se **scores de COMPRA e VENDA** a partir de regras.
- Sinal final = maior score, ou **NEUTRO** se empate.

### Precisão (confiança)

- Baseada no desequilíbrio entre scores de compra e venda.
- Valores próximos → 50%. Mais confluência → mais confiança.

### Previsões Futuras (≤ 30 min)

- Até 5 pontos à frente, baseados em heurística trend vs mean-revert.
- Cada previsão contém `timestamp`, `operacao`, `porcentagem`, `motivo`.

---

## Instalação & Execução

### Rodar Local (sem Docker)

```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 10000
```

### Docker

```bash
docker build -t signals-api .
docker run --rm -p 10000:10000 signals-api
```

### Render.com

**Opção A — Sem Dockerfile**  
Start command:  
`uvicorn main:app --host 0.0.0.0 --port $PORT`

**Opção B — Com Dockerfile**  
Selecionar Docker no Render e usar o `Dockerfile` incluído.

---

## Endpoints

### `POST /analyze`

Entrada:
```json
{
  "T": ["1622494800","1622495100","1622495400"],
  "O": ["100.50","101.00","100.80"],
  "H": ["101.50","101.30","101.20"],
  "L": ["100.40","100.60","100.50"],
  "C": ["101.00","100.90","101.10"],
  "V": ["1200","1500","1400"]
}
```

Saída (resumida):
```json
{
  "signal": "COMPRA",
  "timestamp": 1622495400,
  "precisao_pct": 63.8,
  "motivos": ["EMA curto acima de EMA longo"],
  "indicadores": {...},
  "futuros": [...]
}
```

### `GET /health`

Retorna `{ "status": "ok", "utc": "<timestamp>" }`

---

## Configuração (`config`)

Veja detalhes no README completo com efeitos, ranges e limites.

---

## Perfis prontos

- **Conservador:** menos sinais, filtros mais rígidos.
- **Balanceado:** configuração padrão.
- **Agressivo:** mais sinais, thresholds mais leves.

---

## Aviso Importante

Esta API é **educacional**. Não é recomendação de investimento.
