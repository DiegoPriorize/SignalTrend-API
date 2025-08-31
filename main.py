# main.py
# API de sinais de trading (COMPRA/VENDA/NEUTRO) com múltiplos indicadores
# Execução local:  uvicorn main:app --host 0.0.0.0 --port 8080

from typing import List, Optional, Dict, Any, Literal, Union
from fastapi import FastAPI, Body, Request
from pydantic import BaseModel, Field, validator
import numpy as np
import math
import uvicorn
from datetime import datetime, timezone
import json

app = FastAPI(title="Signals API", version="1.1.0",
              description="Gera sinais (COMPRA/VENDA/NEUTRO) com indicadores técnicos e previsões curtas.")

# -----------------------------
# Utilidades numéricas
# -----------------------------

def to_float_array(xs: List[str]) -> np.ndarray:
    return np.array([float(x) for x in xs], dtype=float)

def ema(arr: np.ndarray, period: int) -> np.ndarray:
    if period <= 1:
        return arr.copy()
    out = np.empty_like(arr); out[:] = np.nan
    if len(arr) == 0: return out
    alpha = 2 / (period + 1)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1 - alpha) * out[i - 1]
    return out

def sma(arr: np.ndarray, period: int) -> np.ndarray:
    if period <= 1:
        return arr.copy()
    if len(arr) < period:
        out = np.empty_like(arr); out[:] = np.nan; return out
    csum = np.cumsum(np.insert(arr, 0, 0.0))
    res = (csum[period:] - csum[:-period]) / period
    pad = np.empty(period - 1); pad[:] = np.nan
    return np.concatenate([pad, res])

def wma(arr: np.ndarray, period: int) -> np.ndarray:
    if period <= 1:
        return arr.copy()
    if len(arr) < period:
        out = np.empty_like(arr); out[:] = np.nan; return out
    weights = np.arange(1, period + 1)
    vals = []
    for i in range(period - 1, len(arr)):
        window = arr[i - period + 1:i + 1]
        vals.append(np.dot(window, weights) / weights.sum())
    pad = np.empty(period - 1); pad[:] = np.nan
    return np.concatenate([pad, np.array(vals)])

def rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    if len(close) < period + 1:
        out = np.empty_like(close); out[:] = np.nan; return out
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = ema(gain, period)
    avg_loss = ema(loss, period)
    rs = np.where(avg_loss == 0, np.nan, avg_gain / avg_loss)
    out = 100 - (100 / (1 + rs))
    out[:period] = np.nan
    return out

def macd(close: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9):
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def bollinger(close: np.ndarray, period: int = 20, mult: float = 2.0):
    mid = sma(close, period)
    if len(close) < period:
        ub = np.empty_like(close); lb = np.empty_like(close); ub[:] = np.nan; lb[:] = np.nan
        return mid, ub, lb
    res_std = []
    for i in range(period - 1, len(close)):
        window = close[i - period + 1:i + 1]
        res_std.append(np.std(window, ddof=0))
    pad = np.empty(period - 1); pad[:] = np.nan
    std = np.concatenate([pad, np.array(res_std)])
    return mid, mid + mult * std, mid - mult * std

def stochastic(high: np.ndarray, low: np.ndarray, close: np.ndarray, k_period: int = 14, d_period: int = 3):
    if len(close) < k_period:
        k = np.empty_like(close); d = np.empty_like(close); k[:] = np.nan; d[:] = np.nan
        return k, d
    k_vals = []
    for i in range(len(close)):
        if i < k_period - 1:
            k_vals.append(np.nan); continue
        lo = np.min(low[i - k_period + 1:i + 1])
        hi = np.max(high[i - k_period + 1:i + 1])
        k_val = 100 * (close[i] - lo) / (hi - lo) if hi != lo else 50.0
        k_vals.append(k_val)
    k = np.array(k_vals)
    d = sma(k, d_period)
    return k, d

def true_range(high, low, close):
    tr = np.empty_like(close)
    tr[0] = high[0] - low[0]
    for i in range(1, len(close)):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    return tr

def adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14):
    if len(close) < period + 1:
        out = np.empty_like(close); out[:] = np.nan
        return out, out, out
    up_move = high[1:] - high[:-1]
    down_move = low[:-1] - low[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = true_range(high, low, close)
    atr = ema(tr, period)
    plus_di = 100 * ema(np.insert(plus_dm, 0, 0.0), period) / atr
    minus_di = 100 * ema(np.insert(minus_dm, 0, 0.0), period) / atr
    dx = 100 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) == 0, np.nan, (plus_di + minus_di))
    adx_line = ema(dx, period)
    return adx_line, plus_di, minus_di

def fib_retracements(high: np.ndarray, low: np.ndarray, lookback: int = 120):
    n = len(high)
    if n == 0:
        return {}
    start = max(0, n - lookback)
    swin_hi = np.max(high[start:])
    swin_lo = np.min(low[start:])
    diff = swin_hi - swin_lo
    return {
        "0.0": float(swin_hi),
        "0.236": float(swin_hi - 0.236 * diff),
        "0.382": float(swin_hi - 0.382 * diff),
        "0.5": float(swin_hi - 0.5 * diff),
        "0.618": float(swin_hi - 0.618 * diff),
        "0.786": float(swin_hi - 0.786 * diff),
        "1.0": float(swin_lo)
    }

def slope(arr: np.ndarray, window: int = 5) -> float:
    if len(arr) < window:
        return 0.0
    y = arr[-window:]
    x = np.arange(window)
    x_mean = np.mean(x); y_mean = np.mean(y)
    num = np.sum((x - x_mean) * (y - y_mean))
    den = np.sum((x - x_mean) ** 2)
    return float(num / den) if den != 0 else 0.0

def pct(x: float) -> float:
    return float(max(0.0, min(100.0, x)))

def try_float(x):
    try:
        if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
            return None
        return float(x)
    except Exception:
        return None

# -----------------------------
# Entrada / Config
# -----------------------------

class Config(BaseModel):
    # Períodos
    sma_short: int = 9
    sma_long: int = 21
    ema_short: int = 12
    ema_long: int = 26
    wma_period: int = 10
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    bb_mult: float = 2.0
    stoch_k: int = 14
    stoch_d: int = 3
    adx_period: int = 14
    fib_lookback: int = 120
    # Pesos
    w_ma_cross: float = 1.2
    w_macd: float = 1.2
    w_rsi: float = 1.0
    w_bb: float = 0.8
    w_stoch: float = 0.8
    w_adx_trend: float = 0.8
    w_volume_confirm: float = 0.6
    # Thresholds
    rsi_buy: int = 30
    rsi_sell: int = 70
    stoch_buy: int = 20
    stoch_sell: int = 80
    adx_trend_min: int = 25
    # Previsões
    future_points: int = 5
    future_max_minutes: int = 30
    forecast_bias: Literal["trend","mean_revert","auto"] = "auto"

class Payload(BaseModel):
    # Aceita lista OU string (CSV/JSON-string)
    T: Union[List[str], str] = Field(..., description="Epoch timestamps (segundos)")
    O: Union[List[str], str]
    H: Union[List[str], str]
    C: Union[List[str], str]
    V: Union[List[str], str]
    L: Optional[Union[List[str], str]] = None
    I: Optional[Union[List[str], str]] = None  # compat: 'I' como low
    config: Optional[Config] = None

    @staticmethod
    def _coerce_to_list_str(v):
        if v is None:
            return None
        if isinstance(v, list):
            return [str(x) for x in v]
        if isinstance(v, (int, float)):
            return [str(v)]
        if isinstance(v, str):
            s = v.strip()
            # JSON-string de array
            if s.startswith('[') and s.endswith(']'):
                try:
                    arr = json.loads(s)
                    return [str(x) for x in arr]
                except Exception:
                    pass
            # CSV (também aceita ;)
            parts = [p.strip() for p in s.replace(';', ',').split(',') if p.strip() != ""]
            return parts
        # fallback
        return [str(v)]

    @validator("T", "O", "H", "C", "V", "L", "I", pre=True, always=True)
    def coerce_fields(cls, v):
        return cls._coerce_to_list_str(v)

    @validator("T","O","H","C","V")
    def same_length_required(cls, v, values, **kwargs):
        if not isinstance(v, list) or len(v) == 0:
            raise ValueError("lista vazia")
        return v

# -----------------------------
# Núcleo de análise
# -----------------------------

def analyze(data: Payload) -> Dict[str, Any]:
    T = np.array([int(t) for t in data.T], dtype=np.int64)
    O = to_float_array(data.O)
    H = to_float_array(data.H)
    C = to_float_array(data.C)
    V = to_float_array(data.V)

    if data.L is not None:
        L = to_float_array(data.L)
    elif data.I is not None:
        L = to_float_array(data.I)
    else:
        raise ValueError("Forneça 'L' (low) ou 'I' (compatibilidade)")

    n = len(T)
    if not (len(O) == len(H) == len(L) == len(C) == len(V) == n):
        raise ValueError("Todos os arrays devem ter o mesmo tamanho")

    cfg = data.config or Config()

    # Indicadores principais
    sma_s = sma(C, cfg.sma_short); sma_l = sma(C, cfg.sma_long)
    ema_s = ema(C, cfg.ema_short); ema_l = ema(C, cfg.ema_long)
    wma_p = wma(C, cfg.wma_period)
    rsi_val = rsi(C, cfg.rsi_period)
    macd_line, macd_sig, macd_hist = macd(C, cfg.macd_fast, cfg.macd_slow, cfg.macd_signal)
    bb_mid, bb_up, bb_lo = bollinger(C, cfg.bb_period, cfg.bb_mult)
    stoch_k, stoch_d = stochastic(H, L, C, cfg.stoch_k, cfg.stoch_d)
    adx_line, plus_di, minus_di = adx(H, L, C, cfg.adx_period)
    fibs = fib_retracements(H, L, cfg.fib_lookback)

    # Volume médio
    vol_ma = sma(V, max(5, min(20, n)))
    vol_boost = 1.0
    reasons = []
    buy_score = 0.0; sell_score = 0.0

    # 1) EMA cross + slope
    if not math.isnan(ema_s[-1]) and not math.isnan(ema_l[-1]):
        if ema_s[-1] > ema_l[-1]:
            buy_score += cfg.w_ma_cross; reasons.append("EMA curto acima de EMA longo (tendência de alta).")
        elif ema_s[-1] < ema_l[-1]:
            sell_score += cfg.w_ma_cross; reasons.append("EMA curto abaixo de EMA longo (tendência de baixa).")
    ema_slope = slope(ema_s, min(5, len(ema_s)))
    if ema_slope > 0: buy_score += 0.2
    elif ema_slope < 0: sell_score += 0.2

    # 2) MACD + hist slope
    if not math.isnan(macd_line[-1]) and not math.isnan(macd_sig[-1])):
        if macd_line[-1] > macd_sig[-1]:
            buy_score += cfg.w_macd; reasons.append("MACD acima da linha de sinal.")
        elif macd_line[-1] < macd_sig[-1]:
            sell_score += cfg.w_macd; reasons.append("MACD abaixo da linha de sinal.")
        hsl = slope(macd_hist, min(5, len(macd_hist)))
        if hsl > 0: buy_score += 0.2
        elif hsl < 0: sell_score += 0.2

    # 3) RSI
    if not math.isnan(rsi_val[-1])):
        if rsi_val[-1] < cfg.rsi_buy:
            buy_score += cfg.w_rsi; reasons.append(f"RSI {rsi_val[-1]:.1f} (sobrevendido).")
        elif rsi_val[-1] > cfg.rsi_sell:
            sell_score += cfg.w_rsi; reasons.append(f"RSI {rsi_val[-1]:.1f} (sobrecomprado).")

    # 4) Bollinger
    if not math.isnan(bb_up[-1]) and not math.isnan(bb_lo[-1])):
        if C[-1] <= bb_lo[-1]:
            buy_score += cfg.w_bb; reasons.append("Preço tocou/abaixo da banda inferior (reversão provável).")
        elif C[-1] >= bb_up[-1]:
            sell_score += cfg.w_bb; reasons.append("Preço tocou/acima da banda superior (reversão provável).")

    # 5) Estocástico
    if not math.isnan(stoch_k[-1]) and not math.isnan(stoch_d[-1])):
        prev_ok = (len(stoch_k) >= 2 and len(stoch_d) >= 2 and not math.isnan(stoch_k[-2]) and not math.isnan(stoch_d[-2]))
        if (stoch_k[-1] < cfg.stoch_buy) and (prev_ok and stoch_k[-2] <= stoch_d[-2]) and (stoch_k[-1] > stoch_d[-1]):
            buy_score += cfg.w_stoch; reasons.append("Estocástico: %K cruzou acima de %D em região de sobrevenda.")
        if (stoch_k[-1] > cfg.stoch_sell) and (prev_ok and stoch_k[-2] >= stoch_d[-2]) and (stoch_k[-1] < stoch_d[-1]):
            sell_score += cfg.w_stoch; reasons.append("Estocástico: %K cruzou abaixo de %D em região de sobrecompra.")

    # 6) ADX reforço
    adx_last = adx_line[-1] if not math.isnan(adx_line[-1]) else np.nan
    if not math.isnan(adx_last) and adx_last >= cfg.adx_trend_min:
        if ema_s[-1] > ema_l[-1] and macd_line[-1] > macd_sig[-1]:
            buy_score += cfg.w_adx_trend; reasons.append(f"ADX {adx_last:.1f} (tendência forte pró-compra).")
        if ema_s[-1] < ema_l[-1] and macd_line[-1] < macd_sig[-1]:
            sell_score += cfg.w_adx_trend; reasons.append(f"ADX {adx_last:.1f} (tendência forte pró-venda).")

    # 7) Volume (multiplicador)
    if not math.isnan(vol_ma[-1]):
        if V[-1] > vol_ma[-1] * 1.2:
            vol_boost = 1.15; reasons.append("Volume acima da média (confirmação).")
        elif V[-1] < vol_ma[-1] * 0.8:
            vol_boost = 0.9; reasons.append("Volume abaixo da média (confirmação fraca).")

    buy_score *= vol_boost; sell_score *= vol_boost

    total = buy_score + sell_score
    if total == 0:
        signal = "NEUTRO"; confidence = 50.0
    else:
        prob_buy = buy_score / total; prob_sell = sell_score / total
        if prob_buy > prob_sell:
            signal = "COMPRA"; confidence = pct(50 + 50 * (prob_buy - prob_sell))
        elif prob_sell > prob_buy:
            signal = "VENDA"; confidence = pct(50 + 50 * (prob_sell - prob_buy))
        else:
            signal = "NEUTRO"; confidence = 50.0

    if len(reasons) == 0:
        reasons.append("Sem confluência suficiente; sinal neutro.")

    future = forecast_short(T, O, H, L, C, V, cfg, adx_last, ema_s, ema_l, macd_line, macd_sig, rsi_val, bb_mid, bb_up, bb_lo)

    snap = {
        "price": try_float(C[-1]),
        "sma_short": try_float(sma_s[-1]),
        "sma_long": try_float(sma_l[-1]),
        "ema_short": try_float(ema_s[-1]),
        "ema_long": try_float(ema_l[-1]),
        "wma": try_float(wma_p[-1]),
        "rsi": try_float(rsi_val[-1]),
        "macd": try_float(macd_line[-1]),
        "macd_signal": try_float(macd_sig[-1]),
        "macd_hist": try_float(macd_hist[-1]),
        "bb_mid": try_float(bb_mid[-1]),
        "bb_up": try_float(bb_up[-1]),
        "bb_lo": try_float(bb_lo[-1]),
        "stoch_k": try_float(stoch_k[-1]),
        "stoch_d": try_float(stoch_d[-1]),
        "adx": try_float(adx_last),
        "plus_di": try_float(plus_di[-1]),
        "minus_di": try_float(minus_di[-1]),
        "volume": try_float(V[-1]),
        "volume_ma": try_float(vol_ma[-1]),
        "fib_levels": fibs
    }

    return {
        "signal": signal,
        "timestamp": int(T[-1]),
        "precisao_pct": round(confidence, 1),
        "motivos": reasons,
        "indicadores": snap,
        "futuros": future
    }

def forecast_short(T, O, H, L, C, V, cfg: Config, adx_last, ema_s, ema_l, macd_line, macd_sig, rsi_val, bb_mid, bb_up, bb_lo):
    # passo temporal (mediana dos deltas)
    if len(T) >= 2:
        deltas = np.diff(T)
        step = int(np.median(deltas))
        if step <= 0: step = int(deltas[-1]) if len(deltas) else 60
    else:
        step = 60
    max_steps = max(1, min(cfg.future_points, int((cfg.future_max_minutes * 60) // step)))
    if max_steps > 5: max_steps = 5
    if max_steps == 0: return []

    mode = cfg.forecast_bias
    if mode == "auto":
        if not math.isnan(adx_last) and adx_last >= cfg.adx_trend_min:
            mode = "trend"
        else:
            mode = "mean_revert"

    last_price = C[-1]
    ema_slope_recent = slope(ema_s, min(5, len(ema_s)))
    macd_slope_recent = slope(macd_line - macd_sig, min(5, len(macd_line)))

    Nvol = min(20, len(C))
    vol = np.std(C[-Nvol:]) if Nvol >= 2 else (0.005 * last_price)

    points = []
    ts = int(T[-1])
    for i in range(1, max_steps + 1):
        ts += step
        if mode == "trend":
            drift = 0.2 * ema_slope_recent + 0.15 * macd_slope_recent
        else:
            if not math.isnan(bb_mid[-1]):
                drift = 0.25 * (bb_mid[-1] - last_price)
            else:
                drift = 0.0
        noise = math.sin(ts % 1000) * (vol * 0.1)
        next_price = max(0.0001, last_price + drift + noise)

        future_signal, prob, why = classify_future(next_price, ema_s[-1], ema_l[-1], rsi_val[-1], bb_up[-1], bb_lo[-1])
        points.append({"timestamp": ts, "operacao": future_signal, "porcentagem": round(prob, 1), "motivo": why})
        last_price = next_price

    return points

def classify_future(price, ema_s_last, ema_l_last, rsi_last, bb_up_last, bb_lo_last):
    score_buy = 0.0; score_sell = 0.0; reasons = []
    if not math.isnan(ema_s_last) and not math.isnan(ema_l_last):
        if price > ema_s_last and ema_s_last > ema_l_last:
            score_buy += 1.0; reasons.append("Preço > EMA curto > EMA longo.")
        if price < ema_s_last and ema_s_last < ema_l_last:
            score_sell += 1.0; reasons.append("Preço < EMA curto < EMA longo.")
    if not math.isnan(rsi_last):
        if rsi_last < 35: score_buy += 0.4
        if rsi_last > 65: score_sell += 0.4
    if not math.isnan(bb_up_last) and not math.isnan(bb_lo_last):
        mid = (bb_up_last + bb_lo_last) / 2.0
        if price <= bb_lo_last: score_buy += 0.6; reasons.append("Preço ≤ banda inferior.")
        if price >= bb_up_last: score_sell += 0.6; reasons.append("Preço ≥ banda superior.")
        if price < mid: score_buy += 0.1
        else: score_sell += 0.1
    total = score_buy + score_sell
    if total == 0:
        return "NEUTRO", 50.0, "Sem confluência."
    if score_buy >= score_sell:
        prob = pct(50 + 50 * ((score_buy - score_sell) / (total if total else 1)))
        return "COMPRA", prob, "; ".join(reasons) if reasons else "Sinais levemente altistas."
    prob = pct(50 + 50 * ((score_sell - score_buy) / (total if total else 1)))
    return "VENDA", prob, "; ".join(reasons) if reasons else "Sinais levemente baixistas."

# -----------------------------
# Endpoints
# -----------------------------

@app.get("/health")
def health():
    return {"status": "ok", "utc": datetime.now(timezone.utc).isoformat()}

@app.post("/analyze")
def analyze_endpoint(payload: Payload = Body(...)) -> Dict[str, Any]:
    try:
        return analyze(payload)
    except Exception as e:
        return {"error": str(e)}

# Alias para compatibilidade (/signal)
@app.post("/signal")
def signal_alias(payload: Payload = Body(...)) -> Dict[str, Any]:
    try:
        return analyze(payload)
    except Exception as e:
        return {"error": str(e)}

# -----------------------------
# Execução local
# -----------------------------
if __name__ == "__main__":
    # Porta padrão 8080 (pode sobrescrever com env PORT ao usar Docker/Render via CMD shell)
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=False)
