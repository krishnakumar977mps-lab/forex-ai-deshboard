"""
Forex AI Dashboard (Mobile friendly)
Indicators: Stochastic (14,3,3) + EMA 47 only

Data source:
  - MT5 (XM) agar Windows PC par MT5 available hai
  - warna TwelveData API (cloud par chalane ke liye, phone se access)

requirements.txt:
    streamlit
    pandas
    pandas-ta
    plotly
    anthropic
    requests
    MetaTrader5   (sirf Windows PC par)

Secrets / env variables:
    ANTHROPIC_API_KEY, TWELVEDATA_API_KEY
"""
import os
import json
import requests
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import anthropic

try:
    import MetaTrader5 as mt5
    HAS_MT5 = True
except Exception:
    HAS_MT5 = False

st.set_page_config(page_title="Forex AI", layout="centered")
MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")

TF_LABELS = ["M15", "M30", "H1", "H4", "D1"]
TD_INTERVAL = {"M15": "15min", "M30": "30min", "H1": "1h", "H4": "4h", "D1": "1day"}
PAIRS = ["EUR/USD", "GBP/USD", "USD/JPY", "XAU/USD", "AUD/USD", "USD/CAD",
         "USD/CHF", "NZD/USD", "EUR/JPY", "GBP/JPY"]


def secret(name):
    val = os.getenv(name)
    if val:
        return val
    try:
        return st.secrets[name]
    except Exception:
        return None


# ---------- Data ----------
def candles_mt5(symbol, tf, n):
    tfmap = {"M15": mt5.TIMEFRAME_M15, "M30": mt5.TIMEFRAME_M30,
             "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4, "D1": mt5.TIMEFRAME_D1}
    if not mt5.initialize():
        return None
    rates = mt5.copy_rates_from_pos(symbol, tfmap[tf], 0, n)
    if rates is None or len(rates) == 0:
        return None
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    return df


def candles_td(pair, tf, n):
    key = secret("TWELVEDATA_API_KEY")
    if not key:
        st.error("TWELVEDATA_API_KEY set karein (twelvedata.com par free key milti hai).")
        return None
    r = requests.get(
        "https://api.twelvedata.com/time_series",
        params={"symbol": pair, "interval": TD_INTERVAL[tf],
                "outputsize": n, "apikey": key},
        timeout=20,
    ).json()
    if "values" not in r:
        st.error(r.get("message", "Data nahi mila"))
        return None
    df = pd.DataFrame(r["values"])
    df["time"] = pd.to_datetime(df["datetime"])
    for c in ["open", "high", "low", "close"]:
        df[c] = df[c].astype(float)
    return df.sort_values("time").reset_index(drop=True)[
        ["time", "open", "high", "low", "close"]]


# ---------- Indicators ----------
def add_indicators(df):
    df["ema47"] = ta.ema(df["close"], length=47)
    st_ = ta.stoch(df["high"], df["low"], df["close"], k=14, d=3, smooth_k=3)
    df["stoch_k"] = st_["STOCHk_14_3_3"]
    df["stoch_d"] = st_["STOCHd_14_3_3"]
    return df


def swing_levels(df, window=5, count=3):
    highs, lows = [], []
    for i in range(window, len(df) - window):
        seg = df.iloc[i - window: i + window + 1]
        if df["high"].iloc[i] == seg["high"].max():
            highs.append(df["high"].iloc[i])
        if df["low"].iloc[i] == seg["low"].min():
            lows.append(df["low"].iloc[i])
    price = df["close"].iloc[-1]
    res = sorted([h for h in highs if h > price])[:count]
    sup = sorted([l for l in lows if l < price], reverse=True)[:count]
    return sup, res


def build_chart(df, sup, res, name):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.72, 0.28], vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(x=df["time"], open=df["open"], high=df["high"],
                                 low=df["low"], close=df["close"], name=name),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=df["time"], y=df["ema47"], name="EMA 47",
                             line=dict(width=1.5, color="orange")), row=1, col=1)
    for s in sup:
        fig.add_hline(y=s, line_dash="dot", line_color="green", row=1, col=1)
    for r in res:
        fig.add_hline(y=r, line_dash="dot", line_color="red", row=1, col=1)
    fig.add_trace(go.Scatter(x=df["time"], y=df["stoch_k"], name="%K",
                             line=dict(width=1.2)), row=2, col=1)
    fig.add_trace(go.Scatter(x=df["time"], y=df["stoch_d"], name="%D",
                             line=dict(width=1.2)), row=2, col=1)
    fig.add_hline(y=80, line_dash="dash", row=2, col=1)
    fig.add_hline(y=20, line_dash="dash", row=2, col=1)
    fig.update_layout(height=560, xaxis_rangeslider_visible=False,
                      margin=dict(l=5, r=5, t=10, b=5),
                      legend=dict(orientation="h", y=1.02))
    return fig


# ---------- Claude ----------
def ask_claude(name, tf, df, sup, res):
    last, prev = df.iloc[-1], df.iloc[-2]
    recent = df.tail(30)[["time", "open", "high", "low", "close"]].copy()
    recent["time"] = recent["time"].astype(str)
    payload = {
        "symbol": name, "timeframe": tf,
        "price": float(last["close"]),
        "ema47": float(last["ema47"]),
        "stoch_k": float(last["stoch_k"]), "stoch_d": float(last["stoch_d"]),
        "prev_stoch_k": float(prev["stoch_k"]), "prev_stoch_d": float(prev["stoch_d"]),
        "support_levels": sup, "resistance_levels": res,
        "last_30_candles": recent.to_dict("records"),
    }
    system = (
        "Tum ek forex technical analyst ho. Sirf diye gaye data se analysis karo, "
        "sirf Stochastic (14,3,3) aur EMA 47 use karo. Hinglish mein, chhota aur "
        "mobile screen ke hisaab se likho. Format: 1) Trend (price vs EMA 47) "
        "2) Stochastic signal (overbought/oversold, %K/%D crossover) "
        "3) Buy zone / Sell zone levels 4) Stop-loss aur take-profit "
        "5) Setup kab fail hoga. Aakhir mein ek line: yeh financial advice nahi hai."
    )
    key = secret("ANTHROPIC_API_KEY")
    if not key:
        return "ANTHROPIC_API_KEY set karein."
    client = anthropic.Anthropic(api_key=key)
    msg = client.messages.create(model=MODEL, max_tokens=1000, system=system,
                                 messages=[{"role": "user", "content": json.dumps(payload)}])
    return msg.content[0].text


# ---------- UI ----------
st.title("Forex AI Dashboard")

source = st.radio("Data source", ["TwelveData (phone/cloud)", "MT5 (XM, PC)"],
                  horizontal=True, index=1 if HAS_MT5 else 0)
c1, c2 = st.columns(2)
tf = c2.selectbox("Timeframe", TF_LABELS, index=2)

if source.startswith("MT5"):
    if not HAS_MT5:
        st.error("MT5 library sirf Windows PC par chalti hai. TwelveData chunein.")
        st.stop()
    symbol = c1.text_input("Symbol (XM jaisa)", "EURUSD")
    df = candles_mt5(symbol, tf, 300)
else:
    symbol = c1.selectbox("Pair", PAIRS)
    df = candles_td(symbol, tf, 300)

if df is None:
    st.stop()

df = add_indicators(df)
sup, res = swing_levels(df)
last = df.iloc[-1]

m1, m2, m3 = st.columns(3)
m1.metric("Price", f"{last['close']:.5f}")
m2.metric("Trend", "Up" if last["close"] > last["ema47"] else "Down")
m3.metric("Stoch %K", f"{last['stoch_k']:.1f}")

st.plotly_chart(build_chart(df, sup, res, symbol), use_container_width=True)
st.caption("Support: " + ", ".join(f"{s:.5f}" for s in sup)
           + "  |  Resistance: " + ", ".join(f"{r:.5f}" for r in res))

if st.button("Claude se analysis karwao", use_container_width=True):
    with st.spinner("Analysis ho raha hai..."):
        st.markdown(ask_claude(symbol, tf, df, sup, res))
