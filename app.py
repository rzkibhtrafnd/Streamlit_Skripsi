import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime, timedelta
from utils import load_artifacts

st.set_page_config(
    page_title="Prediksi IHSG - GRU",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>

.block-container{
    padding-top:2rem;
    padding-bottom:2rem;
}

.main-title{
    font-size:38px;
    font-weight:700;
}

.subtitle{
    color:#6c757d;
    font-size:16px;
    margin-bottom:30px;
}

.section-title{
    margin-top:35px;
    margin-bottom:15px;
    font-size:22px;
    font-weight:600;
}

.pred-card{
    background:#eef6ff;
    border:1px solid #d6e6ff;
    border-radius:12px;
    padding:30px;
    text-align:center;
}

.pred-value{
    font-size:38px;
    font-weight:700;
}

.metric-container{
    display:flex;
    gap:20px;
}

.metric-box{
    flex:1;
    background:#f8f9fa;
    border:1px solid #e6e6e6;
    border-radius:10px;
    padding:20px;
}

.metric-title{
    font-size:14px;
    color:#6c757d;
}

.metric-value{
    font-size:28px;
    font-weight:700;
}

</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">Prediksi IHSG Menggunakan GRU</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Prediksi jangka pendek 1 hari ke depan berbasis model Gated Recurrent Unit dengan data pasar realtime</div>', unsafe_allow_html=True)

ARTIFACT_DIR = "GRU_32-64_LR0.01_DO0.1_DJIA"

model, scaler_target, scaler_features, metadata, history = load_artifacts(ARTIFACT_DIR)
timestep = metadata["timestep"]

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

@st.cache_data
def load_market_data():

    end_date = datetime.today()
    start_date = end_date - timedelta(days=365 * 15)

    stocks = ['^JKSE', '^DJI']

    rename_mapping = {
        '^JKSE': 'IHSG',
        '^DJI': 'DJIA',
    }

    df_list = []

    for stock in stocks:
        data = yf.download(
            stock,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            auto_adjust=False,
            progress=False
        )

        if data.empty:
            continue

        df_close = data[['Close']].copy()
        df_close.columns = [rename_mapping[stock]]

        df_list.append(df_close)

    if not df_list:
        return pd.DataFrame()

    df_combined = pd.concat(df_list, axis=1, join='inner')
    df_combined = df_combined.dropna()
    df_combined = df_combined.reset_index()

    return df_combined

df = load_market_data()

if df.empty:
    st.error("Data tidak berhasil dimuat.")
    st.stop()

st.markdown('<div class="section-title">Pergerakan IHSG 15 Tahun Terakhir</div>', unsafe_allow_html=True)

fig_history = go.Figure()

fig_history.add_trace(go.Scatter(
    x=df["Date"],
    y=df["IHSG"],
    mode="lines",
    name="IHSG",
    line=dict(width=2)
))

fig_history.update_layout(
    height=500,
    template="plotly_white",
    margin=dict(l=20, r=20, t=30, b=20),
    xaxis_title="Tahun",
    yaxis_title="IHSG"
)

st.plotly_chart(fig_history, use_container_width=True)

target = df["IHSG"].values.reshape(-1, 1)
feature = df[["DJIA"]].values

scaled_target = scaler_target.transform(target)
scaled_feature = scaler_features.transform(feature)

combined = np.hstack([scaled_target, scaled_feature])

eval_window = 60
y_true, y_pred = [], []

for i in range(len(combined) - eval_window, len(combined)):
    if i - timestep < 0:
        continue

    X_eval = combined[i - timestep:i].reshape(1, timestep, combined.shape[1])
    pred_scaled = model.predict(X_eval, verbose=0)
    pred_value = scaler_target.inverse_transform(pred_scaled)[0, 0]

    y_pred.append(pred_value)
    y_true.append(target[i][0])

y_true = np.array(y_true)
y_pred = np.array(y_pred)

mape_value = mape(y_true, y_pred)
rmse_value = rmse(y_true, y_pred)
mse_value = mse(y_true, y_pred)

window = combined[-timestep:]
X_future = window.reshape(1, timestep, combined.shape[1])

future_scaled = model.predict(X_future, verbose=0)
future_value = scaler_target.inverse_transform(future_scaled)[0, 0]

last_date = df["Date"].iloc[-1]
next_date = last_date + pd.Timedelta(days=1)

st.markdown('<div class="section-title">Prediksi IHSG Hari Berikutnya</div>', unsafe_allow_html=True)

st.markdown(f"""
<div class="pred-card">
<div class="pred-value">{future_value:,.2f}</div>
Prediksi nilai IHSG untuk tanggal {next_date.date()}
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="section-title">Evaluasi Model</div>', unsafe_allow_html=True)

st.markdown(f"""
<div class="metric-container">

<div class="metric-box">
<div class="metric-title">MAPE</div>
<div class="metric-value">{mape_value:.2f}%</div>
</div>

<div class="metric-box">
<div class="metric-title">RMSE</div>
<div class="metric-value">{rmse_value:.2f}</div>
</div>

<div class="metric-box">
<div class="metric-title">MSE</div>
<div class="metric-value">{mse_value:.2f}</div>
</div>

</div>
""", unsafe_allow_html=True)

st.markdown('<div class="section-title">Evaluasi Prediksi dan Forecast</div>', unsafe_allow_html=True)

left_col, right_col = st.columns(2)

with left_col:

    fig_eval = go.Figure()

    fig_eval.add_trace(go.Scatter(
        x=df["Date"].iloc[-len(y_true):],
        y=y_true,
        mode="lines",
        name="IHSG Aktual"
    ))

    fig_eval.add_trace(go.Scatter(
        x=df["Date"].iloc[-len(y_pred):],
        y=y_pred,
        mode="lines",
        name="IHSG Prediksi",
        line=dict(dash="dash")
    ))

    fig_eval.update_layout(
        height=420,
        template="plotly_white",
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Tanggal",
        yaxis_title="IHSG"
    )

    st.plotly_chart(fig_eval, use_container_width=True)

with right_col:

    hist_df = df.tail(120)

    fig_forecast = go.Figure()

    fig_forecast.add_trace(go.Scatter(
        x=hist_df["Date"],
        y=hist_df["IHSG"],
        mode="lines",
        name="IHSG Aktual"
    ))

    fig_forecast.add_trace(go.Scatter(
        x=[next_date],
        y=[future_value],
        mode="markers+text",
        text=["Forecast"],
        textposition="top center",
        marker=dict(size=12),
        name="Prediksi"
    ))

    fig_forecast.update_layout(
        height=420,
        template="plotly_white",
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Tanggal",
        yaxis_title="IHSG"
    )

    st.plotly_chart(fig_forecast, use_container_width=True)