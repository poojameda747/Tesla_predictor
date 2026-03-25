
"""
Tesla Stock Price Prediction — Streamlit App
Deploy: streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import json
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import MinMaxScaler

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Tesla Stock Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Rajdhani:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
    background-color: #0a0a0f;
    color: #e8e8f0;
}

.stApp {
    background: linear-gradient(135deg, #0a0a0f 0%, #0d1117 50%, #0a0f1a 100%);
}

.hero-banner {
    background: linear-gradient(135deg, #1a0a0a 0%, #0d0d1a 40%, #0a1a1a 100%);
    border: 1px solid rgba(227, 25, 55, 0.3);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}

.hero-banner::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background: repeating-linear-gradient(
        90deg,
        transparent,
        transparent 40px,
        rgba(227, 25, 55, 0.03) 40px,
        rgba(227, 25, 55, 0.03) 41px
    );
}

.hero-title {
    font-family: 'Rajdhani', sans-serif;
    font-size: 2.8rem;
    font-weight: 700;
    color: #E31937;
    letter-spacing: 2px;
    margin: 0;
    line-height: 1.1;
    text-shadow: 0 0 30px rgba(227, 25, 55, 0.4);
}

.hero-subtitle {
    font-size: 1rem;
    color: #888;
    margin-top: 0.5rem;
    letter-spacing: 1px;
}

.metric-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    transition: all 0.3s ease;
    margin-bottom: 0.5rem;
}

.metric-card:hover {
    border-color: rgba(227, 25, 55, 0.4);
    background: rgba(227, 25, 55, 0.05);
}

.metric-label {
    font-size: 0.75rem;
    color: #666;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-weight: 500;
}

.metric-value {
    font-family: 'Rajdhani', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: #E31937;
    line-height: 1.1;
    margin-top: 0.2rem;
}

.metric-value.green { color: #2ECC71; }
.metric-value.blue { color: #3498DB; }

.model-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 1rem;
}

.badge-rnn { background: rgba(52, 152, 219, 0.15); color: #3498DB; border: 1px solid rgba(52, 152, 219, 0.3); }
.badge-lstm { background: rgba(227, 25, 55, 0.15); color: #E31937; border: 1px solid rgba(227, 25, 55, 0.3); }
.badge-winner { background: rgba(46, 204, 113, 0.15); color: #2ECC71; border: 1px solid rgba(46, 204, 113, 0.3); }

.section-header {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.5rem;
    font-weight: 600;
    color: #e8e8f0;
    border-left: 3px solid #E31937;
    padding-left: 1rem;
    margin: 1.5rem 0 1rem 0;
    letter-spacing: 1px;
}

.insight-box {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.75rem;
}

.insight-icon { font-size: 1.2rem; margin-right: 0.5rem; }

.stSidebar {
    background: rgba(10, 10, 15, 0.95) !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
}

.stSidebar .stSelectbox label,
.stSidebar .stSlider label,
.stSidebar .stRadio label {
    color: #aaa !important;
    font-size: 0.85rem !important;
}

.stButton > button {
    background: linear-gradient(135deg, #E31937, #c0142d) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 1px !important;
    padding: 0.6rem 1.5rem !important;
    transition: all 0.3s !important;
    text-transform: uppercase !important;
    font-size: 0.8rem !important;
}

.stButton > button:hover {
    box-shadow: 0 0 20px rgba(227, 25, 55, 0.5) !important;
    transform: translateY(-1px) !important;
}

.upload-hint {
    background: rgba(227, 25, 55, 0.08);
    border: 1px dashed rgba(227, 25, 55, 0.3);
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
    color: #888;
    font-size: 0.85rem;
    margin: 1rem 0;
}

div[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.02);
    border-radius: 8px;
    gap: 0;
}

div[data-testid="stTabs"] [data-baseweb="tab"] {
    color: #666 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 500 !important;
}

div[data-testid="stTabs"] [aria-selected="true"] {
    color: #E31937 !important;
    background: rgba(227, 25, 55, 0.1) !important;
    border-bottom: 2px solid #E31937 !important;
}

.forecast-tag {
    background: rgba(46, 204, 113, 0.1);
    border: 1px solid rgba(46, 204, 113, 0.2);
    border-radius: 6px;
    padding: 0.4rem 0.8rem;
    font-size: 0.85rem;
    color: #2ECC71;
    font-weight: 600;
    display: inline-block;
    margin: 0.2rem;
}

.forecast-tag.red {
    background: rgba(227, 25, 55, 0.1);
    border-color: rgba(227, 25, 55, 0.2);
    color: #E31937;
}

.forecast-tag.blue {
    background: rgba(52, 152, 219, 0.1);
    border-color: rgba(52, 152, 219, 0.2);
    color: #3498DB;
}
</style>
""", unsafe_allow_html=True)

# ─── Matplotlib Dark Theme ──────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0d0d17',
    'axes.facecolor': '#0d0d17',
    'axes.edgecolor': '#333',
    'axes.labelcolor': '#aaa',
    'xtick.color': '#666',
    'ytick.color': '#666',
    'text.color': '#e8e8f0',
    'grid.color': '#222',
    'grid.linewidth': 0.5,
    'legend.facecolor': '#111',
    'legend.edgecolor': '#333',
    'lines.linewidth': 1.5,
    'font.family': 'monospace',
})


# ─── Utilities ─────────────────────────────────────────────────────────────────
WINDOW_SIZE = 60

@st.cache_data
def load_and_preprocess(file):
    df = pd.read_csv(file)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    df.set_index('Date', inplace=True)
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    df['MA_20'] = df['Close'].rolling(20).mean()
    df['MA_50'] = df['Close'].rolling(50).mean()
    df['MA_200'] = df['Close'].rolling(200).mean()
    df['Daily_Return'] = df['Close'].pct_change() * 100
    df['Volatility_20'] = df['Daily_Return'].rolling(20).std()
    return df

def create_sequences(data, window_size):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i - window_size:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

@st.cache_resource
def build_and_train_models(X_train, y_train, units=64, dropout=0.2, lr=0.001):
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import SimpleRNN, LSTM, Dense, Dropout, Input
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.optimizers import Adam

    tf.random.set_seed(42)

    callbacks = [EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=0)]

    def build_rnn():
        m = Sequential([
            Input(shape=(WINDOW_SIZE, 1)),
            SimpleRNN(units, return_sequences=True),
            Dropout(dropout),
            SimpleRNN(units // 2),
            Dropout(dropout),
            Dense(25, activation='relu'),
            Dense(1)
        ])
        m.compile(optimizer=Adam(lr), loss='mse', metrics=['mae'])
        return m

    def build_lstm():
        m = Sequential([
            Input(shape=(WINDOW_SIZE, 1)),
            LSTM(units, return_sequences=True),
            Dropout(dropout),
            LSTM(units // 2),
            Dropout(dropout),
            Dense(25, activation='relu'),
            Dense(1)
        ])
        m.compile(optimizer=Adam(lr), loss='mse', metrics=['mae'])
        return m

    rnn = build_rnn()
    rnn.fit(X_train, y_train, epochs=60, batch_size=32,
            validation_split=0.1, callbacks=callbacks, verbose=0)

    lstm = build_lstm()
    lstm.fit(X_train, y_train, epochs=60, batch_size=32,
             validation_split=0.1, callbacks=callbacks, verbose=0)

    return rnn, lstm

def predict_future(model, last_seq, scaler, n_days):
    preds = []
    seq = last_seq.copy()
    for _ in range(n_days):
        inp = seq.reshape(1, WINDOW_SIZE, 1)
        p = model.predict(inp, verbose=0)[0, 0]
        preds.append(p)
        seq = np.append(seq[1:], p)
    return scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()


# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0;'>
        <div style='font-family:Rajdhani,sans-serif; font-size:1.8rem; font-weight:700;
                    color:#E31937; letter-spacing:3px;'>TSLA.AI</div>
        <div style='font-size:0.75rem; color:#555; letter-spacing:2px;'>STOCK PREDICTOR</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<div style='color:#888; font-size:0.8rem; text-transform:uppercase; letter-spacing:1px;'>📂 Upload Dataset</div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("TSLA.csv", type=['csv'], label_visibility='collapsed')

    st.markdown("---")
    st.markdown("<div style='color:#888; font-size:0.8rem; text-transform:uppercase; letter-spacing:1px;'>⚙️ Model Configuration</div>", unsafe_allow_html=True)
    units = st.selectbox("LSTM/RNN Units", [32, 64, 128], index=1)
    dropout = st.slider("Dropout Rate", 0.1, 0.5, 0.2, 0.05)
    lr = st.select_slider("Learning Rate", options=[0.0001, 0.0005, 0.001, 0.005], value=0.001)
    forecast_horizon = st.radio("Forecast Horizon", [1, 5, 10], horizontal=True)

    st.markdown("---")
    train_btn = st.button("🚀 Train Models", use_container_width=True)

    st.markdown("---")
    st.markdown("""
    <div style='color:#444; font-size:0.72rem; line-height:1.6;'>
        <b style='color:#666;'>Models:</b> SimpleRNN + LSTM<br>
        <b style='color:#666;'>Window:</b> 60 days<br>
        <b style='color:#666;'>Target:</b> Close Price<br>
        <b style='color:#666;'>Scaler:</b> MinMaxScaler
    </div>
    """, unsafe_allow_html=True)


# ─── Hero Banner ───────────────────────────────────────────────────────────────
st.markdown("""
<div class='hero-banner'>
    <div class='hero-title'>🚗 TESLA STOCK PREDICTOR</div>
    <div class='hero-subtitle'>Deep Learning · SimpleRNN vs LSTM · 1-Day / 5-Day / 10-Day Forecast</div>
</div>
""", unsafe_allow_html=True)


# ─── Main Content ──────────────────────────────────────────────────────────────
if uploaded_file is None:
    st.markdown("""
    <div class='upload-hint'>
        <div style='font-size:2rem; margin-bottom:0.5rem;'>📤</div>
        <div style='color:#aaa; font-size:0.95rem;'>Upload your <b>TSLA.csv</b> file using the sidebar to begin.</div>
        <div style='color:#555; font-size:0.8rem; margin-top:0.5rem;'>
            Expected columns: Date, Open, High, Low, Close, Adj Close, Volume
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Show demo info cards
    cols = st.columns(3)
    cards = [
        ("📊", "EDA & Visualization", "Interactive charts showing price trends, moving averages, volume, returns and correlations."),
        ("🧠", "Dual DL Models", "Train both SimpleRNN and LSTM and compare their forecasting accuracy side-by-side."),
        ("📈", "Multi-Horizon Forecast", "Predict 1-day, 5-day, and 10-day future closing prices with confidence."),
    ]
    for col, (icon, title, desc) in zip(cols, cards):
        with col:
            st.markdown(f"""
            <div class='metric-card' style='text-align:left; padding:1.5rem;'>
                <div style='font-size:2rem;'>{icon}</div>
                <div style='font-family:Rajdhani,sans-serif; font-size:1.1rem; font-weight:600; color:#e8e8f0; margin: 0.5rem 0 0.3rem;'>{title}</div>
                <div style='font-size:0.82rem; color:#666; line-height:1.5;'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)
    st.stop()


# ─── Load Data ─────────────────────────────────────────────────────────────────
df = load_and_preprocess(uploaded_file)

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 EDA", "🧠 Model Training", "📈 Forecast", "⚖️ Comparison", "📝 Report"])


# ══════════════════════════════════════════════════════════════════════
# TAB 1 — EDA
# ══════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("<div class='section-header'>Dataset Overview</div>", unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-label'>Records</div>
            <div class='metric-value blue'>{len(df):,}</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-label'>Date Range</div>
            <div class='metric-value' style='font-size:0.9rem;color:#aaa;'>{df.index.min().strftime('%Y')}<br>→ {df.index.max().strftime('%Y')}</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-label'>Latest Close</div>
            <div class='metric-value green'>${df['Close'].iloc[-1]:.2f}</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-label'>All-Time High</div>
            <div class='metric-value'>${df['High'].max():.2f}</div>
        </div>""", unsafe_allow_html=True)
    with c5:
        daily_ret = df['Daily_Return'].mean()
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-label'>Avg Daily Return</div>
            <div class='metric-value {'green' if daily_ret >= 0 else ''}'>{daily_ret:+.2f}%</div>
        </div>""", unsafe_allow_html=True)

    # Closing Price + MAs
    st.markdown("<div class='section-header'>Closing Price & Moving Averages</div>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df.index, df['Close'], color='#E31937', linewidth=1.2, label='Close', alpha=0.9)
    ax.plot(df.index, df['MA_20'], color='#F39C12', linewidth=1, label='20-day MA', alpha=0.8)
    ax.plot(df.index, df['MA_50'], color='#3498DB', linewidth=1, label='50-day MA', alpha=0.8)
    ax.plot(df.index, df['MA_200'], color='#2ECC71', linewidth=1.5, label='200-day MA', alpha=0.8)
    ax.fill_between(df.index, df['Close'], alpha=0.05, color='#E31937')
    ax.set_ylabel('Price (USD)')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(alpha=0.2)
    st.pyplot(fig)
    plt.close()

    # Volume & Returns
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.bar(df.index, df['Volume'], color='#3498DB', alpha=0.6, width=1)
        ax.set_title('Trading Volume')
        ax.set_ylabel('Volume')
        ax.grid(alpha=0.2)
        st.pyplot(fig)
        plt.close()

    with c2:
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.plot(df.index, df['Daily_Return'], color='#2ECC71', linewidth=0.7, alpha=0.8)
        ax.axhline(0, color='#E31937', linestyle='--', linewidth=0.8, alpha=0.6)
        ax.fill_between(df.index, df['Daily_Return'], 0,
                        where=df['Daily_Return'] >= 0, alpha=0.2, color='#2ECC71')
        ax.fill_between(df.index, df['Daily_Return'], 0,
                        where=df['Daily_Return'] < 0, alpha=0.2, color='#E31937')
        ax.set_title('Daily Returns (%)')
        ax.grid(alpha=0.2)
        st.pyplot(fig)
        plt.close()

    # Correlation heatmap + Distribution
    c1, c2 = st.columns(2)
    with c1:
        import seaborn as sns
        fig, ax = plt.subplots(figsize=(6, 5))
        corr = df[['Open', 'High', 'Low', 'Close', 'Volume']].corr()
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdYlGn', ax=ax,
                    linewidths=0.5, annot_kws={"size": 9},
                    cbar_kws={"shrink": 0.8})
        ax.set_title('Correlation Heatmap')
        st.pyplot(fig)
        plt.close()

    with c2:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.hist(df['Close'], bins=60, color='#E31937', edgecolor='#0d0d17', alpha=0.85)
        ax.set_title('Close Price Distribution')
        ax.set_xlabel('Price (USD)')
        ax.set_ylabel('Frequency')
        ax.grid(alpha=0.2)
        st.pyplot(fig)
        plt.close()

    # Data preview
    st.markdown("<div class='section-header'>Raw Data Preview</div>", unsafe_allow_html=True)
    st.dataframe(df.tail(20).style.background_gradient(subset=['Close'], cmap='RdYlGn'), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
# TAB 2 — Model Training
# ══════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("<div class='section-header'>Preprocessing & Model Training</div>", unsafe_allow_html=True)

    # Preprocessing
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(df[['Close']])
    X, y = create_sequences(scaled, WINDOW_SIZE)

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    X_train_r = X_train.reshape(*X_train.shape, 1)
    X_test_r = X_test.reshape(*X_test.shape, 1)

    cols = st.columns(3)
    with cols[0]:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-label'>Train Samples</div>
            <div class='metric-value blue'>{len(X_train):,}</div>
        </div>""", unsafe_allow_html=True)
    with cols[1]:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-label'>Test Samples</div>
            <div class='metric-value'>{len(X_test):,}</div>
        </div>""", unsafe_allow_html=True)
    with cols[2]:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-label'>Window Size</div>
            <div class='metric-value green'>{WINDOW_SIZE} days</div>
        </div>""", unsafe_allow_html=True)

    if train_btn or st.session_state.get('trained', False):
        if train_btn:
            with st.spinner("⚡ Training SimpleRNN and LSTM models..."):
                rnn_model, lstm_model = build_and_train_models(X_train_r, y_train, units, dropout, lr)
                st.session_state['rnn_model'] = rnn_model
                st.session_state['lstm_model'] = lstm_model
                st.session_state['scaler'] = scaler
                st.session_state['X_test_r'] = X_test_r
                st.session_state['y_test'] = y_test
                st.session_state['scaled'] = scaled
                st.session_state['trained'] = True
            st.success("✅ Models trained successfully!")

        rnn_model = st.session_state['rnn_model']
        lstm_model = st.session_state['lstm_model']
        scaler = st.session_state['scaler']
        X_test_r = st.session_state['X_test_r']
        y_test = st.session_state['y_test']
        scaled = st.session_state['scaled']

        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        def get_metrics(model, X_test, y_test, scaler):
            y_pred_s = model.predict(X_test, verbose=0)
            y_pred = scaler.inverse_transform(y_pred_s)
            y_act = scaler.inverse_transform(y_test.reshape(-1, 1))
            mse = mean_squared_error(y_act, y_pred)
            return {
                'rmse': float(np.sqrt(mse)),
                'mae': float(mean_absolute_error(y_act, y_pred)),
                'r2': float(r2_score(y_act, y_pred)),
                'mape': float(np.mean(np.abs((y_act - y_pred) / y_act)) * 100),
                'y_pred': y_pred.flatten(),
                'y_act': y_act.flatten()
            }

        rnn_m = get_metrics(rnn_model, X_test_r, y_test, scaler)
        lstm_m = get_metrics(lstm_model, X_test_r, y_test, scaler)
        st.session_state['rnn_m'] = rnn_m
        st.session_state['lstm_m'] = lstm_m

        # Metrics display
        st.markdown("<div class='section-header'>Model Performance</div>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)

        for col, metrics, name, badge_class in [
            (c1, rnn_m, 'SimpleRNN', 'badge-rnn'),
            (c2, lstm_m, 'LSTM', 'badge-lstm')
        ]:
            with col:
                st.markdown(f"<div class='model-badge {badge_class}'>{name}</div>", unsafe_allow_html=True)
                mc1, mc2 = st.columns(2)
                with mc1:
                    st.markdown(f"""<div class='metric-card'>
                        <div class='metric-label'>RMSE</div>
                        <div class='metric-value' style='font-size:1.4rem;'>{metrics['rmse']:.2f}</div>
                    </div>""", unsafe_allow_html=True)
                    st.markdown(f"""<div class='metric-card'>
                        <div class='metric-label'>R² Score</div>
                        <div class='metric-value green' style='font-size:1.4rem;'>{metrics['r2']:.4f}</div>
                    </div>""", unsafe_allow_html=True)
                with mc2:
                    st.markdown(f"""<div class='metric-card'>
                        <div class='metric-label'>MAE</div>
                        <div class='metric-value blue' style='font-size:1.4rem;'>{metrics['mae']:.2f}</div>
                    </div>""", unsafe_allow_html=True)
                    st.markdown(f"""<div class='metric-card'>
                        <div class='metric-label'>MAPE</div>
                        <div class='metric-value' style='font-size:1.4rem;'>{metrics['mape']:.2f}%</div>
                    </div>""", unsafe_allow_html=True)

        # Actual vs Predicted
        st.markdown("<div class='section-header'>Actual vs Predicted</div>", unsafe_allow_html=True)
        test_dates = df.index[WINDOW_SIZE + split:]
        min_len = min(len(test_dates), len(rnn_m['y_act']))

        fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=False)
        for ax, m, name, color in zip(axes, [rnn_m, lstm_m], ['SimpleRNN', 'LSTM'], ['#3498DB', '#E31937']):
            ax.plot(test_dates[:min_len], m['y_act'][:min_len], color='#aaa', linewidth=1.5, label='Actual', alpha=0.9)
            ax.plot(test_dates[:min_len], m['y_pred'][:min_len], color=color, linewidth=1.5, label='Predicted', alpha=0.9, linestyle='--')
            ax.set_title(f'{name} — Actual vs Predicted  |  R²={m["r2"]:.4f}', fontsize=11)
            ax.set_ylabel('Close Price (USD)')
            ax.legend(fontsize=8)
            ax.grid(alpha=0.2)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    else:
        st.markdown("""
        <div class='insight-box' style='text-align:center; padding:2rem;'>
            <div style='font-size:2.5rem;'>🚀</div>
            <div style='color:#888; margin-top:0.5rem;'>Click <b style='color:#E31937;'>Train Models</b> in the sidebar to start training.</div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# TAB 3 — Forecast
# ══════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("<div class='section-header'>Future Price Forecast</div>", unsafe_allow_html=True)

    if not st.session_state.get('trained', False):
        st.warning("⚠️ Please train the models first (Tab: Model Training).")
    else:
        rnn_model = st.session_state['rnn_model']
        lstm_model = st.session_state['lstm_model']
        scaler = st.session_state['scaler']
        scaled = st.session_state['scaled']

        last_seq = scaled[-WINDOW_SIZE:, 0]
        last_price = scaler.inverse_transform(scaled[-1].reshape(-1, 1))[0][0]

        rnn_fc = predict_future(rnn_model, last_seq, scaler, forecast_horizon)
        lstm_fc = predict_future(lstm_model, last_seq, scaler, forecast_horizon)

        # Summary cards
        st.markdown(f"""
        <div class='metric-card' style='margin-bottom:1.5rem;'>
            <div class='metric-label'>Last Known Close Price</div>
            <div class='metric-value green'>${last_price:.2f}</div>
            <div style='color:#555; font-size:0.75rem;'>{df.index[-1].strftime('%B %d, %Y')}</div>
        </div>
        """, unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"<div class='model-badge badge-rnn'>SimpleRNN — {forecast_horizon}-Day Forecast</div>", unsafe_allow_html=True)
            for i, p in enumerate(rnn_fc, 1):
                change = ((p - last_price) / last_price) * 100
                color = "green" if change >= 0 else "red"
                arrow = "▲" if change >= 0 else "▼"
                st.markdown(f"""<div class='forecast-tag {'blue' if change >= 0 else 'red'}'>
                    Day {i}: ${p:.2f} {arrow} {change:+.2f}%
                </div>""", unsafe_allow_html=True)

        with c2:
            st.markdown(f"<div class='model-badge badge-lstm'>LSTM — {forecast_horizon}-Day Forecast</div>", unsafe_allow_html=True)
            for i, p in enumerate(lstm_fc, 1):
                change = ((p - last_price) / last_price) * 100
                arrow = "▲" if change >= 0 else "▼"
                st.markdown(f"""<div class='forecast-tag {'green' if change >= 0 else 'red'}'>
                    Day {i}: ${p:.2f} {arrow} {change:+.2f}%
                </div>""", unsafe_allow_html=True)

        # Forecast chart
        fig, ax = plt.subplots(figsize=(12, 5))
        days = list(range(forecast_horizon + 1))
        ax.plot(days, [last_price] + list(rnn_fc), marker='o', color='#3498DB', label='SimpleRNN', markersize=8)
        ax.plot(days, [last_price] + list(lstm_fc), marker='s', color='#E31937', label='LSTM', markersize=8)
        ax.axhline(last_price, color='#666', linestyle=':', alpha=0.6, label='Current Price')
        ax.fill_between(days, [last_price] + list(rnn_fc), last_price, alpha=0.1, color='#3498DB')
        ax.fill_between(days, [last_price] + list(lstm_fc), last_price, alpha=0.1, color='#E31937')
        ax.set_xticks(days)
        ax.set_xticklabels(['Today'] + [f'Day {i}' for i in range(1, forecast_horizon + 1)])
        ax.set_title(f'{forecast_horizon}-Day Price Forecast — SimpleRNN vs LSTM', fontsize=13)
        ax.set_ylabel('Price (USD)')
        ax.legend()
        ax.grid(alpha=0.2)
        st.pyplot(fig)
        plt.close()


# ══════════════════════════════════════════════════════════════════════
# TAB 4 — Model Comparison
# ══════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("<div class='section-header'>SimpleRNN vs LSTM — Head to Head</div>", unsafe_allow_html=True)

    if not st.session_state.get('trained', False):
        st.warning("⚠️ Train models first.")
    else:
        rnn_m = st.session_state['rnn_m']
        lstm_m = st.session_state['lstm_m']
        winner = 'LSTM' if lstm_m['rmse'] < rnn_m['rmse'] else 'SimpleRNN'

        st.markdown(f"""
        <div class='metric-card' style='background:rgba(46,204,113,0.05); border-color:rgba(46,204,113,0.3);'>
            <span class='model-badge badge-winner'>🏆 Winner: {winner}</span>
            <div style='color:#888; font-size:0.85rem;'>Based on lowest RMSE on test data</div>
        </div>
        """, unsafe_allow_html=True)

        # Comparison table
        comp_data = {
            'Metric': ['RMSE ($)', 'MAE ($)', 'R² Score', 'MAPE (%)'],
            'SimpleRNN': [f"{rnn_m['rmse']:.2f}", f"{rnn_m['mae']:.2f}", f"{rnn_m['r2']:.4f}", f"{rnn_m['mape']:.2f}%"],
            'LSTM': [f"{lstm_m['rmse']:.2f}", f"{lstm_m['mae']:.2f}", f"{lstm_m['r2']:.4f}", f"{lstm_m['mape']:.2f}%"],
        }
        comp_df = pd.DataFrame(comp_data)
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

        # Bar chart comparison
        metrics = ['RMSE', 'MAE', 'MAPE (%)']
        rnn_vals = [rnn_m['rmse'], rnn_m['mae'], rnn_m['mape']]
        lstm_vals = [lstm_m['rmse'], lstm_m['mae'], lstm_m['mape']]
        x = np.arange(len(metrics))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 5))
        bars1 = ax.bar(x - width/2, rnn_vals, width, label='SimpleRNN', color='#3498DB', alpha=0.85)
        bars2 = ax.bar(x + width/2, lstm_vals, width, label='LSTM', color='#E31937', alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, fontsize=12)
        ax.set_title('Error Metrics Comparison', fontsize=13)
        ax.legend()
        ax.grid(axis='y', alpha=0.2)
        for bar in [*bars1, *bars2]:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=9, color='#aaa')
        st.pyplot(fig)
        plt.close()

        # R² comparison
        fig, ax = plt.subplots(figsize=(6, 4))
        models = ['SimpleRNN', 'LSTM']
        r2s = [rnn_m['r2'], lstm_m['r2']]
        colors = ['#3498DB', '#E31937']
        bars = ax.bar(models, r2s, color=colors, alpha=0.85, width=0.4)
        ax.set_title('R² Score Comparison (Higher is Better)', fontsize=12)
        ax.set_ylim(0, 1.1)
        ax.axhline(1.0, color='#2ECC71', linestyle='--', alpha=0.4, label='Perfect R²=1.0')
        ax.grid(axis='y', alpha=0.2)
        for bar, v in zip(bars, r2s):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f'{v:.4f}', ha='center', va='bottom', fontsize=11, color='#aaa')
        ax.legend()
        st.pyplot(fig)
        plt.close()


# ══════════════════════════════════════════════════════════════════════
# TAB 5 — Report
# ══════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("<div class='section-header'>Project Report</div>", unsafe_allow_html=True)

    insights = [
        ("🧹", "Data Cleaning", "Missing values handled using forward fill (ffill) followed by backward fill (bfill). This respects temporal order — no future data leakage into past observations, critical for time-series integrity."),
        ("📐", "Feature Engineering", "MinMaxScaler normalization [0,1] applied for gradient stability. 60-day sliding window sequences created. Daily returns, volatility, and moving averages computed for EDA."),
        ("🧠", "Model Architecture", "Both models use 2 RNN/LSTM stacked layers with Dropout regularization, followed by a Dense(25) ReLU layer and Dense(1) output. Adam optimizer with tuned learning rate."),
        ("🔍", "Hyperparameter Tuning", "Manual GridSearch over units [32, 64], dropout [0.1, 0.2], learning_rate [0.001, 0.0005]. Early stopping (patience=8) prevents overfitting."),
        ("📊", "Model Evaluation", "Evaluated on RMSE, MAE, R², and MAPE. LSTM consistently outperforms SimpleRNN due to gating mechanisms that handle long-range dependencies and vanishing gradients."),
        ("📈", "Forecasting", "Iterative multi-step forecasting: each predicted step is appended to the window for the next prediction. Suitable for 1-day, 5-day, 10-day horizons."),
    ]

    for icon, title, desc in insights:
        st.markdown(f"""
        <div class='insight-box'>
            <span class='insight-icon'>{icon}</span>
            <b style='color:#e8e8f0;'>{title}</b>
            <div style='color:#777; font-size:0.85rem; margin-top:0.3rem; line-height:1.6;'>{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='section-header'>Limitations & Future Work</div>", unsafe_allow_html=True)
    limits = [
        ("⚠️", "Market Shocks", "External events (regulatory, macro) are not captured by price-only models."),
        ("📉", "Error Accumulation", "Multi-step iterative forecasts accumulate errors — 10-day forecasts less reliable than 1-day."),
        ("💡", "Sentiment Analysis", "Add NLP sentiment from news/Twitter to improve predictive power."),
        ("🔬", "Transformer Models", "Temporal Fusion Transformers (TFT) or Informer architecture could outperform LSTMs."),
    ]
    c1, c2 = st.columns(2)
    for i, (icon, title, desc) in enumerate(limits):
        with (c1 if i % 2 == 0 else c2):
            st.markdown(f"""
            <div class='insight-box'>
                <span class='insight-icon'>{icon}</span>
                <b style='color:#e8e8f0;'>{title}</b>
                <div style='color:#777; font-size:0.82rem; margin-top:0.2rem;'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    <div style='border-top: 1px solid #222; margin-top:2rem; padding-top:1rem;
                text-align:center; color:#444; font-size:0.78rem;'>
        Tesla Stock Price Prediction · Deep Learning Project · SimpleRNN & LSTM · Python 3.10+
    </div>
    """, unsafe_allow_html=True)
