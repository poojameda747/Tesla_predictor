# 🚗 Tesla Stock Price Prediction — Deep Learning Project

## Project Structure

```
tesla-stock-predictor/
├── tesla_stock_prediction.ipynb   # Full Jupyter Notebook
├── app.py                          # Streamlit UI App
├── requirements.txt                # Python dependencies
└── TSLA.csv                        # Dataset (download separately)
```

## Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download Dataset
- Download TSLA.csv from the provided Google Drive link
- Place it in the same folder as the notebook and app

### 3. Run Jupyter Notebook
```bash
jupyter notebook tesla_stock_prediction.ipynb
```
Run all cells in order. This will:
- Perform EDA and data cleaning
- Train SimpleRNN and LSTM models with GridSearch
- Evaluate and compare models
- Generate forecasts for 1-day, 5-day, 10-day horizons
- Save trained models: `best_lstm_model.keras`, `best_rnn_model.keras`
- Save: `scaler.pkl`, `model_metrics.json`, `forecasts.json`

### 4. Run Streamlit App
```bash
streamlit run app.py
```
Then open your browser at: http://localhost:8501

Upload TSLA.csv in the sidebar, configure hyperparameters, and click **Train Models**.

## Deploy to Streamlit Cloud

1. Push this folder to a GitHub repository
2. Go to https://share.streamlit.io
3. Connect your GitHub repo
4. Set `app.py` as the main file
5. Click Deploy

## Models Used
| Model     | Architecture                                  |
|-----------|-----------------------------------------------|
| SimpleRNN | RNN(64) → Dropout → RNN(32) → Dense(25) → Dense(1) |
| LSTM      | LSTM(64) → Dropout → LSTM(32) → Dense(25) → Dense(1) |

## Evaluation Metrics
- **MSE** — Mean Squared Error
- **RMSE** — Root Mean Squared Error
- **MAE** — Mean Absolute Error
- **R²** — Coefficient of Determination
- **MAPE** — Mean Absolute Percentage Error

## Forecast Horizons
- 1-Day prediction
- 5-Day prediction
- 10-Day prediction

## Tech Stack
- Python 3.10+
- TensorFlow / Keras
- Scikit-learn
- Pandas, NumPy
- Matplotlib, Seaborn
- Streamlit
