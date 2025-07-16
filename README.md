# 📈 Quant-Peek

## 🚀 Project Overview

**Quant-Peek** is a stock market forecasting platform that utilizes deep learning techniques to predict next-day stock closing prices.

By leveraging an **LSTM (Long Short-Term Memory)** neural network, the project processes the past 60 days of stock data and forecasts the following day’s price. This enables traders, analysts, or researchers to experiment with sequential data forecasting in financial contexts.

The system also includes tools for **visualizing predictions**, saving models, and optionally integrating predictions into a **web interface**.

---

## 🔧 Technologies Used

* **PyTorch** – Deep Learning Framework
* **yFinance** – Real-time stock data collection
* **NumPy** – Numerical computations
* **scikit-learn** – Data preprocessing (MinMaxScaler)
* **Matplotlib** – Visualization of predictions
* **Python 3.8+**

---

## 📂 Project Structure

```
quant-peek/
│
├── backend/
│   ├── model.py          # LSTM model, training, and saving
│   └── predict.py        # Prediction and visualization functions
│
├── models/               # Saved models (.pth format)
├── frontend/ (optional)  # Web frontend integration (React/Node)
├── requirements.txt      # Project dependencies
├── README.md              # Documentation
```

---

## ⚙️ How to Use

### 1️⃣ Install Dependencies

Use the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

Alternatively, manually install:

```bash
pip install torch yfinance numpy scikit-learn matplotlib
```

---

### 2️⃣ Train the Model

Train an LSTM model on a specific stock (e.g., Apple - AAPL):

```python
from backend.model import train

train(symbol="AAPL", epochs=20, batch_size=32, seq_length=60)
```

The model will be saved automatically to `models/lstm_AAPL.pth`.

---

### 3️⃣ Make a Prediction

Predict the next day's closing price:

```python
from backend.model import predict_next_day

predict_next_day("AAPL")
```

This function will output:

* Predicted closing price
* Last training loss
* Current closing price
* Price change & percentage change

---

### 4️⃣ Visualize Predictions

Display the last 60 days of closing prices along with the predicted next-day price:

```python
from backend.model import plot_prediction

plot_prediction("AAPL")
```

If you want to save the plot instead of displaying it:

```python
plot_prediction("AAPL", save_path="frontend/public/prediction.png")
```

---

## 🌐 Web Integration (Optional)

Graphs and predictions can be served to a frontend using Flask, FastAPI, or any web framework. Save visualizations into `frontend/public/` for frontend rendering:

```html
<img src="/prediction.png" alt="Stock Prediction Graph">
```

---

## 🚀 Future Development

* [ ] Add GRU model option
* [ ] Live data integration via WebSocket
* [ ] Trading signal (buy/sell) generation
* [ ] Backtesting system for strategy evaluation
* [ ] Docker deployment for scalable usage

---

## 👤 Contributor

| Name        | Role                       |
| ----------- | -------------------------- |
| Buse Dikici | Developer & Data Scientist |

---

## 📄 License

This project is licensed under the MIT License.

---

## ⚠️ Disclaimer

This is an educational project and **not financial advice**.
All outputs are for research and learning purposes only.

---

## ⭐ Support

If you like this project, please consider starring the repository on GitHub!
