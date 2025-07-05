import yfinance as yf
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# 1. LSTM Modeli
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# 2. Veri Hazırlama
def load_data(symbol, seq_length=60):
    df = yf.download(symbol, period="5y")
    data = df['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)

    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i])
        y.append(data[i])
    X, y = np.array(X), np.array(y)
    return X, y, scaler

# 3. Eğitim Fonksiyonu
def train(symbol="AAPL", epochs=20, batch_size=32, seq_length=60):
    print(f"--- {symbol} için eğitim başlıyor ---")
    X, y, scaler = load_data(symbol, seq_length)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = LSTMModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    last_loss = None
    for epoch in range(epochs):
        for X_batch, y_batch in loader:
            output = model(X_batch)
            loss = criterion(output, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        last_loss = loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")

    # Modeli kaydet
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler': scaler,
        'last_loss': last_loss
    }, f"lstm_{symbol}.pth")
    print("Model kaydedildi: ", f"lstm_{symbol}.pth\n")
    return last_loss

def predict_next_day(symbol, seq_length=60):
    # Model ve scaler'ı yükle
    checkpoint = torch.load(f"lstm_{symbol}.pth", map_location=torch.device('cpu'))
    model = LSTMModel()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    scaler = checkpoint['scaler']
    last_loss = checkpoint.get('last_loss', None)

    # Son 60 günü çek
    df = yf.download(symbol, period=f"{seq_length+1}d")
    data = df['Close'].values.reshape(-1, 1)
    data_scaled = scaler.transform(data)
    X_input = torch.tensor(data_scaled[-seq_length:], dtype=torch.float32).unsqueeze(0)  # (1, seq_length, 1)

    with torch.no_grad():
        pred_scaled = model(X_input).numpy()
    pred = scaler.inverse_transform(pred_scaled)[0][0]
    pred = float(pred)
    if last_loss is not None:
        print(f"{symbol} için bir sonraki gün tahmini: {pred:.2f} | Son eğitim loss'u: {last_loss:.6f}")
    else:
        print(f"{symbol} için bir sonraki gün tahmini: {pred:.2f}")
    return pred, last_loss

def plot_prediction(symbol, seq_length=60):
    # Model ve scaler'ı yükle
    checkpoint = torch.load(f"lstm_{symbol}.pth", map_location=torch.device('cpu'))
    model = LSTMModel()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    scaler = checkpoint['scaler']

    # Son 60 günü çek
    df = yf.download(symbol, period=f"{seq_length+1}d")
    closes = df['Close'].values
    data = closes.reshape(-1, 1)
    data_scaled = scaler.transform(data)
    X_input = torch.tensor(data_scaled[-seq_length:], dtype=torch.float32).unsqueeze(0)  # (1, seq_length, 1)

    with torch.no_grad():
        pred_scaled = model(X_input).numpy()
    pred = scaler.inverse_transform(pred_scaled)[0][0]

    # Grafik
    plt.figure(figsize=(10, 5))
    plt.plot(range(seq_length), closes[-seq_length:], label='Gerçek Kapanış Fiyatı')
    plt.scatter(seq_length, float(pred), color='red', label='Tahmin (Bir Sonraki Gün)')
    plt.plot([seq_length-1, seq_length], [float(closes[-1]), float(pred)], 'r--', alpha=0.5)
    plt.xlabel('Gün')
    plt.ylabel('Fiyat')
    plt.title(f"{symbol} - Son {seq_length} Gün ve Tahmin")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "TSLA", "NVDA", "AMZN"]
    for ticker in tickers:
        train(symbol=ticker)
        predict_next_day(ticker)
        plot_prediction(ticker)