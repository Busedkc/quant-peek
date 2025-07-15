from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from backend.model import predict_next_day
import yfinance as yf

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    symbol: str


class PredictResponse(BaseModel):
    symbol: str
    prediction: float
    last_loss: Optional[float] = None
    current_price: Optional[float] = None
    change: Optional[float] = None
    change_percent: Optional[float] = None


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """
    Verilen hisse senedi sembolü için bir sonraki gün fiyat tahmini yapar.
    Modelden tahmin, son loss, güncel fiyat, değişim ve değişim yüzdesi döner.
    Hata durumunda HTTPException fırlatır.
    Args:
        req (PredictRequest): Tahmin istek modeli (sembol içerir)
    Returns:
        PredictResponse: Tahmin ve ek bilgiler
    """
    try:
        pred, last_loss, current_price, change, change_percent = predict_next_day(
            req.symbol
        )
        # Güncel fiyatı da çek
        df = yf.download(req.symbol, period="2d")
        current_price = float(df["Close"].values[-1]) if not df.empty else None
        return PredictResponse(
            symbol=req.symbol,
            prediction=pred,
            last_loss=last_loss,
            current_price=current_price,
            change=change,
            change_percent=change_percent,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
