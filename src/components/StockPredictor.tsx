
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { TrendingUp } from "lucide-react";
import StockSymbol from "./StockSymbol";
import { useToast } from "@/hooks/use-toast";

const StockPredictor = () => {
  const [symbol, setSymbol] = useState("");
  const [predictions, setPredictions] = useState<any[]>([]);
  const { toast } = useToast();

  const popularSymbols = [
    "AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "NVDA",
    "META", "NFLX", "AMD", "CRM", "ADBE", "INTC",
    "ORCL", "CSCO", "IBM", "QCOM", "JPM", "BAC"
  ];

  const handleSymbolClick = (clickedSymbol: string) => {
    setSymbol(clickedSymbol);
  };

  const handlePredict = () => {
    if (!symbol) {
      toast({
        title: "Error",
        description: "Please enter a stock symbol",
        variant: "destructive",
      });
      return;
    }

    // Mock prediction data
    const mockPrediction = {
      symbol: symbol.toUpperCase(),
      currentPrice: (Math.random() * 500 + 50).toFixed(2),
      predictedPrice: (Math.random() * 500 + 50).toFixed(2),
      change: (Math.random() * 40 - 20).toFixed(2),
      changePercent: (Math.random() * 20 - 10).toFixed(2),
      confidence: ["Low", "Medium", "High"][Math.floor(Math.random() * 3)],
      updated: new Date().toLocaleString()
    };

    setPredictions(prev => [mockPrediction, ...prev.slice(0, 3)]);
    
    toast({
      title: "Prediction Generated",
      description: `AI prediction for ${symbol.toUpperCase()} has been generated`,
    });
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 text-white">
      <div className="container mx-auto px-4 py-8">
        {/* Main Prediction Section */}
        <div className="text-center mb-12">
          <h1 className="text-5xl font-bold mb-4 bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
            Predict Stock Prices with AI
          </h1>
          <p className="text-xl text-gray-300 mb-8">
            Get next-day stock price predictions powered by LSTM neural networks and comprehensive technical analysis.
          </p>
          
          <div className="flex flex-col sm:flex-row gap-4 justify-center items-center max-w-2xl mx-auto">
            <Input
              placeholder="Enter stock symbol (e.g., AAPL, GOOGL, TSLA)"
              value={symbol}
              onChange={(e) => setSymbol(e.target.value)}
              className="bg-slate-800 border-slate-700 text-white placeholder-gray-400"
            />
            <Button 
              onClick={handlePredict}
              className="bg-blue-600 hover:bg-blue-700 px-8"
            >
              Predict
            </Button>
          </div>
        </div>

        {/* Popular Symbols Section */}
        <Card className="bg-slate-800/50 border-slate-700 mb-8">
          <CardHeader>
            <CardTitle className="text-white flex items-center">
              <span className="w-2 h-2 bg-green-500 rounded-full mr-2"></span>
              Popular Symbols
            </CardTitle>
            <CardDescription className="text-gray-400">
              Click any symbol for instant AI prediction
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-3 sm:grid-cols-6 gap-3">
              {popularSymbols.map((stockSymbol) => (
                <StockSymbol
                  key={stockSymbol}
                  symbol={stockSymbol}
                  onClick={handleSymbolClick}
                />
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Predictions Results */}
        {predictions.length > 0 && (
          <div className="mb-8">
            <div className="flex justify-between items-center mb-4">
              <div>
                <h2 className="text-2xl font-bold text-white">AI Predictions</h2>
                <p className="text-gray-400">{predictions.length} stocks analyzed</p>
              </div>
              <Button
                variant="outline"
                onClick={() => setPredictions([])}
                className="border-slate-600 text-white hover:bg-slate-700"
              >
                Clear All
              </Button>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {predictions.map((prediction, index) => (
                <Card key={index} className="bg-slate-800/50 border-slate-700">
                  <CardHeader className="flex flex-row items-center justify-between">
                    <div>
                      <CardTitle className="text-white text-xl">{prediction.symbol}</CardTitle>
                      <CardDescription className="text-gray-400">
                        Updated: {prediction.updated}
                      </CardDescription>
                    </div>
                    <div className={`p-2 rounded-lg ${
                      parseFloat(prediction.change) >= 0 
                        ? 'bg-green-500/20 text-green-400' 
                        : 'bg-red-500/20 text-red-400'
                    }`}>
                      <TrendingUp className="h-6 w-6" />
                    </div>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <p className="text-gray-400 text-sm">CURRENT PRICE</p>
                        <p className="text-2xl font-bold text-white">${prediction.currentPrice}</p>
                      </div>
                      <div>
                        <p className="text-gray-400 text-sm">AI PREDICTION</p>
                        <p className="text-2xl font-bold text-white">${prediction.predictedPrice}</p>
                      </div>
                      <div>
                        <p className="text-gray-400 text-sm">CHANGE</p>
                        <p className={`text-lg font-semibold ${
                          parseFloat(prediction.change) >= 0 ? 'text-green-400' : 'text-red-400'
                        }`}>
                          ${prediction.change}
                        </p>
                      </div>
                      <div>
                        <p className="text-gray-400 text-sm">CHANGE %</p>
                        <p className={`text-lg font-semibold ${
                          parseFloat(prediction.changePercent) >= 0 ? 'text-green-400' : 'text-red-400'
                        }`}>
                          {parseFloat(prediction.changePercent) >= 0 ? '+' : ''}{prediction.changePercent}%
                        </p>
                      </div>
                    </div>
                    <div className="mt-4 flex items-center justify-between">
                      <div className="flex items-center space-x-2">
                        <span className="w-2 h-2 bg-blue-500 rounded-full"></span>
                        <span className="text-sm text-gray-400">Confidence: {prediction.confidence}</span>
                      </div>
                      <div className="flex items-center space-x-2">
                        <span className="w-2 h-2 bg-purple-500 rounded-full"></span>
                        <span className="text-sm text-gray-400">1 day forecast</span>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default StockPredictor;
