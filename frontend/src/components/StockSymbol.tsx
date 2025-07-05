
import { Button } from "@/components/ui/button";

interface StockSymbolProps {
  symbol: string;
  onClick: (symbol: string) => void;
}

const StockSymbol = ({ symbol, onClick }: StockSymbolProps) => {
  return (
    <Button
      variant="outline"
      className="bg-slate-700 hover:bg-slate-600 text-white border-slate-600 hover:border-slate-500"
      onClick={() => onClick(symbol)}
    >
      {symbol}
    </Button>
  );
};

export default StockSymbol;
