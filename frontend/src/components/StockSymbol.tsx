
import { Button } from "@/components/ui/button";

interface StockSymbolProps {
  symbol: string;
  onClick: (symbol: string) => void;
}

/**
 * StockSymbol bileşeni, verilen sembolü bir buton olarak gösterir ve tıklanınca üst bileşene bildirir.
 * @param symbol Gösterilecek hisse senedi sembolü
 * @param onClick Sembol tıklandığında çağrılacak fonksiyon
 */
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
