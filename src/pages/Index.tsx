
import Header from "@/components/Header";
import StockPredictor from "@/components/StockPredictor";
import StatusPanel from "@/components/StatusPanel";

const Index = () => {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      <Header />
      <div className="container mx-auto px-4 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
          <div className="lg:col-span-3">
            <StockPredictor />
          </div>
          <div className="lg:col-span-1">
            <StatusPanel />
          </div>
        </div>
      </div>
    </div>
  );
};

export default Index;
