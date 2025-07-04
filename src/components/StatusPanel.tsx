
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { TrendingUp } from "lucide-react";

const StatusPanel = () => {
  return (
    <Card className="bg-slate-800/50 border-slate-700">
      <CardHeader>
        <CardTitle className="text-white flex items-center">
          <TrendingUp className="h-5 w-5 mr-2 text-green-500" />
          API Status
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex items-center justify-between">
          <span className="text-gray-400">Status</span>
          <div className="flex items-center">
            <div className="w-2 h-2 bg-green-500 rounded-full mr-2"></div>
            <span className="text-green-400">Healthy</span>
          </div>
        </div>
        
        <div className="flex items-center justify-between">
          <span className="text-gray-400">Model</span>
          <span className="text-green-400">Loaded</span>
        </div>
        
        <div className="flex items-center justify-between">
          <span className="text-gray-400">Device</span>
          <span className="text-white">CPU</span>
        </div>
        
        <div className="mt-6">
          <h4 className="text-white font-semibold mb-2">Model Information</h4>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-gray-400">LSTM with Attention</span>
              <span className="text-white">37</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Regression PyTorch</span>
              <span className="text-white">60</span>
            </div>
          </div>
        </div>
        
        <div className="mt-4 text-xs text-gray-500">
          Trained: Jun 18, 2025, 06:18 PM
        </div>
      </CardContent>
    </Card>
  );
};

export default StatusPanel;
