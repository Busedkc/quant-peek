
import { Button } from "@/components/ui/button";
import { TrendingUp } from "lucide-react";

/**
 * Header bileşeni, uygulamanın üst kısmında başlık ve özellik etiketlerini gösterir.
 */
const Header = () => {
  return (
    <header className="bg-gradient-to-r from-purple-600 to-blue-600 text-white p-4">
      <div className="container mx-auto flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div className="bg-white/20 p-2 rounded-lg">
            <TrendingUp className="h-6 w-6" />
          </div>
          <div>
            <h1 className="text-xl font-bold">QuantPeek</h1>
            <p className="text-sm opacity-90">AI-powered stock price predictions</p>
          </div>
        </div>
        
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2 text-sm">
            <span className="flex items-center">
              <TrendingUp className="h-4 w-4 mr-1" />
              Real-time Data
            </span>
            <span className="flex items-center">
              <TrendingUp className="h-4 w-4 mr-1" />
              Technical Analysis
            </span>
            <span className="flex items-center">
              <TrendingUp className="h-4 w-4 mr-1" />
              AI-Powered
            </span>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;
