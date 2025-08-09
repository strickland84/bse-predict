import { useWebSocket } from '../hooks/useWebSocket'
import { SystemStatus } from '../components/SystemStatus'
import { PredictionGrid } from '../components/PredictionGrid'
import { ModelPerformance } from '../components/ModelPerformance'
import { DataHealth } from '../components/DataHealth'
import { RecentAlerts } from '../components/RecentAlerts'
import { LiveChart } from '../components/LiveChart'
import { PredictionStats } from '../components/PredictionStats'
import { Target, Clock, BarChart3 } from 'lucide-react'

export function Dashboard() {
  const { lastMessage } = useWebSocket()

  return (
    <div className="container mx-auto px-4 py-6">
      {/* Project Explainer */}
      <div className="mb-8">
        <p className="text-sm text-dark-muted mb-4">
          An advanced machine learning system that analyzes technical indicators and market patterns to predict short-term price movements for Bitcoin, Ethereum, and Solana. 
          Our models are trained on historical data using RandomForest algorithms with time-series cross-validation to maintain temporal integrity.
        </p>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-4">
          <div className="flex items-start gap-3">
            <Target className="w-5 h-5 text-success mt-0.5 flex-shrink-0" />
            <div>
              <div className="text-sm font-semibold">Multi-Target Predictions</div>
              <div className="text-xs text-dark-muted">±1%, ±2%, and ±5% price movement targets for each asset</div>
            </div>
          </div>
          <div className="flex items-start gap-3">
            <Clock className="w-5 h-5 text-warning mt-0.5 flex-shrink-0" />
            <div>
              <div className="text-sm font-semibold">Hourly Updates</div>
              <div className="text-xs text-dark-muted">Fresh predictions every hour with confidence scores</div>
            </div>
          </div>
          <div className="flex items-start gap-3">
            <BarChart3 className="w-5 h-5 text-info mt-0.5 flex-shrink-0" />
            <div>
              <div className="text-sm font-semibold">Continuous Learning</div>
              <div className="text-xs text-dark-muted">Models retrain daily with latest market data</div>
            </div>
          </div>
        </div>
        <p className="text-xs text-dark-muted italic">
          <strong>Disclaimer:</strong> This system is for informational purposes only. Cryptocurrency trading involves substantial risk. 
          Past performance does not guarantee future results. Always do your own research and never invest more than you can afford to lose.
        </p>
      </div>

      <div className="space-y-6">
        {/* Main Content */}
        <div className="space-y-6">
          {/* Predictions */}
          <PredictionGrid />
          
          {/* Performance Stats - Full Width */}
          <PredictionStats />
          
          {/* Model Performance - Full Width */}
          <ModelPerformance />
          
          {/* Charts and Alerts */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <RecentAlerts wsMessage={lastMessage} />
            <LiveChart />
          </div>
        </div>

        {/* System Status & Data Health - Now at bottom */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
          <div className="md:col-span-1 lg:col-span-2 xl:col-span-2">
            <SystemStatus />
          </div>
          <div className="md:col-span-1 lg:col-span-1 xl:col-span-2">
            <DataHealth />
          </div>
        </div>
      </div>
    </div>
  )
}