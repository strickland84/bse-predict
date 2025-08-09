import { useQuery } from '@tanstack/react-query'
import { fetchLatestPredictions } from '../api'
import { format } from 'date-fns'
import clsx from 'clsx'
import { Link } from 'react-router-dom'
import { CryptoLogo } from './CryptoLogo'
import { ArrowRight, TrendingUp } from 'lucide-react'

export const PredictionGrid = () => {
  const { data, isLoading, error } = useQuery({
    queryKey: ['latestPredictions'],
    queryFn: fetchLatestPredictions,
  })

  if (isLoading) return <div className="card">Loading predictions...</div>
  if (error) return <div className="card text-error">Failed to load predictions: {String(error)}</div>
  if (!data || !Array.isArray(data)) return <div className="card text-warning">No prediction data available</div>

  const getPredictionColor = (value: number) => {
    if (value > 0) return 'text-success'
    if (value < 0) return 'text-error'
    return 'text-dark-muted'
  }

  const getPredictionLabel = (value: number) => {
    if (value > 0) return '↑ UP'
    if (value < 0) return '↓ DOWN'
    return '→ NEUTRAL'
  }

  const getConfidenceOpacity = (confidence: number) => {
    if (confidence >= 0.8) return 'opacity-100'
    if (confidence >= 0.6) return 'opacity-75'
    return 'opacity-50'
  }

  return (
    <div className="card">
      <div className="flex justify-between items-center mb-4">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-success/10 rounded-lg">
            <TrendingUp className="w-5 h-5 text-success" />
          </div>
          <div>
            <h2 className="text-xl font-bold">Current Predictions</h2>
            <p className="text-xs text-dark-muted">Latest AI predictions</p>
          </div>
        </div>
        <Link 
          to="/predictions" 
          className="text-sm text-info hover:text-info/80 flex items-center gap-1"
        >
          View All <ArrowRight className="w-4 h-4" />
        </Link>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {data?.map((prediction) => (
          <div key={prediction.symbol} className="bg-dark-bg border border-dark-border rounded-lg p-4">
            <div className="flex justify-between items-start mb-3">
              <div className="flex items-center gap-2">
                <CryptoLogo symbol={prediction.symbol} size="sm" />
                <h3 className="text-lg font-semibold">{prediction.symbol.replace('/USDT', '')}</h3>
              </div>
              <span className="text-sm text-dark-muted">
                ${prediction.current_price.toLocaleString()}
              </span>
            </div>
            
            <div className="space-y-2">
              {[
                { target: '1%', value: prediction.target_1pct, confidence: prediction.confidence_1pct },
                { target: '2%', value: prediction.target_2pct, confidence: prediction.confidence_2pct },
                { target: '5%', value: prediction.target_5pct, confidence: prediction.confidence_5pct },
              ].map(({ target, value, confidence }) => (
                <div key={target} className="flex items-center justify-between">
                  <span className="text-sm text-dark-muted">{target}</span>
                  <div className="flex items-center gap-2">
                    <span className={clsx(
                      'text-sm font-semibold',
                      getPredictionColor(value),
                      getConfidenceOpacity(confidence)
                    )}>
                      {getPredictionLabel(value)}
                    </span>
                    <span className="text-xs text-dark-muted">
                      {(confidence * 100).toFixed(0)}%
                    </span>
                  </div>
                </div>
              ))}
            </div>
            
            <div className="mt-3 pt-3 border-t border-dark-border">
              <span className="text-xs text-dark-muted">
                {format(new Date(prediction.timestamp), 'MMM dd HH:mm')}
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}