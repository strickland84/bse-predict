import { useQuery } from '@tanstack/react-query'
import { fetchDataHealth } from '../api'
import { format } from 'date-fns'
import clsx from 'clsx'
import { Link } from 'react-router-dom'
import { ArrowRight, Database } from 'lucide-react'

export const DataHealth = () => {
  const { data, isLoading, error } = useQuery({
    queryKey: ['dataHealth'],
    queryFn: fetchDataHealth,
  })

  if (isLoading) return <div className="card">Loading data health...</div>
  if (error) return <div className="card text-error">Failed to load data health</div>

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy': return 'bg-success'
      case 'gaps_detected': return 'bg-warning'
      case 'critical': return 'bg-error'
      default: return 'bg-gray-500'
    }
  }

  const getCoverageColor = (coverage: number) => {
    if (coverage >= 95) return 'text-success'
    if (coverage >= 80) return 'text-warning'
    return 'text-error'
  }

  return (
    <div className="card">
      <div className="flex justify-between items-center mb-4">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-info/10 rounded-lg">
            <Database className="w-5 h-5 text-info" />
          </div>
          <div>
            <h2 className="text-xl font-bold">Data Health</h2>
            <p className="text-xs text-dark-muted">Data integrity status</p>
          </div>
        </div>
        <Link 
          to="/data" 
          className="text-sm text-info hover:text-info/80 flex items-center gap-1"
        >
          View All <ArrowRight className="w-4 h-4" />
        </Link>
      </div>
      
      <div className="space-y-3">
        {data?.map((item) => (
          <div key={item.symbol} className="bg-dark-bg border border-dark-border rounded-lg p-3">
            <div className="flex justify-between items-center mb-2">
              <h3 className="font-semibold">{item.symbol.replace('/USDT', '')}</h3>
              <div className="flex items-center gap-2">
                <span className={clsx('status-dot', getStatusColor(item.status))} />
                <span className="text-sm capitalize">{item.status.replace('_', ' ')}</span>
              </div>
            </div>
            
            <div className="grid grid-cols-2 gap-2 text-sm">
              <div>
                <span className="text-dark-muted">Total Candles</span>
                <div>{item.total_candles.toLocaleString()}</div>
              </div>
              <div>
                <span className="text-dark-muted">Coverage</span>
                <div className={getCoverageColor(item.coverage_percentage)}>
                  {item.coverage_percentage.toFixed(1)}%
                </div>
              </div>
              <div>
                <span className="text-dark-muted">Gaps Detected</span>
                <div className={item.gaps_detected > 0 ? 'text-warning' : ''}>
                  {item.gaps_detected}
                </div>
              </div>
              <div>
                <span className="text-dark-muted">Latest</span>
                <div className="text-xs">
                  {format(new Date(item.latest_candle), 'HH:mm:ss')}
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}