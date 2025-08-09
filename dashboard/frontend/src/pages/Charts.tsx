import { useQuery } from '@tanstack/react-query'
import { useState } from 'react'
import { format } from 'date-fns'
import { Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Area, ComposedChart } from 'recharts'
import { TrendingUp, TrendingDown, Info } from 'lucide-react'
import { CryptoLogo } from '../components/CryptoLogo'
import { api } from '../api'
import clsx from 'clsx'

interface ChartData {
  symbol: string
  price_data: {
    timestamp: string
    open: number
    high: number
    low: number
    close: number
    volume: number
  }[]
  predictions: {
    [key: string]: {
      timestamp: string
      prediction: 'UP' | 'DOWN' | 'NEUTRAL'
      confidence: number
      probability: number
      status: 'correct' | 'incorrect' | 'pending' | 'expired'
      actual_outcome: number | null
      max_favorable: number | null
      max_adverse: number | null
    }[]
  }
  statistics: {
    total_predictions: number
    correct_predictions: number
    pending_predictions: number
    accuracy: number
  }
}

// Custom dot for predictions
const PredictionDot = (props: any) => {
  const { cx, cy, payload, dataKey } = props
  const prediction = payload[dataKey]
  
  // Skip if no prediction
  if (!prediction) return null
  
  // Fixed size for all dots
  const size = 8
  
  // Calculate opacity based on confidence (0.4 to 1.0)
  const confidenceOpacity = 0.4 + (prediction.confidence / 100) * 0.6
  
  // Determine colors based on status
  let fillColor = '#6B7280' // Default gray
  let strokeColor = '#4B5563'
  
  if (prediction.status === 'correct') {
    fillColor = '#10B981' // Green for hits
    strokeColor = '#059669'
  } else if (prediction.status === 'incorrect') {
    fillColor = '#EF4444' // Red for misses
    strokeColor = '#DC2626'
  } else if (prediction.status === 'pending') {
    fillColor = '#6B7280' // Gray for pending
    strokeColor = '#4B5563'
  }
  
  const isUp = prediction.prediction === 'UP'
  
  return (
    <g>
      {/* Main circle with confidence-based opacity */}
      <circle 
        cx={cx} 
        cy={cy} 
        r={size} 
        fill={fillColor} 
        fillOpacity={confidenceOpacity}
        stroke={strokeColor} 
        strokeWidth="1.5" 
      />
      
      {/* Add arrow for prediction direction */}
      {isUp ? (
        // Up arrow
        <path
          d={`M ${cx} ${cy - 4} L ${cx - 3} ${cy + 1} L ${cx - 1} ${cy + 1} L ${cx - 1} ${cy + 4} L ${cx + 1} ${cy + 4} L ${cx + 1} ${cy + 1} L ${cx + 3} ${cy + 1} Z`}
          fill="white"
          fillOpacity="0.9"
        />
      ) : (
        // Down arrow
        <path
          d={`M ${cx} ${cy + 4} L ${cx - 3} ${cy - 1} L ${cx - 1} ${cy - 1} L ${cx - 1} ${cy - 4} L ${cx + 1} ${cy - 4} L ${cx + 1} ${cy - 1} L ${cx + 3} ${cy - 1} Z`}
          fill="white"
          fillOpacity="0.9"
        />
      )}
    </g>
  )
}

// Custom tooltip
const CustomTooltip = ({ active, payload, label }: any) => {
  if (!active || !payload || !payload.length) return null
  
  const data = payload[0].payload
  const prediction = data.prediction
  
  return (
    <div className="bg-dark-card border border-dark-border rounded-lg p-3 shadow-xl">
      <div className="text-xs text-dark-muted mb-2">
        {format(new Date(label), 'MMM dd, HH:mm')}
      </div>
      
      <div className="space-y-1">
        <div className="flex items-center justify-between gap-4">
          <span className="text-xs">Price:</span>
          <span className="text-sm font-mono font-bold">${data.close?.toFixed(2)}</span>
        </div>
        
        {prediction && (
          <div className="flex items-center justify-between gap-4 pt-1 border-t border-dark-border">
            <span className="text-xs">Prediction:</span>
            <div className="flex items-center gap-1">
              {prediction.prediction === 'UP' ? 
                <TrendingUp className="w-3 h-3 text-success" /> : 
                <TrendingDown className="w-3 h-3 text-error" />
              }
              <span className="text-xs font-medium">{prediction.prediction}</span>
              <span className="text-xs">({prediction.confidence.toFixed(0)}%)</span>
              {prediction.status === 'correct' && <span className="text-xs text-success">✓</span>}
              {prediction.status === 'incorrect' && <span className="text-xs text-error">✗</span>}
              {prediction.status === 'pending' && <span className="text-xs text-warning">•</span>}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export function Charts() {
  const [hours, setHours] = useState(72)
  const [selectedSymbol, setSelectedSymbol] = useState<string | null>(null)
  
  const { data, isLoading, error } = useQuery({
    queryKey: ['charts-data', hours, selectedSymbol],
    queryFn: async () => {
      const params = new URLSearchParams()
      params.append('hours', hours.toString())
      if (selectedSymbol) params.append('symbol', selectedSymbol)
      const { data } = await api.get(`/api/charts/price-predictions?${params}`)
      return data
    },
    refetchInterval: 60000 // Refresh every minute
  })
  
  if (isLoading) {
    return (
      <div className="container mx-auto px-4 py-6">
        <div className="card">
          <div className="flex items-center gap-3">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-info"></div>
            <span className="text-dark-muted">Loading chart data...</span>
          </div>
        </div>
      </div>
    )
  }
  
  if (error || !data) {
    return (
      <div className="container mx-auto px-4 py-6">
        <div className="card text-error">Failed to load chart data</div>
      </div>
    )
  }
  
  // Process data for recharts with specific target
  const processChartData = (chart: ChartData, targetKey: string) => {
    const priceMap = new Map()
    
    // First, add all price data
    chart.price_data.forEach(candle => {
      const hour = format(new Date(candle.timestamp), 'yyyy-MM-dd HH:00')
      priceMap.set(hour, {
        timestamp: candle.timestamp,
        close: candle.close,
        high: candle.high,
        low: candle.low,
        open: candle.open,
        volume: candle.volume,
        prediction: null
      })
    })
    
    // Then, add predictions for specific target
    const targetPreds = chart.predictions[targetKey] || []
    targetPreds.forEach(pred => {
      const hour = format(new Date(pred.timestamp), 'yyyy-MM-dd HH:00')
      const existing = priceMap.get(hour)
      if (existing) {
        existing.prediction = pred
      }
    })
    
    // Convert to array and sort
    return Array.from(priceMap.values()).sort((a, b) => 
      new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime()
    )
  }
  
  return (
    <div className="container mx-auto px-4 py-6">
      <div className="mb-6">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h1 className="text-3xl font-bold mb-2">Price & Predictions Chart</h1>
            <p className="text-dark-muted text-sm">
              Interactive visualization of price movements with prediction overlays
            </p>
          </div>
          
          {/* Time Range Selector */}
          <div className="flex items-center gap-2">
            <span className="text-xs text-dark-muted">Time Range:</span>
            {[24, 48, 72, 168].map(h => (
              <button
                key={h}
                onClick={() => setHours(h)}
                className={clsx(
                  'px-3 py-1 rounded-lg text-xs font-medium transition-all',
                  hours === h
                    ? 'bg-gradient-to-r from-info to-purple-500 text-white shadow-lg shadow-info/25'
                    : 'bg-dark-card border border-dark-border hover:bg-dark-border'
                )}
              >
                {h < 168 ? `${h}h` : '1w'}
              </button>
            ))}
          </div>
        </div>
        
        {/* Symbol Filter */}
        <div className="flex gap-2 mb-6">
          <button
            onClick={() => setSelectedSymbol(null)}
            className={clsx(
              'px-4 py-2 rounded-lg transition-all duration-300',
              !selectedSymbol ? 'btn-primary' : 'btn-secondary'
            )}
          >
            All Assets
          </button>
          {['BTC/USDT', 'ETH/USDT', 'SOL/USDT'].map((symbol) => (
            <button
              key={symbol}
              onClick={() => setSelectedSymbol(symbol)}
              className={clsx(
                'px-4 py-2 rounded-lg transition-all duration-300 flex items-center gap-2',
                selectedSymbol === symbol ? 'btn-primary' : 'btn-secondary'
              )}
            >
              <CryptoLogo symbol={symbol} size="sm" />
              {symbol.replace('/USDT', '')}
            </button>
          ))}
        </div>
        
        {/* Legend */}
        <div className="card mb-4 bg-gradient-to-r from-dark-card to-dark-border/20">
          <div className="flex items-start gap-2">
            <Info className="w-4 h-4 text-info mt-0.5" />
            <div className="text-xs text-dark-muted">
              <div className="font-medium text-white mb-1">How to read these charts:</div>
              <ul className="space-y-1">
                <li>• Each chart shows predictions for a specific target (±1%, ±2%, or ±5%)</li>
                <li>• The <span className="text-info">blue line</span> shows the asset price over time</li>
                <li>• <span className="text-success">Green dots</span> = Correct predictions (hits)</li>
                <li>• <span className="text-error">Red dots</span> = Incorrect predictions (misses)</li>
                <li>• <span className="text-gray-400">Gray dots</span> = Pending (waiting for outcome)</li>
                <li>• <span className="font-medium">↑</span> = UP prediction, <span className="font-medium">↓</span> = DOWN prediction</li>
                <li>• <span className="font-medium">Opacity</span> = Confidence level (darker = more confident)</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
      
      {/* Charts Grid */}
      <div className="space-y-6">
        {data.charts.map((chart: ChartData) => {
          const targets = ['1pct', '2pct', '5pct']
          const latestPrice = chart.price_data[chart.price_data.length - 1]?.close || 0
          
          return (
            <div key={chart.symbol} className="space-y-4">
              {/* Asset Header */}
              <div className="flex items-center gap-3 mb-2">
                <CryptoLogo symbol={chart.symbol} size="lg" />
                <div>
                  <h3 className="text-xl font-bold">{chart.symbol.replace('/USDT', '')}</h3>
                  <div className="text-sm text-dark-muted">
                    Latest: <span className="font-mono font-bold text-white">${latestPrice.toFixed(2)}</span>
                  </div>
                </div>
              </div>
              
              {/* Separate chart for each target */}
              {targets.map(targetKey => {
                const targetPreds = chart.predictions[targetKey] || []
                const chartData = processChartData(chart, targetKey)
                const targetLabel = targetKey.replace('pct', '')
                
                // Calculate stats for this target
                const targetStats = {
                  total: targetPreds.length,
                  correct: targetPreds.filter(p => p.status === 'correct').length,
                  pending: targetPreds.filter(p => p.status === 'pending').length,
                  accuracy: 0
                }
                const completed = targetStats.total - targetStats.pending
                if (completed > 0) {
                  targetStats.accuracy = (targetStats.correct / completed) * 100
                }
                
                return (
                  <div key={targetKey} className="card-glow">
                    {/* Chart Header */}
                    <div className="flex items-center justify-between mb-4">
                      <div className="flex items-center gap-2">
                        <span className="px-3 py-1 bg-gradient-to-r from-info/20 to-purple-500/20 text-info rounded-full text-sm font-bold border border-info/30">
                          ±{targetLabel}%
                        </span>
                        <span className="text-sm text-dark-muted">Target</span>
                      </div>
                      
                      {/* Statistics for this target */}
                      <div className="flex items-center gap-4 text-sm">
                        <div className="text-center">
                          <div className="text-xs text-dark-muted">Predictions</div>
                          <div className="font-bold">{targetStats.total}</div>
                        </div>
                        <div className="text-center">
                          <div className="text-xs text-dark-muted">Correct</div>
                          <div className="font-bold text-success">{targetStats.correct}</div>
                        </div>
                        <div className="text-center">
                          <div className="text-xs text-dark-muted">Pending</div>
                          <div className="font-bold text-warning">{targetStats.pending}</div>
                        </div>
                        <div className="text-center">
                          <div className="text-xs text-dark-muted">Accuracy</div>
                          <div className="font-bold text-info">{targetStats.accuracy.toFixed(1)}%</div>
                        </div>
                      </div>
                    </div>
              
                    {/* Chart */}
                    <div className="h-80">
                      <ResponsiveContainer width="100%" height="100%">
                        <ComposedChart data={chartData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                          <defs>
                            <linearGradient id={`colorPrice${chart.symbol}${targetKey}`} x1="0" y1="0" x2="0" y2="1">
                              <stop offset="5%" stopColor="#60A5FA" stopOpacity={0.3}/>
                              <stop offset="95%" stopColor="#60A5FA" stopOpacity={0}/>
                            </linearGradient>
                          </defs>
                          
                          <CartesianGrid 
                            strokeDasharray="3 3" 
                            stroke="#374151" 
                            opacity={0.2}
                            horizontal={true}
                            vertical={false}
                          />
                          
                          <XAxis 
                            dataKey="timestamp"
                            tickFormatter={(value) => format(new Date(value), 'HH:mm')}
                            stroke="#6B7280"
                            tick={{ fontSize: 9, fill: '#6B7280' }}
                            axisLine={{ stroke: '#374151' }}
                            tickLine={{ stroke: '#374151' }}
                            interval="preserveStartEnd"
                            minTickGap={50}
                          />
                          
                          <YAxis 
                            domain={[(dataMin: number) => Math.floor(dataMin * 0.995), (dataMax: number) => Math.ceil(dataMax * 1.005)]}
                            stroke="#6B7280"
                            tick={{ fontSize: 9, fill: '#6B7280' }}
                            axisLine={{ stroke: '#374151' }}
                            tickLine={{ stroke: '#374151' }}
                            tickFormatter={(value) => {
                              // Format based on price range
                              if (value >= 10000) {
                                return `$${(value / 1000).toFixed(0)}k`
                              } else if (value >= 1000) {
                                return `$${value.toFixed(0)}`
                              } else if (value >= 100) {
                                return `$${value.toFixed(0)}`
                              } else {
                                return `$${value.toFixed(1)}`
                              }
                            }}
                            tickCount={6}
                            allowDecimals={false}
                            scale="linear"
                          />
                          
                          <Tooltip content={<CustomTooltip />} />
                          
                          {/* Price Area */}
                          <Area
                            type="monotone"
                            dataKey="close"
                            stroke="#60A5FA"
                            fill={`url(#colorPrice${chart.symbol}${targetKey})`}
                            strokeWidth={2}
                          />
                          
                          {/* Price Line with Prediction Dots */}
                          <Line
                            type="monotone"
                            dataKey="close"
                            stroke="#60A5FA"
                            strokeWidth={2}
                            dot={(props: any) => {
                              const { cx, cy, payload } = props
                              if (payload.prediction) {
                                return (
                                  <PredictionDot
                                    cx={cx}
                                    cy={cy}
                                    payload={payload}
                                    dataKey="prediction"
                                  />
                                )
                              }
                              // Return an invisible dot instead of null
                              return <circle cx={cx} cy={cy} r={0} fill="none" />
                            }}
                            activeDot={{ r: 4 }}
                          />
                        </ComposedChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                )
              })}
            </div>
          )
        })}
      </div>
    </div>
  )
}