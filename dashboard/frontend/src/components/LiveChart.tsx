import { useQuery } from '@tanstack/react-query'
import { fetchLatestCandles } from '../api'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts'
import { format } from 'date-fns'
import { useState } from 'react'
import clsx from 'clsx'
import { LineChart as ChartIcon } from 'lucide-react'

export const LiveChart = () => {
  const [selectedSymbol, setSelectedSymbol] = useState('BTC/USDT')
  
  const { data, isLoading } = useQuery({
    queryKey: ['latestCandles', selectedSymbol],
    queryFn: () => fetchLatestCandles(selectedSymbol, 50),
  })

  const symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']

  const chartData = data?.candles?.map((candle: any) => ({
    time: format(new Date(candle.timestamp), 'HH:mm'),
    price: candle.close,
    high: candle.high,
    low: candle.low,
  })).reverse()

  return (
    <div className="card">
      <div className="flex justify-between items-center mb-4">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-purple-500/10 rounded-lg">
            <ChartIcon className="w-5 h-5 text-purple-500" />
          </div>
          <div>
            <h2 className="text-xl font-bold">Live Price Chart</h2>
            <p className="text-xs text-dark-muted">Real-time market data</p>
          </div>
        </div>
        <div className="flex gap-2">
          {symbols.map((symbol) => (
            <button
              key={symbol}
              onClick={() => setSelectedSymbol(symbol)}
              className={clsx(
                'px-3 py-1 rounded text-sm transition-colors',
                selectedSymbol === symbol
                  ? 'bg-info text-white'
                  : 'bg-dark-bg border border-dark-border hover:bg-dark-border'
              )}
            >
              {symbol.replace('/USDT', '')}
            </button>
          ))}
        </div>
      </div>

      {isLoading ? (
        <div className="h-64 flex items-center justify-center text-dark-muted">
          Loading chart data...
        </div>
      ) : (
        <ResponsiveContainer width="100%" height={250}>
          <LineChart data={chartData} margin={{ top: 5, right: 5, left: 5, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#262626" />
            <XAxis 
              dataKey="time" 
              stroke="#a3a3a3"
              tick={{ fontSize: 12 }}
            />
            <YAxis 
              stroke="#a3a3a3"
              tick={{ fontSize: 12 }}
              domain={['dataMin - 50', 'dataMax + 50']}
            />
            <Tooltip 
              contentStyle={{ 
                backgroundColor: '#141414', 
                border: '1px solid #262626',
                borderRadius: '8px'
              }}
              labelStyle={{ color: '#e5e5e5' }}
            />
            <Legend />
            <Line 
              type="monotone" 
              dataKey="price" 
              stroke="#3b82f6" 
              strokeWidth={2}
              dot={false}
              name="Price"
            />
            <Line 
              type="monotone" 
              dataKey="high" 
              stroke="#10b981" 
              strokeWidth={1}
              dot={false}
              strokeDasharray="3 3"
              name="High"
            />
            <Line 
              type="monotone" 
              dataKey="low" 
              stroke="#ef4444" 
              strokeWidth={1}
              dot={false}
              strokeDasharray="3 3"
              name="Low"
            />
          </LineChart>
        </ResponsiveContainer>
      )}
    </div>
  )
}