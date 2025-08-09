import { useQuery } from '@tanstack/react-query'
import { useState } from 'react'
import { format } from 'date-fns'
import { Database, AlertTriangle, CheckCircle, Activity, TrendingUp, Clock, DollarSign } from 'lucide-react'
import { CryptoLogo } from '../components/CryptoLogo'
import { XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar } from 'recharts'
import { api } from '../api'

export function DataIntegrityDetail() {
  const [selectedSymbol, setSelectedSymbol] = useState<string | null>(null)
  
  const { data, isLoading, error } = useQuery({
    queryKey: ['data-integrity-detailed', selectedSymbol],
    queryFn: async () => {
      const params = selectedSymbol ? `?symbol=${selectedSymbol}` : ''
      const { data } = await api.get(`/api/data/integrity-detailed${params}`)
      return data
    },
  })

  const { data: futuresData } = useQuery({
    queryKey: ['futures-data', selectedSymbol],
    queryFn: async () => {
      const params = selectedSymbol ? `?symbol=${selectedSymbol}` : ''
      const { data } = await api.get(`/api/data/futures${params}`)
      return data
    },
  })

  const getCompletenessColor = (completeness: number) => {
    if (completeness >= 95) return 'text-success'
    if (completeness >= 80) return 'text-warning'
    return 'text-error'
  }

  const getGapSeverity = (hours: number) => {
    if (hours < 2) return { color: 'text-warning', label: 'Minor' }
    if (hours < 6) return { color: 'text-orange-500', label: 'Moderate' }
    return { color: 'text-error', label: 'Severe' }
  }

  if (isLoading) return <div className="card">Loading data integrity report...</div>
  if (error) return <div className="card text-error">Failed to load data integrity report</div>

  return (
    <div className="container mx-auto px-4 py-6">
      <div className="mb-6">
        <h1 className="text-3xl font-bold mb-4 flex items-center gap-3">
          <Database className="w-8 h-8 text-info" />
          Data Integrity Report
        </h1>
        
        {/* Symbol Filter */}
        <div className="flex gap-2 mb-6">
          <button
            onClick={() => setSelectedSymbol(null)}
            className={`px-4 py-2 rounded-lg transition-colors ${
              !selectedSymbol ? 'bg-info text-white' : 'bg-dark-card border border-dark-border hover:bg-dark-border'
            }`}
          >
            All Assets
          </button>
          {['BTC/USDT', 'ETH/USDT', 'SOL/USDT'].map((symbol) => (
            <button
              key={symbol}
              onClick={() => setSelectedSymbol(symbol)}
              className={`px-4 py-2 rounded-lg transition-colors flex items-center gap-2 ${
                selectedSymbol === symbol ? 'bg-info text-white' : 'bg-dark-card border border-dark-border hover:bg-dark-border'
              }`}
            >
              <CryptoLogo symbol={symbol} size="sm" />
              {symbol.replace('/USDT', '')}
            </button>
          ))}
        </div>
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
        <div className="card">
          <div className="flex items-center justify-between mb-2">
            <span className="text-dark-muted">Data Completeness</span>
            <Activity className="w-5 h-5 text-info" />
          </div>
          <div className={`text-3xl font-bold ${getCompletenessColor(data?.summary?.data_completeness || 0)}`}>
            {data?.summary?.data_completeness?.toFixed(1)}%
          </div>
          <div className="text-sm text-dark-muted mt-1">
            Average across all days
          </div>
        </div>

        <div className="card">
          <div className="flex items-center justify-between mb-2">
            <span className="text-dark-muted">Total Gaps Detected</span>
            <AlertTriangle className="w-5 h-5 text-warning" />
          </div>
          <div className="text-3xl font-bold text-warning">
            {data?.summary?.total_gaps || 0}
          </div>
          <div className="text-sm text-dark-muted mt-1">
            In the last 30 days
          </div>
        </div>

        <div className="card">
          <div className="flex items-center justify-between mb-2">
            <span className="text-dark-muted">Average Gap Duration</span>
            <Clock className="w-5 h-5 text-error" />
          </div>
          <div className="text-3xl font-bold">
            {data?.summary?.avg_gap_hours?.toFixed(1)} hrs
          </div>
          <div className="text-sm text-dark-muted mt-1">
            When gaps occur
          </div>
        </div>
      </div>

      {/* Daily Completeness Chart */}
      {data?.daily_stats && data.daily_stats.length > 0 && (
        <div className="card mb-6">
          <h2 className="text-xl font-semibold mb-4">Daily Data Completeness</h2>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={data.daily_stats.slice(0, 30).reverse()}>
              <CartesianGrid strokeDasharray="3 3" stroke="#262626" />
              <XAxis 
                dataKey="date" 
                stroke="#a3a3a3"
                tick={{ fontSize: 12 }}
                tickFormatter={(value) => format(new Date(value), 'MMM dd')}
              />
              <YAxis 
                stroke="#a3a3a3"
                tick={{ fontSize: 12 }}
                domain={[0, 100]}
              />
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: '#141414', 
                  border: '1px solid #262626',
                  borderRadius: '8px'
                }}
                labelFormatter={(value) => format(new Date(value), 'MMM dd, yyyy')}
                formatter={(value: any) => [`${value.toFixed(1)}%`, 'Completeness']}
              />
              <Bar 
                dataKey="completeness" 
                fill="#3b82f6"
                radius={[4, 4, 0, 0]}
              />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Gaps Table */}
      <div className="card mb-6">
        <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
          <AlertTriangle className="w-5 h-5 text-warning" />
          Data Gaps Detected
        </h2>
        
        {data?.gaps && data.gaps.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="border-b border-dark-border">
                <tr className="text-left">
                  <th className="p-3 font-semibold">Asset</th>
                  <th className="p-3 font-semibold">Gap Start</th>
                  <th className="p-3 font-semibold">Gap End</th>
                  <th className="p-3 font-semibold">Duration</th>
                  <th className="p-3 font-semibold">Price at Gap</th>
                  <th className="p-3 font-semibold">Severity</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-dark-border">
                {data.gaps.map((gap: any, idx: number) => {
                  const severity = getGapSeverity(gap.gap_hours)
                  return (
                    <tr key={idx} className="hover:bg-dark-border/30 transition-colors">
                      <td className="p-3">
                        <div className="flex items-center gap-2">
                          <CryptoLogo symbol={gap.symbol} size="sm" />
                          <span>{gap.symbol.replace('/USDT', '')}</span>
                        </div>
                      </td>
                      <td className="p-3">
                        <div className="text-sm">
                          <div>{format(new Date(gap.gap_start), 'MMM dd')}</div>
                          <div className="text-dark-muted">{format(new Date(gap.gap_start), 'HH:mm')}</div>
                        </div>
                      </td>
                      <td className="p-3">
                        <div className="text-sm">
                          <div>{format(new Date(gap.gap_end), 'MMM dd')}</div>
                          <div className="text-dark-muted">{format(new Date(gap.gap_end), 'HH:mm')}</div>
                        </div>
                      </td>
                      <td className="p-3">
                        <span className="font-mono">{gap.gap_hours} hrs</span>
                      </td>
                      <td className="p-3">
                        <span className="font-mono">${gap.price_at_gap.toLocaleString()}</span>
                      </td>
                      <td className="p-3">
                        <span className={`px-2 py-1 rounded text-xs font-semibold ${severity.color}`}>
                          {severity.label}
                        </span>
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="text-center py-8 text-dark-muted">
            <CheckCircle className="w-12 h-12 mx-auto mb-3 text-success" />
            <p>No data gaps detected in the selected period</p>
          </div>
        )}
      </div>

      {/* Futures Data Summary */}
      {futuresData?.summary && futuresData.summary.length > 0 ? (
        <div className="card-glow mb-6 border-2 border-accent/50">
          <h2 className="text-xl font-semibold mb-4 flex items-center gap-2 glow-text">
            <DollarSign className="w-5 h-5 text-accent animate-pulse" />
            Futures Data Summary
          </h2>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
            {futuresData.summary.map((futures: any) => (
              <div key={futures.symbol} className="bg-dark-bg rounded-lg p-4 border border-dark-border">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <CryptoLogo symbol={futures.symbol} size="sm" />
                    <span className="font-semibold">{futures.symbol.replace('/USDT', '')}</span>
                  </div>
                  <span className="text-xs text-dark-muted">{futures.total_records} records</span>
                </div>
                
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-dark-muted">Avg Open Interest:</span>
                    <span className="font-mono">${(futures.avg_open_interest / 1000000).toFixed(2)}M</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-dark-muted">Avg Funding Rate:</span>
                    <span className={`font-mono ${
                      futures.avg_funding_rate_bps > 0 ? 'text-success' : 
                      futures.avg_funding_rate_bps < 0 ? 'text-error' : 'text-dark-text'
                    }`}>
                      {futures.avg_funding_rate_bps.toFixed(2)} bps
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-dark-muted">Funding Range:</span>
                    <span className="font-mono text-xs">
                      {futures.min_funding_rate_bps.toFixed(2)} to {futures.max_funding_rate_bps.toFixed(2)} bps
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-dark-muted">Funding Volatility:</span>
                    <span className="font-mono">{futures.funding_volatility_bps.toFixed(2)} bps</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-dark-muted">Latest Update:</span>
                    <span className="text-xs">
                      {format(new Date(futures.latest_record), 'MMM dd HH:mm')}
                    </span>
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* Recent Futures Data Chart */}
          {futuresData.recent_data && futuresData.recent_data.length > 0 && (
            <div className="mt-6">
              <h3 className="text-lg font-semibold mb-3">Recent Funding Rates</h3>
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={futuresData.recent_data.slice(0, 48).reverse()}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#262626" />
                  <XAxis 
                    dataKey="timestamp" 
                    stroke="#7da3c0"
                    tick={{ fontSize: 10 }}
                    tickFormatter={(value) => format(new Date(value), 'HH:mm')}
                  />
                  <YAxis 
                    stroke="#7da3c0"
                    tick={{ fontSize: 10 }}
                    domain={['dataMin', 'dataMax']}
                  />
                  <Tooltip 
                    contentStyle={{ 
                      backgroundColor: '#0a1628', 
                      border: '1px solid #1e3a5f',
                      borderRadius: '8px'
                    }}
                    labelFormatter={(value) => format(new Date(value), 'MMM dd, HH:mm')}
                    formatter={(value: any) => [`${value.toFixed(4)} bps`, 'Funding Rate']}
                  />
                  <Bar 
                    dataKey="funding_rate_bps" 
                    fill="#00b4d8"
                    radius={[2, 2, 0, 0]}
                  />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>
      ) : (
        <div className="card mb-6 border border-warning/30">
          <div className="flex items-center gap-3 text-warning">
            <AlertTriangle className="w-5 h-5" />
            <span>No futures data available</span>
          </div>
        </div>
      )}

      {/* Daily Statistics */}
      <div className="card">
        <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
          <TrendingUp className="w-5 h-5 text-info" />
          Daily Statistics
        </h2>
        
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="border-b border-dark-border">
              <tr className="text-left">
                <th className="p-3 font-semibold">Date</th>
                <th className="p-3 font-semibold">Asset</th>
                <th className="p-3 font-semibold">Candles</th>
                <th className="p-3 font-semibold">Completeness</th>
                <th className="p-3 font-semibold">Price Range</th>
                <th className="p-3 font-semibold">Avg Volume</th>
                <th className="p-3 font-semibold">Volatility</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-dark-border">
              {data?.daily_stats?.slice(0, 20).map((stat: any, idx: number) => (
                <tr key={idx} className="hover:bg-dark-border/30 transition-colors">
                  <td className="p-3">
                    {format(new Date(stat.date), 'MMM dd')}
                  </td>
                  <td className="p-3">
                    <div className="flex items-center gap-2">
                      <CryptoLogo symbol={stat.symbol} size="sm" />
                      <span>{stat.symbol.replace('/USDT', '')}</span>
                    </div>
                  </td>
                  <td className="p-3">
                    <span className={stat.candles_count < 24 ? 'text-warning' : ''}>
                      {stat.candles_count}/{stat.expected_candles}
                    </span>
                  </td>
                  <td className="p-3">
                    <div className="flex items-center gap-2">
                      <div className="w-16 bg-dark-border rounded-full h-2">
                        <div 
                          className={`h-2 rounded-full ${
                            stat.completeness >= 95 ? 'bg-success' : 
                            stat.completeness >= 80 ? 'bg-warning' : 'bg-error'
                          }`}
                          style={{ width: `${stat.completeness}%` }}
                        />
                      </div>
                      <span className={`text-sm ${getCompletenessColor(stat.completeness)}`}>
                        {stat.completeness.toFixed(1)}%
                      </span>
                    </div>
                  </td>
                  <td className="p-3">
                    <div className="text-sm">
                      <div className="font-mono">${stat.day_low.toLocaleString()}</div>
                      <div className="font-mono">${stat.day_high.toLocaleString()}</div>
                    </div>
                  </td>
                  <td className="p-3">
                    <span className="font-mono text-sm">
                      {(stat.avg_volume / 1000000).toFixed(2)}M
                    </span>
                  </td>
                  <td className="p-3">
                    <span className="font-mono text-sm">
                      {stat.price_volatility.toFixed(2)}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}