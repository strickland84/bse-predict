import { useQuery } from '@tanstack/react-query'
import { useState, useEffect, useRef } from 'react'
import { format } from 'date-fns'
import { ArrowUpCircle, ArrowDownCircle, MinusCircle, CheckCircle, XCircle, Clock } from 'lucide-react'
import { CryptoLogo } from '../components/CryptoLogo'
import { Pagination } from '../components/Pagination'
import { FilterPanel, SortControl } from '../components/FilterPanel'
import { api } from '../api'

export function PredictionsDetail() {
  const [selectedSymbol, setSelectedSymbol] = useState<string | null>(null)
  const [statsPeriod, setStatsPeriod] = useState(7)
  const [currentPage, setCurrentPage] = useState(1)
  const perPage = 50
  const scrollPositionRef = useRef(0)
  const shouldRestoreScroll = useRef(false)
  
  // Filter states
  const [filters, setFilters] = useState<Record<string, any>>({
    outcome: null,
    target: null,
    minConfidence: null
  })
  
  // Sort states
  const [sortBy, setSortBy] = useState('timestamp')
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc')
  
  const { data, isLoading, error } = useQuery({
    queryKey: ['predictions-detailed', selectedSymbol, currentPage, filters, sortBy, sortOrder],
    queryFn: async () => {
      // Save scroll position before fetching
      scrollPositionRef.current = window.scrollY
      shouldRestoreScroll.current = true
      
      const params = new URLSearchParams()
      params.append('page', currentPage.toString())
      params.append('per_page', perPage.toString())
      if (selectedSymbol) params.append('symbol', selectedSymbol)
      if (filters.outcome) params.append('outcome_filter', filters.outcome)
      if (filters.target) params.append('target_filter', filters.target)
      if (filters.minConfidence) params.append('min_confidence', filters.minConfidence)
      params.append('sort_by', sortBy)
      params.append('sort_order', sortOrder)
      const { data } = await api.get(`/api/predictions/detailed?${params}`)
      return data
    },
  })
  
  // Restore scroll position after data loads
  useEffect(() => {
    if (!isLoading && shouldRestoreScroll.current) {
      window.scrollTo(0, scrollPositionRef.current)
      shouldRestoreScroll.current = false
    }
  }, [isLoading])

  const { data: stats, isLoading: statsLoading } = useQuery({
    queryKey: ['predictions-statistics', statsPeriod, selectedSymbol],
    queryFn: async () => {
      const params = new URLSearchParams({ days: statsPeriod.toString() })
      if (selectedSymbol) params.append('symbol', selectedSymbol)
      const { data } = await api.get(`/api/predictions/statistics?${params}`)
      return data
    },
  })

  const getPredictionIcon = (prediction: string) => {
    switch (prediction) {
      case 'UP':
        return <ArrowUpCircle className="w-5 h-5 text-success" />
      case 'DOWN':
        return <ArrowDownCircle className="w-5 h-5 text-error" />
      default:
        return <MinusCircle className="w-5 h-5 text-dark-muted" />
    }
  }

  const getOutcomeIcon = (outcome: number | null) => {
    if (outcome === null) return <Clock className="w-4 h-4 text-warning" />
    return outcome === 1 ? 
      <CheckCircle className="w-4 h-4 text-success" /> : 
      <XCircle className="w-4 h-4 text-error" />
  }

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 80) return 'text-success'
    if (confidence >= 60) return 'text-warning'
    return 'text-error'
  }

  if (isLoading) return <div className="card">Loading predictions history...</div>
  if (error) return <div className="card text-error">Failed to load predictions history</div>

  return (
    <div className="container mx-auto px-4 py-6">
      <div className="mb-6">
        <h1 className="text-3xl font-bold mb-4 glow-text">Predictions History</h1>
        
        {/* Symbol Filter */}
        <div className="flex gap-2 mb-6">
          <button
            onClick={() => {
              scrollPositionRef.current = 0  // Reset scroll for symbol change
              setSelectedSymbol(null)
              setCurrentPage(1)
            }}
            className={`px-4 py-2 rounded-lg transition-all duration-300 ${
              !selectedSymbol ? 'btn-primary' : 'btn-secondary'
            }`}
          >
            All Assets
          </button>
          {['BTC/USDT', 'ETH/USDT', 'SOL/USDT'].map((symbol) => (
            <button
              key={symbol}
              onClick={() => {
                scrollPositionRef.current = 0  // Reset scroll for symbol change
                setSelectedSymbol(symbol)
                setCurrentPage(1)
              }}
              className={`px-4 py-2 rounded-lg transition-all duration-300 flex items-center gap-2 ${
                selectedSymbol === symbol ? 'btn-primary' : 'btn-secondary'
              }`}
            >
              <CryptoLogo symbol={symbol} size="sm" />
              {symbol.replace('/USDT', '')}
            </button>
          ))}
        </div>
      </div>

      {/* Statistics Section */}
      {stats && !statsLoading && (
        <div className="mb-6">
          {/* Period Selector */}
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-xl font-bold">Performance Statistics</h2>
            <div className="flex gap-2">
              {[1, 7, 14, 30].map((days) => (
                <button
                  key={days}
                  onClick={() => setStatsPeriod(days)}
                  className={`px-3 py-1 rounded-lg text-xs transition-all ${
                    statsPeriod === days ? 'btn-primary' : 'btn-secondary'
                  }`}
                >
                  {days}D
                </button>
              ))}
            </div>
          </div>

          {/* Overall Statistics */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
            <div className="card p-4">
              <div className="text-xs text-dark-muted mb-1">Overall Hit Rate</div>
              <div className="text-2xl font-bold text-primary">
                {stats.overall.overall_hit_rate?.toFixed(1)}%
              </div>
              <div className="text-xs text-dark-muted mt-1">
                {stats.overall.wins}/{stats.overall.wins + stats.overall.losses} wins
              </div>
            </div>

            <div className="card p-4">
              <div className="text-xs text-dark-muted mb-1">Win/Loss/Expired</div>
              <div className="flex items-center gap-2">
                <span className="text-success font-bold">{stats.overall.wins}</span>
                <span className="text-dark-muted">/</span>
                <span className="text-error font-bold">{stats.overall.losses}</span>
                <span className="text-dark-muted">/</span>
                <span className="text-warning font-bold">{stats.overall.expired}</span>
              </div>
              <div className="text-xs text-dark-muted mt-1">
                Total: {stats.overall.tracked_outcomes}
              </div>
            </div>

            <div className="card p-4">
              <div className="text-xs text-dark-muted mb-1">Avg Win Time</div>
              <div className="text-2xl font-bold text-success">
                {stats.overall.timing.avg_time_to_win_hours?.toFixed(1)}h
              </div>
              <div className="text-xs text-dark-muted mt-1">
                Range: {stats.overall.timing.min_time_to_win_hours?.toFixed(1)}-{stats.overall.timing.max_time_to_win_hours?.toFixed(1)}h
              </div>
            </div>

            <div className="card p-4">
              <div className="text-xs text-dark-muted mb-1">MFE/MAE</div>
              <div className="flex items-center gap-2">
                <span className="text-success">+{stats.overall.excursions.avg_mfe?.toFixed(2)}%</span>
                <span className="text-error">{stats.overall.excursions.avg_mae?.toFixed(2)}%</span>
              </div>
              <div className="text-xs text-dark-muted mt-1">
                Avg favorable/adverse
              </div>
            </div>
          </div>

          {/* Model Breakdown */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-6">
            <div className="card p-4">
              <h3 className="text-sm font-bold mb-3">Performance by Model</h3>
              <div className="space-y-2">
                {stats.by_model?.map((model: any) => (
                  <div key={`${model.symbol}-${model.target_pct}`} className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <CryptoLogo symbol={model.symbol} size="sm" />
                      <span className="text-xs">{model.symbol.replace('/USDT', '')} {model.target_pct}</span>
                    </div>
                    <div className="flex items-center gap-3">
                      <span className={`text-xs font-bold ${model.hit_rate >= 60 ? 'text-success' : model.hit_rate >= 40 ? 'text-warning' : 'text-error'}`}>
                        {model.hit_rate?.toFixed(1)}%
                      </span>
                      <span className="text-xs text-dark-muted">
                        {model.wins}W/{model.losses}L
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Confidence Cohorts */}
            <div className="card p-4">
              <h3 className="text-sm font-bold mb-3">Confidence Cohort Analysis</h3>
              <div className="space-y-2">
                {stats.confidence_cohorts?.map((cohort: any) => (
                  <div key={cohort.confidence_range} className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <div className={`w-2 h-2 rounded-full ${
                        cohort.accuracy >= 70 ? 'bg-success' : 
                        cohort.accuracy >= 50 ? 'bg-warning' : 'bg-error'
                      }`} />
                      <span className="text-xs">{cohort.confidence_range}</span>
                    </div>
                    <div className="flex items-center gap-3">
                      <span className={`text-xs font-bold ${
                        cohort.accuracy >= 70 ? 'text-success' : 
                        cohort.accuracy >= 50 ? 'text-warning' : 'text-error'
                      }`}>
                        {cohort.accuracy?.toFixed(1)}%
                      </span>
                      <span className="text-xs text-dark-muted">
                        ({cohort.correct}/{cohort.total_predictions})
                      </span>
                    </div>
                  </div>
                ))}
                {stats.confidence_cohorts?.length === 0 && (
                  <div className="text-xs text-dark-muted">No cohort data available</div>
                )}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Filters and Sorting */}
      <FilterPanel
        filters={[
          {
            id: 'outcome',
            label: 'Outcome',
            type: 'select',
            options: [
              { label: 'Hit', value: 'win' },
              { label: 'Miss', value: 'loss' },
              { label: 'Pending', value: 'pending' }
            ]
          },
          {
            id: 'target',
            label: 'Target',
            type: 'select',
            options: [
              { label: '±1%', value: '1' },
              { label: '±2%', value: '2' },
              { label: '±5%', value: '5' }
            ]
          },
          {
            id: 'minConfidence',
            label: 'Min Confidence',
            type: 'range',
            min: 0,
            max: 100,
            step: 5,
            suffix: '%'
          }
        ]}
        values={filters}
        onChange={(id, value) => {
          setFilters(prev => ({ ...prev, [id]: value }))
          setCurrentPage(1)
        }}
        onReset={() => {
          setFilters({ outcome: null, target: null, minConfidence: null })
          setCurrentPage(1)
        }}
        className="mb-6"
      />

      {/* Predictions Table */}
      <div className="card-glow overflow-hidden">
        <div className="p-4 border-b border-dark-border flex justify-between items-center">
          <h3 className="font-bold">Prediction History</h3>
          <SortControl
            sortBy={sortBy}
            sortOrder={sortOrder}
            options={[
              { label: 'Time', value: 'timestamp' },
              { label: 'Confidence', value: 'confidence' },
              { label: 'Target', value: 'target_pct' },
              { label: 'Symbol', value: 'symbol' }
            ]}
            onSortChange={(field, order) => {
              setSortBy(field)
              setSortOrder(order)
            }}
          />
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead className="border-b border-dark-border">
              <tr className="text-left text-dark-muted">
                <th className="p-2 font-medium">Asset</th>
                <th className="p-2 font-medium">Time</th>
                <th className="p-2 font-medium">Target</th>
                <th className="p-2 font-medium">Prediction</th>
                <th className="p-2 font-medium">Confidence</th>
                <th className="p-2 font-medium">Max Move</th>
                <th className="p-2 font-medium">Time to Target</th>
                <th className="p-2 font-medium">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-dark-border">
              {data?.predictions?.map((pred: any) => (
                <tr key={pred.id} className="hover:bg-dark-border/30 transition-colors">
                  <td className="p-2">
                    <div className="flex items-center gap-1">
                      <CryptoLogo symbol={pred.symbol} size="sm" />
                      <span className="font-medium">{pred.symbol.replace('/USDT', '')}</span>
                    </div>
                  </td>
                  <td className="p-2">
                    <div className="text-xs">
                      <div>{format(new Date(pred.prediction_time), 'MMM dd')}</div>
                      <div className="text-dark-muted">{format(new Date(pred.prediction_time), 'HH:mm')}</div>
                    </div>
                  </td>
                  <td className="p-2">
                    <span className="px-2 py-1 bg-gradient-to-r from-blue-glow/20 to-accent/20 rounded text-xs font-medium border border-accent/30">
                      ±{pred.target_pct}%
                    </span>
                  </td>
                  <td className="p-2">
                    <div className="flex items-center gap-1">
                      {getPredictionIcon(pred.prediction)}
                      <span className="font-medium">{pred.prediction}</span>
                    </div>
                  </td>
                  <td className="p-2">
                    <div>
                      <div className={`font-semibold ${getConfidenceColor(pred.confidence)}`}>
                        {pred.confidence.toFixed(1)}%
                      </div>
                    </div>
                  </td>
                  <td className="p-2">
                    {pred.max_favorable !== null ? (
                      <div>
                        <div className="text-success text-xs">+{pred.max_favorable?.toFixed(2)}%</div>
                        <div className="text-error text-xs">{pred.max_adverse?.toFixed(2)}%</div>
                      </div>
                    ) : (
                      <span className="text-dark-muted">-</span>
                    )}
                  </td>
                  <td className="p-2">
                    {pred.time_to_target !== null ? (
                      <span className="font-mono text-xs">{pred.time_to_target?.toFixed(1)}h</span>
                    ) : (
                      <span className="text-dark-muted">-</span>
                    )}
                  </td>
                  <td className="p-2">
                    <div className="flex items-center gap-1">
                      {getOutcomeIcon(pred.actual_outcome)}
                      <span className="text-xs">
                        {pred.actual_outcome === null ? 'Pending' : pred.actual_outcome === 1 ? 'Hit' : 'Miss'}
                      </span>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        
        {/* Pagination and Summary Stats */}
        <div className="p-4 border-t border-dark-border bg-gradient-to-r from-dark-card/50 to-dark-bg/50">
          {data?.pagination && (
            <Pagination
              page={currentPage}
              totalPages={data.pagination.total_pages}
              onPageChange={setCurrentPage}
              perPage={perPage}
              total={data.pagination.total}
            />
          )}
          {data?.predictions && (
            <div className="flex gap-4 text-sm mt-4 justify-center">
              <span className="text-success">
                Hit: {data.predictions.filter((p: any) => p.actual_outcome === 1).length}
              </span>
              <span className="text-error">
                Miss: {data.predictions.filter((p: any) => p.actual_outcome === 0).length}
              </span>
              <span className="text-warning">
                Pending: {data.predictions.filter((p: any) => p.actual_outcome === null).length}
              </span>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}