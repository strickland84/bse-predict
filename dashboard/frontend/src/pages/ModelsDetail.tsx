import { useQuery } from '@tanstack/react-query'
import { useState, useEffect, useRef } from 'react'
import { format } from 'date-fns'
import { Brain, TrendingUp, Target, BarChart3, Package, Zap, Clock, CheckCircle, AlertCircle, Activity, ChevronDown, ChevronUp } from 'lucide-react'
import { CryptoLogo } from '../components/CryptoLogo'
import { Pagination } from '../components/Pagination'
import { FilterPanel, SortControl } from '../components/FilterPanel'
import { api } from '../api'
import clsx from 'clsx'

// Crypto logos as simple SVG components (same as ModelPerformance)
const CryptoLogos = {
  BTC: () => (
    <svg viewBox="0 0 32 32" className="w-10 h-10">
      <circle cx="16" cy="16" r="16" fill="#F7931A"/>
      <path fill="white" d="M23.189 14.02c.314-2.096-1.283-3.223-3.465-3.975l.708-2.84-1.728-.43-.69 2.765c-.454-.114-.92-.22-1.385-.326l.695-2.783L15.596 6l-.708 2.839c-.376-.086-.746-.17-1.104-.26l.002-.009-2.384-.595-.46 1.846s1.283.294 1.256.312c.7.175.826.638.805 1.006l-.806 3.235c.048.012.11.03.18.057l-.183-.045-1.13 4.532c-.086.212-.303.531-.793.41.018.025-1.256-.313-1.256-.313l-.858 1.978 2.25.561c.418.105.828.215 1.231.318l-.715 2.872 1.727.43.708-2.84c.472.127.93.245 1.378.357l-.706 2.828 1.728.43.715-2.866c2.948.558 5.164.333 6.097-2.333.752-2.146-.037-3.385-1.588-4.192 1.13-.26 1.98-1.003 2.207-2.538zm-3.95 5.538c-.533 2.147-4.148.986-5.32.695l.95-3.805c1.172.293 4.929.872 4.37 3.11zm.535-5.569c-.487 1.953-3.495.96-4.47.717l.86-3.45c.975.243 4.118.696 3.61 2.733z"/>
    </svg>
  ),
  ETH: () => (
    <svg viewBox="0 0 32 32" className="w-10 h-10">
      <circle cx="16" cy="16" r="16" fill="#627EEA"/>
      <g fill="white">
        <path fillOpacity="0.6" d="M16 11v5.586l4.998 2.232L16 11z"/>
        <path d="M16 11l-4.998 7.818L16 16.586V11z"/>
        <path fillOpacity="0.6" d="M16 20.573v4.423L21 17.682l-5 2.891z"/>
        <path d="M16 24.996v-4.423l-5-2.891 5 7.314z"/>
        <path fillOpacity="0.2" d="M16 19.573l4.998-2.755L16 14.586v4.987z"/>
        <path fillOpacity="0.6" d="M11 16.818l5 2.755v-4.987l-5 2.232z"/>
      </g>
    </svg>
  ),
  SOL: () => (
    <svg viewBox="0 0 32 32" className="w-10 h-10">
      <circle cx="16" cy="16" r="16" fill="url(#solana-gradient-models)"/>
      <defs>
        <linearGradient id="solana-gradient-models" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor="#00FFA3"/>
          <stop offset="100%" stopColor="#DC1FFF"/>
        </linearGradient>
      </defs>
      <path fill="white" d="M9.5 20.5c.2-.2.5-.3.8-.3h11.4c.5 0 .7.6.4.9l-2.6 2.6c-.2.2-.5.3-.8.3H7.3c-.5 0-.7-.6-.4-.9l2.6-2.6zm0-9c.2-.2.5-.3.8-.3h11.4c.5 0 .7.6.4.9l-2.6 2.6c-.2.2-.5.3-.8.3H7.3c-.5 0-.7-.6-.4-.9l2.6-2.6zm13 4.5c-.2.2-.5.3-.8.3H10.3c-.5 0-.7-.6-.4-.9l2.6-2.6c.2-.2.5-.3.8-.3h11.4c.5 0 .7.6.4.9l-2.6 2.6z"/>
    </svg>
  )
}

// Progress bar component
const ProgressBar = ({ value, max = 100, className = '', showLabel = true, color = 'bg-info', height = 'h-2' }: {
  value: number
  max?: number
  className?: string
  showLabel?: boolean
  color?: string
  height?: string
}) => {
  const percentage = Math.min(100, Math.max(0, (value / max) * 100))
  
  return (
    <div className={clsx('relative', className)}>
      <div className={clsx('w-full bg-dark-border rounded-full overflow-hidden', height)}>
        <div 
          className={clsx(
            'h-full rounded-full transition-all duration-500 ease-out',
            color
          )}
          style={{ width: `${percentage}%` }}
        />
      </div>
      {showLabel && (
        <span className="absolute -top-1 right-0 text-xs font-medium">
          {value.toFixed(1)}%
        </span>
      )}
    </div>
  )
}

// Performance badge component
const PerformanceBadge = ({ score }: { score: number }) => {
  const getConfig = () => {
    if (score >= 80) return { 
      color: 'text-success bg-success/10 border-success/20', 
      icon: CheckCircle,
      label: 'Excellent'
    }
    if (score >= 70) return { 
      color: 'text-info bg-info/10 border-info/20', 
      icon: TrendingUp,
      label: 'Good'
    }
    if (score >= 60) return { 
      color: 'text-warning bg-warning/10 border-warning/20', 
      icon: Activity,
      label: 'Fair'
    }
    return { 
      color: 'text-error bg-error/10 border-error/20', 
      icon: AlertCircle,
      label: 'Needs Work'
    }
  }
  
  const config = getConfig()
  const Icon = config.icon
  
  return (
    <div className={clsx(
      'inline-flex items-center gap-1 px-3 py-1.5 rounded-full text-sm font-medium border',
      config.color
    )}>
      <Icon className="w-4 h-4" />
      {config.label}
    </div>
  )
}

const getScoreColor = (score: number) => {
  if (score >= 80) return 'bg-success'
  if (score >= 70) return 'bg-info'
  if (score >= 60) return 'bg-warning'
  return 'bg-error'
}

const getMetricTextColor = (value: number) => {
  if (value >= 80) return 'text-success'
  if (value >= 70) return 'text-info'
  if (value >= 60) return 'text-warning'
  return 'text-error'
}

export function ModelsDetail() {
  const [selectedSymbol, setSelectedSymbol] = useState<string | null>(null)
  const [expandedRow, setExpandedRow] = useState<number | null>(null)
  const [currentPage, setCurrentPage] = useState(1)
  const perPage = 20
  const scrollPositionRef = useRef(0)
  const shouldRestoreScroll = useRef(false)
  
  // Filter states
  const [filters, setFilters] = useState<Record<string, any>>({
    target: null,
    minAccuracy: null,
    daysAgo: null
  })
  
  // Sort states
  const [sortBy, setSortBy] = useState('trained_at')
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc')
  
  const { data, isLoading, error } = useQuery({
    queryKey: ['models-detailed', selectedSymbol, currentPage, filters, sortBy, sortOrder],
    queryFn: async () => {
      // Save scroll position before fetching
      scrollPositionRef.current = window.scrollY
      shouldRestoreScroll.current = true
      
      const params = new URLSearchParams()
      params.append('page', currentPage.toString())
      params.append('per_page', perPage.toString())
      if (selectedSymbol) params.append('symbol', selectedSymbol)
      if (filters.target) params.append('target_filter', filters.target)
      if (filters.minAccuracy) params.append('min_accuracy', filters.minAccuracy)
      if (filters.daysAgo) params.append('days_ago', filters.daysAgo)
      params.append('sort_by', sortBy)
      params.append('sort_order', sortOrder)
      const { data } = await api.get(`/api/models/training-history-detailed?${params}`)
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

  const getCryptoLogo = (symbol: string) => {
    const base = symbol.split('/')[0]
    const Logo = CryptoLogos[base as keyof typeof CryptoLogos]
    return Logo ? <Logo /> : null
  }

  if (isLoading) {
    return (
      <div className="container mx-auto px-4 py-6">
        <div className="card">
          <div className="flex items-center gap-3">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-info"></div>
            <span className="text-dark-muted">Loading model training history...</span>
          </div>
        </div>
      </div>
    )
  }
  
  if (error) {
    return (
      <div className="container mx-auto px-4 py-6">
        <div className="card">
          <div className="flex items-center gap-3 text-error">
            <AlertCircle className="w-5 h-5" />
            <span>Failed to load model training history</span>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="container mx-auto px-4 py-6">
      <div className="mb-6">
        <div className="flex items-center gap-4 mb-6">
          <div className="p-3 bg-gradient-to-br from-info/20 to-purple-500/20 rounded-xl">
            <Brain className="w-8 h-8 text-info" />
          </div>
          <div>
            <h1 className="text-3xl font-bold">Model Training History</h1>
            <p className="text-dark-muted">Detailed performance metrics and training logs</p>
          </div>
        </div>
        
        {/* Symbol Filter */}
        <div className="flex gap-2 mb-6">
          <button
            onClick={() => {
              scrollPositionRef.current = 0  // Reset scroll for symbol change
              setSelectedSymbol(null)
              setCurrentPage(1)
              setExpandedRow(null)
            }}
            className={clsx(
              'px-4 py-2 rounded-lg transition-all font-medium',
              !selectedSymbol 
                ? 'bg-gradient-to-r from-info to-purple-500 text-white shadow-lg shadow-info/25' 
                : 'bg-dark-card border border-dark-border hover:bg-dark-border'
            )}
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
                setExpandedRow(null)
              }}
              className={clsx(
                'px-4 py-2 rounded-lg transition-all font-medium flex items-center gap-2',
                selectedSymbol === symbol 
                  ? 'bg-gradient-to-r from-info to-purple-500 text-white shadow-lg shadow-info/25' 
                  : 'bg-dark-card border border-dark-border hover:bg-dark-border'
              )}
            >
              <CryptoLogo symbol={symbol} size="sm" />
              {symbol.replace('/USDT', '')}
            </button>
          ))}
        </div>
      </div>

      {/* Filters and Sorting */}
      <FilterPanel
        filters={[
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
            id: 'minAccuracy',
            label: 'Min Accuracy',
            type: 'range',
            min: 0,
            max: 100,
            step: 5,
            suffix: '%'
          },
          {
            id: 'daysAgo',
            label: 'Days Ago',
            type: 'select',
            options: [
              { label: 'Last 7 days', value: '7' },
              { label: 'Last 14 days', value: '14' },
              { label: 'Last 30 days', value: '30' },
              { label: 'Last 90 days', value: '90' }
            ]
          }
        ]}
        values={filters}
        onChange={(id, value) => {
          setFilters(prev => ({ ...prev, [id]: value }))
          setCurrentPage(1)
          setExpandedRow(null)
        }}
        onReset={() => {
          setFilters({ target: null, minAccuracy: null, daysAgo: null })
          setCurrentPage(1)
          setExpandedRow(null)
        }}
        className="mb-6"
      />

      {/* Sort Control */}
      <div className="mb-4 flex justify-end">
        <SortControl
          sortBy={sortBy}
          sortOrder={sortOrder}
          options={[
            { label: 'Training Date', value: 'trained_at' },
            { label: 'Accuracy', value: 'final_accuracy' },
            { label: 'F1 Score', value: 'f1_score' },
            { label: 'Precision', value: 'precision' },
            { label: 'Recall', value: 'recall' },
            { label: 'Training Samples', value: 'training_samples' }
          ]}
          onSortChange={(field, order) => {
            setSortBy(field)
            setSortOrder(order)
            setExpandedRow(null)
          }}
        />
      </div>

      {/* Training History Cards */}
      <div className="space-y-4">
        {data?.training_history?.map((training: any, idx: number) => (
          <div key={training.id} className="card bg-gradient-to-br from-dark-card via-dark-card to-dark-border/20 hover:shadow-xl transition-all">
            <div 
              className="cursor-pointer"
              onClick={() => setExpandedRow(expandedRow === idx ? null : idx)}
            >
              {/* Header */}
              <div className="flex items-center justify-between mb-6">
                <div className="flex items-center gap-4">
                  {getCryptoLogo(training.symbol)}
                  <div>
                    <div className="flex items-center gap-2 sm:gap-3 flex-wrap">
                      <h3 className="text-xl sm:text-2xl font-bold">{training.symbol.replace('/USDT', '')}</h3>
                      <span className="px-2 sm:px-3 py-1 bg-gradient-to-r from-info/20 to-purple-500/20 text-info rounded-full text-xs sm:text-sm font-bold border border-info/30">
                        ±{training.target_pct}%
                      </span>
                    </div>
                    <div className="mt-2 sm:mt-1">
                      <PerformanceBadge score={training.metrics.final_accuracy} />
                    </div>
                    <div className="flex items-center gap-2 mt-1 text-sm text-dark-muted">
                      <Clock className="w-4 h-4" />
                      {format(new Date(training.trained_at), 'MMM dd, yyyy HH:mm')}
                    </div>
                  </div>
                </div>
                
                <div className="flex items-center gap-2">
                  <div className="text-right">
                    <div className="text-3xl font-bold">{training.metrics.final_accuracy.toFixed(1)}%</div>
                    <div className="text-xs text-dark-muted uppercase tracking-wider">Accuracy</div>
                  </div>
                  {expandedRow === idx ? <ChevronUp className="w-5 h-5 text-dark-muted" /> : <ChevronDown className="w-5 h-5 text-dark-muted" />}
                </div>
              </div>

              {/* Main Metrics with Progress Bars */}
              <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-4">
                <div className="bg-dark-bg/50 rounded-xl p-4 border border-dark-border/50">
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-2">
                      <Target className="w-4 h-4 text-success" />
                      <span className="text-sm font-medium">Precision</span>
                    </div>
                    <span className={clsx('text-lg font-bold', getMetricTextColor(training.metrics.precision))}>
                      {training.metrics.precision.toFixed(1)}%
                    </span>
                  </div>
                  <ProgressBar 
                    value={training.metrics.precision} 
                    showLabel={false}
                    color={getScoreColor(training.metrics.precision)}
                    height="h-3"
                  />
                  <p className="text-xs text-dark-muted mt-2">When model predicts UP/DOWN, how often it's correct</p>
                </div>
                
                <div className="bg-dark-bg/50 rounded-xl p-4 border border-dark-border/50">
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-2">
                      <TrendingUp className="w-4 h-4 text-info" />
                      <span className="text-sm font-medium">Recall</span>
                    </div>
                    <span className={clsx('text-lg font-bold', getMetricTextColor(training.metrics.recall))}>
                      {training.metrics.recall.toFixed(1)}%
                    </span>
                  </div>
                  <ProgressBar 
                    value={training.metrics.recall} 
                    showLabel={false}
                    color={getScoreColor(training.metrics.recall)}
                    height="h-3"
                  />
                  <p className="text-xs text-dark-muted mt-2">Percentage of actual price moves correctly identified</p>
                </div>
                
                <div className="bg-dark-bg/50 rounded-xl p-4 border border-dark-border/50">
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-2">
                      <Zap className="w-4 h-4 text-warning" />
                      <span className="text-sm font-medium">F1 Score</span>
                    </div>
                    <span className={clsx('text-lg font-bold', getMetricTextColor(training.metrics.f1_score))}>
                      {training.metrics.f1_score.toFixed(1)}%
                    </span>
                  </div>
                  <ProgressBar 
                    value={training.metrics.f1_score} 
                    showLabel={false}
                    color={getScoreColor(training.metrics.f1_score)}
                    height="h-3"
                  />
                  <p className="text-xs text-dark-muted mt-2">Harmonic mean of precision and recall (overall balance)</p>
                </div>
                
                <div className="bg-dark-bg/50 rounded-xl p-4 border border-dark-border/50">
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-2">
                      <BarChart3 className="w-4 h-4 text-purple-500" />
                      <span className="text-sm font-medium">CV Score</span>
                    </div>
                    <span className={clsx('text-lg font-bold', getMetricTextColor(training.metrics.cv_accuracy))}>
                      {training.metrics.cv_accuracy.toFixed(1)}%
                    </span>
                  </div>
                  <ProgressBar 
                    value={training.metrics.cv_accuracy} 
                    showLabel={false}
                    color={getScoreColor(training.metrics.cv_accuracy)}
                    height="h-3"
                  />
                  <div className="text-xs text-dark-muted mt-1">±{training.metrics.cv_std.toFixed(1)} std</div>
                  <p className="text-xs text-dark-muted mt-2">Cross-validation accuracy (tested on unseen data)</p>
                </div>
              </div>

              {/* Training Stats */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
                <div className="flex items-center gap-2 group/stat relative">
                  <Package className="w-4 h-4 text-dark-muted" />
                  <span className="text-dark-muted">Samples:</span>
                  <span className="font-bold">{training.training_samples.toLocaleString()}</span>
                  <div className="absolute bottom-full left-0 mb-1 px-2 py-1 bg-dark-card border border-dark-border rounded text-xs text-dark-muted whitespace-nowrap opacity-0 group-hover/stat:opacity-100 transition-opacity pointer-events-none z-10">
                    Number of data points used for training
                  </div>
                </div>
                <div className="flex items-center gap-2 group/stat relative">
                  <Activity className="w-4 h-4 text-dark-muted" />
                  <span className="text-dark-muted">Features:</span>
                  <span className="font-bold">{training.features_count}</span>
                  <div className="absolute bottom-full left-0 mb-1 px-2 py-1 bg-dark-card border border-dark-border rounded text-xs text-dark-muted whitespace-nowrap opacity-0 group-hover/stat:opacity-100 transition-opacity pointer-events-none z-10">
                    Technical indicators used by the model
                  </div>
                </div>
                <div className="flex items-center gap-2 group/stat relative">
                  <Clock className="w-4 h-4 text-dark-muted" />
                  <span className="text-dark-muted">Period:</span>
                  <span className="font-bold">
                    {format(new Date(training.date_range.start), 'MMM dd')} - {format(new Date(training.date_range.end), 'MMM dd')}
                  </span>
                  <div className="absolute bottom-full left-0 mb-1 px-2 py-1 bg-dark-card border border-dark-border rounded text-xs text-dark-muted whitespace-nowrap opacity-0 group-hover/stat:opacity-100 transition-opacity pointer-events-none z-10">
                    Date range of training data
                  </div>
                </div>
                <div className="flex items-center gap-2 group/stat relative">
                  <TrendingUp className="w-4 h-4 text-dark-muted" />
                  <span className="text-dark-muted">Price:</span>
                  <span className="font-bold">
                    ${training.price_range.min.toLocaleString()} - ${training.price_range.max.toLocaleString()}
                  </span>
                  <div className="absolute bottom-full left-0 mb-1 px-2 py-1 bg-dark-card border border-dark-border rounded text-xs text-dark-muted whitespace-nowrap opacity-0 group-hover/stat:opacity-100 transition-opacity pointer-events-none z-10">
                    Price range seen during training
                  </div>
                </div>
              </div>
            </div>

            {/* Expanded Details */}
            {expandedRow === idx && (
              <div className="mt-6 pt-6 border-t border-dark-border animate-fadeIn">
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  {/* Target Distribution */}
                  <div className="bg-dark-bg/30 rounded-xl p-5 border border-dark-border/50">
                    <h4 className="font-bold mb-4 flex items-center gap-2">
                      <div className="p-1.5 bg-warning/10 rounded-lg">
                        <Zap className="w-4 h-4 text-warning" />
                      </div>
                      Target Distribution
                      <span className="text-xs font-normal text-dark-muted ml-2">Training data class balance</span>
                    </h4>
                    <div className="space-y-3">
                      {Object.entries(training.target_distribution || {}).map(([key, value]: [string, any]) => {
                        // Normalize keys: 0, '0', 0.0 all mean Down; 1, '1', 1.0 all mean Up
                        const keyStr = String(key)
                        const isUp = keyStr === '1' || keyStr === '1.0'
                        const isDown = keyStr === '0' || keyStr === '0.0' || keyStr === '-1'
                        
                        const label = isUp ? 'Up' : isDown ? 'Down' : `Class ${key}`
                        const percentage = (value / training.training_samples) * 100
                        
                        return (
                          <div key={key} className="flex items-center gap-3">
                            <span className={clsx(
                              "text-sm font-medium w-16",
                              isUp ? "text-success" : isDown ? "text-error" : ""
                            )}>
                              {label}
                            </span>
                            <div className="flex-1">
                              <ProgressBar 
                                value={percentage} 
                                showLabel={false}
                                color={isUp ? 'bg-success' : 'bg-error'}
                                height="h-6"
                              />
                            </div>
                            <span className="text-sm font-mono w-20 text-right">
                              {value.toLocaleString()} ({percentage.toFixed(1)}%)
                            </span>
                          </div>
                        )
                      })}
                    </div>
                  </div>

                  {/* Top Features */}
                  <div className="bg-dark-bg/30 rounded-xl p-5 border border-dark-border/50">
                    <h4 className="font-bold mb-4 flex items-center gap-2">
                      <div className="p-1.5 bg-info/10 rounded-lg">
                        <BarChart3 className="w-4 h-4 text-info" />
                      </div>
                      Top Features by Importance
                      <span className="text-xs font-normal text-dark-muted ml-2">Most influential indicators</span>
                    </h4>
                    <div className="space-y-3">
                      {Array.isArray(training.top_features) ? (
                        training.top_features.slice(0, 5).map((item: any, idx: number) => (
                          <div key={idx} className="flex items-center gap-3">
                            <span className="text-xs font-bold text-info w-6">#{idx + 1}</span>
                            <span className="text-sm font-mono flex-1 truncate">{item.feature}</span>
                            <div className="w-32">
                              <ProgressBar 
                                value={item.importance * 100} 
                                showLabel={false}
                                color="bg-gradient-to-r from-info to-purple-500"
                                height="h-6"
                              />
                            </div>
                            <span className="text-xs font-bold w-12 text-right">
                              {(item.importance * 100).toFixed(1)}%
                            </span>
                          </div>
                        ))
                      ) : (
                        <div className="text-sm text-dark-muted">No feature importance data</div>
                      )}
                    </div>
                  </div>
                </div>

                {/* Model Configuration */}
                <div className="mt-6 bg-dark-bg/30 rounded-xl p-5 border border-dark-border/50">
                  <h4 className="font-bold mb-3 flex items-center gap-2">
                    <div className="p-1.5 bg-purple-500/10 rounded-lg">
                      <Brain className="w-4 h-4 text-purple-500" />
                    </div>
                    Model Configuration
                  </h4>
                  <div className="bg-dark-bg rounded-lg p-4 border border-dark-border/50">
                    <pre className="text-xs text-dark-muted overflow-x-auto font-mono">
                      {JSON.stringify(training.model_config, null, 2)}
                    </pre>
                  </div>
                </div>
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Pagination and Summary Footer */}
      <div className="card mt-6 bg-gradient-to-r from-dark-card to-dark-border/20">
        {data?.pagination && (
          <div className="mb-4">
            <Pagination
              page={currentPage}
              totalPages={data.pagination.total_pages}
              onPageChange={(page) => {
                setCurrentPage(page)
                setExpandedRow(null)
              }}
              perPage={perPage}
              total={data.pagination.total}
            />
          </div>
        )}
        <div className="flex justify-between items-center">
          <div className="flex items-center gap-3">
            <Activity className="w-5 h-5 text-info" />
            <span className="text-sm text-dark-muted">
              Total training sessions: <span className="font-bold text-white">{data?.pagination?.total || 0}</span>
            </span>
          </div>
          <div className="text-xs text-dark-muted">
            Last updated: {format(new Date(), 'MMM dd, yyyy HH:mm')}
          </div>
        </div>
      </div>
    </div>
  )
}