import { useQuery } from '@tanstack/react-query'
import { fetchModelPerformance } from '../api'
import { format } from 'date-fns'
import { Link } from 'react-router-dom'
import { ArrowRight, TrendingUp, Activity, Target, Zap, CheckCircle, AlertCircle } from 'lucide-react'
import clsx from 'clsx'

// Crypto logos as simple SVG components
const CryptoLogos = {
  BTC: () => (
    <svg viewBox="0 0 32 32" className="w-8 h-8">
      <circle cx="16" cy="16" r="16" fill="#F7931A"/>
      <path fill="white" d="M23.189 14.02c.314-2.096-1.283-3.223-3.465-3.975l.708-2.84-1.728-.43-.69 2.765c-.454-.114-.92-.22-1.385-.326l.695-2.783L15.596 6l-.708 2.839c-.376-.086-.746-.17-1.104-.26l.002-.009-2.384-.595-.46 1.846s1.283.294 1.256.312c.7.175.826.638.805 1.006l-.806 3.235c.048.012.11.03.18.057l-.183-.045-1.13 4.532c-.086.212-.303.531-.793.41.018.025-1.256-.313-1.256-.313l-.858 1.978 2.25.561c.418.105.828.215 1.231.318l-.715 2.872 1.727.43.708-2.84c.472.127.93.245 1.378.357l-.706 2.828 1.728.43.715-2.866c2.948.558 5.164.333 6.097-2.333.752-2.146-.037-3.385-1.588-4.192 1.13-.26 1.98-1.003 2.207-2.538zm-3.95 5.538c-.533 2.147-4.148.986-5.32.695l.95-3.805c1.172.293 4.929.872 4.37 3.11zm.535-5.569c-.487 1.953-3.495.96-4.47.717l.86-3.45c.975.243 4.118.696 3.61 2.733z"/>
    </svg>
  ),
  ETH: () => (
    <svg viewBox="0 0 32 32" className="w-8 h-8">
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
    <svg viewBox="0 0 32 32" className="w-8 h-8">
      <circle cx="16" cy="16" r="16" fill="url(#solana-gradient)"/>
      <defs>
        <linearGradient id="solana-gradient" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor="#00FFA3"/>
          <stop offset="100%" stopColor="#DC1FFF"/>
        </linearGradient>
      </defs>
      <path fill="white" d="M9.5 20.5c.2-.2.5-.3.8-.3h11.4c.5 0 .7.6.4.9l-2.6 2.6c-.2.2-.5.3-.8.3H7.3c-.5 0-.7-.6-.4-.9l2.6-2.6zm0-9c.2-.2.5-.3.8-.3h11.4c.5 0 .7.6.4.9l-2.6 2.6c-.2.2-.5.3-.8.3H7.3c-.5 0-.7-.6-.4-.9l2.6-2.6zm13 4.5c-.2.2-.5.3-.8.3H10.3c-.5 0-.7-.6-.4-.9l2.6-2.6c.2-.2.5-.3.8-.3h11.4c.5 0 .7.6.4.9l-2.6 2.6z"/>
    </svg>
  )
}

// Progress bar component
const ProgressBar = ({ value, max = 100, className = '', showLabel = true, color = 'bg-info' }: {
  value: number
  max?: number
  className?: string
  showLabel?: boolean
  color?: string
}) => {
  const percentage = Math.min(100, Math.max(0, (value / max) * 100))
  
  return (
    <div className={clsx('relative', className)}>
      <div className="w-full bg-dark-border rounded-full h-2 overflow-hidden">
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
      'inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium border',
      config.color
    )}>
      <Icon className="w-3 h-3" />
      {config.label}
    </div>
  )
}

export const ModelPerformance = () => {
  const { data, isLoading, error } = useQuery({
    queryKey: ['modelPerformance'],
    queryFn: fetchModelPerformance,
  })

  if (isLoading) {
    return (
      <div className="card">
        <div className="flex items-center gap-3">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-info"></div>
          <span className="text-dark-muted">Loading model performance...</span>
        </div>
      </div>
    )
  }
  
  if (error) {
    return (
      <div className="card">
        <div className="flex items-center gap-3 text-error">
          <AlertCircle className="w-5 h-5" />
          <span>Failed to load model performance</span>
        </div>
      </div>
    )
  }

  const getScoreColor = (score: number) => {
    if (score >= 80) return 'bg-success'
    if (score >= 70) return 'bg-info'
    if (score >= 60) return 'bg-warning'
    return 'bg-error'
  }

  // Group by symbol and calculate averages
  const groupedData = data?.reduce((acc, item) => {
    if (!acc[item.symbol]) {
      acc[item.symbol] = {
        models: [],
        avgAccuracy: 0,
        avgPrecision: 0,
        avgRecall: 0,
        avgF1: 0
      }
    }
    acc[item.symbol].models.push(item)
    return acc
  }, {} as Record<string, { 
    models: typeof data, 
    avgAccuracy: number,
    avgPrecision: number,
    avgRecall: number,
    avgF1: number
  }>)

  // Calculate averages
  Object.values(groupedData || {}).forEach(group => {
    const len = group.models.length
    group.avgAccuracy = group.models.reduce((sum, m) => sum + m.accuracy, 0) / len
    group.avgPrecision = group.models.reduce((sum, m) => sum + m.precision, 0) / len
    group.avgRecall = group.models.reduce((sum, m) => sum + m.recall, 0) / len
    group.avgF1 = group.models.reduce((sum, m) => sum + m.f1_score, 0) / len
  })

  const getCryptoLogo = (symbol: string) => {
    const base = symbol.split('/')[0]
    const Logo = CryptoLogos[base as keyof typeof CryptoLogos]
    return Logo ? <Logo /> : null
  }

  return (
    <div className="card">
      <div className="flex justify-between items-center mb-6">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-info/10 rounded-lg">
            <Target className="w-5 h-5 text-info" />
          </div>
          <div>
            <h2 className="text-xl font-bold">Model Performance</h2>
            <p className="text-xs text-dark-muted">Real-time accuracy metrics</p>
          </div>
        </div>
        <Link 
          to="/models" 
          className="text-sm text-info hover:text-info/80 flex items-center gap-1 transition-colors"
        >
          View Details <ArrowRight className="w-4 h-4" />
        </Link>
      </div>
      
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {Object.entries(groupedData || {}).map(([symbol, group]) => (
          <div key={symbol} className="bg-gradient-to-br from-dark-bg via-dark-bg to-dark-border/20 border border-dark-border rounded-xl p-5 hover:border-dark-border/60 transition-all">
            {/* Header with logo and symbol */}
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-3">
                {getCryptoLogo(symbol)}
                <div>
                  <h3 className="text-lg font-bold">{symbol.replace('/USDT', '')}</h3>
                  <div className="text-xs text-dark-muted">
                    {group.models.reduce((sum, m) => sum + m.training_samples, 0).toLocaleString()} samples
                  </div>
                </div>
              </div>
              <PerformanceBadge score={group.avgAccuracy} />
            </div>
            
            {/* Overall performance meter */}
            <div className="mb-4 p-3 bg-dark-card/50 rounded-lg border border-dark-border/50">
              <div className="flex justify-between items-center mb-2">
                <span className="text-sm font-medium text-dark-muted">Overall Accuracy</span>
                <span className="text-lg font-bold">{group.avgAccuracy.toFixed(1)}%</span>
              </div>
              <ProgressBar 
                value={group.avgAccuracy} 
                showLabel={false}
                color={getScoreColor(group.avgAccuracy)}
              />
              <p className="text-xs text-dark-muted mt-2">Average correctness across all ±1%, ±2%, ±5% predictions</p>
            </div>
            
            {/* Target breakdowns */}
            <div className="space-y-3">
              {group.models.map((model) => (
                <div key={model.target} className="group">
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-2">
                      <div className={clsx(
                        'w-2 h-2 rounded-full',
                        model.accuracy >= 70 ? 'bg-success animate-pulse' : 'bg-warning'
                      )} />
                      <span className="text-sm font-medium">{model.target} Target</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <Zap className={clsx(
                        'w-3 h-3',
                        model.accuracy >= 70 ? 'text-success' : 'text-warning'
                      )} />
                      <span className="text-sm font-bold">{model.accuracy}%</span>
                    </div>
                  </div>
                  
                  {/* Metrics grid with mini progress bars */}
                  <div className="grid grid-cols-3 gap-2">
                    <div className="text-center group/metric relative">
                      <div className="text-xs text-dark-muted mb-1">Precision</div>
                      <ProgressBar 
                        value={model.precision} 
                        showLabel={false}
                        color={getScoreColor(model.precision)}
                        className="mb-1"
                      />
                      <div className="text-xs font-medium">{model.precision}%</div>
                      <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-2 py-1 bg-dark-card border border-dark-border rounded text-xs text-dark-muted whitespace-nowrap opacity-0 group-hover/metric:opacity-100 transition-opacity pointer-events-none z-10">
                        When predicts UP/DOWN, how often correct
                      </div>
                    </div>
                    <div className="text-center group/metric relative">
                      <div className="text-xs text-dark-muted mb-1">Recall</div>
                      <ProgressBar 
                        value={model.recall} 
                        showLabel={false}
                        color={getScoreColor(model.recall)}
                        className="mb-1"
                      />
                      <div className="text-xs font-medium">{model.recall}%</div>
                      <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-2 py-1 bg-dark-card border border-dark-border rounded text-xs text-dark-muted whitespace-nowrap opacity-0 group-hover/metric:opacity-100 transition-opacity pointer-events-none z-10">
                        % of actual moves correctly identified
                      </div>
                    </div>
                    <div className="text-center group/metric relative">
                      <div className="text-xs text-dark-muted mb-1">F1</div>
                      <ProgressBar 
                        value={model.f1_score} 
                        showLabel={false}
                        color={getScoreColor(model.f1_score)}
                        className="mb-1"
                      />
                      <div className="text-xs font-medium">{model.f1_score}%</div>
                      <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-2 py-1 bg-dark-card border border-dark-border rounded text-xs text-dark-muted whitespace-nowrap opacity-0 group-hover/metric:opacity-100 transition-opacity pointer-events-none z-10">
                        Balance of precision and recall
                      </div>
                    </div>
                  </div>
                  
                  {/* Last trained info */}
                  <div className="mt-2 flex items-center justify-between text-xs text-dark-muted">
                    <span>Last trained</span>
                    <span>{format(new Date(model.last_trained), 'MMM dd, HH:mm')}</span>
                  </div>
                </div>
              ))}
            </div>
            
            {/* Summary stats */}
            <div className="mt-4 pt-4 border-t border-dark-border/50 grid grid-cols-2 gap-2 text-xs">
              <div className="flex justify-between">
                <span className="text-dark-muted">Avg Precision</span>
                <span className="font-medium">{group.avgPrecision.toFixed(1)}%</span>
              </div>
              <div className="flex justify-between">
                <span className="text-dark-muted">Avg F1</span>
                <span className="font-medium">{group.avgF1.toFixed(1)}%</span>
              </div>
            </div>
          </div>
        ))}
      </div>
      
      {/* Footer with legend */}
      <div className="mt-6 pt-4 border-t border-dark-border flex items-center justify-between text-xs text-dark-muted">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded bg-success" />
            <span>≥80% Excellent</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded bg-info" />
            <span>≥70% Good</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded bg-warning" />
            <span>≥60% Fair</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded bg-error" />
            <span>&lt;60% Needs Work</span>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <Activity className="w-3 h-3" />
          <span>Auto-refreshes every 5 minutes</span>
        </div>
      </div>
    </div>
  )
}