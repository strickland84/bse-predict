import { useQuery } from '@tanstack/react-query'
import { api } from '../api'
import { Activity } from 'lucide-react'
import clsx from 'clsx'

export const PredictionStats = () => {
  const { data, isLoading } = useQuery({
    queryKey: ['predictions-statistics', 7],
    queryFn: async () => {
      const { data } = await api.get('/api/predictions/statistics?days=7')
      return data
    },
    refetchInterval: 60000, // Refresh every minute
  })

  if (isLoading || !data) {
    return (
      <div className="card">
        <div className="animate-pulse">
          <div className="h-6 bg-dark-border rounded w-1/3 mb-4"></div>
          <div className="space-y-3">
            <div className="h-4 bg-dark-border rounded"></div>
            <div className="h-4 bg-dark-border rounded"></div>
          </div>
        </div>
      </div>
    )
  }

  const stats = data.overall

  return (
    <div className="card">
      <div className="flex justify-between items-center mb-4">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-info/10 rounded-lg">
            <Activity className="w-5 h-5 text-info" />
          </div>
          <div>
            <h2 className="text-xl font-bold">Performance Stats</h2>
            <p className="text-xs text-dark-muted">Last 7 days</p>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
        <div className="bg-dark-bg/50 rounded-lg p-3 border border-dark-border/50">
          <div className="text-2xl font-bold text-success">
            {stats.overall_hit_rate?.toFixed(1) || '0'}%
          </div>
          <div className="text-xs text-dark-muted">Hit Rate</div>
        </div>
        
        <div className="bg-dark-bg/50 rounded-lg p-3 border border-dark-border/50">
          <div className="flex items-center gap-1">
            <span className="text-lg font-bold text-success">{stats.wins || 0}</span>
            <span className="text-dark-muted">/</span>
            <span className="text-lg font-bold text-error">{stats.losses || 0}</span>
          </div>
          <div className="text-xs text-dark-muted">Wins/Losses</div>
        </div>

        <div className="bg-dark-bg/50 rounded-lg p-3 border border-dark-border/50">
          <div className="text-2xl font-bold text-info">
            {stats.timing?.avg_time_to_win_hours?.toFixed(1) || '0'}h
          </div>
          <div className="text-xs text-dark-muted">Avg Win Time</div>
        </div>

        <div className="bg-dark-bg/50 rounded-lg p-3 border border-dark-border/50">
          <div className="flex items-center gap-1">
            <span className="text-sm font-bold text-success">+{stats.excursions?.avg_mfe?.toFixed(2) || '0'}%</span>
            <span className="text-sm font-bold text-error">{stats.excursions?.avg_mae?.toFixed(2) || '0'}%</span>
          </div>
          <div className="text-xs text-dark-muted">MFE/MAE</div>
        </div>
      </div>

      {/* Confidence Cohort Analysis */}
      {data.confidence_cohorts && data.confidence_cohorts.length > 0 && (
        <div className="mt-4 pt-4 border-t border-dark-border">
          <h3 className="text-sm font-bold mb-3 text-dark-muted">Confidence Analysis</h3>
          <div className="space-y-2">
            {data.confidence_cohorts.map((cohort: any) => (
              <div key={cohort.confidence_range} className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <div className={clsx(
                    'w-2 h-2 rounded-full',
                    cohort.confidence_range === '90-100%' ? 'bg-success' :
                    cohort.confidence_range === '80-90%' ? 'bg-info' :
                    cohort.accuracy >= 70 ? 'bg-success' : 
                    cohort.accuracy >= 50 ? 'bg-warning' : 'bg-error'
                  )} />
                  <span className="text-xs text-dark-muted">{cohort.confidence_range}</span>
                </div>
                <div className="flex items-center gap-3">
                  <div className="w-24 bg-dark-border rounded-full h-1.5">
                    <div 
                      className={clsx(
                        'h-full rounded-full transition-all',
                        cohort.confidence_range === '90-100%' ? 'bg-success' :
                        cohort.confidence_range === '80-90%' ? 'bg-info' :
                        cohort.accuracy >= 70 ? 'bg-success' : 
                        cohort.accuracy >= 50 ? 'bg-warning' : 'bg-error'
                      )}
                      style={{ width: `${cohort.accuracy}%` }}
                    />
                  </div>
                  <span className={clsx(
                    'text-xs font-bold w-12 text-right',
                    cohort.confidence_range === '90-100%' ? 'text-success' :
                    cohort.confidence_range === '80-90%' ? 'text-info' :
                    cohort.accuracy >= 70 ? 'text-success' : 
                    cohort.accuracy >= 50 ? 'text-warning' : 'text-error'
                  )}>
                    {cohort.accuracy?.toFixed(1)}%
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Model Breakdown Summary */}
      {data.by_model && data.by_model.length > 0 && (
        <div className="mt-4 pt-4 border-t border-dark-border">
          <h3 className="text-sm font-bold mb-3 text-dark-muted">By Target</h3>
          <div className="grid grid-cols-3 gap-2">
            {['1.0%', '2.0%', '5.0%'].map(target => {
              const models = data.by_model.filter((m: any) => m.target_pct === target)
              const avgHitRate = models.reduce((acc: number, m: any) => acc + (m.hit_rate || 0), 0) / (models.length || 1)
              
              return (
                <div key={target} className="bg-dark-bg/50 rounded-lg p-2 border border-dark-border/50 text-center">
                  <div className="text-xs text-dark-muted mb-1">±{target}</div>
                  <div className={clsx(
                    'text-lg font-bold',
                    avgHitRate >= 60 ? 'text-success' : 
                    avgHitRate >= 40 ? 'text-warning' : 'text-error'
                  )}>
                    {avgHitRate.toFixed(1)}%
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      )}
    </div>
  )
}