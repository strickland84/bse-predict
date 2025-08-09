import { useEffect, useState } from 'react'
import { format } from 'date-fns'
import clsx from 'clsx'

interface Alert {
  id: string
  type: 'prediction' | 'system' | 'data'
  level: 'info' | 'warning' | 'error' | 'success'
  message: string
  timestamp: Date
}

export const RecentAlerts = ({ wsMessage }: { wsMessage: any }) => {
  const [alerts, setAlerts] = useState<Alert[]>([])

  useEffect(() => {
    if (wsMessage) {
      const newAlert: Alert = {
        id: Date.now().toString(),
        type: wsMessage.type === 'prediction_update' ? 'prediction' : 
              wsMessage.type === 'system_status' ? 'system' : 'data',
        level: 'info',
        message: `${wsMessage.type}: ${JSON.stringify(wsMessage.data).substring(0, 100)}...`,
        timestamp: new Date(wsMessage.timestamp),
      }
      
      setAlerts(prev => [newAlert, ...prev].slice(0, 10)) // Keep only last 10 alerts
    }
  }, [wsMessage])

  const getLevelColor = (level: string) => {
    switch (level) {
      case 'success': return 'text-success border-success'
      case 'warning': return 'text-warning border-warning'
      case 'error': return 'text-error border-error'
      default: return 'text-info border-info'
    }
  }

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'prediction': return '📊'
      case 'system': return '⚙️'
      case 'data': return '💾'
      default: return '📌'
    }
  }

  return (
    <div className="card">
      <h2 className="text-xl font-bold mb-4">Recent Alerts</h2>
      
      <div className="space-y-2 max-h-96 overflow-y-auto">
        {alerts.length === 0 ? (
          <div className="text-dark-muted text-center py-4">No recent alerts</div>
        ) : (
          alerts.map((alert) => (
            <div
              key={alert.id}
              className={clsx(
                'border-l-2 pl-3 py-2',
                getLevelColor(alert.level)
              )}
            >
              <div className="flex items-start gap-2">
                <span className="text-lg">{getTypeIcon(alert.type)}</span>
                <div className="flex-1">
                  <p className="text-sm">{alert.message}</p>
                  <span className="text-xs text-dark-muted">
                    {format(alert.timestamp, 'HH:mm:ss')}
                  </span>
                </div>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  )
}