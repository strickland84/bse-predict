import { useQuery } from '@tanstack/react-query'
import { fetchSystemStatus } from '../api'
import { format } from 'date-fns'
import { Server } from 'lucide-react'

export const SystemStatus = () => {
  const { data, isLoading, error } = useQuery({
    queryKey: ['systemStatus'],
    queryFn: fetchSystemStatus,
  })

  if (isLoading) return <div className="card">Loading system status...</div>
  if (error) return <div className="card text-error">Failed to load system status</div>

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy': return 'bg-success'
      case 'warning': return 'bg-warning'
      case 'error': return 'bg-error'
      default: return 'bg-gray-500'
    }
  }

  const getServiceStatusColor = (status: string) => {
    return status === 'running' ? 'bg-success' : 'bg-error'
  }

  return (
    <div className="card">
      <div className="flex items-center gap-3 mb-4">
        <div className="p-2 bg-warning/10 rounded-lg">
          <Server className="w-5 h-5 text-warning" />
        </div>
        <div>
          <h2 className="text-xl font-bold">System Status</h2>
          <p className="text-xs text-dark-muted">Service health monitoring</p>
        </div>
      </div>
      
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <span className="text-dark-muted">Overall Status</span>
          <div className="flex items-center gap-2">
            <span className={`status-dot ${getStatusColor(data?.status || '')}`} />
            <span className="capitalize">{data?.status}</span>
          </div>
        </div>

        <div className="flex items-center justify-between">
          <span className="text-dark-muted">Database</span>
          <div className="flex items-center gap-2">
            <span className={`status-dot ${data?.database ? 'bg-success' : 'bg-error'}`} />
            <span>{data?.database ? 'Connected' : 'Disconnected'}</span>
          </div>
        </div>

        <div className="border-t border-dark-border pt-3">
          <h3 className="text-sm font-semibold mb-2">Services</h3>
          <div className="space-y-1">
            {Object.entries(data?.services || {}).map(([name, status]) => (
              <div key={name} className="flex items-center justify-between text-sm">
                <span className="text-dark-muted truncate">{name}</span>
                <div className="flex items-center gap-2">
                  <span className={`status-dot ${getServiceStatusColor(status)}`} />
                  <span className="capitalize">{status}</span>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="border-t border-dark-border pt-3 text-sm">
          <div className="space-y-1">
            {data?.last_data_fetch && (
              <div className="flex justify-between">
                <span className="text-dark-muted">Last Data Fetch</span>
                <span>{format(new Date(data.last_data_fetch), 'HH:mm:ss')}</span>
              </div>
            )}
            {data?.last_prediction && (
              <div className="flex justify-between">
                <span className="text-dark-muted">Last Prediction</span>
                <span>{format(new Date(data.last_prediction), 'HH:mm:ss')}</span>
              </div>
            )}
            {data?.last_model_training && (
              <div className="flex justify-between">
                <span className="text-dark-muted">Last Training</span>
                <span>{format(new Date(data.last_model_training), 'MMM dd HH:mm')}</span>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}