import { Filter, X } from 'lucide-react'
import clsx from 'clsx'

interface FilterOption {
  label: string
  value: string | number
}

interface FilterConfig {
  id: string
  label: string
  type: 'select' | 'range' | 'multiselect'
  options?: FilterOption[]
  min?: number
  max?: number
  step?: number
  suffix?: string
}

interface FilterPanelProps {
  filters: FilterConfig[]
  values: Record<string, any>
  onChange: (id: string, value: any) => void
  onReset: () => void
  className?: string
}

export function FilterPanel({ filters, values, onChange, onReset, className }: FilterPanelProps) {
  const hasActiveFilters = Object.values(values).some(v => v !== null && v !== undefined && v !== '')
  
  return (
    <div className={clsx('card', className)}>
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2 text-sm font-medium">
          <Filter className="w-4 h-4" />
          Filters
          {hasActiveFilters && (
            <span className="px-2 py-0.5 bg-info/20 text-info rounded-full text-xs">
              {Object.values(values).filter(v => v !== null && v !== undefined && v !== '').length}
            </span>
          )}
        </div>
        
        {hasActiveFilters && (
          <button
            onClick={onReset}
            className="flex items-center gap-1 text-xs text-dark-muted hover:text-error transition-colors"
          >
            <X className="w-3 h-3" />
            Clear filters
          </button>
        )}
      </div>
      
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
        {filters.map((filter) => (
            <div key={filter.id} className="space-y-1">
              <label className="text-xs font-medium text-dark-muted">
                {filter.label}
              </label>
              
              {filter.type === 'select' && (
                <select
                  value={values[filter.id] || ''}
                  onChange={(e) => onChange(filter.id, e.target.value || null)}
                  className="w-full px-3 py-1.5 bg-dark-bg border border-dark-border rounded-lg text-sm focus:border-info focus:outline-none transition-colors"
                >
                  <option value="">All</option>
                  {filter.options?.map((option) => (
                    <option key={option.value} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </select>
              )}
              
              {filter.type === 'range' && (
                <div className="flex items-center gap-2">
                  <input
                    type="range"
                    min={filter.min || 0}
                    max={filter.max || 100}
                    step={filter.step || 1}
                    value={values[filter.id] || filter.min || 0}
                    onChange={(e) => onChange(filter.id, Number(e.target.value) || null)}
                    className="flex-1 accent-info"
                  />
                  <span className="text-sm font-mono w-12 text-right">
                    {values[filter.id] || filter.min || 0}{filter.suffix || ''}
                  </span>
                </div>
              )}
              
              {filter.type === 'multiselect' && (
                <div className="flex flex-wrap gap-1">
                  {filter.options?.map((option) => {
                    const selected = Array.isArray(values[filter.id]) && 
                                   values[filter.id].includes(option.value)
                    return (
                      <button
                        key={option.value}
                        onClick={() => {
                          const current = Array.isArray(values[filter.id]) ? values[filter.id] : []
                          const updated = selected
                            ? current.filter((v: any) => v !== option.value)
                            : [...current, option.value]
                          onChange(filter.id, updated.length > 0 ? updated : null)
                        }}
                        className={clsx(
                          'px-2 py-0.5 rounded text-xs transition-all',
                          selected
                            ? 'bg-info/20 text-info border border-info/30'
                            : 'bg-dark-bg border border-dark-border hover:border-info/50'
                        )}
                      >
                        {option.label}
                      </button>
                    )
                  })}
                </div>
              )}
            </div>
          ))}
      </div>
    </div>
  )
}

interface SortControlProps {
  sortBy: string
  sortOrder: 'asc' | 'desc'
  options: FilterOption[]
  onSortChange: (field: string, order: 'asc' | 'desc') => void
  className?: string
}

export function SortControl({ sortBy, sortOrder, options, onSortChange, className }: SortControlProps) {
  const getSortLabel = () => {
    if (sortBy === 'timestamp' || sortBy === 'trained_at') {
      return sortOrder === 'desc' ? '↓ Newest' : '↑ Oldest'
    } else {
      return sortOrder === 'desc' ? '↓ Highest' : '↑ Lowest'
    }
  }
  
  return (
    <div className={clsx('flex items-center gap-2', className)}>
      <span className="text-xs font-medium text-dark-muted">Sort by:</span>
      <select
        value={sortBy}
        onChange={(e) => onSortChange(e.target.value, sortOrder)}
        className="px-3 py-1 bg-dark-bg border border-dark-border rounded-lg text-sm focus:border-info focus:outline-none transition-colors"
      >
        {options.map((option) => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
      <button
        onClick={() => onSortChange(sortBy, sortOrder === 'asc' ? 'desc' : 'asc')}
        className="px-3 py-1 bg-dark-bg border border-dark-border rounded-lg text-sm hover:border-info focus:border-info focus:outline-none transition-colors"
      >
        {getSortLabel()}
      </button>
    </div>
  )
}