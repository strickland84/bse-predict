export interface SystemStatus {
  status: 'healthy' | 'warning' | 'error'
  database: boolean
  services: Record<string, string>
  last_data_fetch: string | null
  last_prediction: string | null
  last_model_training: string | null
}

export interface PredictionData {
  symbol: string
  timestamp: string
  target_1pct: number
  target_2pct: number
  target_5pct: number
  confidence_1pct: number
  confidence_2pct: number
  confidence_5pct: number
  current_price: number
}

export interface ModelPerformance {
  symbol: string
  target: string
  accuracy: number
  precision: number
  recall: number
  f1_score: number
  last_trained: string
  training_samples: number
}

export interface DataHealth {
  symbol: string
  total_candles: number
  latest_candle: string
  oldest_candle: string
  gaps_detected: number
  coverage_percentage: number
  status: 'healthy' | 'gaps_detected' | 'critical'
}