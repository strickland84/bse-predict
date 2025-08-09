import axios from 'axios'
import { SystemStatus, PredictionData, ModelPerformance, DataHealth } from './types'

const API_BASE = import.meta.env.VITE_API_URL || ''

export const api = axios.create({
  baseURL: API_BASE,
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
  withCredentials: false, // Explicitly set to avoid CORS issues
})

export const fetchSystemStatus = async (): Promise<SystemStatus> => {
  const { data } = await api.get('/api/system/status')
  return data
}

export const fetchLatestPredictions = async (): Promise<PredictionData[]> => {
  const { data } = await api.get('/api/predictions/latest')
  return data
}

export const fetchPredictionHistory = async (hours: number = 24) => {
  const { data } = await api.get(`/api/predictions/history?hours=${hours}`)
  return data
}

export const fetchPredictionAccuracy = async (hours: number = 24) => {
  const { data } = await api.get(`/api/predictions/accuracy?hours=${hours}`)
  return data
}

export const fetchModelPerformance = async (): Promise<ModelPerformance[]> => {
  const { data } = await api.get('/api/models/performance')
  return data
}

export const fetchDataHealth = async (): Promise<DataHealth[]> => {
  const { data } = await api.get('/api/data/health')
  return data
}

export const fetchLatestCandles = async (symbol?: string, limit: number = 100) => {
  const params = new URLSearchParams()
  if (symbol) params.append('symbol', symbol)
  params.append('limit', limit.toString())
  
  const { data } = await api.get(`/api/data/candles/latest?${params}`)
  return data
}