import { useEffect, useRef, useState } from 'react'

interface WebSocketMessage {
  type: string
  data: any
  timestamp: string
}

export const useWebSocket = () => {
  const [isConnected, setIsConnected] = useState(false)
  const [lastMessage, setLastMessage] = useState<WebSocketMessage | null>(null)
  const ws = useRef<WebSocket | null>(null)
  
  useEffect(() => {
    // Use WebSocket through nginx proxy
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const host = window.location.host
    // If VITE_WS_URL is set and not empty, use it; otherwise use relative path
    const wsUrl = import.meta.env.VITE_WS_URL && import.meta.env.VITE_WS_URL !== '' 
      ? import.meta.env.VITE_WS_URL 
      : `${protocol}//${host}/ws`
    
    const connect = () => {
      try {
        ws.current = new WebSocket(wsUrl)
      } catch (error) {
        console.error('Failed to create WebSocket:', error)
        setIsConnected(false)
        return
      }
      
      ws.current.onopen = () => {
        setIsConnected(true)
        console.log('WebSocket connected')
      }
      
      ws.current.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data)
          setLastMessage(message)
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error)
        }
      }
      
      ws.current.onclose = () => {
        setIsConnected(false)
        console.log('WebSocket disconnected, reconnecting...')
        setTimeout(connect, 5000)
      }
      
      ws.current.onerror = (error) => {
        console.error('WebSocket error:', error)
      }
    }
    
    connect()
    
    // Ping interval to keep connection alive
    const pingInterval = setInterval(() => {
      if (ws.current?.readyState === WebSocket.OPEN) {
        ws.current.send('ping')
      }
    }, 30000)
    
    return () => {
      clearInterval(pingInterval)
      ws.current?.close()
    }
  }, [])
  
  return { isConnected, lastMessage }
}