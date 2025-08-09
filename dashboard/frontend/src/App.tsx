import { BrowserRouter as Router, Routes, Route, Link, useLocation } from 'react-router-dom'
import { useWebSocket } from './hooks/useWebSocket'
import { Dashboard } from './pages/Dashboard'
import { PredictionsDetail } from './pages/PredictionsDetail'
import { ModelsDetail } from './pages/ModelsDetail'
import { DataIntegrityDetail } from './pages/DataIntegrityDetail'
import { Charts } from './pages/Charts'
import { format } from 'date-fns'
import clsx from 'clsx'
import { Home, TrendingUp, Brain, Database, LineChart, Menu, X, Send, Github } from 'lucide-react'
import { useState } from 'react'

function Navigation() {
  const location = useLocation()
  const { isConnected } = useWebSocket()
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false)
  
  const navItems = [
    { path: '/', label: 'Dashboard', icon: Home },
    { path: '/predictions', label: 'Predictions', icon: TrendingUp },
    { path: '/charts', label: 'Charts', icon: LineChart },
    { path: '/models', label: 'Models', icon: Brain },
    { path: '/data', label: 'Data Integrity', icon: Database },
  ]

  return (
    <header className={clsx(
      "border-b border-dark-border bg-gradient-to-r from-dark-card/90 to-dark-bg/90 backdrop-blur-lg shadow-glow relative",
      isMobileMenuOpen && "z-[9998]"
    )}>
      <div className="container mx-auto px-4">
        <div className="flex justify-between items-center py-3">
          <div className="flex items-center gap-2 md:gap-4 lg:gap-6">
            <Link to="/" className="text-xl md:text-2xl font-bold flex items-center gap-2 group">
              <img 
                src="/logo.png" 
                alt="BSE Predict" 
                className="w-7 h-7 md:w-9 md:h-9 object-contain"
              />
              <span className="bg-gradient-to-r from-accent to-blue-glow bg-clip-text text-transparent">
                BSE Predict
              </span>
            </Link>
            
            <nav className="hidden md:flex gap-1">
              {navItems.map(({ path, label, icon: Icon }) => (
                <Link
                  key={path}
                  to={path}
                  className={clsx(
                    'px-3 py-2 rounded-lg transition-all duration-300 flex items-center gap-1.5 relative overflow-hidden group text-sm',
                    location.pathname === path
                      ? 'bg-gradient-to-r from-blue-glow/20 to-accent/20 text-accent border border-accent/50 shadow-neon'
                      : 'hover:bg-dark-border/50 text-dark-muted hover:text-dark-text border border-transparent hover:border-dark-border'
                  )}
                >
                  <Icon className="w-4 h-4 relative z-10" />
                  <span className="relative z-10">{label}</span>
                  {location.pathname === path && (
                    <div className="absolute inset-0 bg-gradient-to-r from-blue-glow/10 to-accent/10 animate-pulse-slow" />
                  )}
                </Link>
              ))}
            </nav>
          </div>
          
          <div className="flex items-center gap-1.5 md:gap-2">
            <a
              href="https://github.com/strickland84/bse-predict"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-gradient-to-r from-gray-600/20 to-gray-700/20 border border-gray-500/30 hover:border-gray-400/50 transition-all group"
              title="View on GitHub"
            >
              <Github className="w-4 h-4 text-gray-400 group-hover:text-gray-300" />
              <span className="hidden lg:inline text-xs font-medium text-gray-400 group-hover:text-gray-300">GitHub</span>
            </a>
            <a
              href="https://t.me/+0sWqLhHXhgNjZDdk"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-gradient-to-r from-blue-500/20 to-blue-600/20 border border-blue-500/30 hover:border-blue-400/50 transition-all group"
              title="Join our Telegram channel"
            >
              <Send className="w-4 h-4 text-blue-400 group-hover:text-blue-300" />
              <span className="hidden lg:inline text-xs font-medium text-blue-400 group-hover:text-blue-300">Telegram</span>
            </a>
            <div className="hidden sm:flex items-center gap-1.5 px-3 py-1 rounded-full bg-dark-card/50 border border-dark-border">
              <span className={clsx(
                'status-dot animate-pulse',
                isConnected ? 'bg-success shadow-[0_0_10px_#00ff88]' : 'bg-error shadow-[0_0_10px_#ff3366]'
              )} />
              <span className="text-xs font-medium">
                {isConnected ? 'ONLINE' : 'OFFLINE'}
              </span>
            </div>
            <div className="hidden xl:block text-xs text-accent font-mono bg-dark-card/50 px-3 py-1 rounded border border-dark-border">
              {format(new Date(), 'MMM dd | HH:mm:ss')}
            </div>
            
            <button
              onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
              className="md:hidden p-2 rounded-lg hover:bg-dark-border/50 transition-colors"
              aria-label="Toggle menu"
            >
              {isMobileMenuOpen ? (
                <X className="w-6 h-6" />
              ) : (
                <Menu className="w-6 h-6" />
              )}
            </button>
          </div>
        </div>
        
        {isMobileMenuOpen && (
          <div className="md:hidden absolute top-full left-0 right-0 bg-dark-card/95 backdrop-blur-lg border-b border-dark-border z-[9999] shadow-xl">
            <nav className="flex flex-col p-4 gap-2">
              {navItems.map(({ path, label, icon: Icon }) => (
                <Link
                  key={path}
                  to={path}
                  onClick={() => setIsMobileMenuOpen(false)}
                  className={clsx(
                    'px-4 py-3 rounded-lg transition-all duration-300 flex items-center gap-3',
                    location.pathname === path
                      ? 'bg-gradient-to-r from-blue-glow/20 to-accent/20 text-accent border border-accent/50'
                      : 'hover:bg-dark-border/50 text-dark-muted hover:text-dark-text'
                  )}
                >
                  <Icon className="w-5 h-5" />
                  <span className="font-medium">{label}</span>
                </Link>
              ))}
              
              <div className="mt-2 pt-2 border-t border-dark-border">
                <a
                  href="https://github.com/strickland84/bse-predict"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center gap-3 px-4 py-3 rounded-lg bg-gradient-to-r from-gray-600/20 to-gray-700/20 border border-gray-500/30 hover:border-gray-400/50 transition-all"
                  onClick={() => setIsMobileMenuOpen(false)}
                >
                  <Github className="w-5 h-5 text-gray-400" />
                  <span className="font-medium text-gray-400">View on GitHub</span>
                </a>
                <a
                  href="https://t.me/+0sWqLhHXhgNjZDdk"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center gap-3 px-4 py-3 rounded-lg bg-gradient-to-r from-blue-500/20 to-blue-600/20 border border-blue-500/30 hover:border-blue-400/50 transition-all mt-2"
                  onClick={() => setIsMobileMenuOpen(false)}
                >
                  <Send className="w-5 h-5 text-blue-400" />
                  <span className="font-medium text-blue-400">Join Telegram Channel</span>
                </a>
              </div>
              
              <div className="sm:hidden mt-2 pt-2 border-t border-dark-border">
                <div className="flex items-center gap-2 px-4 py-2">
                  <span className={clsx(
                    'status-dot animate-pulse',
                    isConnected ? 'bg-success shadow-[0_0_10px_#00ff88]' : 'bg-error shadow-[0_0_10px_#ff3366]'
                  )} />
                  <span className="text-sm">
                    Connection: {isConnected ? 'ONLINE' : 'OFFLINE'}
                  </span>
                </div>
                <div className="lg:hidden px-4 py-2 text-sm text-accent font-mono">
                  {format(new Date(), 'MMM dd | HH:mm:ss')}
                </div>
              </div>
            </nav>
          </div>
        )}
      </div>
    </header>
  )
}

function App() {
  return (
    <Router>
      <div className="min-h-screen bg-dark-bg text-dark-text">
        <Navigation />
        
        <main>
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/predictions" element={<PredictionsDetail />} />
            <Route path="/charts" element={<Charts />} />
            <Route path="/models" element={<ModelsDetail />} />
            <Route path="/data" element={<DataIntegrityDetail />} />
          </Routes>
        </main>
      </div>
    </Router>
  )
}

export default App