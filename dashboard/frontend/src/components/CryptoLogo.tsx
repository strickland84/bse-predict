interface CryptoLogoProps {
  symbol: string
  size?: 'sm' | 'md' | 'lg'
}

export const CryptoLogo = ({ symbol, size = 'md' }: CryptoLogoProps) => {
  const sizeClasses = {
    sm: 'w-6 h-6',
    md: 'w-8 h-8',
    lg: 'w-12 h-12'
  }

  const getLogoUrl = (sym: string) => {
    const cleanSymbol = sym.replace('/USDT', '').toLowerCase()
    // Using CoinGecko's public CDN for crypto logos
    switch (cleanSymbol) {
      case 'btc':
        return 'https://assets.coingecko.com/coins/images/1/small/bitcoin.png'
      case 'eth':
        return 'https://assets.coingecko.com/coins/images/279/small/ethereum.png'
      case 'sol':
        return 'https://assets.coingecko.com/coins/images/4128/small/solana.png'
      default:
        return null
    }
  }

  const getGradientColors = (sym: string) => {
    const cleanSymbol = sym.replace('/USDT', '').toLowerCase()
    switch (cleanSymbol) {
      case 'btc':
        return 'from-orange-400 to-yellow-500'
      case 'eth':
        return 'from-blue-400 to-purple-500'
      case 'sol':
        return 'from-purple-400 to-pink-500'
      default:
        return 'from-blue-glow to-accent'
    }
  }

  const logoUrl = getLogoUrl(symbol)
  const text = symbol.replace('/USDT', '')

  if (!logoUrl) {
    // Futuristic fallback with gradient and glow
    return (
      <div className={`${sizeClasses[size]} rounded-full bg-gradient-to-br ${getGradientColors(symbol)} flex items-center justify-center text-white font-bold text-xs shadow-neon relative`}>
        <span className="relative z-10">{text.slice(0, 3)}</span>
        <div className={`absolute inset-0 rounded-full bg-gradient-to-br ${getGradientColors(symbol)} animate-pulse-slow opacity-50`} />
      </div>
    )
  }

  interface CryptoLogoImgProps {
    logoUrl: string
    symbol: string
    sizeClass: string
  }

  const CryptoLogoImg: React.FC<CryptoLogoImgProps> = ({ logoUrl, symbol, sizeClass }) => (
    <div className={`${sizeClass} relative group`}>
      <img 
        src={logoUrl} 
        alt={symbol} 
        className={`${sizeClass} rounded-full relative z-10 transition-transform group-hover:scale-110`}
        onError={(e: React.SyntheticEvent<HTMLImageElement, Event>) => {
          // Hide image on error and show fallback
          e.currentTarget.style.display = 'none'
        }}
      />
      <div className={`absolute inset-0 rounded-full bg-gradient-to-br ${getGradientColors(symbol)} opacity-30 blur-md group-hover:opacity-50 transition-opacity`} />
    </div>
  )

  return (
    <CryptoLogoImg logoUrl={logoUrl} symbol={symbol} sizeClass={sizeClasses[size]} />
  )
}