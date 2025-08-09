/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        dark: {
          bg: '#040811',
          card: '#0a1628',
          border: '#1e3a5f',
          text: '#e8f4ff',
          muted: '#7da3c0',
        },
        success: '#00ff88',
        error: '#ff3366',
        warning: '#ffaa00',
        info: '#00b4d8',
        accent: '#00ffff',
        purple: '#a855f7',
        'blue-glow': '#0099ff',
      },
      backgroundImage: {
        'gradient-radial': 'radial-gradient(var(--tw-gradient-stops))',
        'gradient-cyber': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
        'gradient-neon': 'linear-gradient(135deg, #00ffff 0%, #0099ff 50%, #0066ff 100%)',
        'gradient-dark': 'linear-gradient(180deg, #0a1628 0%, #040811 100%)',
        'gradient-card': 'linear-gradient(135deg, rgba(10, 22, 40, 0.8) 0%, rgba(30, 58, 95, 0.3) 100%)',
      },
      boxShadow: {
        'glow': '0 0 20px rgba(0, 153, 255, 0.5)',
        'glow-sm': '0 0 10px rgba(0, 153, 255, 0.3)',
        'glow-lg': '0 0 30px rgba(0, 153, 255, 0.6)',
        'neon': '0 0 15px rgba(0, 255, 255, 0.5), 0 0 30px rgba(0, 255, 255, 0.3)',
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'glow': 'glow 2s ease-in-out infinite alternate',
        'border-beam': 'border-beam 3s linear infinite',
      },
      keyframes: {
        glow: {
          'from': { boxShadow: '0 0 10px rgba(0, 153, 255, 0.3)' },
          'to': { boxShadow: '0 0 20px rgba(0, 153, 255, 0.6), 0 0 30px rgba(0, 153, 255, 0.3)' }
        },
        'border-beam': {
          '0%, 100%': { backgroundPosition: '0% 50%' },
          '50%': { backgroundPosition: '100% 50%' }
        }
      }
    },
  },
  plugins: [],
}