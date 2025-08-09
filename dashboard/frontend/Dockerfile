# Development Dockerfile - Serve React app directly without nginx
FROM node:18-alpine

WORKDIR /app

# Install serve package globally  
RUN npm install -g serve

# Copy package files
COPY package.json ./
RUN npm install

# Copy source code
COPY . .

# Build the app - use localhost:8001 for development
ENV VITE_API_URL=http://localhost:8001
ENV VITE_WS_URL=ws://localhost:8001/ws
RUN npm run build

EXPOSE 3000

# Serve the built app
CMD ["serve", "-s", "dist", "-l", "3000"]