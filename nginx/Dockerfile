# Build frontend
FROM node:18-alpine as frontend-builder

WORKDIR /app

# Copy frontend source
COPY dashboard/frontend/package*.json ./
RUN npm ci

COPY dashboard/frontend/ ./

# Build with correct environment variables
# Empty URL since we'll use relative paths
ENV VITE_API_URL=""
ENV VITE_WS_URL=""
RUN npm run build

# Production nginx image
FROM nginx:alpine

# Remove default nginx config
RUN rm /etc/nginx/conf.d/default.conf

# Copy nginx config
COPY nginx/nginx.prod.conf /etc/nginx/conf.d/default.conf

# Copy built frontend
COPY --from=frontend-builder /app/dist /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]