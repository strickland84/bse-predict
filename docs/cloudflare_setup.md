# Cloudflare Tunnel Setup for HTTPS API Access

This guide will help you set up Cloudflare Tunnel to provide HTTPS access to your API backend, solving the mixed content issue when deploying the frontend to Vercel.

## Why Cloudflare Tunnel?

- **Free HTTPS**: No need to manage SSL certificates
- **No port forwarding**: Works behind firewalls/NAT
- **Automatic SSL**: Valid certificates managed by Cloudflare
- **Quick setup**: 5-10 minutes to get running

## Prerequisites

1. A Cloudflare account (free tier is fine)
2. Your domain added to Cloudflare (e.g., bse-predict.com)

## Step-by-Step Setup

### 1. Create a Cloudflare Tunnel

1. Go to [Cloudflare Dashboard](https://one.dash.cloudflare.com/)
2. Navigate to **Zero Trust** → **Networks** → **Tunnels**
3. Click **Create a tunnel**
4. Choose **Cloudflared** as the connector
5. Name your tunnel: `bse-predict-api`
6. Click **Save tunnel**

### 2. Configure the Tunnel

After creating the tunnel, you'll see a token. This is your `CLOUDFLARE_TUNNEL_TOKEN`.

1. **Copy the token** (it looks like a long string)
2. Add it to your `.env` file on the server:
```bash
echo "CLOUDFLARE_TUNNEL_TOKEN=your-token-here" >> .env
```

### 3. Add a Public Hostname

In the Cloudflare dashboard:

1. Click on your tunnel name
2. Go to **Public Hostname** tab
3. Click **Add a public hostname**
4. Configure:
   - **Subdomain**: `api` (or whatever you prefer)
   - **Domain**: Select `bse-predict.com`
   - **Type**: `HTTP`
   - **URL**: `dashboard-backend:8000` (this is the Docker service name)

### 4. Deploy with Docker Compose

On your server:

```bash
# Pull the latest code
git pull

# Make sure your .env has the token
cat .env | grep CLOUDFLARE_TUNNEL_TOKEN

# Deploy with the production compose file
docker-compose -f docker-compose.prod.yml up -d

# Check if cloudflared is running
docker logs bse-cloudflared-prod
```

### 5. Update Frontend Environment Variables

In your Vercel dashboard:

1. Go to your project settings
2. Navigate to **Environment Variables**
3. Update:
   - `VITE_API_URL` = `https://api.bse-predict.com`
   - `VITE_WS_URL` = `wss://api.bse-predict.com`
4. Redeploy your frontend

## Verification

1. Test the API endpoint:
```bash
curl https://api.bse-predict.com/api/system/status
```

2. Check tunnel status in Cloudflare dashboard:
   - Go to **Zero Trust** → **Networks** → **Tunnels**
   - Your tunnel should show as **Active**

## Troubleshooting

### Tunnel not connecting
```bash
# Check logs
docker logs bse-cloudflared-prod

# Restart the service
docker-compose -f docker-compose.prod.yml restart cloudflared
```

### API not responding
```bash
# Check if backend is running
docker ps | grep dashboard-backend

# Check backend logs
docker logs bse-dashboard-backend-prod
```

### DNS not resolving
- Make sure the subdomain (api.bse-predict.com) is configured in Cloudflare
- DNS propagation can take a few minutes

## Alternative: Manual Setup (Without Docker)

If you prefer to run cloudflared directly on the server:

```bash
# Install cloudflared
wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
sudo dpkg -i cloudflared-linux-amd64.deb

# Login to Cloudflare
cloudflared tunnel login

# Create tunnel
cloudflared tunnel create bse-predict-api

# Create config file
cat > ~/.cloudflared/config.yml << EOF
tunnel: bse-predict-api
credentials-file: /home/your-user/.cloudflared/<tunnel-id>.json

ingress:
  - hostname: api.bse-predict.com
    service: http://localhost:8001
  - service: http_status:404
EOF

# Route DNS
cloudflared tunnel route dns bse-predict-api api.bse-predict.com

# Run as service
sudo cloudflared service install
sudo systemctl start cloudflared
```

## Benefits

✅ **No SSL certificate management**
✅ **Works immediately with Vercel frontend**
✅ **Free for personal/small projects**
✅ **Built-in DDoS protection from Cloudflare**
✅ **Can add authentication later via Cloudflare Access**

## Next Steps

Once your API is accessible via HTTPS:
1. Frontend will work without mixed content errors
2. WebSocket connections will work over WSS
3. You can add Cloudflare Access for authentication if needed
4. Consider adding rate limiting via Cloudflare Rules