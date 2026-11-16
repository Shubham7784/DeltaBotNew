# Stage 1: Build the React Application
FROM node:20-alpine AS builder

WORKDIR /app

COPY package*.json ./
RUN npm install

COPY . .

# Set empty API base URL so the frontend makes relative /api calls routed by Nginx
ENV VITE_API_BASE_URL=""
RUN npm run build

# Stage 2: Serve with Production Nginx Reverse Proxy
FROM nginx:alpine

COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

HEALTHCHECK --interval=10s --timeout=3s --retries=3 \
  CMD wget -q -O /dev/null http://127.0.0.1/ || exit 1

CMD ["nginx", "-g", "daemon off;"]
