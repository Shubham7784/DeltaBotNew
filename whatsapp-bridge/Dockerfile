FROM node:20-alpine

# Install git needed by baileys dependencies
RUN apk add --no-cache git

WORKDIR /app

# Install dependencies
COPY package*.json ./
RUN npm install --omit=dev

# Copy application source
COPY . .

EXPOSE 3001

CMD ["node", "server.js"]
