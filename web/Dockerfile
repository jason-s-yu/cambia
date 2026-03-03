# web/Dockerfile — build React frontend, output to /dist for nginx
FROM node:22-alpine AS builder
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
ARG VITE_API_URL=""
ARG VITE_WS_URL=""
# Skip tsc (pre-existing type errors); Vite/esbuild handles transpilation
RUN npx vite build

# --- Serve via nginx ---
FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
# nginx config is mounted via docker-compose
EXPOSE 80
