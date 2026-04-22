FROM node:20-alpine
WORKDIR /app
ARG NPM_REGISTRY
RUN if [ -n "$NPM_REGISTRY" ]; then npm config set registry "$NPM_REGISTRY"; fi
COPY package.json package-lock.json* ./
RUN npm ci --include=dev
COPY . .
ARG VITE_WORKFLOW_WS_URL
ARG VITE_WORKFLOW_GRAPH_URL
ENV VITE_WORKFLOW_WS_URL=$VITE_WORKFLOW_WS_URL
ENV VITE_WORKFLOW_GRAPH_URL=$VITE_WORKFLOW_GRAPH_URL
RUN npm run build
EXPOSE 10000
CMD ["npm", "run", "preview", "--", "--host", "0.0.0.0", "--port", "10000"]
