# Stage 1: Build React Frontend
FROM node:18-alpine AS frontend-builder
WORKDIR /app/frontend
# Copy only package files first for caching npm install
COPY frontend/package*.json ./
RUN npm ci
# Copy the rest of the frontend source
COPY frontend/ .
RUN npm run build

# Stage 2: Final Python Backend
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code (including backend)
COPY . .

# Copy the built React app from Stage 1
COPY --from=frontend-builder /app/frontend/dist /app/frontend/dist

# Create necessary directories
RUN mkdir -p data/papers/easy data/papers/medium data/papers/hard logs checkpoints data/tmp

# Expose port (Hugging Face Spaces uses 7860)
EXPOSE 7860

# Set environment variables
ENV HOST="0.0.0.0"
ENV PORT=7860

# Run FastAPI app
CMD ["python", "server/api.py"]
