# API Documentation

## Overview

The Deep Learning Platform API is built with FastAPI, providing a RESTful interface for managing machine learning workflows. The API includes automatic documentation, type validation, and high-performance async support.

## Base URL

```
http://localhost:8000
```

## Authentication

The API uses JWT (JSON Web Tokens) for authentication. Include the token in the Authorization header:

```
Authorization: Bearer <your_jwt_token>
```

## API Endpoints

### Health Check

#### GET /
Returns a welcome message and confirms the API is running.

**Response:**
```json
{
  "message": "Welcome to the Deep Learning Platform"
}
```

### Authentication Endpoints

#### POST /auth/register
Register a new user account.

**Request Body:**
```json
{
  "username": "string",
  "email": "user@example.com",
  "password": "string",
  "full_name": "string"
}
```

**Response (201):**
```json
{
  "id": "uuid",
  "username": "string",
  "email": "user@example.com",
  "full_name": "string",
  "created_at": "2024-01-01T00:00:00Z"
}
```

#### POST /auth/login
Authenticate user and receive JWT tokens.

**Request Body:**
```json
{
  "username": "string",
  "password": "string"
}
```

**Response (200):**
```json
{
  "access_token": "string",
  "refresh_token": "string",
  "token_type": "bearer",
  "expires_in": 1800
}
```

#### POST /auth/refresh
Refresh expired access token.

**Request Body:**
```json
{
  "refresh_token": "string"
}
```

**Response (200):**
```json
{
  "access_token": "string",
  "token_type": "bearer",
  "expires_in": 1800
}
```

### Dataset Management

#### POST /datasets/upload
Upload a new dataset for training.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Authentication: Required

**Form Data:**
- `file`: Dataset file (CSV, JSON, or ZIP)
- `name`: Dataset name
- `description`: Dataset description (optional)
- `dataset_type`: Type of dataset (classification, regression, etc.)

**Response (201):**
```json
{
  "id": "uuid",
  "name": "string",
  "description": "string",
  "dataset_type": "string",
  "file_size": 1024000,
  "uploaded_at": "2024-01-01T00:00:00Z",
  "status": "processing"
}
```

#### GET /datasets
List all available datasets for the authenticated user.

**Query Parameters:**
- `page`: Page number (default: 1)
- `limit`: Items per page (default: 10)
- `sort_by`: Sort field (name, created_at, size)
- `order`: Sort order (asc, desc)

**Response (200):**
```json
{
  "total": 50,
  "page": 1,
  "limit": 10,
  "datasets": [
    {
      "id": "uuid",
      "name": "string",
      "description": "string",
      "dataset_type": "string",
      "file_size": 1024000,
      "uploaded_at": "2024-01-01T00:00:00Z",
      "status": "ready"
    }
  ]
}
```

#### GET /datasets/{dataset_id}
Get detailed information about a specific dataset.

**Response (200):**
```json
{
  "id": "uuid",
  "name": "string",
  "description": "string",
  "dataset_type": "string",
  "file_size": 1024000,
  "uploaded_at": "2024-01-01T00:00:00Z",
  "status": "ready",
  "metadata": {
    "rows": 10000,
    "columns": 50,
    "features": ["feature1", "feature2"],
    "target": "label"
  }
}
```

#### DELETE /datasets/{dataset_id}
Delete a dataset.

**Response (204):**
No content

### Model Training

#### POST /training/start
Start a new model training job.

**Request Body:**
```json
{
  "dataset_id": "uuid",
  "model_type": "resnet50",
  "hyperparameters": {
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 100,
    "optimizer": "adam"
  },
  "validation_split": 0.2,
  "early_stopping": {
    "enabled": true,
    "patience": 10,
    "monitor": "val_loss"
  }
}
```

**Response (201):**
```json
{
  "job_id": "uuid",
  "status": "queued",
  "created_at": "2024-01-01T00:00:00Z",
  "estimated_time": 3600
}
```

#### GET /training/status/{job_id}
Check the status of a training job.

**Response (200):**
```json
{
  "job_id": "uuid",
  "status": "running",
  "progress": 45,
  "current_epoch": 45,
  "total_epochs": 100,
  "metrics": {
    "loss": 0.234,
    "accuracy": 0.912,
    "val_loss": 0.301,
    "val_accuracy": 0.889
  },
  "started_at": "2024-01-01T00:00:00Z",
  "elapsed_time": 1800
}
```

#### GET /training/results/{job_id}
Get final training results.

**Response (200):**
```json
{
  "job_id": "uuid",
  "status": "completed",
  "model_id": "uuid",
  "final_metrics": {
    "train_loss": 0.123,
    "train_accuracy": 0.956,
    "val_loss": 0.234,
    "val_accuracy": 0.912
  },
  "training_history": {
    "loss": [0.9, 0.7, 0.5],
    "accuracy": [0.6, 0.75, 0.85],
    "val_loss": [0.95, 0.8, 0.6],
    "val_accuracy": [0.55, 0.7, 0.82]
  },
  "duration": 3600,
  "completed_at": "2024-01-01T01:00:00Z"
}
```

#### POST /training/stop/{job_id}
Stop a running training job.

**Response (200):**
```json
{
  "job_id": "uuid",
  "status": "stopped",
  "message": "Training job stopped successfully"
}
```

### Model Management

#### GET /models
List all trained models.

**Query Parameters:**
- `page`: Page number (default: 1)
- `limit`: Items per page (default: 10)
- `model_type`: Filter by model type
- `sort_by`: Sort field (created_at, accuracy, name)

**Response (200):**
```json
{
  "total": 25,
  "page": 1,
  "limit": 10,
  "models": [
    {
      "id": "uuid",
      "name": "ResNet50_v1",
      "model_type": "resnet50",
      "accuracy": 0.912,
      "created_at": "2024-01-01T00:00:00Z",
      "size_mb": 98.5
    }
  ]
}
```

#### GET /models/{model_id}
Get detailed information about a model.

**Response (200):**
```json
{
  "id": "uuid",
  "name": "ResNet50_v1",
  "model_type": "resnet50",
  "architecture": {
    "input_shape": [224, 224, 3],
    "output_classes": 10,
    "parameters": 25600000
  },
  "performance": {
    "accuracy": 0.912,
    "precision": 0.905,
    "recall": 0.918,
    "f1_score": 0.911
  },
  "training_config": {
    "dataset_id": "uuid",
    "epochs": 100,
    "batch_size": 32,
    "optimizer": "adam"
  },
  "created_at": "2024-01-01T00:00:00Z"
}
```

#### POST /models/{model_id}/predict
Make predictions using a trained model.

**Request Body (for single prediction):**
```json
{
  "input": [0.5, 0.3, 0.2, 0.1]
}
```

**Request Body (for batch prediction):**
```json
{
  "inputs": [
    [0.5, 0.3, 0.2, 0.1],
    [0.4, 0.2, 0.3, 0.1]
  ]
}
```

**Response (200):**
```json
{
  "predictions": [
    {
      "class": "cat",
      "confidence": 0.892
    },
    {
      "class": "dog",
      "confidence": 0.756
    }
  ]
}
```

#### DELETE /models/{model_id}
Delete a trained model.

**Response (204):**
No content

## Error Responses

The API uses standard HTTP status codes and returns detailed error messages:

### 400 Bad Request
```json
{
  "error": "validation_error",
  "message": "Invalid input data",
  "details": [
    {
      "field": "learning_rate",
      "message": "Must be between 0 and 1"
    }
  ]
}
```

### 401 Unauthorized
```json
{
  "error": "unauthorized",
  "message": "Invalid or expired token"
}
```

### 403 Forbidden
```json
{
  "error": "forbidden",
  "message": "You don't have permission to access this resource"
}
```

### 404 Not Found
```json
{
  "error": "not_found",
  "message": "Resource not found"
}
```

### 429 Too Many Requests
```json
{
  "error": "rate_limit_exceeded",
  "message": "Too many requests",
  "retry_after": 60
}
```

### 500 Internal Server Error
```json
{
  "error": "internal_error",
  "message": "An unexpected error occurred",
  "request_id": "uuid"
}
```

## Rate Limiting

The API implements rate limiting to ensure fair usage:

- **Authenticated users**: 1000 requests per hour
- **Unauthenticated users**: 100 requests per hour
- **Training jobs**: 5 concurrent jobs per user

Rate limit information is included in response headers:
```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 950
X-RateLimit-Reset: 1609459200
```

## Webhooks

Configure webhooks to receive real-time updates about training jobs:

### Webhook Events

- `training.started`: Training job has started
- `training.progress`: Training progress update (every 10%)
- `training.completed`: Training job completed successfully
- `training.failed`: Training job failed
- `dataset.processed`: Dataset processing completed

### Webhook Payload Example
```json
{
  "event": "training.completed",
  "timestamp": "2024-01-01T00:00:00Z",
  "data": {
    "job_id": "uuid",
    "model_id": "uuid",
    "accuracy": 0.912
  }
}
```

## SDK Examples

### Python
```python
import requests

# Authentication
response = requests.post(
    "http://localhost:8000/auth/login",
    json={"username": "user", "password": "pass"}
)
token = response.json()["access_token"]

# Upload dataset
files = {"file": open("data.csv", "rb")}
data = {"name": "My Dataset", "dataset_type": "classification"}
headers = {"Authorization": f"Bearer {token}"}

response = requests.post(
    "http://localhost:8000/datasets/upload",
    files=files,
    data=data,
    headers=headers
)
dataset_id = response.json()["id"]

# Start training
training_config = {
    "dataset_id": dataset_id,
    "model_type": "resnet50",
    "hyperparameters": {
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 100
    }
}

response = requests.post(
    "http://localhost:8000/training/start",
    json=training_config,
    headers=headers
)
job_id = response.json()["job_id"]
```

### JavaScript/TypeScript
```javascript
// Authentication
const loginResponse = await fetch('http://localhost:8000/auth/login', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ username: 'user', password: 'pass' })
});
const { access_token } = await loginResponse.json();

// Upload dataset
const formData = new FormData();
formData.append('file', fileInput.files[0]);
formData.append('name', 'My Dataset');
formData.append('dataset_type', 'classification');

const uploadResponse = await fetch('http://localhost:8000/datasets/upload', {
  method: 'POST',
  headers: { 'Authorization': `Bearer ${access_token}` },
  body: formData
});
const { id: datasetId } = await uploadResponse.json();

// Start training
const trainingResponse = await fetch('http://localhost:8000/training/start', {
  method: 'POST',
  headers: {
    'Authorization': `Bearer ${access_token}`,
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    dataset_id: datasetId,
    model_type: 'resnet50',
    hyperparameters: {
      learning_rate: 0.001,
      batch_size: 32,
      epochs: 100
    }
  })
});
const { job_id } = await trainingResponse.json();
```

## API Versioning

The API uses URL path versioning. The current version is v1 (implicit). Future versions will be explicitly versioned:

- Current: `http://localhost:8000/datasets`
- Future: `http://localhost:8000/v2/datasets`

## OpenAPI Schema

The complete OpenAPI schema is available at:
- JSON: `http://localhost:8000/openapi.json`
- Interactive documentation: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`