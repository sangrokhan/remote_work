# Development Guide

## Table of Contents
1. [Development Environment Setup](#development-environment-setup)
2. [Project Structure](#project-structure)
3. [Coding Standards](#coding-standards)
4. [Development Workflow](#development-workflow)
5. [Testing](#testing)
6. [Debugging](#debugging)
7. [Database Management](#database-management)
8. [API Development](#api-development)
9. [ML Model Development](#ml-model-development)
10. [Best Practices](#best-practices)

## Development Environment Setup

### Prerequisites

Ensure you have the following installed:
- Python 3.9 or higher
- PostgreSQL 13+
- Redis 6.0+
- Git
- Docker and Docker Compose (optional but recommended)

### Initial Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/dl-platform.git
cd dl-platform
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Development dependencies
```

4. **Set up environment variables**
```bash
cp .env.example .env
```

Edit `.env` file with your configuration:
```env
# Database
DATABASE_URL=postgresql://dluser:dlpass@localhost:5432/dlplatform

# Redis
REDIS_URL=redis://localhost:6379/0

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=True

# Security
SECRET_KEY=your-secret-key-for-development
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# File Storage
UPLOAD_DIR=./data/uploads
MODEL_DIR=./trained_models

# ML Configuration
MAX_WORKERS=4
GPU_ENABLED=False
```

5. **Initialize database**
```bash
# Create database
createdb dlplatform

# Run migrations (when implemented)
alembic upgrade head
```

6. **Start services**
```bash
# Start Redis
redis-server

# Start API server
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Start Celery worker
celery -A src.worker worker --loglevel=info
```

### Using Docker Compose

For a simpler setup, use Docker Compose:

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## Project Structure

```
dl-platform/
├── src/
│   ├── api/                 # FastAPI application
│   │   ├── main.py          # Application entry point
│   │   ├── endpoints/       # API endpoints
│   │   │   ├── __init__.py
│   │   │   ├── auth.py      # Authentication endpoints
│   │   │   ├── datasets.py  # Dataset management
│   │   │   ├── models.py    # Model management
│   │   │   └── training.py  # Training endpoints
│   │   ├── schemas/         # Pydantic models
│   │   │   ├── __init__.py
│   │   │   ├── user.py      # User schemas
│   │   │   ├── dataset.py   # Dataset schemas
│   │   │   ├── model.py     # Model schemas
│   │   │   └── training.py  # Training schemas
│   │   ├── middleware/      # Custom middleware
│   │   └── dependencies.py  # Dependency injection
│   ├── core/                # Business logic
│   │   ├── config.py        # Configuration
│   │   ├── security.py      # Security utilities
│   │   ├── services/        # Business services
│   │   ├── repositories/    # Data repositories
│   │   └── exceptions.py    # Custom exceptions
│   ├── ml/                  # Machine learning
│   │   ├── models/          # Model architectures
│   │   ├── training/        # Training logic
│   │   ├── evaluation/      # Evaluation metrics
│   │   └── preprocessing/   # Data preprocessing
│   └── worker/              # Background tasks
│       ├── tasks.py         # Celery tasks
│       └── celery.py        # Celery configuration
├── tests/                   # Test suite
├── docs/                    # Documentation
├── scripts/                 # Utility scripts
└── migrations/              # Database migrations
```

## Coding Standards

### Python Style Guide

We follow PEP 8 with some modifications:

```python
# Good
def calculate_model_accuracy(
    predictions: List[float],
    labels: List[float],
    threshold: float = 0.5
) -> float:
    """
    Calculate accuracy of model predictions.
    
    Args:
        predictions: Model predictions
        labels: Ground truth labels
        threshold: Classification threshold
        
    Returns:
        Accuracy score between 0 and 1
    """
    pass

# Bad
def calc_acc(pred,lbl,t=0.5):
    pass
```

### Type Hints

Always use type hints for better code clarity:

```python
from typing import List, Dict, Optional, Union

async def get_user_models(
    user_id: str,
    limit: Optional[int] = None,
    offset: int = 0
) -> List[Dict[str, Union[str, float]]]:
    pass
```

### Docstrings

Use Google-style docstrings:

```python
def train_model(config: TrainingConfig) -> Model:
    """Train a machine learning model.
    
    Args:
        config: Training configuration object containing
            hyperparameters and dataset information.
    
    Returns:
        Trained model instance.
        
    Raises:
        TrainingError: If training fails.
        ValidationError: If config is invalid.
    
    Examples:
        >>> config = TrainingConfig(epochs=10)
        >>> model = train_model(config)
    """
    pass
```

### Import Organization

```python
# Standard library imports
import os
import sys
from typing import List, Optional

# Third-party imports
import numpy as np
import torch
from fastapi import FastAPI, Depends

# Local imports
from src.core.config import settings
from src.ml.models import ResNet50
```

## Development Workflow

### Git Workflow

We use Git Flow:

1. **Feature Development**
```bash
# Create feature branch
git checkout -b feature/new-feature

# Make changes and commit
git add .
git commit -m "feat: add new feature"

# Push to remote
git push origin feature/new-feature

# Create pull request
```

2. **Commit Message Format**
```
<type>(<scope>): <subject>

<body>

<footer>
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Testing
- `chore`: Maintenance

Example:
```
feat(api): add dataset upload endpoint

- Implement multipart file upload
- Add file validation
- Store metadata in database

Closes #123
```

### Code Review Process

1. Create pull request with description
2. Ensure all tests pass
3. Request review from team members
4. Address feedback
5. Merge after approval

## Testing

### Test Structure

```
tests/
├── unit/              # Unit tests
│   ├── test_models.py
│   └── test_services.py
├── integration/       # Integration tests
│   ├── test_api.py
│   └── test_database.py
├── e2e/              # End-to-end tests
│   └── test_workflows.py
└── conftest.py       # Pytest fixtures
```

### Writing Tests

```python
# tests/unit/test_services.py
import pytest
from src.core.services import UserService

class TestUserService:
    @pytest.fixture
    def user_service(self):
        return UserService()
    
    def test_create_user(self, user_service):
        user = user_service.create_user(
            username="testuser",
            email="test@example.com"
        )
        assert user.username == "testuser"
        assert user.email == "test@example.com"
    
    def test_create_duplicate_user(self, user_service):
        user_service.create_user(
            username="testuser",
            email="test@example.com"
        )
        with pytest.raises(ValidationError):
            user_service.create_user(
                username="testuser",
                email="test@example.com"
            )
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/unit/test_models.py

# Run with verbose output
pytest -v

# Run only marked tests
pytest -m "slow"
```

### Test Coverage

Maintain minimum 80% code coverage:

```bash
# Generate coverage report
coverage run -m pytest
coverage report
coverage html  # Generate HTML report
```

## Debugging

### Using debugger

```python
# Using Python debugger
import pdb

def complex_function():
    x = 10
    pdb.set_trace()  # Debugger will stop here
    y = x * 2
    return y
```

### Using VS Code

`.vscode/launch.json`:
```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "FastAPI",
            "type": "python",
            "request": "launch",
            "module": "uvicorn",
            "args": [
                "src.api.main:app",
                "--reload",
                "--port", "8000"
            ],
            "jinja": true
        }
    ]
}
```

### Logging

```python
import logging
from src.core.logging import get_logger

logger = get_logger(__name__)

def process_data(data):
    logger.info(f"Processing {len(data)} items")
    try:
        result = transform(data)
        logger.debug(f"Transformation result: {result}")
        return result
    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)
        raise
```

## Database Management

### Migrations

Using Alembic for database migrations:

```bash
# Create new migration
alembic revision --autogenerate -m "Add user table"

# Apply migrations
alembic upgrade head

# Rollback one version
alembic downgrade -1

# View migration history
alembic history
```

### Database Models

```python
# src/core/models/user.py
from sqlalchemy import Column, String, DateTime
from sqlalchemy.dialects.postgresql import UUID
from src.core.database import Base

class User(Base):
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True)
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    created_at = Column(DateTime, server_default=func.now())
```

## API Development

### Creating New Endpoints

1. **Define Schema** (`src/api/schemas/`)
```python
from pydantic import BaseModel

class DatasetCreate(BaseModel):
    name: str
    description: Optional[str] = None
    
class DatasetResponse(BaseModel):
    id: str
    name: str
    created_at: datetime
    
    class Config:
        orm_mode = True
```

2. **Create Endpoint** (`src/api/endpoints/`)
```python
from fastapi import APIRouter, Depends, File, UploadFile
from src.api.schemas import DatasetCreate, DatasetResponse

router = APIRouter(prefix="/datasets", tags=["datasets"])

@router.post("/", response_model=DatasetResponse)
async def create_dataset(
    dataset: DatasetCreate,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user)
):
    """Create a new dataset."""
    return await dataset_service.create(dataset, file, current_user)
```

3. **Register Router** (`src/api/main.py`)
```python
from src.api.endpoints import datasets

app.include_router(datasets.router)
```

### API Testing

```python
# tests/integration/test_api.py
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_create_dataset():
    response = client.post(
        "/datasets/",
        json={"name": "Test Dataset"},
        files={"file": ("test.csv", b"data", "text/csv")}
    )
    assert response.status_code == 201
    assert response.json()["name"] == "Test Dataset"
```

## ML Model Development

### Adding New Models

1. **Define Model Architecture** (`src/ml/models/`)
```python
import torch
import torch.nn as nn

class CustomModel(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, output_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)
```

2. **Create Training Pipeline** (`src/ml/training/`)
```python
def train_model(
    model: nn.Module,
    dataloader: DataLoader,
    config: TrainingConfig
) -> Dict[str, float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(config.epochs):
        for batch in dataloader:
            optimizer.zero_grad()
            outputs = model(batch.inputs)
            loss = criterion(outputs, batch.labels)
            loss.backward()
            optimizer.step()
    
    return {"loss": loss.item()}
```

### Model Testing

```python
# tests/unit/test_models.py
import torch
from src.ml.models import CustomModel

def test_model_forward():
    model = CustomModel(input_dim=10, output_dim=5)
    x = torch.randn(32, 10)
    output = model(x)
    assert output.shape == (32, 5)
```

## Best Practices

### Security

1. **Never commit secrets**
```python
# Bad
API_KEY = "sk-1234567890"

# Good
API_KEY = os.getenv("API_KEY")
```

2. **Validate all inputs**
```python
from pydantic import validator

class UserCreate(BaseModel):
    email: str
    
    @validator("email")
    def validate_email(cls, v):
        if "@" not in v:
            raise ValueError("Invalid email")
        return v
```

3. **Use parameterized queries**
```python
# Bad
query = f"SELECT * FROM users WHERE id = {user_id}"

# Good
query = "SELECT * FROM users WHERE id = :user_id"
result = db.execute(query, {"user_id": user_id})
```

### Performance

1. **Use async/await for I/O operations**
```python
async def fetch_data():
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.json()
```

2. **Implement caching**
```python
from functools import lru_cache

@lru_cache(maxsize=128)
def expensive_computation(x: int) -> int:
    # Complex calculation
    return result
```

3. **Use batch operations**
```python
# Bad
for item in items:
    db.add(item)
    db.commit()

# Good
db.add_all(items)
db.commit()
```

### Error Handling

```python
from src.core.exceptions import DLPlatformException

try:
    result = risky_operation()
except SpecificError as e:
    logger.error(f"Operation failed: {e}")
    raise DLPlatformException("User-friendly error message") from e
finally:
    cleanup()
```

### Documentation

1. Document all public functions
2. Keep README up to date
3. Add examples for complex features
4. Document breaking changes

## Troubleshooting

### Common Issues

1. **Database connection errors**
```bash
# Check PostgreSQL is running
pg_isready

# Check connection string
psql $DATABASE_URL
```

2. **Redis connection errors**
```bash
# Check Redis is running
redis-cli ping

# Check Redis connection
redis-cli -h localhost -p 6379
```

3. **Import errors**
```bash
# Ensure you're in the project root
cd dl-platform

# Add project to Python path
export PYTHONPATH="${PYTHONPATH}:${PWD}"
```

4. **Port already in use**
```bash
# Find process using port 8000
lsof -i :8000

# Kill process
kill -9 <PID>
```

## Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Celery Documentation](https://docs.celeryproject.org/)
- [SQLAlchemy Documentation](https://www.sqlalchemy.org/)
- [Python Style Guide (PEP 8)](https://www.python.org/dev/peps/pep-0008/)
- [Type Hints (PEP 484)](https://www.python.org/dev/peps/pep-0484/)