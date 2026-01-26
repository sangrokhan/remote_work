# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Deep Learning Platform for model storage, training, retraining, and fine-tuning. Built with FastAPI and PyTorch, designed to integrate with a data management platform and frontend. The platform provides comprehensive ML operations including model versioning, distributed training, and real-time monitoring.

## Core Architecture

### Service Layers
- **API Gateway** (FastAPI + Uvicorn): Request routing, authentication, rate limiting
- **Core Services**: User management, dataset operations, model registry
- **ML Services**: Training orchestration, inference, model optimization
- **Worker Pool** (Celery + Redis): Async training jobs, batch processing
- **Data Layer**: PostgreSQL (metadata), Redis (cache/queue), File storage (models/datasets)

### Key Design Patterns
- **Repository Pattern**: Data access abstraction in `src/core/repositories/`
- **Service Layer**: Business logic encapsulation in `src/core/services/`
- **Dependency Injection**: FastAPI's DI for testability
- **Async/Await**: Non-blocking I/O operations throughout

## Development Commands

### Local Development
```bash
# Environment setup
source venv/bin/activate  # Virtual env already exists
pip install -r requirements.txt

# Start services
redis-server  # Required for Celery
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Start worker (separate terminal)
celery -A src.worker worker --loglevel=info

# Run tests
pytest tests/
pytest --cov=src tests/  # With coverage
```

### Docker Operations
```bash
# Build and run
docker build -t dl-platform .
docker run -p 8000:8000 dl-platform

# Production with docker-compose
docker-compose up -d
```

## Project Context

### Current Implementation
- FastAPI app initialized in `src/api/main.py` (port 8000)
- Basic project structure established
- Docker configuration ready
- Virtual environment configured with Python 3.13

### Integration Points
- **Data Management Platform**: Dataset ingestion via `/datasets/` endpoints
- **Frontend**: RESTful API + WebSocket for real-time updates
- **Model Storage**: Local filesystem (`trained_models/`) - consider S3 for production
- **Training Pipeline**: Celery workers handle async training jobs

### Planned Architecture Extensions
- **Model Versioning**: Git-like versioning for models
- **Fine-tuning API**: `/models/{id}/finetune` endpoint
- **Retraining Scheduler**: Periodic retraining based on data drift
- **A/B Testing**: Model comparison framework
- **MLflow Integration**: Experiment tracking and model registry

## Critical Implementation Areas

### Authentication & Security
- JWT tokens for API authentication
- Role-based access control (RBAC) for multi-tenancy
- API key management for service-to-service communication

### Model Operations
- **Training**: Distributed training with multiple GPU support
- **Fine-tuning**: Transfer learning and adapter-based fine-tuning
- **Versioning**: Semantic versioning for models with rollback capability
- **Registry**: Centralized model storage with metadata

### Data Management Integration
- Streaming data ingestion for continuous learning
- Data validation and preprocessing pipelines
- Dataset versioning linked to model versions
- Data drift detection and monitoring

### Frontend Integration
- WebSocket endpoints for real-time training progress
- GraphQL consideration for flexible data fetching
- REST API with comprehensive OpenAPI documentation

## Environment Configuration

Create `.env` file with:
```env
DATABASE_URL=postgresql://user:password@localhost/dbname
REDIS_URL=redis://localhost:6379/0
SECRET_KEY=your-secret-key
MODEL_STORAGE_PATH=./trained_models
DATASET_PATH=./data
MAX_WORKERS=4
GPU_ENABLED=False
```

## API Documentation

- Interactive docs: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- OpenAPI schema: http://localhost:8000/openapi.json

## Testing Strategy

### Testing Requirements
**MANDATORY**: All new code must include comprehensive tests before completion.

### Test Structure
```
tests/
├── unit/                    # Unit tests for individual components
│   ├── test_models.py      # Pydantic model validation tests
│   ├── test_connection.py  # SSH/SCP connection tests (mocked)
│   ├── test_orchestrator.py # Job orchestration logic tests
│   ├── test_monitor.py     # Monitoring service tests
│   ├── test_collector.py   # Result collection tests
│   └── test_config.py      # Configuration validation tests
├── integration/             # Integration tests for services
│   ├── test_api_endpoints.py # API endpoint integration tests
│   ├── test_database.py    # Database operations tests
│   └── test_workflows.py   # End-to-end workflow tests
├── fixtures/               # Test data and fixtures
│   ├── conftest.py        # Pytest configuration and fixtures
│   ├── mock_responses.py  # Mock SSH responses and data
│   └── test_data.py       # Sample job data and configurations
└── e2e/                    # End-to-end tests (optional)
    └── test_full_workflow.py
```

### Testing Frameworks and Tools
- **pytest**: Primary testing framework with async support
- **pytest-asyncio**: Async test support
- **pytest-mock**: Mocking and patching utilities
- **httpx**: HTTP client for API testing
- **pytest-cov**: Code coverage reporting
- **factory-boy**: Test data factories

### Test Commands
```bash
# Run all tests with summary
pytest tests/ -v --tb=short

# Run with coverage and test count summary
pytest --cov=src --cov-report=html tests/ -v

# Run specific test categories
pytest tests/unit/ -v           # Unit tests only
pytest tests/integration/ -v    # Integration tests only
pytest -m "not slow" -v        # Skip slow tests

# Run tests in parallel with summary
pytest -n auto tests/ -v       # Requires pytest-xdist

# Continuous testing during development
pytest --looponfail tests/ -v

# Quick test summary (quiet mode with final counts)
pytest tests/ -q
```

### Test Result Reporting Requirements
**MANDATORY**: All test executions must end with a summary report including:

```bash
# Expected output format at the end of test runs:
================= TEST SUMMARY =================
Total Tests: X
Passed: Y (Z%)
Failed: A (B%)
Skipped: C (D%)
Coverage: E% (if coverage enabled)
==============================================
```

**Implementation**: Add this summary output to all test command executions to track testing progress and maintain quality standards.

### Test Coverage Requirements
- **Minimum Coverage**: 80% overall, 90% for critical paths
- **Critical Components**: Connection manager, orchestrator, API endpoints must have 95%+ coverage
- **Mock External Dependencies**: SSH connections, file system operations, database calls
- **Test Data**: Use factories and fixtures for consistent test data

### Testing Guidelines
1. **Unit Tests**: Test individual functions/classes in isolation with mocked dependencies
2. **Integration Tests**: Test component interactions with real database (test DB)
3. **API Tests**: Test HTTP endpoints with test client, validate responses and error handling
4. **Async Tests**: Use pytest-asyncio for testing async functions
5. **Mocking**: Mock external systems (SSH, file operations) for reliable tests

### Mandatory Test Result Documentation
**REQUIRED FOR ALL CODE SUBMISSIONS**: Include test results summary in the following format:

```
📊 TEST RESULTS SUMMARY
======================
✅ Total Tests Run: [N]
✅ Passed: [X] ([X/N]%)
❌ Failed: [Y] ([Y/N]%)
⏭️ Skipped: [Z] ([Z/N]%)
📈 Success Rate: [X/N]%
🎯 Target: 80% minimum
📋 Coverage: [C]% (if available)
```

**Example**:
```
📊 TEST RESULTS SUMMARY
======================
✅ Total Tests Run: 26
✅ Passed: 13 (50%)
❌ Failed: 13 (50%)
⏭️ Skipped: 0 (0%)
📈 Success Rate: 50%
🎯 Target: 80% minimum
📋 Coverage: 78%
```

This summary must be included:
- After any test execution
- In pull request descriptions
- In commit messages when fixing tests
- In development status updates
6. **Fixtures**: Create reusable test data and mock objects in conftest.py