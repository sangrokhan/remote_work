# Deep Learning Platform

A comprehensive, production-ready deep learning platform built with FastAPI and PyTorch for training and managing machine learning models at scale.

## 🚀 Features

- **RESTful API**: Built with FastAPI for high-performance, automatic documentation, and type validation
- **ML Framework Support**: PyTorch integration with support for TorchVision and TorchAudio
- **Asynchronous Processing**: Celery-based task queue for distributed model training
- **Database Integration**: SQLAlchemy ORM with PostgreSQL for persistent storage
- **Containerization**: Docker support for easy deployment and scaling
- **Auto-documentation**: Interactive API documentation with Swagger UI and ReDoc

## 📋 Requirements

- Python 3.9+
- PostgreSQL 13+
- Redis 6.0+
- Docker (optional, for containerized deployment)

## 🛠️ Technology Stack

| Component | Technology |
|-----------|------------|
| **Web Framework** | FastAPI + Uvicorn |
| **ML Framework** | PyTorch, TorchVision, TorchAudio |
| **Task Queue** | Celery + Redis |
| **Database** | PostgreSQL + SQLAlchemy |
| **Containerization** | Docker |
| **Testing** | Pytest (planned) |

## 📂 Project Structure

```
dl-platform/
├── src/
│   ├── api/              # FastAPI application
│   │   ├── main.py       # Application entry point
│   │   ├── endpoints/    # API route handlers
│   │   └── schemas/      # Pydantic models
│   ├── core/             # Core business logic
│   ├── ml/               # Machine learning components
│   │   └── models/       # PyTorch model definitions
│   └── worker/           # Background task workers
├── tests/                # Test suite
├── data/                 # Dataset storage
├── notebooks/            # Jupyter notebooks
├── trained_models/       # Saved model artifacts
├── training_outputs/     # Training logs and results
├── requirements.txt      # Python dependencies
├── Dockerfile           # Container definition
└── CLAUDE.md            # AI assistant guidelines
```

## 🚀 Quick Start

### Local Development Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd dl-platform
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. **Start Redis** (required for Celery)
   ```bash
   redis-server
   ```

6. **Run the development server**
   ```bash
   uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
   ```

7. **Access the application**
   - API: http://localhost:8000
   - Interactive docs: http://localhost:8000/docs
   - Alternative docs: http://localhost:8000/redoc

### Docker Deployment

1. **Build the Docker image**
   ```bash
   docker build -t dl-platform .
   ```

2. **Run the container**
   ```bash
   docker run -p 8000:8000 dl-platform
   ```

### Docker Compose (recommended for production)
```bash
docker-compose up -d
```

## 🔌 API Endpoints

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check and welcome message |
| GET | `/docs` | Interactive API documentation |
| GET | `/redoc` | Alternative API documentation |

### Planned Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| **Authentication** | | |
| POST | `/auth/register` | User registration |
| POST | `/auth/login` | User authentication |
| POST | `/auth/refresh` | Refresh access token |
| **Datasets** | | |
| POST | `/datasets/upload` | Upload training dataset |
| GET | `/datasets` | List available datasets |
| GET | `/datasets/{id}` | Get dataset details |
| DELETE | `/datasets/{id}` | Delete dataset |
| **Model Training** | | |
| POST | `/training/start` | Initiate model training |
| GET | `/training/status/{job_id}` | Check training status |
| GET | `/training/results/{job_id}` | Get training results |
| POST | `/training/stop/{job_id}` | Stop training job |
| **Models** | | |
| GET | `/models` | List trained models |
| GET | `/models/{id}` | Get model details |
| POST | `/models/{id}/predict` | Make predictions |
| DELETE | `/models/{id}` | Delete model |

## 🧪 Testing

Run the test suite:
```bash
pytest tests/
```

Run with coverage:
```bash
pytest --cov=src tests/
```

## 🔧 Configuration

Configuration is managed through environment variables. Create a `.env` file in the project root:

```env
# Database
DATABASE_URL=postgresql://user:password@localhost/dbname

# Redis
REDIS_URL=redis://localhost:6379/0

# API
API_HOST=0.0.0.0
API_PORT=8000

# Security
SECRET_KEY=your-secret-key
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
```

## 🚢 Deployment

### Production Considerations

1. **Environment Variables**: Use secure secret management (AWS Secrets Manager, Vault, etc.)
2. **Database**: Use managed PostgreSQL service (RDS, Cloud SQL, etc.)
3. **Redis**: Use managed Redis service (ElastiCache, Redis Cloud, etc.)
4. **Monitoring**: Implement logging and monitoring (ELK stack, Prometheus, etc.)
5. **Scaling**: Use container orchestration (Kubernetes, ECS, etc.)

### Health Checks

The platform provides health check endpoints for monitoring:
- `/health` - Basic health check
- `/health/ready` - Readiness probe
- `/health/live` - Liveness probe

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 Development Roadmap

### Phase 1: Core Infrastructure ✅
- [x] FastAPI setup
- [x] Project structure
- [x] Docker configuration
- [x] Basic routing

### Phase 2: Authentication & Authorization 🚧
- [ ] User registration/login
- [ ] JWT token implementation
- [ ] Role-based access control
- [ ] API key management

### Phase 3: Data Management 📋
- [ ] Dataset upload API
- [ ] Data validation
- [ ] Storage management
- [ ] Data preprocessing pipelines

### Phase 4: Model Training 📋
- [ ] Training job queue
- [ ] Model architecture selection
- [ ] Hyperparameter configuration
- [ ] Training monitoring

### Phase 5: Model Serving 📋
- [ ] Model registry
- [ ] Inference API
- [ ] Batch prediction
- [ ] Model versioning

### Phase 6: Monitoring & Analytics 📋
- [ ] Training metrics dashboard
- [ ] Model performance tracking
- [ ] Resource utilization monitoring
- [ ] Alerting system

## 📚 Documentation

- [API Documentation](http://localhost:8000/docs) - Interactive API documentation
- [Architecture Guide](docs/ARCHITECTURE.md) - System architecture details
- [Development Guide](docs/DEVELOPMENT.md) - Development best practices
- [Deployment Guide](docs/DEPLOYMENT.md) - Production deployment instructions

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- FastAPI for the excellent web framework
- PyTorch team for the powerful ML framework
- The open-source community for continuous inspiration

## 📞 Support

For support, please:
1. Check the [documentation](docs/)
2. Search [existing issues](https://github.com/yourusername/dl-platform/issues)
3. Create a new issue if needed

---

**Built with ❤️ for the ML community**
