from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .endpoints import gpu_cluster
from ..core.config import settings

app = FastAPI(
    title=settings.app_name,
    description="API for training and managing deep learning models.",
    version=settings.app_version,
    debug=settings.debug
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.security.cors_origins,
    allow_credentials=settings.security.cors_allow_credentials,
    allow_methods=settings.security.cors_allow_methods,
    allow_headers=settings.security.cors_allow_headers,
)

@app.get("/")
async def root():
    return {"message": "Welcome to the Deep Learning Platform"}

# GPU Cluster endpoints
app.include_router(gpu_cluster.router)

# Configuration endpoint
@app.get("/api/v1/config")
async def get_config():
    """시스템 설정 정보 조회"""
    from ..core.config import get_config_summary
    return get_config_summary()

# Here you will add routers for other endpoints
# from .endpoints import training, users
#
# app.include_router(training.router)
# app.include_router(users.router)
