"""
Database Repository Layer

Repository pattern implementation for GPU cluster data access with
async SQLAlchemy operations and business logic separation.
"""

from datetime import datetime
from typing import Dict, List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete, func, and_, or_
from sqlalchemy.orm import selectinload

from .db_models import (
    GPUJobDB, JobResultDB, JobLogDB, JobMetricDB, 
    ClusterConfigDB, JobTemplateDB, JobQueueDB, User
)
from ..ml.cluster.models import GPUJob, JobStatus, JobResults, CodeSource


class GPUJobRepository:
    """GPU 작업 데이터 액세스 레이어"""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def create_job(self, job: GPUJob, user_id: str) -> GPUJobDB:
        """작업 생성"""
        from .db_models import gpu_job_to_db
        
        db_job = gpu_job_to_db(job, user_id)
        self.session.add(db_job)
        await self.session.commit()
        await self.session.refresh(db_job)
        return db_job
    
    async def get_job(self, job_id: str) -> Optional[GPUJobDB]:
        """작업 조회"""
        result = await self.session.execute(
            select(GPUJobDB)
            .options(selectinload(GPUJobDB.user))
            .where(GPUJobDB.job_id == job_id)
        )
        return result.scalar_one_or_none()
    
    async def update_job_status(self, job_id: str, status: JobStatus, **kwargs) -> bool:
        """작업 상태 업데이트"""
        update_data = {"status": status.value, "updated_at": datetime.utcnow()}
        update_data.update(kwargs)
        
        result = await self.session.execute(
            update(GPUJobDB)
            .where(GPUJobDB.job_id == job_id)
            .values(**update_data)
        )
        
        await self.session.commit()
        return result.rowcount > 0
    
    async def update_job_progress(self, job_id: str, progress_data: Dict) -> bool:
        """작업 진행 상황 업데이트"""
        result = await self.session.execute(
            update(GPUJobDB)
            .where(GPUJobDB.job_id == job_id)
            .values(**progress_data, updated_at=datetime.utcnow())
        )
        
        await self.session.commit()
        return result.rowcount > 0
    
    async def list_jobs(
        self,
        status: Optional[JobStatus] = None,
        user_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[GPUJobDB]:
        """작업 목록 조회"""
        query = select(GPUJobDB).options(selectinload(GPUJobDB.user))
        
        if status:
            query = query.where(GPUJobDB.status == status.value)
        if user_id:
            query = query.where(GPUJobDB.user_id == user_id)
        
        query = query.order_by(GPUJobDB.created_at.desc()).offset(offset).limit(limit)
        
        result = await self.session.execute(query)
        return result.scalars().all()
    
    async def delete_job(self, job_id: str) -> bool:
        """작업 삭제"""
        result = await self.session.execute(
            delete(GPUJobDB).where(GPUJobDB.job_id == job_id)
        )
        await self.session.commit()
        return result.rowcount > 0
    
    async def get_jobs_by_status(self, status: JobStatus) -> List[GPUJobDB]:
        """상태별 작업 조회"""
        result = await self.session.execute(
            select(GPUJobDB)
            .where(GPUJobDB.status == status.value)
            .order_by(GPUJobDB.created_at)
        )
        return result.scalars().all()
    
    async def get_user_job_count(self, user_id: str, status: Optional[JobStatus] = None) -> int:
        """사용자 작업 개수"""
        query = select(func.count(GPUJobDB.job_id)).where(GPUJobDB.user_id == user_id)
        
        if status:
            query = query.where(GPUJobDB.status == status.value)
        
        result = await self.session.execute(query)
        return result.scalar()
    
    async def get_active_gpu_usage(self) -> int:
        """현재 사용 중인 GPU 수"""
        result = await self.session.execute(
            select(func.sum(GPUJobDB.gpu_count))
            .where(GPUJobDB.status.in_([JobStatus.RUNNING.value, JobStatus.QUEUED.value]))
        )
        total_gpus = result.scalar()
        return total_gpus or 0


class JobResultRepository:
    """작업 결과 데이터 액세스 레이어"""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def create_result(self, result: JobResults) -> JobResultDB:
        """결과 생성"""
        db_result = JobResultDB(
            job_id=result.job_id,
            model_files=result.model_files,
            log_files=result.log_files,
            output_files=result.output_files,
            final_metrics=result.final_metrics,
            training_history=result.training_history,
            total_size_mb=result.total_size_mb,
            collected_at=result.collected_at
        )
        
        self.session.add(db_result)
        await self.session.commit()
        await self.session.refresh(db_result)
        return db_result
    
    async def get_result(self, job_id: str) -> Optional[JobResultDB]:
        """결과 조회"""
        result = await self.session.execute(
            select(JobResultDB)
            .options(selectinload(JobResultDB.job))
            .where(JobResultDB.job_id == job_id)
        )
        return result.scalar_one_or_none()
    
    async def update_result(self, job_id: str, update_data: Dict) -> bool:
        """결과 업데이트"""
        result = await self.session.execute(
            update(JobResultDB)
            .where(JobResultDB.job_id == job_id)
            .values(**update_data)
        )
        
        await self.session.commit()
        return result.rowcount > 0
    
    async def list_results(self, limit: int = 50, offset: int = 0) -> List[JobResultDB]:
        """결과 목록 조회"""
        result = await self.session.execute(
            select(JobResultDB)
            .options(selectinload(JobResultDB.job))
            .order_by(JobResultDB.collected_at.desc())
            .offset(offset)
            .limit(limit)
        )
        return result.scalars().all()
    
    async def delete_result(self, job_id: str) -> bool:
        """결과 삭제"""
        result = await self.session.execute(
            delete(JobResultDB).where(JobResultDB.job_id == job_id)
        )
        await self.session.commit()
        return result.rowcount > 0


class JobMetricRepository:
    """작업 메트릭 데이터 액세스 레이어"""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def add_metric(self, job_id: str, metric_name: str, value: float, **kwargs) -> JobMetricDB:
        """메트릭 추가"""
        metric = JobMetricDB(
            job_id=job_id,
            metric_name=metric_name,
            metric_value=value,
            **kwargs
        )
        
        self.session.add(metric)
        await self.session.commit()
        await self.session.refresh(metric)
        return metric
    
    async def get_job_metrics(self, job_id: str, metric_name: Optional[str] = None) -> List[JobMetricDB]:
        """작업 메트릭 조회"""
        query = select(JobMetricDB).where(JobMetricDB.job_id == job_id)
        
        if metric_name:
            query = query.where(JobMetricDB.metric_name == metric_name)
        
        query = query.order_by(JobMetricDB.recorded_at)
        
        result = await self.session.execute(query)
        return result.scalars().all()
    
    async def get_metric_history(self, job_id: str, metric_name: str) -> List[Dict]:
        """메트릭 히스토리 조회"""
        result = await self.session.execute(
            select(JobMetricDB.epoch, JobMetricDB.metric_value, JobMetricDB.recorded_at)
            .where(and_(JobMetricDB.job_id == job_id, JobMetricDB.metric_name == metric_name))
            .order_by(JobMetricDB.epoch, JobMetricDB.recorded_at)
        )
        
        return [
            {"epoch": epoch, "value": value, "timestamp": timestamp}
            for epoch, value, timestamp in result.all()
        ]
    
    async def delete_job_metrics(self, job_id: str) -> bool:
        """작업 메트릭 삭제"""
        result = await self.session.execute(
            delete(JobMetricDB).where(JobMetricDB.job_id == job_id)
        )
        await self.session.commit()
        return result.rowcount > 0


class ClusterConfigRepository:
    """클러스터 설정 데이터 액세스 레이어"""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def get_active_config(self) -> Optional[ClusterConfigDB]:
        """활성 클러스터 설정 조회"""
        result = await self.session.execute(
            select(ClusterConfigDB)
            .where(ClusterConfigDB.is_active == True)
            .order_by(ClusterConfigDB.created_at.desc())
        )
        return result.scalar_one_or_none()
    
    async def create_config(self, config_data: Dict) -> ClusterConfigDB:
        """설정 생성"""
        config = ClusterConfigDB(**config_data)
        self.session.add(config)
        await self.session.commit()
        await self.session.refresh(config)
        return config
    
    async def update_health_status(self, cluster_name: str, status: str) -> bool:
        """헬스 상태 업데이트"""
        result = await self.session.execute(
            update(ClusterConfigDB)
            .where(ClusterConfigDB.cluster_name == cluster_name)
            .values(
                health_status=status,
                last_health_check=datetime.utcnow()
            )
        )
        
        await self.session.commit()
        return result.rowcount > 0


class UserRepository:
    """사용자 데이터 액세스 레이어"""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def create_user(self, user_data: Dict) -> User:
        """사용자 생성"""
        user = User(**user_data)
        self.session.add(user)
        await self.session.commit()
        await self.session.refresh(user)
        return user
    
    async def get_user(self, user_id: str) -> Optional[User]:
        """사용자 조회"""
        result = await self.session.execute(
            select(User).where(User.id == user_id)
        )
        return result.scalar_one_or_none()
    
    async def get_user_by_username(self, username: str) -> Optional[User]:
        """사용자명으로 조회"""
        result = await self.session.execute(
            select(User).where(User.username == username)
        )
        return result.scalar_one_or_none()
    
    async def update_user(self, user_id: str, update_data: Dict) -> bool:
        """사용자 정보 업데이트"""
        update_data["updated_at"] = datetime.utcnow()
        
        result = await self.session.execute(
            update(User)
            .where(User.id == user_id)
            .values(**update_data)
        )
        
        await self.session.commit()
        return result.rowcount > 0