"""Initial GPU cluster tables

Revision ID: 001
Revises: 
Create Date: 2025-08-25

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create users table
    op.create_table('users',
    sa.Column('id', sa.String(), nullable=False),
    sa.Column('username', sa.String(length=255), nullable=False),
    sa.Column('email', sa.String(length=255), nullable=False),
    sa.Column('full_name', sa.String(length=255), nullable=True),
    sa.Column('is_active', sa.Boolean(), nullable=False),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.Column('updated_at', sa.DateTime(), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_users_active', 'users', ['is_active'], unique=False)
    op.create_index('ix_users_email', 'users', ['email'], unique=False)
    op.create_index('ix_users_username', 'users', ['username'], unique=False)
    op.create_unique_constraint(None, 'users', ['username'])
    op.create_unique_constraint(None, 'users', ['email'])

    # Create cluster_configs table
    op.create_table('cluster_configs',
    sa.Column('id', sa.String(), nullable=False),
    sa.Column('cluster_name', sa.String(length=255), nullable=False),
    sa.Column('cluster_host', sa.String(length=255), nullable=False),
    sa.Column('cluster_port', sa.Integer(), nullable=False),
    sa.Column('username', sa.String(length=255), nullable=False),
    sa.Column('private_key_path', sa.String(length=500), nullable=True),
    sa.Column('password_encrypted', sa.String(length=500), nullable=True),
    sa.Column('base_path', sa.String(length=500), nullable=False),
    sa.Column('max_concurrent_jobs', sa.Integer(), nullable=False),
    sa.Column('max_gpu_per_job', sa.Integer(), nullable=False),
    sa.Column('default_timeout_hours', sa.Integer(), nullable=False),
    sa.Column('is_active', sa.Boolean(), nullable=False),
    sa.Column('last_health_check', sa.DateTime(), nullable=True),
    sa.Column('health_status', sa.String(length=50), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.Column('updated_at', sa.DateTime(), nullable=False),
    sa.CheckConstraint('cluster_port > 0 AND cluster_port <= 65535', name='check_port_range'),
    sa.CheckConstraint('max_concurrent_jobs > 0', name='check_max_jobs'),
    sa.CheckConstraint('max_gpu_per_job > 0', name='check_max_gpu'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_cluster_configs_active', 'cluster_configs', ['is_active'], unique=False)
    op.create_index('ix_cluster_configs_name', 'cluster_configs', ['cluster_name'], unique=False)
    op.create_unique_constraint(None, 'cluster_configs', ['cluster_name'])

    # Create gpu_jobs table
    op.create_table('gpu_jobs',
    sa.Column('job_id', sa.String(length=255), nullable=False),
    sa.Column('user_id', sa.String(), nullable=False),
    sa.Column('job_name', sa.String(length=255), nullable=False),
    sa.Column('gpu_count', sa.Integer(), nullable=False),
    sa.Column('code_source', sa.String(length=50), nullable=False),
    sa.Column('entry_script', sa.String(length=500), nullable=False),
    sa.Column('git_url', sa.String(length=500), nullable=True),
    sa.Column('git_branch', sa.String(length=255), nullable=True),
    sa.Column('git_commit', sa.String(length=255), nullable=True),
    sa.Column('manual_files', sa.JSON(), nullable=True),
    sa.Column('model_files', sa.JSON(), nullable=True),
    sa.Column('data_files', sa.JSON(), nullable=True),
    sa.Column('environment_vars', sa.JSON(), nullable=True),
    sa.Column('python_packages', sa.JSON(), nullable=True),
    sa.Column('docker_image', sa.String(length=255), nullable=True),
    sa.Column('status', sa.String(length=50), nullable=False),
    sa.Column('cluster_job_id', sa.String(length=255), nullable=True),
    sa.Column('cluster_path', sa.String(length=500), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.Column('submitted_at', sa.DateTime(), nullable=True),
    sa.Column('started_at', sa.DateTime(), nullable=True),
    sa.Column('completed_at', sa.DateTime(), nullable=True),
    sa.Column('current_epoch', sa.Integer(), nullable=True),
    sa.Column('total_epochs', sa.Integer(), nullable=True),
    sa.Column('current_loss', sa.Float(), nullable=True),
    sa.Column('current_accuracy', sa.Float(), nullable=True),
    sa.Column('memory_usage_mb', sa.Float(), nullable=True),
    sa.Column('gpu_usage_percent', sa.Float(), nullable=True),
    sa.Column('error_message', sa.Text(), nullable=True),
    sa.Column('exit_code', sa.Integer(), nullable=True),
    sa.Column('final_metrics', sa.JSON(), nullable=True),
    sa.Column('output_size_mb', sa.Float(), nullable=True),
    sa.CheckConstraint('gpu_count >= 1 AND gpu_count <= 8', name='check_gpu_count'),
    sa.CheckConstraint('current_epoch >= 0', name='check_current_epoch'),
    sa.CheckConstraint('total_epochs >= 1', name='check_total_epochs'),
    sa.CheckConstraint('memory_usage_mb >= 0', name='check_memory_usage'),
    sa.CheckConstraint('gpu_usage_percent >= 0 AND gpu_usage_percent <= 100', name='check_gpu_usage'),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
    sa.PrimaryKeyConstraint('job_id')
    )
    op.create_index('ix_gpu_jobs_cluster_id', 'gpu_jobs', ['cluster_job_id'], unique=False)
    op.create_index('ix_gpu_jobs_created', 'gpu_jobs', ['created_at'], unique=False)
    op.create_index('ix_gpu_jobs_status', 'gpu_jobs', ['status'], unique=False)
    op.create_index('ix_gpu_jobs_user', 'gpu_jobs', ['user_id'], unique=False)

    # Create job_templates table
    op.create_table('job_templates',
    sa.Column('id', sa.String(), nullable=False),
    sa.Column('template_name', sa.String(length=255), nullable=False),
    sa.Column('description', sa.Text(), nullable=True),
    sa.Column('category', sa.String(length=100), nullable=False),
    sa.Column('default_gpu_count', sa.Integer(), nullable=False),
    sa.Column('default_code_source', sa.String(length=50), nullable=False),
    sa.Column('default_entry_script', sa.String(length=500), nullable=False),
    sa.Column('default_python_packages', sa.JSON(), nullable=True),
    sa.Column('default_environment_vars', sa.JSON(), nullable=True),
    sa.Column('default_docker_image', sa.String(length=255), nullable=True),
    sa.Column('is_public', sa.Boolean(), nullable=False),
    sa.Column('created_by', sa.String(), nullable=False),
    sa.Column('usage_count', sa.Integer(), nullable=False),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.Column('updated_at', sa.DateTime(), nullable=False),
    sa.CheckConstraint('default_gpu_count >= 1 AND default_gpu_count <= 8', name='check_template_gpu_count'),
    sa.ForeignKeyConstraint(['created_by'], ['users.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_job_templates_category', 'job_templates', ['category'], unique=False)
    op.create_index('ix_job_templates_creator', 'job_templates', ['created_by'], unique=False)
    op.create_index('ix_job_templates_name', 'job_templates', ['template_name'], unique=False)
    op.create_index('ix_job_templates_public', 'job_templates', ['is_public'], unique=False)
    op.create_unique_constraint(None, 'job_templates', ['template_name'])

    # Create job_results table
    op.create_table('job_results',
    sa.Column('id', sa.String(), nullable=False),
    sa.Column('job_id', sa.String(length=255), nullable=False),
    sa.Column('model_files', sa.JSON(), nullable=False),
    sa.Column('log_files', sa.JSON(), nullable=False),
    sa.Column('output_files', sa.JSON(), nullable=False),
    sa.Column('final_metrics', sa.JSON(), nullable=False),
    sa.Column('training_history', sa.JSON(), nullable=False),
    sa.Column('total_size_mb', sa.Float(), nullable=False),
    sa.Column('archive_path', sa.String(length=500), nullable=True),
    sa.Column('collected_at', sa.DateTime(), nullable=False),
    sa.Column('collection_duration_seconds', sa.Float(), nullable=True),
    sa.CheckConstraint('total_size_mb >= 0', name='check_total_size'),
    sa.ForeignKeyConstraint(['job_id'], ['gpu_jobs.job_id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_job_results_collected', 'job_results', ['collected_at'], unique=False)
    op.create_index('ix_job_results_job_id', 'job_results', ['job_id'], unique=False)
    op.create_unique_constraint(None, 'job_results', ['job_id'])

    # Create job_logs table
    op.create_table('job_logs',
    sa.Column('id', sa.String(), nullable=False),
    sa.Column('job_id', sa.String(length=255), nullable=False),
    sa.Column('log_type', sa.String(length=50), nullable=False),
    sa.Column('log_level', sa.String(length=20), nullable=False),
    sa.Column('timestamp', sa.DateTime(), nullable=False),
    sa.Column('message', sa.Text(), nullable=False),
    sa.Column('source', sa.String(length=255), nullable=True),
    sa.Column('metadata', sa.JSON(), nullable=True),
    sa.ForeignKeyConstraint(['job_id'], ['gpu_jobs.job_id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_job_logs_job_id', 'job_logs', ['job_id'], unique=False)
    op.create_index('ix_job_logs_level', 'job_logs', ['log_level'], unique=False)
    op.create_index('ix_job_logs_timestamp', 'job_logs', ['timestamp'], unique=False)
    op.create_index('ix_job_logs_type', 'job_logs', ['log_type'], unique=False)

    # Create job_metrics table
    op.create_table('job_metrics',
    sa.Column('id', sa.String(), nullable=False),
    sa.Column('job_id', sa.String(length=255), nullable=False),
    sa.Column('metric_name', sa.String(length=100), nullable=False),
    sa.Column('metric_value', sa.Float(), nullable=False),
    sa.Column('epoch', sa.Integer(), nullable=True),
    sa.Column('step', sa.Integer(), nullable=True),
    sa.Column('metric_type', sa.String(length=50), nullable=False),
    sa.Column('recorded_at', sa.DateTime(), nullable=False),
    sa.Column('metadata', sa.JSON(), nullable=True),
    sa.ForeignKeyConstraint(['job_id'], ['gpu_jobs.job_id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_job_metrics_epoch', 'job_metrics', ['epoch'], unique=False)
    op.create_index('ix_job_metrics_job', 'job_metrics', ['job_id'], unique=False)
    op.create_index('ix_job_metrics_job_epoch', 'job_metrics', ['job_id', 'epoch'], unique=False)
    op.create_index('ix_job_metrics_job_name', 'job_metrics', ['job_id', 'metric_name'], unique=False)
    op.create_index('ix_job_metrics_name', 'job_metrics', ['metric_name'], unique=False)
    op.create_index('ix_job_metrics_recorded', 'job_metrics', ['recorded_at'], unique=False)
    op.create_index('ix_job_metrics_type', 'job_metrics', ['metric_type'], unique=False)

    # Create job_queue table
    op.create_table('job_queue',
    sa.Column('id', sa.String(), nullable=False),
    sa.Column('job_id', sa.String(length=255), nullable=False),
    sa.Column('priority', sa.Integer(), nullable=False),
    sa.Column('queue_position', sa.Integer(), nullable=True),
    sa.Column('estimated_start_time', sa.DateTime(), nullable=True),
    sa.Column('estimated_duration_minutes', sa.Integer(), nullable=True),
    sa.Column('queue_status', sa.String(length=50), nullable=False),
    sa.Column('depends_on_jobs', sa.JSON(), nullable=True),
    sa.Column('queued_at', sa.DateTime(), nullable=False),
    sa.Column('scheduled_at', sa.DateTime(), nullable=True),
    sa.Column('started_at', sa.DateTime(), nullable=True),
    sa.Column('completed_at', sa.DateTime(), nullable=True),
    sa.CheckConstraint('priority >= 1 AND priority <= 10', name='check_priority_range'),
    sa.ForeignKeyConstraint(['job_id'], ['gpu_jobs.job_id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_job_queue_job', 'job_queue', ['job_id'], unique=False)
    op.create_index('ix_job_queue_priority', 'job_queue', ['priority'], unique=False)
    op.create_index('ix_job_queue_queued', 'job_queue', ['queued_at'], unique=False)
    op.create_index('ix_job_queue_status', 'job_queue', ['queue_status'], unique=False)

    # Create system_events table
    op.create_table('system_events',
    sa.Column('id', sa.String(), nullable=False),
    sa.Column('event_type', sa.String(length=100), nullable=False),
    sa.Column('event_level', sa.String(length=20), nullable=False),
    sa.Column('message', sa.Text(), nullable=False),
    sa.Column('job_id', sa.String(length=255), nullable=True),
    sa.Column('user_id', sa.String(), nullable=True),
    sa.Column('cluster_name', sa.String(length=255), nullable=True),
    sa.Column('event_data', sa.JSON(), nullable=True),
    sa.Column('timestamp', sa.DateTime(), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_system_events_job', 'system_events', ['job_id'], unique=False)
    op.create_index('ix_system_events_level', 'system_events', ['event_level'], unique=False)
    op.create_index('ix_system_events_timestamp', 'system_events', ['timestamp'], unique=False)
    op.create_index('ix_system_events_type', 'system_events', ['event_type'], unique=False)
    op.create_index('ix_system_events_user', 'system_events', ['user_id'], unique=False)


def downgrade() -> None:
    op.drop_table('system_events')
    op.drop_table('job_queue')
    op.drop_table('job_metrics')
    op.drop_table('job_logs')
    op.drop_table('job_results')
    op.drop_table('job_templates')
    op.drop_table('gpu_jobs')
    op.drop_table('cluster_configs')
    op.drop_table('users')