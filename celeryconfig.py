#!/usr/bin/env python3
# Celery Configuration

import os

# Broker settings
broker_url = os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0')
result_backend = os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0')

# Task settings
task_serializer = 'json'
accept_content = ['json']
result_serializer = 'json'
timezone = 'UTC'
enable_utc = True

# Worker settings
worker_prefetch_multiplier = 1
task_acks_late = True
worker_max_tasks_per_child = 1000

# Task routing
task_routes = {
    'tasks.process_match_async': {'queue': 'video_processing'},
    'tasks.generate_highlights': {'queue': 'video_processing'},
    'tasks.generate_match_report': {'queue': 'reports'},
}

# Task time limits
task_soft_time_limit = 300  # 5 minutes
task_time_limit = 600       # 10 minutes