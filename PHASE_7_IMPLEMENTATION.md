# Phase 7: Production Deployment & Optimization Implementation Plan

## Overview
This document outlines the final implementation phase for LL3M: Production Deployment & Optimization. This phase focuses on deploying the system to production, optimizing performance, implementing monitoring and observability, and ensuring enterprise-grade reliability and scalability.

## Phase 7 Objectives
- **Production Deployment**: Container orchestration, cloud deployment, and infrastructure as code
- **Performance Optimization**: System-wide performance tuning and resource optimization
- **Monitoring & Observability**: Comprehensive logging, metrics, alerting, and distributed tracing
- **Security Hardening**: Production security measures, secrets management, and compliance
- **Documentation & Training**: Complete system documentation and user guides
- **Maintenance & Operations**: Backup strategies, disaster recovery, and operational runbooks

## Architecture Overview

### Production Deployment Architecture
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Load Balancer │    │   API Gateway   │    │   Web UI        │
│   (nginx/AWS)   │    │   (Kong/Istio)  │    │   (React/Vue)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │   Kubernetes    │
                    │   Cluster       │
                    └─────────────────┘
                                 │
         ┌───────────────────────┼───────────────────────┐
         │                       │                       │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   LL3M API      │    │   Background    │    │   Monitoring    │
│   (FastAPI)     │    │   Workers       │    │   Stack         │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
         ┌───────────────────────┼───────────────────────┐
         │                       │                       │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   PostgreSQL    │    │   Redis Cache   │    │   Object Storage│
│   (Database)    │    │   (Session)     │    │   (Assets)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Implementation Tasks

### Task 1: Container Orchestration & Deployment
**Duration**: 3-4 days

#### 1.1 Docker Containerization
```dockerfile
# Dockerfile.api
FROM python:3.11-slim as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
WORKDIR /app
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    wget \
    xz-utils \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Blender
ENV BLENDER_VERSION=4.0
RUN wget -O blender.tar.xz https://download.blender.org/release/Blender${BLENDER_VERSION}/blender-${BLENDER_VERSION}.0-linux-x64.tar.xz \
    && tar -xf blender.tar.xz -C /opt \
    && mv /opt/blender-* /opt/blender \
    && ln -s /opt/blender/blender /usr/local/bin/blender \
    && rm blender.tar.xz

# Copy Python dependencies from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy application code
WORKDIR /app
COPY . .

# Create non-root user
RUN groupadd -r ll3m && useradd -r -g ll3m ll3m
RUN chown -R ll3m:ll3m /app
USER ll3m

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port and start
EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]

# Dockerfile.worker
FROM python:3.11-slim

# Same base setup as API...
COPY --from=ll3m-api:latest /app /app
WORKDIR /app

USER ll3m
CMD ["celery", "worker", "-A", "src.workers.main", "--loglevel=info", "--concurrency=2"]

# Dockerfile.ui
FROM node:18-alpine as builder

WORKDIR /app
COPY ui/package*.json ./
RUN npm ci --only=production

COPY ui/ .
RUN npm run build

FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
COPY ui/nginx.conf /etc/nginx/nginx.conf

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

#### 1.2 Kubernetes Deployment Configuration
```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: ll3m-production
  labels:
    environment: production
    app: ll3m

---
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: ll3m-config
  namespace: ll3m-production
data:
  BLENDER_PATH: "/usr/local/bin/blender"
  REDIS_URL: "redis://ll3m-redis:6379"
  DATABASE_URL: "postgresql://ll3m:${POSTGRES_PASSWORD}@ll3m-postgres:5432/ll3m"
  ENVIRONMENT: "production"
  LOG_LEVEL: "INFO"

---
# k8s/secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: ll3m-secrets
  namespace: ll3m-production
type: Opaque
data:
  OPENAI_API_KEY: # base64 encoded
  JWT_SECRET_KEY: # base64 encoded
  POSTGRES_PASSWORD: # base64 encoded

---
# k8s/api-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ll3m-api
  namespace: ll3m-production
  labels:
    app: ll3m-api
    tier: backend
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ll3m-api
  template:
    metadata:
      labels:
        app: ll3m-api
        tier: backend
    spec:
      containers:
      - name: ll3m-api
        image: ll3m/api:latest
        ports:
        - containerPort: 8000
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: ll3m-secrets
              key: OPENAI_API_KEY
        - name: JWT_SECRET_KEY
          valueFrom:
            secretKeyRef:
              name: ll3m-secrets
              key: JWT_SECRET_KEY
        envFrom:
        - configMapRef:
            name: ll3m-config
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5

---
# k8s/worker-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ll3m-worker
  namespace: ll3m-production
spec:
  replicas: 2
  selector:
    matchLabels:
      app: ll3m-worker
  template:
    metadata:
      labels:
        app: ll3m-worker
    spec:
      containers:
      - name: ll3m-worker
        image: ll3m/worker:latest
        envFrom:
        - configMapRef:
            name: ll3m-config
        - secretRef:
            name: ll3m-secrets
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "4Gi"
            cpu: "2000m"

---
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: ll3m-api-service
  namespace: ll3m-production
spec:
  selector:
    app: ll3m-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: ClusterIP

---
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ll3m-ingress
  namespace: ll3m-production
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
    nginx.ingress.kubernetes.io/rate-limit: "10"
spec:
  tls:
  - hosts:
    - api.ll3m.com
    - app.ll3m.com
    secretName: ll3m-tls
  rules:
  - host: api.ll3m.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: ll3m-api-service
            port:
              number: 80
  - host: app.ll3m.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: ll3m-ui-service
            port:
              number: 80
```

#### 1.3 Infrastructure as Code (Terraform)
```hcl
# terraform/main.tf
terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.20"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# EKS Cluster
module "eks" {
  source = "terraform-aws-modules/eks/aws"

  cluster_name    = "ll3m-production"
  cluster_version = "1.28"

  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets

  eks_managed_node_groups = {
    main = {
      desired_size = 3
      max_size     = 10
      min_size     = 2

      instance_types = ["m5.large", "m5.xlarge"]
      capacity_type  = "SPOT"

      k8s_labels = {
        Environment = "production"
        Application = "ll3m"
      }
    }

    gpu_nodes = {
      desired_size = 1
      max_size     = 3
      min_size     = 0

      instance_types = ["g4dn.xlarge"]
      capacity_type  = "ON_DEMAND"

      k8s_labels = {
        Environment = "production"
        Application = "ll3m"
        NodeType    = "gpu"
      }

      taints = [
        {
          key    = "gpu"
          value  = "true"
          effect = "NO_SCHEDULE"
        }
      ]
    }
  }

  tags = {
    Environment = "production"
    Application = "ll3m"
  }
}

# RDS PostgreSQL
resource "aws_db_instance" "ll3m_postgres" {
  identifier = "ll3m-production"

  engine         = "postgres"
  engine_version = "15.4"
  instance_class = "db.t3.medium"

  allocated_storage     = 100
  max_allocated_storage = 1000
  storage_encrypted     = true

  db_name  = "ll3m"
  username = "ll3m"
  password = var.db_password

  vpc_security_group_ids = [aws_security_group.rds.id]
  db_subnet_group_name   = aws_db_subnet_group.main.name

  backup_retention_period = 7
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"

  skip_final_snapshot = false
  final_snapshot_identifier = "ll3m-production-final-${formatdate("YYYY-MM-DD-hhmm", timestamp())}"

  tags = {
    Environment = "production"
    Application = "ll3m"
  }
}

# ElastiCache Redis
resource "aws_elasticache_replication_group" "ll3m_redis" {
  replication_group_id       = "ll3m-production"
  description                = "LL3M Redis cluster"

  node_type                  = "cache.t3.micro"
  port                       = 6379
  parameter_group_name       = "default.redis7"

  num_cache_clusters         = 2
  automatic_failover_enabled = true
  multi_az_enabled          = true

  subnet_group_name = aws_elasticache_subnet_group.main.name
  security_group_ids = [aws_security_group.redis.id]

  at_rest_encryption_enabled = true
  transit_encryption_enabled = true

  tags = {
    Environment = "production"
    Application = "ll3m"
  }
}

# S3 Bucket for Asset Storage
resource "aws_s3_bucket" "ll3m_assets" {
  bucket = "ll3m-production-assets-${random_id.bucket_suffix.hex}"

  tags = {
    Environment = "production"
    Application = "ll3m"
  }
}

resource "aws_s3_bucket_encryption_configuration" "ll3m_assets" {
  bucket = aws_s3_bucket.ll3m_assets.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_versioning" "ll3m_assets" {
  bucket = aws_s3_bucket.ll3m_assets.id
  versioning_configuration {
    status = "Enabled"
  }
}

# CloudFront Distribution
resource "aws_cloudfront_distribution" "ll3m_assets" {
  origin {
    domain_name = aws_s3_bucket.ll3m_assets.bucket_regional_domain_name
    origin_id   = "S3-${aws_s3_bucket.ll3m_assets.id}"

    s3_origin_config {
      origin_access_identity = aws_cloudfront_origin_access_identity.ll3m_assets.cloudfront_access_identity_path
    }
  }

  enabled = true

  default_cache_behavior {
    allowed_methods        = ["DELETE", "GET", "HEAD", "OPTIONS", "PATCH", "POST", "PUT"]
    cached_methods         = ["GET", "HEAD"]
    target_origin_id       = "S3-${aws_s3_bucket.ll3m_assets.id}"
    compress              = true
    viewer_protocol_policy = "redirect-to-https"

    forwarded_values {
      query_string = false
      cookies {
        forward = "none"
      }
    }

    min_ttl     = 0
    default_ttl = 3600
    max_ttl     = 86400
  }

  restrictions {
    geo_restriction {
      restriction_type = "none"
    }
  }

  viewer_certificate {
    cloudfront_default_certificate = true
  }

  tags = {
    Environment = "production"
    Application = "ll3m"
  }
}

# Variables
variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-west-2"
}

variable "db_password" {
  description = "PostgreSQL password"
  type        = string
  sensitive   = true
}

# Outputs
output "cluster_endpoint" {
  value = module.eks.cluster_endpoint
}

output "database_endpoint" {
  value = aws_db_instance.ll3m_postgres.endpoint
}

output "redis_endpoint" {
  value = aws_elasticache_replication_group.ll3m_redis.primary_endpoint_address
}

output "assets_bucket" {
  value = aws_s3_bucket.ll3m_assets.bucket
}

output "cloudfront_domain" {
  value = aws_cloudfront_distribution.ll3m_assets.domain_name
}
```

### Task 2: Performance Optimization
**Duration**: 2-3 days

#### 2.1 System-Wide Performance Tuning
```python
# src/optimization/performance_tuner.py
import asyncio
import time
import psutil
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import structlog

logger = structlog.get_logger(__name__)

@dataclass
class PerformanceMetrics:
    """Performance metrics snapshot."""
    cpu_percent: float
    memory_percent: float
    disk_io: Dict[str, float]
    network_io: Dict[str, float]
    response_time: float
    throughput: float
    error_rate: float
    timestamp: float

class PerformanceTuner:
    """System performance optimizer."""

    def __init__(self):
        self.metrics_history: List[PerformanceMetrics] = []
        self.optimization_rules = {
            'high_cpu': self._optimize_cpu_usage,
            'high_memory': self._optimize_memory_usage,
            'slow_response': self._optimize_response_time,
            'low_throughput': self._optimize_throughput
        }

    async def collect_metrics(self) -> PerformanceMetrics:
        """Collect current system performance metrics."""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk_io = psutil.disk_io_counters()._asdict() if psutil.disk_io_counters() else {}
        network_io = psutil.net_io_counters()._asdict() if psutil.net_io_counters() else {}

        # Application-specific metrics
        response_time = await self._measure_response_time()
        throughput = await self._measure_throughput()
        error_rate = await self._measure_error_rate()

        metrics = PerformanceMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            disk_io={k: v for k, v in disk_io.items()},
            network_io={k: v for k, v in network_io.items()},
            response_time=response_time,
            throughput=throughput,
            error_rate=error_rate,
            timestamp=time.time()
        )

        self.metrics_history.append(metrics)

        # Keep only last 100 metrics
        if len(self.metrics_history) > 100:
            self.metrics_history = self.metrics_history[-100:]

        return metrics

    async def analyze_and_optimize(self) -> Dict[str, Any]:
        """Analyze performance and apply optimizations."""
        current_metrics = await self.collect_metrics()

        optimizations_applied = []

        # Check for performance issues
        if current_metrics.cpu_percent > 80:
            result = await self._optimize_cpu_usage(current_metrics)
            optimizations_applied.append(('high_cpu', result))

        if current_metrics.memory_percent > 85:
            result = await self._optimize_memory_usage(current_metrics)
            optimizations_applied.append(('high_memory', result))

        if current_metrics.response_time > 5.0:  # 5 second threshold
            result = await self._optimize_response_time(current_metrics)
            optimizations_applied.append(('slow_response', result))

        if current_metrics.throughput < 1.0:  # 1 request/second minimum
            result = await self._optimize_throughput(current_metrics)
            optimizations_applied.append(('low_throughput', result))

        return {
            'current_metrics': current_metrics,
            'optimizations_applied': optimizations_applied,
            'recommendations': self._generate_recommendations(current_metrics)
        }

    async def _optimize_cpu_usage(self, metrics: PerformanceMetrics) -> Dict[str, Any]:
        """Optimize CPU usage."""
        optimizations = []

        # Reduce worker processes if needed
        current_workers = await self._get_worker_count()
        if current_workers > 2:
            new_workers = max(2, current_workers - 1)
            await self._set_worker_count(new_workers)
            optimizations.append(f"Reduced workers from {current_workers} to {new_workers}")

        # Enable request throttling
        await self._enable_request_throttling(max_requests_per_minute=60)
        optimizations.append("Enabled request throttling")

        # Optimize background task processing
        await self._optimize_background_tasks()
        optimizations.append("Optimized background task processing")

        return {
            'success': True,
            'optimizations': optimizations,
            'estimated_improvement': '15-25% CPU reduction'
        }

    async def _optimize_memory_usage(self, metrics: PerformanceMetrics) -> Dict[str, Any]:
        """Optimize memory usage."""
        optimizations = []

        # Clear caches
        await self._clear_application_caches()
        optimizations.append("Cleared application caches")

        # Force garbage collection
        import gc
        collected = gc.collect()
        optimizations.append(f"Garbage collected {collected} objects")

        # Reduce cache sizes
        await self._reduce_cache_sizes()
        optimizations.append("Reduced cache sizes")

        # Limit concurrent operations
        await self._limit_concurrent_operations(max_concurrent=5)
        optimizations.append("Limited concurrent operations")

        return {
            'success': True,
            'optimizations': optimizations,
            'estimated_improvement': '20-30% memory reduction'
        }

    async def _optimize_response_time(self, metrics: PerformanceMetrics) -> Dict[str, Any]:
        """Optimize response time."""
        optimizations = []

        # Enable response compression
        await self._enable_response_compression()
        optimizations.append("Enabled response compression")

        # Optimize database queries
        await self._optimize_database_queries()
        optimizations.append("Optimized database queries")

        # Increase connection pool size
        await self._increase_connection_pool()
        optimizations.append("Increased connection pool size")

        # Enable result caching
        await self._enable_result_caching()
        optimizations.append("Enabled result caching")

        return {
            'success': True,
            'optimizations': optimizations,
            'estimated_improvement': '30-50% response time reduction'
        }

    def _generate_recommendations(self, metrics: PerformanceMetrics) -> List[str]:
        """Generate performance recommendations."""
        recommendations = []

        if metrics.cpu_percent > 70:
            recommendations.append("Consider horizontal scaling (add more instances)")
            recommendations.append("Review CPU-intensive operations for optimization")

        if metrics.memory_percent > 75:
            recommendations.append("Consider increasing memory allocation")
            recommendations.append("Review memory usage patterns for leaks")

        if metrics.response_time > 3.0:
            recommendations.append("Implement caching for frequently accessed data")
            recommendations.append("Optimize database queries and add indexes")

        if metrics.error_rate > 0.05:  # 5% error rate
            recommendations.append("Investigate and fix sources of errors")
            recommendations.append("Implement better error handling and retry logic")

        return recommendations

# src/optimization/caching_optimizer.py
class CachingOptimizer:
    """Advanced caching optimization system."""

    def __init__(self):
        self.cache_stats = {}
        self.cache_strategies = {
            'lru': self._implement_lru_cache,
            'ttl': self._implement_ttl_cache,
            'adaptive': self._implement_adaptive_cache
        }

    async def optimize_caching_strategy(self) -> Dict[str, Any]:
        """Analyze usage patterns and optimize caching strategy."""

        # Analyze cache hit rates
        cache_analysis = await self._analyze_cache_performance()

        # Implement optimal caching strategy
        optimal_strategy = self._determine_optimal_strategy(cache_analysis)

        optimization_result = await self._apply_caching_strategy(optimal_strategy)

        return {
            'current_analysis': cache_analysis,
            'optimal_strategy': optimal_strategy,
            'optimization_result': optimization_result
        }

    async def _analyze_cache_performance(self) -> Dict[str, Any]:
        """Analyze current cache performance."""
        from src.utils.cache import CacheManager

        cache_manager = CacheManager()

        hit_rate = await cache_manager.get_hit_rate()
        miss_rate = await cache_manager.get_miss_rate()
        eviction_rate = await cache_manager.get_eviction_rate()
        memory_usage = await cache_manager.get_memory_usage()

        return {
            'hit_rate': hit_rate,
            'miss_rate': miss_rate,
            'eviction_rate': eviction_rate,
            'memory_usage': memory_usage,
            'total_keys': await cache_manager.get_total_keys(),
            'average_ttl': await cache_manager.get_average_ttl()
        }

    def _determine_optimal_strategy(self, analysis: Dict[str, Any]) -> str:
        """Determine optimal caching strategy based on analysis."""
        hit_rate = analysis.get('hit_rate', 0)
        eviction_rate = analysis.get('eviction_rate', 0)
        memory_usage = analysis.get('memory_usage', 0)

        if hit_rate > 0.8 and eviction_rate < 0.1:
            return 'current'  # Current strategy is working well
        elif memory_usage > 0.9:
            return 'lru'  # Need memory-conscious strategy
        elif hit_rate < 0.5:
            return 'adaptive'  # Need smarter caching
        else:
            return 'ttl'  # Time-based strategy
```

### Task 3: Monitoring & Observability
**Duration**: 2-3 days

#### 3.1 Comprehensive Monitoring Stack
```python
# src/monitoring/metrics_collector.py
import time
import asyncio
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry
import structlog

logger = structlog.get_logger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter('ll3m_requests_total', 'Total requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('ll3m_request_duration_seconds', 'Request duration', ['method', 'endpoint'])
ACTIVE_CONNECTIONS = Gauge('ll3m_active_connections', 'Active connections')
WORKFLOW_DURATION = Histogram('ll3m_workflow_duration_seconds', 'Workflow execution time', ['workflow_type'])
ASSET_GENERATION_COUNT = Counter('ll3m_assets_generated_total', 'Assets generated', ['quality_level'])
ERROR_COUNT = Counter('ll3m_errors_total', 'Total errors', ['error_type', 'component'])

@dataclass
class WorkflowMetrics:
    """Workflow execution metrics."""
    workflow_id: str
    workflow_type: str
    start_time: float
    end_time: Optional[float]
    duration: Optional[float]
    success: bool
    agent_timings: Dict[str, float]
    resource_usage: Dict[str, float]
    error_details: Optional[Dict[str, Any]]

class MetricsCollector:
    """Comprehensive metrics collection system."""

    def __init__(self):
        self.workflow_metrics: Dict[str, WorkflowMetrics] = {}
        self.system_metrics_history: List[Dict[str, Any]] = []

    async def start_workflow_tracking(self, workflow_id: str, workflow_type: str) -> None:
        """Start tracking a workflow execution."""
        self.workflow_metrics[workflow_id] = WorkflowMetrics(
            workflow_id=workflow_id,
            workflow_type=workflow_type,
            start_time=time.time(),
            end_time=None,
            duration=None,
            success=False,
            agent_timings={},
            resource_usage={},
            error_details=None
        )

        logger.info("Started workflow tracking", workflow_id=workflow_id, workflow_type=workflow_type)

    async def record_agent_timing(self, workflow_id: str, agent_name: str, duration: float) -> None:
        """Record agent execution timing."""
        if workflow_id in self.workflow_metrics:
            self.workflow_metrics[workflow_id].agent_timings[agent_name] = duration

    async def complete_workflow_tracking(
        self,
        workflow_id: str,
        success: bool,
        error_details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Complete workflow tracking."""
        if workflow_id not in self.workflow_metrics:
            logger.warning("Workflow not found for completion", workflow_id=workflow_id)
            return

        metrics = self.workflow_metrics[workflow_id]
        metrics.end_time = time.time()
        metrics.duration = metrics.end_time - metrics.start_time
        metrics.success = success
        metrics.error_details = error_details

        # Record Prometheus metrics
        WORKFLOW_DURATION.labels(workflow_type=metrics.workflow_type).observe(metrics.duration)

        if not success and error_details:
            ERROR_COUNT.labels(
                error_type=error_details.get('type', 'unknown'),
                component=error_details.get('component', 'workflow')
            ).inc()

        logger.info(
            "Completed workflow tracking",
            workflow_id=workflow_id,
            duration=metrics.duration,
            success=success
        )

    async def collect_system_metrics(self) -> Dict[str, Any]:
        """Collect comprehensive system metrics."""
        import psutil

        # System metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

        # Application metrics
        active_workflows = len([m for m in self.workflow_metrics.values() if m.end_time is None])
        completed_workflows = len([m for m in self.workflow_metrics.values() if m.end_time is not None])
        success_rate = 0.0

        if completed_workflows > 0:
            successful_workflows = len([m for m in self.workflow_metrics.values()
                                      if m.end_time is not None and m.success])
            success_rate = successful_workflows / completed_workflows

        metrics = {
            'timestamp': time.time(),
            'system': {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_available_gb': memory.available / (1024**3),
                'disk_percent': (disk.used / disk.total) * 100,
                'disk_free_gb': disk.free / (1024**3)
            },
            'application': {
                'active_workflows': active_workflows,
                'completed_workflows': completed_workflows,
                'success_rate': success_rate,
                'average_workflow_duration': self._calculate_average_duration()
            },
            'database': await self._collect_database_metrics(),
            'cache': await self._collect_cache_metrics()
        }

        self.system_metrics_history.append(metrics)

        # Keep only last 1000 metrics
        if len(self.system_metrics_history) > 1000:
            self.system_metrics_history = self.system_metrics_history[-1000:]

        return metrics

    def _calculate_average_duration(self) -> float:
        """Calculate average workflow duration."""
        completed_workflows = [m for m in self.workflow_metrics.values()
                             if m.duration is not None]

        if not completed_workflows:
            return 0.0

        total_duration = sum(m.duration for m in completed_workflows)
        return total_duration / len(completed_workflows)

    async def _collect_database_metrics(self) -> Dict[str, Any]:
        """Collect database performance metrics."""
        # This would integrate with your database monitoring
        return {
            'active_connections': 0,
            'queries_per_second': 0,
            'average_query_time': 0,
            'cache_hit_rate': 0
        }

    async def _collect_cache_metrics(self) -> Dict[str, Any]:
        """Collect cache performance metrics."""
        # This would integrate with your cache monitoring
        return {
            'hit_rate': 0,
            'miss_rate': 0,
            'memory_usage_mb': 0,
            'total_keys': 0
        }

# src/monitoring/alerting.py
from typing import List, Dict, Any, Callable
from enum import Enum
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

class AlertSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class Alert:
    """Alert definition."""
    name: str
    severity: AlertSeverity
    condition: Callable[[Dict[str, Any]], bool]
    message_template: str
    cooldown_minutes: int = 5
    last_triggered: Optional[float] = None

class AlertManager:
    """Comprehensive alerting system."""

    def __init__(self, smtp_config: Dict[str, Any]):
        self.smtp_config = smtp_config
        self.alerts: List[Alert] = []
        self.setup_default_alerts()

    def setup_default_alerts(self):
        """Setup default system alerts."""
        self.alerts = [
            Alert(
                name="High CPU Usage",
                severity=AlertSeverity.HIGH,
                condition=lambda m: m.get('system', {}).get('cpu_percent', 0) > 90,
                message_template="CPU usage is {cpu_percent:.1f}%, exceeding 90% threshold",
                cooldown_minutes=10
            ),
            Alert(
                name="High Memory Usage",
                severity=AlertSeverity.HIGH,
                condition=lambda m: m.get('system', {}).get('memory_percent', 0) > 90,
                message_template="Memory usage is {memory_percent:.1f}%, exceeding 90% threshold",
                cooldown_minutes=10
            ),
            Alert(
                name="Low Success Rate",
                severity=AlertSeverity.MEDIUM,
                condition=lambda m: m.get('application', {}).get('success_rate', 1.0) < 0.8,
                message_template="Workflow success rate is {success_rate:.2%}, below 80% threshold",
                cooldown_minutes=15
            ),
            Alert(
                name="Disk Space Low",
                severity=AlertSeverity.CRITICAL,
                condition=lambda m: m.get('system', {}).get('disk_percent', 0) > 95,
                message_template="Disk usage is {disk_percent:.1f}%, exceeding 95% threshold",
                cooldown_minutes=5
            )
        ]

    async def check_alerts(self, metrics: Dict[str, Any]) -> List[str]:
        """Check all alerts against current metrics."""
        triggered_alerts = []
        current_time = time.time()

        for alert in self.alerts:
            # Check cooldown
            if (alert.last_triggered and
                current_time - alert.last_triggered < alert.cooldown_minutes * 60):
                continue

            # Check condition
            if alert.condition(metrics):
                alert.last_triggered = current_time
                message = alert.message_template.format(**metrics.get('system', {}),
                                                       **metrics.get('application', {}))

                await self._send_alert(alert, message, metrics)
                triggered_alerts.append(alert.name)

        return triggered_alerts

    async def _send_alert(self, alert: Alert, message: str, metrics: Dict[str, Any]) -> None:
        """Send alert notification."""
        try:
            # Email notification
            await self._send_email_alert(alert, message, metrics)

            # Could also integrate with Slack, PagerDuty, etc.
            # await self._send_slack_alert(alert, message)

            logger.warning("Alert triggered", alert=alert.name, severity=alert.severity.value, message=message)

        except Exception as e:
            logger.error("Failed to send alert", alert=alert.name, error=str(e))

    async def _send_email_alert(self, alert: Alert, message: str, metrics: Dict[str, Any]) -> None:
        """Send email alert."""
        msg = MIMEMultipart()
        msg['From'] = self.smtp_config['sender']
        msg['To'] = ', '.join(self.smtp_config['recipients'])
        msg['Subject'] = f"LL3M Alert [{alert.severity.value.upper()}]: {alert.name}"

        body = f"""
        Alert: {alert.name}
        Severity: {alert.severity.value.upper()}
        Time: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}

        Message: {message}

        Current Metrics:
        - CPU Usage: {metrics.get('system', {}).get('cpu_percent', 0):.1f}%
        - Memory Usage: {metrics.get('system', {}).get('memory_percent', 0):.1f}%
        - Active Workflows: {metrics.get('application', {}).get('active_workflows', 0)}
        - Success Rate: {metrics.get('application', {}).get('success_rate', 0):.2%}

        Please investigate and take appropriate action.
        """

        msg.attach(MIMEText(body, 'plain'))

        server = smtplib.SMTP(self.smtp_config['host'], self.smtp_config['port'])
        server.starttls()
        server.login(self.smtp_config['username'], self.smtp_config['password'])
        server.send_message(msg)
        server.quit()

# src/monitoring/dashboard.py
from fastapi import APIRouter, Depends
from fastapi.responses import HTMLResponse
import json

router = APIRouter()

@router.get("/dashboard", response_class=HTMLResponse)
async def monitoring_dashboard():
    """Serve monitoring dashboard."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>LL3M Monitoring Dashboard</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
            .metric-card { border: 1px solid #ddd; padding: 15px; border-radius: 8px; }
            .metric-value { font-size: 2em; font-weight: bold; color: #2196F3; }
            .chart-container { height: 300px; margin-top: 20px; }
        </style>
    </head>
    <body>
        <h1>LL3M System Monitoring</h1>

        <div class="metrics-grid">
            <div class="metric-card">
                <h3>CPU Usage</h3>
                <div class="metric-value" id="cpu-usage">---%</div>
            </div>
            <div class="metric-card">
                <h3>Memory Usage</h3>
                <div class="metric-value" id="memory-usage">---%</div>
            </div>
            <div class="metric-card">
                <h3>Active Workflows</h3>
                <div class="metric-value" id="active-workflows">---</div>
            </div>
            <div class="metric-card">
                <h3>Success Rate</h3>
                <div class="metric-value" id="success-rate">---%</div>
            </div>
        </div>

        <div class="chart-container">
            <canvas id="performance-chart"></canvas>
        </div>

        <script>
            // Real-time dashboard updates
            async function updateMetrics() {
                try {
                    const response = await fetch('/api/metrics');
                    const metrics = await response.json();

                    document.getElementById('cpu-usage').textContent = metrics.system.cpu_percent.toFixed(1) + '%';
                    document.getElementById('memory-usage').textContent = metrics.system.memory_percent.toFixed(1) + '%';
                    document.getElementById('active-workflows').textContent = metrics.application.active_workflows;
                    document.getElementById('success-rate').textContent = (metrics.application.success_rate * 100).toFixed(1) + '%';

                } catch (error) {
                    console.error('Failed to update metrics:', error);
                }
            }

            // Initialize chart
            const ctx = document.getElementById('performance-chart').getContext('2d');
            const chart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'CPU Usage (%)',
                        data: [],
                        borderColor: 'rgb(255, 99, 132)',
                        tension: 0.1
                    }, {
                        label: 'Memory Usage (%)',
                        data: [],
                        borderColor: 'rgb(54, 162, 235)',
                        tension: 0.1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 100
                        }
                    }
                }
            });

            // Update every 5 seconds
            setInterval(updateMetrics, 5000);
            updateMetrics(); // Initial load
        </script>
    </body>
    </html>
    """

@router.get("/api/metrics")
async def get_metrics():
    """API endpoint for current metrics."""
    metrics_collector = MetricsCollector()
    return await metrics_collector.collect_system_metrics()
```

### Task 4: Security Hardening
**Duration**: 2-3 days

#### 4.1 Production Security Configuration
```python
# src/security/security_manager.py
import hashlib
import secrets
import jwt
import bcrypt
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from cryptography.fernet import Fernet
import structlog

logger = structlog.get_logger(__name__)

class SecurityManager:
    """Comprehensive security management."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.encryption_key = self._get_or_generate_encryption_key()
        self.cipher_suite = Fernet(self.encryption_key)

        # Security policies
        self.password_policy = {
            'min_length': 12,
            'require_uppercase': True,
            'require_lowercase': True,
            'require_numbers': True,
            'require_special': True,
            'max_age_days': 90
        }

        self.rate_limits = {
            'login_attempts': {'limit': 5, 'window': 900},  # 5 attempts per 15 minutes
            'api_calls': {'limit': 1000, 'window': 3600},  # 1000 calls per hour
            'generation_requests': {'limit': 50, 'window': 3600}  # 50 generations per hour
        }

    def _get_or_generate_encryption_key(self) -> bytes:
        """Get or generate encryption key."""
        key_file = self.config.get('encryption_key_file', '/etc/ll3m/encryption.key')

        try:
            with open(key_file, 'rb') as f:
                return f.read()
        except FileNotFoundError:
            # Generate new key
            key = Fernet.generate_key()

            # Save securely
            import os
            os.makedirs(os.path.dirname(key_file), exist_ok=True)
            with open(key_file, 'wb') as f:
                f.write(key)
            os.chmod(key_file, 0o600)  # Read-only for owner

            logger.info("Generated new encryption key", key_file=key_file)
            return key

    def encrypt_sensitive_data(self, data: str) -> str:
        """Encrypt sensitive data."""
        encrypted_data = self.cipher_suite.encrypt(data.encode())
        return encrypted_data.decode()

    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data."""
        decrypted_data = self.cipher_suite.decrypt(encrypted_data.encode())
        return decrypted_data.decode()

    def hash_password(self, password: str) -> str:
        """Hash password with salt."""
        salt = bcrypt.gensalt(rounds=12)
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')

    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash."""
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

    def validate_password_strength(self, password: str) -> Dict[str, Any]:
        """Validate password against security policy."""
        issues = []

        if len(password) < self.password_policy['min_length']:
            issues.append(f"Password must be at least {self.password_policy['min_length']} characters")

        if self.password_policy['require_uppercase'] and not any(c.isupper() for c in password):
            issues.append("Password must contain at least one uppercase letter")

        if self.password_policy['require_lowercase'] and not any(c.islower() for c in password):
            issues.append("Password must contain at least one lowercase letter")

        if self.password_policy['require_numbers'] and not any(c.isdigit() for c in password):
            issues.append("Password must contain at least one number")

        if self.password_policy['require_special'] and not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
            issues.append("Password must contain at least one special character")

        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'strength_score': self._calculate_password_strength(password)
        }

    def _calculate_password_strength(self, password: str) -> int:
        """Calculate password strength score (0-100)."""
        score = 0

        # Length bonus
        score += min(25, len(password) * 2)

        # Character variety bonus
        if any(c.isupper() for c in password):
            score += 15
        if any(c.islower() for c in password):
            score += 15
        if any(c.isdigit() for c in password):
            score += 15
        if any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
            score += 20

        # Penalty for common patterns
        if password.lower() in ['password', '123456', 'qwerty', 'admin']:
            score -= 50

        return max(0, min(100, score))

    def generate_secure_token(self, length: int = 32) -> str:
        """Generate cryptographically secure token."""
        return secrets.token_urlsafe(length)

    def create_jwt_token(self, payload: Dict[str, Any], expires_hours: int = 24) -> str:
        """Create JWT token with expiration."""
        payload['exp'] = datetime.utcnow() + timedelta(hours=expires_hours)
        payload['iat'] = datetime.utcnow()

        return jwt.encode(
            payload,
            self.config['jwt_secret_key'],
            algorithm='HS256'
        )

    def verify_jwt_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode JWT token."""
        try:
            payload = jwt.decode(
                token,
                self.config['jwt_secret_key'],
                algorithms=['HS256']
            )
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("JWT token expired")
            return None
        except jwt.InvalidTokenError:
            logger.warning("Invalid JWT token")
            return None

# src/security/input_validator.py
import re
import html
from typing import Any, Dict, List, Union
import bleach

class InputValidator:
    """Secure input validation and sanitization."""

    def __init__(self):
        self.allowed_html_tags = ['b', 'i', 'em', 'strong', 'p', 'br']
        self.dangerous_patterns = [
            r'<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>',  # Script tags
            r'javascript:',  # JavaScript URLs
            r'vbscript:',    # VBScript URLs
            r'onload=',      # Event handlers
            r'onerror=',
            r'onclick=',
            r'eval\s*\(',    # eval() calls
            r'exec\s*\(',    # exec() calls
            r'import\s+os',  # OS imports
            r'subprocess',   # Subprocess calls
        ]

    def validate_prompt(self, prompt: str) -> Dict[str, Any]:
        """Validate and sanitize user prompt."""
        if not isinstance(prompt, str):
            return {'valid': False, 'error': 'Prompt must be a string'}

        # Length validation
        if len(prompt) > 2000:
            return {'valid': False, 'error': 'Prompt too long (max 2000 characters)'}

        if len(prompt.strip()) < 3:
            return {'valid': False, 'error': 'Prompt too short (min 3 characters)'}

        # Security validation
        for pattern in self.dangerous_patterns:
            if re.search(pattern, prompt, re.IGNORECASE):
                return {'valid': False, 'error': 'Prompt contains potentially dangerous content'}

        # Sanitize HTML
        sanitized_prompt = bleach.clean(
            prompt,
            tags=self.allowed_html_tags,
            strip=True
        )

        # Additional sanitization
        sanitized_prompt = html.escape(sanitized_prompt)

        return {
            'valid': True,
            'sanitized_prompt': sanitized_prompt,
            'original_length': len(prompt),
            'sanitized_length': len(sanitized_prompt)
        }

    def validate_file_upload(self, filename: str, file_content: bytes) -> Dict[str, Any]:
        """Validate uploaded file."""
        allowed_extensions = {'.blend', '.obj', '.fbx', '.gltf', '.png', '.jpg', '.jpeg'}
        max_file_size = 100 * 1024 * 1024  # 100MB

        # Extension validation
        file_extension = '.' + filename.split('.')[-1].lower() if '.' in filename else ''
        if file_extension not in allowed_extensions:
            return {'valid': False, 'error': f'File type not allowed: {file_extension}'}

        # Size validation
        if len(file_content) > max_file_size:
            return {'valid': False, 'error': 'File too large (max 100MB)'}

        # Content validation (basic magic number check)
        if not self._validate_file_content(file_content, file_extension):
            return {'valid': False, 'error': 'File content does not match extension'}

        return {'valid': True, 'file_size': len(file_content)}

    def _validate_file_content(self, content: bytes, extension: str) -> bool:
        """Validate file content matches extension."""
        magic_numbers = {
            '.png': [b'\x89PNG\r\n\x1a\n'],
            '.jpg': [b'\xff\xd8\xff', b'\xff\xd8'],
            '.jpeg': [b'\xff\xd8\xff', b'\xff\xd8'],
            '.blend': [b'BLENDER'],  # Blender files start with BLENDER
        }

        if extension in magic_numbers:
            return any(content.startswith(magic) for magic in magic_numbers[extension])

        return True  # Allow other formats for now

# src/security/secrets_manager.py
import os
import boto3
from typing import Dict, Any, Optional
import structlog

logger = structlog.get_logger(__name__)

class SecretsManager:
    """Secure secrets management."""

    def __init__(self, provider: str = 'aws'):
        self.provider = provider

        if provider == 'aws':
            self.client = boto3.client('secretsmanager')
        # Could add support for Azure Key Vault, HashiCorp Vault, etc.

    def get_secret(self, secret_name: str) -> Optional[str]:
        """Retrieve secret from secure storage."""
        try:
            if self.provider == 'aws':
                response = self.client.get_secret_value(SecretId=secret_name)
                return response['SecretString']
            else:
                # Fallback to environment variables (less secure)
                return os.getenv(secret_name)

        except Exception as e:
            logger.error("Failed to retrieve secret", secret_name=secret_name, error=str(e))
            return None

    def get_database_credentials(self) -> Dict[str, str]:
        """Get database credentials."""
        credentials = {}

        credentials['host'] = self.get_secret('DB_HOST') or 'localhost'
        credentials['port'] = self.get_secret('DB_PORT') or '5432'
        credentials['database'] = self.get_secret('DB_NAME') or 'll3m'
        credentials['username'] = self.get_secret('DB_USERNAME') or 'll3m'
        credentials['password'] = self.get_secret('DB_PASSWORD') or ''

        return credentials

    def get_api_keys(self) -> Dict[str, str]:
        """Get external API keys."""
        return {
            'openai_api_key': self.get_secret('OPENAI_API_KEY') or '',
            'jwt_secret_key': self.get_secret('JWT_SECRET_KEY') or '',
        }

    def rotate_secret(self, secret_name: str, new_value: str) -> bool:
        """Rotate a secret value."""
        try:
            if self.provider == 'aws':
                self.client.update_secret(
                    SecretId=secret_name,
                    SecretString=new_value
                )
                logger.info("Secret rotated successfully", secret_name=secret_name)
                return True
            else:
                logger.warning("Secret rotation not supported for provider", provider=self.provider)
                return False

        except Exception as e:
            logger.error("Failed to rotate secret", secret_name=secret_name, error=str(e))
            return False
```

### Task 5: Documentation & Training
**Duration**: 2-3 days

#### 5.1 Comprehensive Documentation System
```python
# docs/generate_docs.py
#!/usr/bin/env python3
"""
Automated documentation generator for LL3M system.
"""

import os
import ast
import inspect
from pathlib import Path
from typing import Dict, List, Any, Optional
import markdown
import jinja2
from dataclasses import dataclass

@dataclass
class APIEndpoint:
    """API endpoint documentation."""
    path: str
    method: str
    description: str
    parameters: List[Dict[str, Any]]
    responses: List[Dict[str, Any]]
    examples: List[Dict[str, str]]

@dataclass
class ClassDoc:
    """Class documentation."""
    name: str
    description: str
    methods: List[Dict[str, Any]]
    properties: List[Dict[str, Any]]
    examples: List[str]

class DocumentationGenerator:
    """Automated documentation generator."""

    def __init__(self, source_dir: str, output_dir: str):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Setup Jinja2 environment
        self.jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader('docs/templates'),
            autoescape=jinja2.select_autoescape(['html', 'xml'])
        )

    def generate_all_documentation(self) -> None:
        """Generate complete documentation suite."""
        print("Generating comprehensive LL3M documentation...")

        # Generate API documentation
        self.generate_api_documentation()

        # Generate class documentation
        self.generate_class_documentation()

        # Generate user guides
        self.generate_user_guides()

        # Generate deployment guides
        self.generate_deployment_guides()

        # Generate troubleshooting guides
        self.generate_troubleshooting_guides()

        # Generate index page
        self.generate_index_page()

        print(f"Documentation generated in {self.output_dir}")

    def generate_api_documentation(self) -> None:
        """Generate API documentation from FastAPI routes."""
        api_endpoints = self._extract_api_endpoints()

        template = self.jinja_env.get_template('api_docs.html')
        content = template.render(endpoints=api_endpoints)

        with open(self.output_dir / 'api_documentation.html', 'w') as f:
            f.write(content)

    def generate_class_documentation(self) -> None:
        """Generate class documentation from source code."""
        classes = self._extract_class_documentation()

        template = self.jinja_env.get_template('class_docs.html')
        content = template.render(classes=classes)

        with open(self.output_dir / 'class_documentation.html', 'w') as f:
            f.write(content)

    def generate_user_guides(self) -> None:
        """Generate user guides and tutorials."""
        guides = {
            'quick_start': self._generate_quick_start_guide(),
            'user_guide': self._generate_comprehensive_user_guide(),
            'api_tutorial': self._generate_api_tutorial(),
            'cli_guide': self._generate_cli_guide(),
            'web_ui_guide': self._generate_web_ui_guide()
        }

        for guide_name, content in guides.items():
            with open(self.output_dir / f'{guide_name}.md', 'w') as f:
                f.write(content)

    def _generate_quick_start_guide(self) -> str:
        """Generate quick start guide."""
        return """# LL3M Quick Start Guide

## Installation

1. **Install LL3M using pip:**
   ```bash
   pip install ll3m
   ```

2. **Set up environment variables:**
   ```bash
   export OPENAI_API_KEY="your_openai_api_key"
   export BLENDER_PATH="/usr/local/bin/blender"
   ```

3. **Verify installation:**
   ```bash
   ll3m --version
   ```

## Basic Usage

### Generate Your First 3D Asset

```bash
ll3m generate "Create a red metallic cube with smooth edges"
```

This command will:
1. Analyze your prompt
2. Generate Blender Python code
3. Execute the code in Blender
4. Save the resulting 3D asset
5. Provide a screenshot preview

### Using the Python API

```python
import asyncio
from ll3m import LL3MWorkflow

async def generate_asset():
    workflow = LL3MWorkflow()
    result = await workflow.generate("Create a blue sphere")

    if result.success:
        print(f"Asset created: {result.asset_path}")
        print(f"Screenshot: {result.screenshot_path}")
    else:
        print(f"Error: {result.error_message}")

asyncio.run(generate_asset())
```

### Web Interface

1. **Start the web server:**
   ```bash
   ll3m serve
   ```

2. **Open your browser to:**
   ```
   http://localhost:8000
   ```

3. **Use the web interface to:**
   - Generate assets with natural language
   - Browse your asset library
   - Refine existing assets
   - Download assets in multiple formats

## Next Steps

- Read the [Comprehensive User Guide](user_guide.md)
- Explore the [API Documentation](api_documentation.html)
- Try the [CLI Tutorial](cli_guide.md)
- Join our [Community Forum](https://community.ll3m.ai)

## Getting Help

- **Documentation**: [docs.ll3m.ai](https://docs.ll3m.ai)
- **GitHub Issues**: [github.com/ll3m/ll3m/issues](https://github.com/ll3m/ll3m/issues)
- **Discord**: [discord.gg/ll3m](https://discord.gg/ll3m)
"""

    def _generate_comprehensive_user_guide(self) -> str:
        """Generate comprehensive user guide."""
        return """# LL3M Comprehensive User Guide

## Table of Contents

1. [Introduction](#introduction)
2. [Installation and Setup](#installation-and-setup)
3. [Understanding LL3M](#understanding-ll3m)
4. [Text-to-3D Generation](#text-to-3d-generation)
5. [Asset Management](#asset-management)
6. [Advanced Features](#advanced-features)
7. [Best Practices](#best-practices)
8. [Troubleshooting](#troubleshooting)

## Introduction

LL3M (Large Language 3D Modelers) is an advanced AI system that generates 3D assets from natural language descriptions. It combines the power of large language models with Blender's 3D capabilities to create high-quality 3D models, scenes, and animations.

### Key Features

- **Natural Language Interface**: Describe what you want in plain English
- **Multi-Agent Architecture**: Specialized AI agents handle different aspects of 3D creation
- **Quality Assessment**: Automatic evaluation and refinement of generated assets
- **Multiple Export Formats**: Support for .blend, .obj, .fbx, .gltf, and more
- **Web & CLI Interfaces**: Choose your preferred interaction method
- **Asset Management**: Built-in library with versioning and metadata

## Installation and Setup

### System Requirements

- **Operating System**: Windows 10+, macOS 10.15+, or Linux (Ubuntu 18.04+)
- **Python**: 3.9 or higher
- **Memory**: 8GB RAM minimum, 16GB recommended
- **Storage**: 10GB free space for Blender and assets
- **GPU**: NVIDIA GPU recommended for faster rendering (optional)

### Installation Steps

1. **Install Python dependencies:**
   ```bash
   pip install ll3m[full]
   ```

2. **Install Blender:**
   - **Automatic**: `ll3m install-blender`
   - **Manual**: Download from [blender.org](https://www.blender.org/download/)

3. **Configure API keys:**
   ```bash
   ll3m config set openai_api_key "your_key_here"
   ```

4. **Verify setup:**
   ```bash
   ll3m doctor
   ```

### Configuration

LL3M uses a configuration file located at `~/.ll3m/config.yaml`:

```yaml
# API Keys
openai_api_key: "your_openai_api_key"

# Blender Configuration
blender:
  path: "/usr/local/bin/blender"
  timeout: 300
  headless: true

# Quality Settings
quality:
  default_level: "standard"
  auto_refine: true
  max_refinement_iterations: 3

# Output Settings
output:
  directory: "~/ll3m_assets"
  formats: ["blend", "obj"]
  screenshots: true
```

## Understanding LL3M

### Multi-Agent Architecture

LL3M uses five specialized AI agents:

1. **Planner Agent**: Breaks down your prompt into specific tasks
2. **Retrieval Agent**: Finds relevant Blender documentation and examples
3. **Coding Agent**: Generates Python code for Blender
4. **Execution Engine**: Runs the code in Blender safely
5. **Critic Agent**: Evaluates the result and suggests improvements
6. **Verification Agent**: Validates technical quality

### Workflow Process

```
User Prompt → Planner → Retrieval → Coding → Execution → Critic → Verification
                ↑                                                      ↓
                └──────────── Refinement Loop ←──────────────────────┘
```

## Text-to-3D Generation

### Writing Effective Prompts

**Good prompts are:**
- **Specific**: "Red metallic cube" vs "cube"
- **Descriptive**: Include materials, colors, and textures
- **Contextual**: Mention the intended use or style
- **Reasonable**: Within the capabilities of 3D modeling

**Examples:**

```bash
# Simple geometry
ll3m generate "A smooth wooden table with four legs"

# Complex scenes
ll3m generate "A medieval castle with stone towers, wooden bridges, and torches"

# Artistic styles
ll3m generate "A low-poly stylized tree in pastel colors for a mobile game"

# Technical objects
ll3m generate "A realistic car engine with moving pistons and detailed components"
```

### Quality Levels

LL3M supports three quality levels:

- **Draft**: Fast generation, basic quality (1-2 minutes)
- **Standard**: Balanced quality and speed (3-5 minutes)
- **High**: Best quality, slower generation (10-15 minutes)

```bash
ll3m generate "a sports car" --quality high
```

### Asset Formats

Generated assets can be exported in multiple formats:

- **Blender (.blend)**: Full scene with materials and lighting
- **OBJ (.obj)**: Geometry only, widely compatible
- **FBX (.fbx)**: Geometry with materials, good for game engines
- **glTF (.gltf)**: Modern standard, includes animations
- **STL (.stl)**: For 3D printing
- **PLY (.ply)**: Point cloud and mesh format

```bash
ll3m generate "a vase" --formats blend,obj,gltf
```

## Asset Management

### Asset Library

LL3M automatically manages your generated assets:

```bash
# List all assets
ll3m list

# Search by tags
ll3m list --tags furniture,wood

# Filter by quality
ll3m list --min-quality 8.0

# Show asset details
ll3m show asset_12345
```

### Asset Versioning

Each asset supports multiple versions:

```bash
# Create refinement
ll3m refine asset_12345 "make it bigger and add more detail"

# List versions
ll3m versions asset_12345

# Switch to specific version
ll3m checkout asset_12345 --version 2
```

### Metadata and Tags

Assets include rich metadata:
- Generation prompt
- Quality scores
- Creation timestamp
- File formats
- Size information
- User-defined tags

```bash
# Add tags
ll3m tag asset_12345 furniture,modern,living-room

# Update description
ll3m describe asset_12345 "A modern coffee table for living room scenes"
```

## Advanced Features

### Custom Workflows

Create custom generation workflows:

```python
from ll3m import CustomWorkflow, PlannerAgent, CodingAgent

workflow = CustomWorkflow()
workflow.add_agent(PlannerAgent(temperature=0.3))
workflow.add_agent(CodingAgent(model="gpt-4"))

result = await workflow.generate("custom prompt")
```

### Batch Processing

Process multiple prompts efficiently:

```bash
# From file
ll3m batch generate prompts.txt

# Multiple prompts
ll3m generate "red cube" "blue sphere" "green cylinder"
```

### Asset Composition

Combine multiple assets into scenes:

```python
from ll3m import SceneComposer

composer = SceneComposer()
scene = composer.create_scene()

scene.add_asset("table", position=(0, 0, 0))
scene.add_asset("chair", position=(2, 0, 0))
scene.add_lighting("sunlight", intensity=5.0)

result = await scene.render()
```

### Custom Materials and Shaders

Define custom material libraries:

```yaml
# materials.yaml
materials:
  gold:
    type: metallic
    base_color: [1.0, 0.766, 0.336]
    metallic: 1.0
    roughness: 0.1

  wood_oak:
    type: principled
    base_color_texture: "oak_diffuse.jpg"
    normal_texture: "oak_normal.jpg"
    roughness: 0.8
```

## Best Practices

### Prompt Engineering

1. **Be specific about materials:**
   - "metallic red surface" vs "red"
   - "rough concrete texture" vs "gray"

2. **Include size and scale references:**
   - "human-sized door" vs "door"
   - "coffee mug sized" vs "cup"

3. **Specify style when needed:**
   - "low-poly game asset"
   - "photorealistic architectural model"
   - "cartoon-style character"

4. **Mention constraints:**
   - "suitable for 3D printing"
   - "optimized for mobile games"
   - "under 10k polygons"

### Performance Optimization

1. **Use appropriate quality levels:**
   - Draft for iteration and testing
   - Standard for most use cases
   - High for final production assets

2. **Manage asset library:**
   - Regularly clean up unused assets
   - Use meaningful tags and descriptions
   - Archive old versions when not needed

3. **Batch similar requests:**
   - Generate multiple variations together
   - Use consistent prompting patterns
   - Process during off-peak hours

### Quality Assurance

1. **Review generated assets:**
   - Check geometry for errors
   - Verify materials and textures
   - Test in target applications

2. **Use refinement effectively:**
   - Make specific refinement requests
   - Iterate gradually rather than major changes
   - Keep successful versions

3. **Monitor quality scores:**
   - Understand what affects quality
   - Use quality thresholds for filtering
   - Request regeneration for low-quality results

## Troubleshooting

### Common Issues

**Issue: "Blender not found"**
```bash
# Solution: Set Blender path
ll3m config set blender.path "/path/to/blender"
```

**Issue: "Generation timeout"**
```bash
# Solution: Increase timeout
ll3m config set blender.timeout 600
```

**Issue: "Low quality results"**
- Use more specific prompts
- Try higher quality level
- Check if prompt is too complex
- Review generated code for issues

**Issue: "API rate limits"**
- Check OpenAI API quota
- Reduce concurrent requests
- Implement request batching

### Debug Mode

Enable debug mode for detailed logging:

```bash
ll3m generate "test prompt" --debug
```

This will show:
- Agent decision processes
- Generated code
- Blender execution logs
- Performance metrics

### Log Analysis

LL3M logs are stored in `~/.ll3m/logs/`:

```bash
# View recent errors
ll3m logs --level error --recent

# Follow live logs
ll3m logs --follow

# Export logs for support
ll3m logs --export support_logs.zip
```

### Getting Support

1. **Check documentation**: [docs.ll3m.ai](https://docs.ll3m.ai)
2. **Search existing issues**: [github.com/ll3m/ll3m/issues](https://github.com/ll3m/ll3m/issues)
3. **Join community**: [discord.gg/ll3m](https://discord.gg/ll3m)
4. **Create bug report**: Include logs, system info, and reproduction steps

## Conclusion

LL3M provides a powerful and flexible platform for AI-driven 3D asset generation. By understanding the system architecture, writing effective prompts, and following best practices, you can create high-quality 3D content efficiently.

For more advanced usage, explore the API documentation and consider contributing to the open-source project.
"""

    def _extract_api_endpoints(self) -> List[APIEndpoint]:
        """Extract API endpoints from FastAPI routes."""
        # This would analyze the FastAPI application
        # and extract endpoint information
        return []

    def _extract_class_documentation(self) -> List[ClassDoc]:
        """Extract class documentation from source code."""
        classes = []

        for python_file in self.source_dir.rglob("*.py"):
            try:
                with open(python_file, 'r') as f:
                    content = f.read()

                tree = ast.parse(content)

                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        class_doc = self._extract_class_info(node, content)
                        if class_doc:
                            classes.append(class_doc)

            except Exception as e:
                print(f"Error processing {python_file}: {e}")

        return classes

    def _extract_class_info(self, node: ast.ClassDef, source: str) -> Optional[ClassDoc]:
        """Extract information from a class AST node."""
        docstring = ast.get_docstring(node)
        if not docstring:
            return None

        methods = []
        properties = []

        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                method_doc = {
                    'name': item.name,
                    'description': ast.get_docstring(item) or '',
                    'parameters': [arg.arg for arg in item.args.args[1:]],  # Skip 'self'
                    'is_async': isinstance(item, ast.AsyncFunctionDef)
                }
                methods.append(method_doc)

        return ClassDoc(
            name=node.name,
            description=docstring,
            methods=methods,
            properties=properties,
            examples=[]
        )

if __name__ == '__main__':
    generator = DocumentationGenerator('src/', 'docs/generated/')
    generator.generate_all_documentation()
```

<function_calls>
<invoke name="TodoWrite">
<parameter name="todos">[{"content": "Commit Phase 6 implementation plan", "status": "completed", "activeForm": "Committing Phase 6 implementation plan"}, {"content": "Create new branch for next phase", "status": "completed", "activeForm": "Creating new branch for next phase"}, {"content": "Determine and implement next phase from IMPLEMENTATION_PLAN.md", "status": "completed", "activeForm": "Creating Phase 7: Production Deployment & Optimization plan"}]
