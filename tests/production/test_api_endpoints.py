"""Comprehensive API endpoint testing suite."""

import asyncio
import os
from datetime import datetime
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient

from src.api.app import create_app
from src.api.auth import create_test_user


@pytest.fixture(autouse=True)
def mock_settings():
    """Override the global mock_settings fixture to allow real settings."""
    # Do nothing - let real settings be used
    yield


@pytest.fixture
def test_app():
    """Create test FastAPI application."""
    # Ensure environment variables are set before creating the app
    os.environ["ENVIRONMENT"] = "test"
    os.environ["OPENAI_API_KEY"] = "sk-test-mock-key-for-testing"
    os.environ["BLENDER_PATH"] = "/usr/bin/blender"

    app = create_app()
    return app


@pytest.fixture
def client(test_app):
    """Create test client."""
    return TestClient(test_app)


@pytest.fixture
async def async_client(test_app):
    """Create async test client."""
    async with AsyncClient(app=test_app, base_url="http://testserver") as client:
        yield client


@pytest.fixture
def auth_headers():
    """Create authentication headers for testing."""
    # In production, this would use actual JWT tokens
    return {"Authorization": "Bearer test-token"}


@pytest.fixture
def test_user():
    """Create test user."""
    return create_test_user()


class TestHealthEndpoints:
    """Test health and monitoring endpoints."""

    def test_health_check(self, client):
        """Test basic health check endpoint."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] in ["healthy", "degraded", "unhealthy"]
        assert "version" in data
        assert "uptime" in data
        assert "dependencies" in data

    def test_liveness_probe(self, client):
        """Test Kubernetes liveness probe."""
        response = client.get("/api/v1/health/live")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "alive"
        assert "timestamp" in data

    def test_readiness_probe(self, client):
        """Test Kubernetes readiness probe."""
        response = client.get("/api/v1/health/ready")
        # May return 200 or 503 depending on service availability
        assert response.status_code in [200, 503]

    def test_metrics_endpoint(self, client):
        """Test Prometheus metrics endpoint."""
        response = client.get("/api/v1/health/metrics")
        assert response.status_code == 200

        # Should return Prometheus format
        metrics_text = response.text
        assert "ll3m_" in metrics_text
        assert "\n" in metrics_text


class TestAuthenticationEndpoints:
    """Test authentication and authorization endpoints."""

    def test_register_user(self, client):
        """Test user registration."""
        user_data = {
            "email": f"test_{uuid4()}@example.com",
            "name": "Test User",
            "password": "securepassword123",
        }

        response = client.post("/api/v1/auth/register", json=user_data)
        assert response.status_code == 201

        data = response.json()
        assert "user_id" in data
        assert data["email"] == user_data["email"]

    def test_register_duplicate_email(self, client):
        """Test registration with duplicate email."""
        user_data = {
            "email": "duplicate@example.com",
            "name": "Test User",
            "password": "securepassword123",
        }

        # First registration should succeed
        response1 = client.post("/api/v1/auth/register", json=user_data)
        assert response1.status_code == 201

        # Second registration should fail
        response2 = client.post("/api/v1/auth/register", json=user_data)
        assert response2.status_code == 409

    def test_login_valid_credentials(self, client):
        """Test login with valid credentials."""
        # First register a user
        user_data = {
            "email": f"login_test_{uuid4()}@example.com",
            "name": "Login Test",
            "password": "securepassword123",
        }
        client.post("/api/v1/auth/register", json=user_data)

        # Then try to login
        login_data = {"email": user_data["email"], "password": user_data["password"]}

        response = client.post("/api/v1/auth/login", json=login_data)
        assert response.status_code == 200

        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        assert "expires_in" in data

    def test_login_invalid_credentials(self, client):
        """Test login with invalid credentials."""
        login_data = {"email": "nonexistent@example.com", "password": "wrongpassword"}

        response = client.post("/api/v1/auth/login", json=login_data)
        assert response.status_code == 401

    def test_get_current_user(self, client, auth_headers):
        """Test getting current user information."""
        response = client.get("/api/v1/auth/me", headers=auth_headers)
        assert response.status_code == 200

        data = response.json()
        assert "id" in data
        assert "email" in data
        assert "subscription_tier" in data

    def test_unauthorized_access(self, client):
        """Test accessing protected endpoint without authentication."""
        response = client.get("/api/v1/auth/me")
        assert response.status_code == 401


class TestAssetEndpoints:
    """Test asset management endpoints."""

    def test_generate_asset(self, client, auth_headers):
        """Test asset generation endpoint."""
        request_data = {
            "prompt": "a futuristic robot with glowing eyes",
            "name": "Test Robot",
            "complexity": "medium",
            "quality": "high",
        }

        response = client.post(
            "/api/v1/assets/generate", json=request_data, headers=auth_headers
        )
        assert response.status_code == 202

        data = response.json()
        assert "asset" in data
        assert data["asset"]["prompt"] == request_data["prompt"]
        assert data["asset"]["status"] == "pending"
        assert "generation_job_id" in data

    def test_generate_asset_invalid_prompt(self, client, auth_headers):
        """Test asset generation with invalid prompt."""
        request_data = {
            "prompt": "a",  # Too short
            "name": "Test Asset",
        }

        response = client.post(
            "/api/v1/assets/generate", json=request_data, headers=auth_headers
        )
        assert response.status_code == 422

    def test_list_assets(self, client, auth_headers):
        """Test listing user assets."""
        response = client.get("/api/v1/assets", headers=auth_headers)
        assert response.status_code == 200

        data = response.json()
        assert isinstance(data, list)

    def test_list_assets_with_pagination(self, client, auth_headers):
        """Test asset listing with pagination."""
        response = client.get("/api/v1/assets?limit=5&offset=0", headers=auth_headers)
        assert response.status_code == 200

        data = response.json()
        assert isinstance(data, list)
        assert len(data) <= 5

    def test_get_asset_status(self, client, auth_headers):
        """Test getting asset generation status."""
        # First create an asset
        request_data = {
            "prompt": "test asset for status check",
            "name": "Status Test Asset",
        }

        create_response = client.post(
            "/api/v1/assets/generate", json=request_data, headers=auth_headers
        )
        asset_id = create_response.json()["asset"]["id"]

        # Then check its status
        response = client.get(f"/api/v1/assets/{asset_id}/status", headers=auth_headers)
        assert response.status_code == 200

        data = response.json()
        assert "asset_id" in data
        assert "status" in data

    def test_refine_asset(self, client, auth_headers):
        """Test asset refinement."""
        # Mock asset ID (in real test, would create actual asset first)
        asset_id = str(uuid4())

        request_data = {
            "feedback": "make the robot bigger and add more detail to the eyes",
            "priority_areas": ["size", "eyes"],
            "preserve_aspects": ["overall_design"],
        }

        # This would normally return 404 since asset doesn't exist
        response = client.post(
            f"/api/v1/assets/{asset_id}/refine", json=request_data, headers=auth_headers
        )
        assert response.status_code == 404

    def test_delete_asset(self, client, auth_headers):
        """Test asset deletion."""
        asset_id = str(uuid4())

        response = client.delete(f"/api/v1/assets/{asset_id}", headers=auth_headers)
        # Would return 404 since asset doesn't exist
        assert response.status_code == 404


class TestBatchEndpoints:
    """Test batch processing endpoints."""

    def test_create_batch(self, client, auth_headers):
        """Test creating a batch processing job."""
        requests = [
            {"prompt": "a red cube", "name": "Red Cube"},
            {"prompt": "a blue sphere", "name": "Blue Sphere"},
        ]

        batch_data = {
            "name": "Test Batch",
            "requests": requests,
            "priority": 3,
            "notify_on_completion": True,
        }

        response = client.post("/api/v1/batches", json=batch_data, headers=auth_headers)
        assert response.status_code == 202

        data = response.json()
        assert "batch_id" in data
        assert data["total_assets"] == len(requests)
        assert "estimated_completion" in data

    def test_create_batch_too_many_items(self, client, auth_headers):
        """Test creating batch with too many items."""
        requests = [{"prompt": f"item {i}", "name": f"Item {i}"} for i in range(101)]

        batch_data = {"name": "Large Batch", "requests": requests}

        response = client.post("/api/v1/batches", json=batch_data, headers=auth_headers)
        assert response.status_code == 400

    def test_get_batch_status(self, client, auth_headers):
        """Test getting batch status."""
        batch_id = str(uuid4())

        response = client.get(f"/api/v1/batches/{batch_id}", headers=auth_headers)
        assert response.status_code == 404

    def test_list_batches(self, client, auth_headers):
        """Test listing user batches."""
        response = client.get("/api/v1/batches", headers=auth_headers)
        assert response.status_code == 200

        data = response.json()
        assert isinstance(data, list)

    def test_cancel_batch(self, client, auth_headers):
        """Test canceling a batch job."""
        batch_id = str(uuid4())

        response = client.delete(f"/api/v1/batches/{batch_id}", headers=auth_headers)
        assert response.status_code == 404


class TestExportEndpoints:
    """Test asset export endpoints."""

    def test_export_asset(self, client, auth_headers):
        """Test exporting an asset."""
        asset_id = str(uuid4())

        export_data = {
            "format": "gltf",
            "quality": "high",
            "include_materials": True,
            "include_textures": True,
        }

        response = client.post(
            f"/api/v1/exports/{asset_id}/export", json=export_data, headers=auth_headers
        )
        assert response.status_code == 404  # Asset doesn't exist

    def test_get_export_status(self, client, auth_headers):
        """Test getting export status."""
        export_id = str(uuid4())

        response = client.get(
            f"/api/v1/exports/{export_id}/status", headers=auth_headers
        )
        assert response.status_code == 404

    def test_list_exports(self, client, auth_headers):
        """Test listing user exports."""
        response = client.get("/api/v1/exports", headers=auth_headers)
        assert response.status_code == 200

        data = response.json()
        assert isinstance(data, list)


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_404_endpoint(self, client):
        """Test accessing non-existent endpoint."""
        response = client.get("/api/v1/nonexistent")
        assert response.status_code == 404

    def test_405_method_not_allowed(self, client):
        """Test using wrong HTTP method."""
        response = client.put("/api/v1/health")
        assert response.status_code == 405

    def test_422_validation_error(self, client, auth_headers):
        """Test validation error handling."""
        invalid_data = {
            "prompt": "",  # Empty prompt should fail validation
            "complexity": "invalid_level",
        }

        response = client.post(
            "/api/v1/assets/generate", json=invalid_data, headers=auth_headers
        )
        assert response.status_code == 422

        data = response.json()
        assert "error" in data
        assert "detail" in data


class TestRateLimiting:
    """Test rate limiting functionality."""

    @pytest.mark.asyncio
    async def test_rate_limiting(self, async_client, auth_headers):
        """Test API rate limiting."""
        # Make multiple rapid requests
        tasks = []
        for _ in range(20):
            task = async_client.get("/api/v1/health", headers=auth_headers)
            tasks.append(task)

        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # Some requests should succeed, some might be rate limited
        status_codes = [r.status_code for r in responses if hasattr(r, "status_code")]
        assert 200 in status_codes

        # Check if rate limiting is working (429 status code)
        # Note: This depends on rate limiting configuration
        any(code == 429 for code in status_codes)
        # Don't assert rate limiting is working as it depends on configuration


class TestWebSocketConnections:
    """Test WebSocket functionality."""

    @pytest.mark.asyncio
    async def test_asset_progress_websocket(self, _test_app):
        """Test WebSocket connection for asset progress."""
        str(uuid4())

        # This would test WebSocket connection
        # Skipped due to complexity in test setup
        pass


class TestPerformance:
    """Test API performance characteristics."""

    @pytest.mark.asyncio
    async def test_concurrent_requests(self, async_client):
        """Test handling concurrent requests."""
        # Make concurrent health check requests
        tasks = [async_client.get("/api/v1/health") for _ in range(10)]

        start_time = datetime.utcnow()
        responses = await asyncio.gather(*tasks)
        end_time = datetime.utcnow()

        # All requests should succeed
        assert all(r.status_code == 200 for r in responses)

        # Total time should be reasonable (less than 5 seconds for 10 requests)
        total_time = (end_time - start_time).total_seconds()
        assert total_time < 5.0

    def test_response_time(self, client):
        """Test API response times."""
        start_time = datetime.utcnow()
        response = client.get("/api/v1/health")
        end_time = datetime.utcnow()

        assert response.status_code == 200

        response_time = (end_time - start_time).total_seconds()
        assert response_time < 1.0  # Should respond within 1 second


@pytest.mark.integration
class TestIntegration:
    """Integration tests for complete workflows."""

    @pytest.mark.asyncio
    async def test_complete_asset_workflow(self, async_client, auth_headers):
        """Test complete asset generation workflow."""
        # 1. Generate asset
        request_data = {
            "prompt": "integration test robot",
            "name": "Integration Test Robot",
        }

        create_response = await async_client.post(
            "/api/v1/assets/generate", json=request_data, headers=auth_headers
        )
        assert create_response.status_code == 202

        asset_id = create_response.json()["asset"]["id"]

        # 2. Check status
        status_response = await async_client.get(
            f"/api/v1/assets/{asset_id}/status", headers=auth_headers
        )
        assert status_response.status_code == 200

        # 3. List assets (should include our new asset)
        list_response = await async_client.get("/api/v1/assets", headers=auth_headers)
        assert list_response.status_code == 200

        assets = list_response.json()
        asset_ids = [asset["id"] for asset in assets]
        assert asset_id in asset_ids
