/**
 * K6 Load Testing Script for LL3M API
 * 
 * This script tests the performance and reliability of the LL3M API
 * under various load conditions.
 */

import http from 'k6/http';
import { check, group, sleep } from 'k6';
import { Rate, Trend, Counter } from 'k6/metrics';

// Custom metrics
const errorRate = new Rate('errors');
const responseTimeTrend = new Trend('response_time');
const requestCounter = new Counter('requests_total');

// Test configuration
export const options = {
  stages: [
    // Ramp-up
    { duration: '2m', target: 10 }, // Ramp up to 10 users over 2 minutes
    { duration: '5m', target: 10 }, // Stay at 10 users for 5 minutes
    { duration: '2m', target: 20 }, // Ramp up to 20 users over 2 minutes
    { duration: '5m', target: 20 }, // Stay at 20 users for 5 minutes
    { duration: '2m', target: 50 }, // Ramp up to 50 users over 2 minutes
    { duration: '5m', target: 50 }, // Stay at 50 users for 5 minutes
    // Ramp-down
    { duration: '2m', target: 0 },  // Ramp down to 0 users over 2 minutes
  ],
  
  thresholds: {
    http_req_duration: ['p(95)<2000'], // 95% of requests must complete below 2s
    http_req_failed: ['rate<0.05'],    // Error rate must be below 5%
    errors: ['rate<0.05'],             // Custom error rate must be below 5%
  },
};

// Test data
const BASE_URL = __ENV.API_BASE_URL || 'http://localhost:8000';

// Mock authentication token (in real tests, would obtain via login)
const AUTH_TOKEN = 'test-token';
const AUTH_HEADERS = {
  'Authorization': `Bearer ${AUTH_TOKEN}`,
  'Content-Type': 'application/json',
};

// Test prompts for asset generation
const TEST_PROMPTS = [
  'a futuristic robot with glowing eyes',
  'a medieval castle on a mountain',
  'a modern chair design',
  'an abstract sculpture',
  'a fantasy sword with runes',
  'a steampunk airship',
  'a minimalist table',
  'a cyberpunk car',
  'an organic building',
  'a crystal formation'
];

// Get random test prompt
function getRandomPrompt() {
  return TEST_PROMPTS[Math.floor(Math.random() * TEST_PROMPTS.length)];
}

// Generate random asset request
function generateAssetRequest() {
  return {
    prompt: getRandomPrompt(),
    name: `Load Test Asset ${Math.random().toString(36).substring(7)}`,
    complexity: ['simple', 'medium', 'complex'][Math.floor(Math.random() * 3)],
    quality: ['low', 'medium', 'high'][Math.floor(Math.random() * 3)],
  };
}

export default function () {
  requestCounter.add(1);
  
  group('Health Check', () => {
    const response = http.get(`${BASE_URL}/api/v1/health`);
    
    const success = check(response, {
      'health check status is 200': (r) => r.status === 200,
      'health check response time < 500ms': (r) => r.timings.duration < 500,
      'health check has status field': (r) => JSON.parse(r.body).status !== undefined,
    });
    
    errorRate.add(!success);
    responseTimeTrend.add(response.timings.duration);
    
    if (!success) {
      console.error(`Health check failed: ${response.status} ${response.body}`);
    }
  });
  
  sleep(1);
  
  group('Authentication Flow', () => {
    // Test getting current user (simulates authenticated request)
    const userResponse = http.get(`${BASE_URL}/api/v1/auth/me`, {
      headers: AUTH_HEADERS,
    });
    
    const success = check(userResponse, {
      'get current user status is 200 or 401': (r) => [200, 401].includes(r.status),
      'get current user response time < 1000ms': (r) => r.timings.duration < 1000,
    });
    
    errorRate.add(!success);
    responseTimeTrend.add(userResponse.timings.duration);
  });
  
  sleep(1);
  
  group('Asset Generation', () => {
    const assetRequest = generateAssetRequest();
    
    const response = http.post(
      `${BASE_URL}/api/v1/assets/generate`,
      JSON.stringify(assetRequest),
      { headers: AUTH_HEADERS }
    );
    
    const success = check(response, {
      'asset generation status is 202 or 401': (r) => [202, 401].includes(r.status),
      'asset generation response time < 3000ms': (r) => r.timings.duration < 3000,
      'asset generation response has asset field': (r) => {
        if (r.status === 202) {
          return JSON.parse(r.body).asset !== undefined;
        }
        return true; // Skip check for 401
      },
    });
    
    errorRate.add(!success);
    responseTimeTrend.add(response.timings.duration);
    
    // If asset generation was successful, test status check
    if (response.status === 202) {
      const responseBody = JSON.parse(response.body);
      if (responseBody.asset && responseBody.asset.id) {
        sleep(0.5); // Brief pause before status check
        
        const statusResponse = http.get(
          `${BASE_URL}/api/v1/assets/${responseBody.asset.id}/status`,
          { headers: AUTH_HEADERS }
        );
        
        const statusSuccess = check(statusResponse, {
          'asset status check is 200': (r) => r.status === 200,
          'asset status response time < 1000ms': (r) => r.timings.duration < 1000,
        });
        
        errorRate.add(!statusSuccess);
        responseTimeTrend.add(statusResponse.timings.duration);
      }
    }
  });
  
  sleep(2);
  
  group('Asset Listing', () => {
    const response = http.get(`${BASE_URL}/api/v1/assets?limit=10`, {
      headers: AUTH_HEADERS,
    });
    
    const success = check(response, {
      'asset list status is 200 or 401': (r) => [200, 401].includes(r.status),
      'asset list response time < 2000ms': (r) => r.timings.duration < 2000,
      'asset list response is array': (r) => {
        if (r.status === 200) {
          return Array.isArray(JSON.parse(r.body));
        }
        return true; // Skip check for 401
      },
    });
    
    errorRate.add(!success);
    responseTimeTrend.add(response.timings.duration);
  });
  
  sleep(1);
  
  group('Batch Processing', () => {
    const batchRequest = {
      name: `Load Test Batch ${Math.random().toString(36).substring(7)}`,
      requests: [
        generateAssetRequest(),
        generateAssetRequest(),
      ],
      priority: Math.floor(Math.random() * 5) + 1,
    };
    
    const response = http.post(
      `${BASE_URL}/api/v1/batches`,
      JSON.stringify(batchRequest),
      { headers: AUTH_HEADERS }
    );
    
    const success = check(response, {
      'batch creation status is 202 or 401': (r) => [202, 401].includes(r.status),
      'batch creation response time < 2000ms': (r) => r.timings.duration < 2000,
    });
    
    errorRate.add(!success);
    responseTimeTrend.add(response.timings.duration);
  });
  
  sleep(1);
  
  // Occasional stress test with export functionality
  if (Math.random() < 0.1) { // 10% chance
    group('Export Functionality', () => {
      // Create a mock asset ID for testing
      const mockAssetId = '550e8400-e29b-41d4-a716-446655440000';
      
      const exportRequest = {
        format: 'gltf',
        quality: 'medium',
        include_materials: true,
        include_textures: true,
      };
      
      const response = http.post(
        `${BASE_URL}/api/v1/exports/${mockAssetId}/export`,
        JSON.stringify(exportRequest),
        { headers: AUTH_HEADERS }
      );
      
      const success = check(response, {
        'export request status is 202, 401, or 404': (r) => [202, 401, 404].includes(r.status),
        'export request response time < 3000ms': (r) => r.timings.duration < 3000,
      });
      
      errorRate.add(!success);
      responseTimeTrend.add(response.timings.duration);
    });
  }
  
  sleep(Math.random() * 3 + 1); // Random sleep between 1-4 seconds
}

// Setup function - runs once before the test
export function setup() {
  console.log('Starting LL3M API load test...');
  console.log(`Target URL: ${BASE_URL}`);
  
  // Verify API is accessible
  const response = http.get(`${BASE_URL}/api/v1/health`);
  if (response.status !== 200) {
    throw new Error(`API health check failed: ${response.status}`);
  }
  
  console.log('API health check passed, starting load test...');
  return {};
}

// Teardown function - runs once after the test
export function teardown(data) {
  console.log('LL3M API load test completed');
  
  // Log final statistics
  console.log(`Total requests: ${requestCounter.value}`);
  console.log(`Error rate: ${(errorRate.rate * 100).toFixed(2)}%`);
  console.log(`Average response time: ${responseTimeTrend.avg.toFixed(2)}ms`);
  console.log(`95th percentile response time: ${responseTimeTrend.p95.toFixed(2)}ms`);
}

// Scenario-specific load tests
export const scenarios = {
  // Smoke test - light load to verify basic functionality
  smoke: {
    executor: 'constant-vus',
    vus: 1,
    duration: '1m',
    tags: { test_type: 'smoke' },
  },
  
  // Load test - normal expected load
  load: {
    executor: 'ramping-vus',
    startVUs: 0,
    stages: [
      { duration: '5m', target: 10 },
      { duration: '10m', target: 10 },
      { duration: '5m', target: 0 },
    ],
    tags: { test_type: 'load' },
  },
  
  // Stress test - beyond normal capacity
  stress: {
    executor: 'ramping-vus',
    startVUs: 0,
    stages: [
      { duration: '2m', target: 10 },
      { duration: '5m', target: 20 },
      { duration: '5m', target: 50 },
      { duration: '5m', target: 100 },
      { duration: '10m', target: 100 },
      { duration: '5m', target: 0 },
    ],
    tags: { test_type: 'stress' },
  },
  
  // Spike test - sudden load increase
  spike: {
    executor: 'ramping-vus',
    startVUs: 0,
    stages: [
      { duration: '1m', target: 10 },
      { duration: '1m', target: 100 }, // Sudden spike
      { duration: '3m', target: 100 },
      { duration: '1m', target: 10 },
      { duration: '1m', target: 0 },
    ],
    tags: { test_type: 'spike' },
  },
};