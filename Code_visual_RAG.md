# Code Visual RAG: Retrieval-Augmented Generation for Coding Tasks

## Table of Contents

- [Introduction](#introduction)
- [LLM Configuration](#llm-configuration)
- [Agentic RAG Features](#agentic-rag-features)
  - [Question Generation from Documentation](#question-generation-from-documentation)
  - [Query Reformulation & Expansion](#query-reformulation--expansion)
  - [Code Quality Assurance](#code-quality-assurance)
  - [Metadata Management](#metadata-management)
- [RAG for General Coding Tasks](#rag-for-general-coding-tasks)
  - [Indexing](#indexing)
  - [Retrieval](#retrieval)
  - [Graph Structure](#graph-structure)
  - [Minimal LangGraph Example](#minimal-langgraph-example)
- [Blender Python Code Snippets RAG](#blender-python-code-snippets-rag)
  - [Ingestion](#ingestion)
  - [Retrieval](#retrieval-1)
  - [Validation Toolchain](#validation-toolchain)
  - [vLLM Critic](#vllm-critic)
  - [LangGraph Wiring](#langgraph-wiring)
  - [Generation Prompt](#generation-prompt)
  - [Critic Prompt](#critic-prompt)
  - [Configuration Files](#configuration-files)
  - [Example Usage](#example-usage)
  - [Acceptance Policy](#acceptance-policy)
  - [Runtime with Healing](#runtime-with-healing)
  - [CI Considerations](#ci-considerations)
  - [Error Examples & Fixes](#error-examples--fixes)
  - [Simple Operations](#simple-operations)
  - [Intermediate Operations](#intermediate-operations)
  - [Advanced Operations](#advanced-operations)
  - [Error Patterns & Solutions](#error-patterns--solutions)

## Introduction

This comprehensive guide outlines the implementation of a Retrieval-Augmented Generation (RAG) system for coding tasks, with specialized focus on generating and validating Blender Python code snippets. The system leverages modern AI orchestration frameworks and integrates multiple validation layers to ensure code quality and correctness.

### Key Components

- **LangGraph**: Orchestration framework for building stateful agents (v0.6.0)
- **Pydantic**: Data validation library for robust type checking
- **FAISS**: Efficient similarity search for vector retrieval
- **Multiple Validation Layers**: Static analysis (ruff, mypy, Pyright) and runtime testing
- **Healing Mechanism**: Automatic code correction using LLM feedback

### Architecture Overview

The system follows a graph-based architecture where:
1. **Ingestion** processes documentation and codebases into searchable chunks
2. **Retrieval** finds relevant context using hybrid search techniques
3. **Generation** creates code using retrieved context and validation
4. **Validation** ensures code correctness through multiple layers
5. **Healing** automatically fixes issues when validation fails

This approach ensures production-ready code generation with comprehensive quality assurance.

## LLM Configuration

The system supports multiple LLM endpoints for flexibility across different deployment scenarios:

### Supported Endpoints

- **Local LMStudio**: Run models locally for privacy and cost control
- **OpenRouter**: Access various models through a unified API
- **OpenAI**: Direct access to GPT models

### Configuration Implementation

```python
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod
import httpx
import openai
from openai import OpenAI

class LLMClient(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt."""
        pass

class LMStudioClient(LLMClient):
    """Client for local LMStudio server."""

    def __init__(self, base_url: str = "http://localhost:1234", model: str = "local-model"):
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.client = httpx.Client(timeout=60)

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate using LMStudio local API."""
        try:
            response = self.client.post(
                f"{self.base_url}/v1/chat/completions",
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": kwargs.get("temperature", 0.7),
                    "max_tokens": kwargs.get("max_tokens", 2000)
                }
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"LMStudio Error: {e}"

class OpenRouterClient(LLMClient):
    """Client for OpenRouter API."""

    def __init__(self, api_key: str, model: str = "anthropic/claude-3-haiku"):
        self.api_key = api_key
        self.model = model
        self.client = httpx.Client(timeout=60)

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate using OpenRouter API."""
        try:
            response = self.client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": kwargs.get("temperature", 0.7),
                    "max_tokens": kwargs.get("max_tokens", 2000)
                }
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"OpenRouter Error: {e}"

class OpenAIClient(LLMClient):
    """Client for OpenAI API."""

    def __init__(self, api_key: str, model: str = "gpt-4"):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate using OpenAI API."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=kwargs.get("temperature", 0.7),
                max_tokens=kwargs.get("max_tokens", 2000)
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"OpenAI Error: {e}"

class LLMManager:
    """Manager for different LLM endpoints."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.clients = {}
        self._initialize_clients()

    def _initialize_clients(self):
        """Initialize available LLM clients."""
        if "lmstudio" in self.config:
            self.clients["lmstudio"] = LMStudioClient(**self.config["lmstudio"])

        if "openrouter" in self.config:
            self.clients["openrouter"] = OpenRouterClient(**self.config["openrouter"])

        if "openai" in self.config:
            self.clients["openai"] = OpenAIClient(**self.config["openai"])

    def generate(self, prompt: str, provider: str = "auto", **kwargs) -> str:
        """Generate text using specified or auto-selected provider."""
        if provider == "auto":
            # Auto-select: prefer local, then OpenRouter, then OpenAI
            for preferred in ["lmstudio", "openrouter", "openai"]:
                if preferred in self.clients:
                    provider = preferred
                    break

        if provider not in self.clients:
            return f"Error: Provider '{provider}' not configured"

        return self.clients[provider].generate(prompt, **kwargs)

    def list_providers(self) -> list:
        """List available providers."""
        return list(self.clients.keys())

# Configuration example
LLM_CONFIG = {
    "lmstudio": {
        "base_url": "http://localhost:1234",
        "model": "codellama-34b-instruct"
    },
    "openrouter": {
        "api_key": "your-openrouter-key",
        "model": "anthropic/claude-3-sonnet"
    },
    "openai": {
        "api_key": "your-openai-key",
        "model": "gpt-4-turbo-preview"
    }
}

# Initialize global LLM manager
llm_manager = LLMManager(LLM_CONFIG)

def llm_call(provider: str, prompt: str, **kwargs) -> str:
    """Convenience function for LLM calls."""
    return llm_manager.generate(prompt, provider, **kwargs)
```

### Configuration Best Practices

- **Local First**: Use LMStudio for development and testing to reduce costs
- **Fallback Strategy**: Configure multiple providers for reliability
- **Model Selection**: Choose models based on task complexity and cost requirements
- **Rate Limiting**: Implement proper rate limiting for API-based providers
- **Error Handling**: Always handle API failures gracefully with retries

## Agentic RAG Features

The system incorporates advanced agentic capabilities that transform passive retrieval into active, intelligent code generation and validation. These features enable the RAG system to proactively understand, refine, and validate code requests.

### Question Generation from Documentation

The agentic system generates targeted questions from documentation to better understand user intent and fill knowledge gaps.

```python
from typing import List, Dict, Any
from pydantic import BaseModel

class DocumentationQuestion(BaseModel):
    question: str
    category: str  # 'clarification', 'prerequisites', 'alternatives', 'edge_cases'
    priority: int  # 1-5, higher = more important
    context_needed: List[str]  # Related documentation sections

class QuestionGenerator:
    """Generate intelligent questions from documentation context."""

    def __init__(self):
        self.templates = {
            'clarification': [
                "What specific {concept} behavior are you trying to achieve?",
                "Are you working with {version} or newer?",
                "Do you need this to work in {context} mode?",
            ],
            'prerequisites': [
                "Have you ensured {requirement} is properly configured?",
                "Are all necessary {dependencies} installed?",
                "Is your {environment} set up correctly?",
            ],
            'alternatives': [
                "Have you considered using {alternative} instead?",
                "Would {method} be more appropriate for your use case?",
                "Could {pattern} provide better performance?",
            ],
            'edge_cases': [
                "How should the system handle {edge_case}?",
                "What happens when {condition} is not met?",
                "Are there any {constraint} limitations to consider?",
            ]
        }

    def generate_questions(self, user_query: str, context_docs: List[Dict],
                          max_questions: int = 5) -> List[DocumentationQuestion]:
        """Generate relevant questions based on user query and documentation."""

        prompt = self._build_question_generation_prompt(user_query, context_docs)

        response = llm_call("auto", prompt, temperature=0.3, max_tokens=1000)

        return self._parse_questions_response(response)[:max_questions]

    def _build_question_generation_prompt(self, user_query: str,
                                        context_docs: List[Dict]) -> str:
        """Build prompt for question generation."""

        context_str = "\n".join([
            f"Section: {doc.get('title', 'Unknown')}\n"
            f"Content: {doc.get('content', '')[:500]}..."
            for doc in context_docs[:3]  # Limit context
        ])

        return f"""Analyze this user query and available documentation to generate clarifying questions.

User Query: {user_query}

Available Documentation Context:
{context_str}

Generate 3-5 specific questions that would help:
1. Clarify the user's exact requirements
2. Identify missing prerequisites or dependencies
3. Suggest alternative approaches
4. Uncover edge cases or constraints

Format each question as:
- Question: [specific question]
- Category: [clarification|prerequisites|alternatives|edge_cases]
- Priority: [1-5]
- Context: [related documentation concepts]

Questions:"""

    def _parse_questions_response(self, response: str) -> List[DocumentationQuestion]:
        """Parse LLM response into structured questions."""
        questions = []
        # Implementation for parsing structured response
        return questions

# Usage
question_gen = QuestionGenerator()
questions = question_gen.generate_questions(
    "How do I create a spiral staircase?",
    context_docs=[{"title": "Curve Objects", "content": "..."}]
)

for q in questions:
    print(f"Q: {q.question} (Priority: {q.priority})")
```

### Query Reformulation & Expansion

The system intelligently reformulates and expands user queries to improve retrieval accuracy and code generation quality.

```python
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import re

class ReformulatedQuery(BaseModel):
    original_query: str
    reformulated_query: str
    expanded_queries: List[str]
    search_terms: List[str]
    complexity_level: str  # 'basic', 'intermediate', 'advanced'
    domain_context: List[str]  # 'modeling', 'animation', 'rendering', etc.
    technical_requirements: List[str]

class QueryReformulator:
    """Reformulate and expand user queries for better retrieval."""

    def __init__(self):
        self.domain_keywords = {
            'modeling': ['mesh', 'vertices', 'edges', 'faces', 'modifiers', 'subdivision'],
            'animation': ['keyframes', 'timeline', 'armature', 'bones', 'constraints', 'drivers'],
            'rendering': ['camera', 'lighting', 'materials', 'textures', 'render', 'cycles'],
            'physics': ['rigid body', 'soft body', 'cloth', 'simulation', 'collision'],
            'scripting': ['operators', 'handlers', 'callbacks', 'automation', 'batch']
        }

    def reformulate_query(self, user_query: str,
                         context_docs: Optional[List[Dict]] = None) -> ReformulatedQuery:
        """Reformulate user query with expansion and context analysis."""

        prompt = self._build_reformulation_prompt(user_query, context_docs)

        response = llm_call("auto", prompt, temperature=0.2, max_tokens=800)

        return self._parse_reformulation_response(user_query, response)

    def _build_reformulation_prompt(self, user_query: str,
                                  context_docs: Optional[List[Dict]] = None) -> str:
        """Build prompt for query reformulation."""

        context_str = ""
        if context_docs:
            context_str = "\n".join([
                f"- {doc.get('title', '')}: {doc.get('content', '')[:200]}..."
                for doc in context_docs[:3]
            ])

        return f"""Reformulate and expand this Blender Python query for better code generation.

Original Query: {user_query}

Available Context:
{context_str}

Please provide:
1. A clear, specific reformulation of the query
2. 2-3 expanded queries that capture different aspects
3. Key search terms for documentation retrieval
4. Complexity assessment (basic/intermediate/advanced)
5. Relevant Blender domains (modeling, animation, rendering, etc.)
6. Technical requirements or constraints

Format as JSON with these fields:
- reformulated_query: string
- expanded_queries: array of strings
- search_terms: array of strings
- complexity_level: string
- domain_context: array of strings
- technical_requirements: array of strings

Response:"""

    def _parse_reformulation_response(self, original_query: str,
                                    response: str) -> ReformulatedQuery:
        """Parse reformulation response."""
        # Implementation for parsing JSON response
        # This would include error handling and fallback parsing
        pass

    def extract_domain_context(self, query: str) -> List[str]:
        """Extract Blender domains from query using keyword matching."""
        domains = []
        query_lower = query.lower()

        for domain, keywords in self.domain_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                domains.append(domain)

        return domains or ['general']

# Usage
reformulator = QueryReformulator()
result = reformulator.reformulate_query("create spiral staircase")

print(f"Reformulated: {result.reformulated_query}")
print(f"Domains: {result.domain_context}")
print(f"Complexity: {result.complexity_level}")
```

### Code Quality Assurance

Comprehensive code quality assurance with linting, formatting, testing, and smoke testing capabilities.

```python
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import subprocess
import tempfile
import pathlib
import time

class CodeQualityReport(BaseModel):
    linting_passed: bool
    linting_errors: List[str]
    formatting_applied: bool
    formatting_changes: List[str]
    static_analysis_passed: bool
    static_analysis_issues: List[Dict]
    smoke_test_passed: bool
    smoke_test_output: str
    smoke_test_errors: str
    performance_score: float  # 0-1
    quality_score: float  # 0-1

class CodeQualityAssurance:
    """Comprehensive code quality assurance system."""

    def __init__(self, blender_path: str = "blender"):
        self.blender_path = blender_path
        self.linters = ['ruff', 'flake8', 'pylint']
        self.formatters = ['black', 'autopep8']
        self.type_checkers = ['mypy', 'pyright']

    def run_full_quality_check(self, code: str,
                             blender_version: str = "current") -> CodeQualityReport:
        """Run complete quality assurance pipeline."""

        report = CodeQualityReport(
            linting_passed=True,
            linting_errors=[],
            formatting_applied=False,
            formatting_changes=[],
            static_analysis_passed=True,
            static_analysis_issues=[],
            smoke_test_passed=True,
            smoke_test_output="",
            smoke_test_errors="",
            performance_score=1.0,
            quality_score=1.0
        )

        # 1. Linting
        report.linting_passed, report.linting_errors = self._run_linting(code)

        # 2. Formatting
        formatted_code, report.formatting_changes = self._apply_formatting(code)
        if formatted_code != code:
            report.formatting_applied = True
            code = formatted_code

        # 3. Static Analysis
        report.static_analysis_passed, report.static_analysis_issues = self._run_static_analysis(code)

        # 4. Smoke Testing
        report.smoke_test_passed, report.smoke_test_output, report.smoke_test_errors = self._run_smoke_test(code)

        # 5. Performance Scoring
        report.performance_score = self._calculate_performance_score(code)

        # 6. Overall Quality Scoring
        report.quality_score = self._calculate_quality_score(report)

        return report

    def _run_linting(self, code: str) -> tuple[bool, List[str]]:
        """Run multiple linters and collect results."""
        errors = []

        for linter in self.linters:
            try:
                result = subprocess.run(
                    [linter, '--stdin-filename', 'code.py', '-'],
                    input=code,
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if result.returncode != 0:
                    errors.extend(result.stdout.split('\n'))
                    errors.extend(result.stderr.split('\n'))
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue  # Linter not available

        return len(errors) == 0, [e for e in errors if e.strip()]

    def _apply_formatting(self, code: str) -> tuple[str, List[str]]:
        """Apply code formatting and track changes."""
        original_lines = code.split('\n')
        changes = []

        for formatter in self.formatters:
            try:
                result = subprocess.run(
                    [formatter, '--stdin-filename', 'code.py', '-'],
                    input=code,
                    capture_output=True,
                    text=True,
                    timeout=15
                )
                if result.returncode == 0:
                    formatted_code = result.stdout
                    if formatted_code != code:
                        changes.append(f"Applied {formatter} formatting")
                        code = formatted_code
                    break
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue

        return code, changes

    def _run_static_analysis(self, code: str) -> tuple[bool, List[Dict]]:
        """Run static analysis tools."""
        issues = []

        for checker in self.type_checkers:
            try:
                # Write code to temporary file for analysis
                with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                    f.write(code)
                    temp_path = f.name

                result = subprocess.run(
                    [checker, temp_path],
                    capture_output=True,
                    text=True,
                    timeout=20
                )

                if result.returncode != 0:
                    issues.extend(self._parse_checker_output(result.stdout, checker))

                pathlib.Path(temp_path).unlink(missing_ok=True)

            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue

        return len(issues) == 0, issues

    def _run_smoke_test(self, code: str, timeout: int = 30) -> tuple[bool, str, str]:
        """Run smoke test in Blender."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            script_path = f.name

        try:
            result = subprocess.run([
                self.blender_path, '-b', '--python', script_path
            ], capture_output=True, text=True, timeout=timeout)

            success = result.returncode == 0
            return success, result.stdout, result.stderr

        except subprocess.TimeoutExpired:
            return False, "", f"Smoke test timed out after {timeout}s"
        finally:
            pathlib.Path(script_path).unlink(missing_ok=True)

    def _calculate_performance_score(self, code: str) -> float:
        """Calculate performance score based on code patterns."""
        score = 1.0

        # Penalize inefficient patterns
        if 'for ' in code and 'range(len(' in code:
            score -= 0.1  # Prefer enumerate or direct iteration

        if 'import *' in code:
            score -= 0.2  # Wildcard imports

        if len(code.split('\n')) > 100:
            score -= 0.1  # Very long functions

        return max(0.0, score)

    def _calculate_quality_score(self, report: CodeQualityReport) -> float:
        """Calculate overall quality score."""
        score = 1.0

        if not report.linting_passed:
            score -= 0.3

        if not report.static_analysis_passed:
            score -= 0.3

        if not report.smoke_test_passed:
            score -= 0.4

        score = score * 0.7 + report.performance_score * 0.3

        return max(0.0, score)

    def _parse_checker_output(self, output: str, checker: str) -> List[Dict]:
        """Parse checker output into structured issues."""
        issues = []
        # Implementation for parsing different checker formats
        return issues

# Usage
quality_assurance = CodeQualityAssurance()
report = quality_assurance.run_full_quality_check(generated_code)

print(f"Quality Score: {report.quality_score:.2f}")
print(f"Linting Passed: {report.linting_passed}")
print(f"Smoke Test Passed: {report.smoke_test_passed}")
```

### Metadata Management

Comprehensive metadata management to track Blender versions, platforms, and breaking changes for each code snippet.

```python
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from datetime import datetime
import json
import hashlib

class SnippetMetadata(BaseModel):
    snippet_id: str
    blender_version: str
    blender_platform: str  # 'windows', 'macos', 'linux'
    python_version: str
    creation_date: datetime
    last_validated: Optional[datetime]
    author: Optional[str]
    tags: List[str]
    dependencies: List[str]
    breaking_changes: List[Dict]  # Version -> changes
    compatibility_score: float  # 0-1
    test_results: Dict[str, Any]
    performance_metrics: Dict[str, float]
    usage_statistics: Dict[str, int]

class MetadataManager:
    """Manage comprehensive metadata for code snippets."""

    def __init__(self, metadata_store_path: str = "./snippet_metadata.json"):
        self.store_path = pathlib.Path(metadata_store_path)
        self.metadata_store = self._load_metadata_store()

    def generate_snippet_id(self, code: str) -> str:
        """Generate unique ID for code snippet."""
        code_hash = hashlib.md5(code.encode()).hexdigest()[:12]
        timestamp = int(time.time())
        return f"snippet_{timestamp}_{code_hash}"

    def create_metadata(self, code: str, blender_version: str,
                       platform: str, python_version: str) -> SnippetMetadata:
        """Create comprehensive metadata for a new snippet."""

        snippet_id = self.generate_snippet_id(code)

        # Detect dependencies
        dependencies = self._extract_dependencies(code)

        # Check for breaking changes
        breaking_changes = self._check_breaking_changes(blender_version, dependencies)

        # Initial compatibility assessment
        compatibility_score = self._calculate_compatibility(blender_version, breaking_changes)

        metadata = SnippetMetadata(
            snippet_id=snippet_id,
            blender_version=blender_version,
            blender_platform=platform,
            python_version=python_version,
            creation_date=datetime.now(),
            last_validated=None,
            author=None,
            tags=self._generate_tags(code),
            dependencies=dependencies,
            breaking_changes=breaking_changes,
            compatibility_score=compatibility_score,
            test_results={},
            performance_metrics={},
            usage_statistics={
                'times_used': 0,
                'success_rate': 0.0,
                'avg_execution_time': 0.0
            }
        )

        self.metadata_store[snippet_id] = metadata.dict()
        self._save_metadata_store()

        return metadata

    def update_validation_results(self, snippet_id: str,
                                test_results: Dict[str, Any],
                                performance_metrics: Dict[str, float]):
        """Update metadata with validation results."""

        if snippet_id not in self.metadata_store:
            return

        metadata = self.metadata_store[snippet_id]
        metadata['last_validated'] = datetime.now().isoformat()
        metadata['test_results'] = test_results
        metadata['performance_metrics'] = performance_metrics

        # Update usage statistics
        if test_results.get('success', False):
            metadata['usage_statistics']['times_used'] += 1
            # Recalculate success rate
            total_runs = metadata['usage_statistics']['times_used']
            success_runs = sum(1 for result in [test_results] if result.get('success'))
            metadata['usage_statistics']['success_rate'] = success_runs / total_runs

        self._save_metadata_store()

    def check_compatibility(self, snippet_id: str,
                          target_blender_version: str,
                          target_platform: str) -> Dict[str, Any]:
        """Check if snippet is compatible with target environment."""

        if snippet_id not in self.metadata_store:
            return {'compatible': False, 'issues': ['Snippet not found']}

        metadata = self.metadata_store[snippet_id]

        issues = []
        compatible = True

        # Check Blender version compatibility
        current_version = self._parse_version(metadata['blender_version'])
        target_version = self._parse_version(target_blender_version)

        if target_version < current_version:
            compatible = False
            issues.append(f"Target Blender {target_blender_version} is older than snippet version {metadata['blender_version']}")

        # Check platform compatibility
        if metadata['blender_platform'] != target_platform:
            # Some platform-specific checks
            if metadata['blender_platform'] == 'windows' and target_platform == 'linux':
                issues.append("Potential path separator issues when moving from Windows to Linux")

        # Check for breaking changes between versions
        breaking_changes = self._get_breaking_changes_between(
            metadata['blender_version'], target_blender_version
        )

        if breaking_changes:
            compatible = False
            issues.extend([change['description'] for change in breaking_changes])

        return {
            'compatible': compatible,
            'issues': issues,
            'breaking_changes': breaking_changes,
            'compatibility_score': metadata['compatibility_score']
        }

    def _extract_dependencies(self, code: str) -> List[str]:
        """Extract dependencies from code."""
        dependencies = []

        # Check for bpy imports
        if 'import bpy' in code or 'from bpy' in code:
            dependencies.append('bpy')

        # Check for other common Blender-related imports
        common_imports = ['bmesh', 'gpu', 'mathutils', 'aud', 'bgl']
        for imp in common_imports:
            if f'import {imp}' in code or f'from {imp}' in code:
                dependencies.append(imp)

        # Check for third-party libraries
        third_party = ['numpy', 'scipy', 'pillow', 'opencv']
        for lib in third_party:
            if f'import {lib}' in code or f'from {lib}' in code:
                dependencies.append(lib)

        return dependencies

    def _check_breaking_changes(self, blender_version: str,
                              dependencies: List[str]) -> List[Dict]:
        """Check for breaking changes in dependencies."""
        breaking_changes = []

        # Known breaking changes between Blender versions
        breaking_change_db = {
            '3.0': [
                {'dependency': 'bpy', 'description': 'Context API changes in operators'},
                {'dependency': 'bmesh', 'description': 'BMesh API updates'}
            ],
            '3.1': [
                {'dependency': 'gpu', 'description': 'GPU module API changes'}
            ],
            '3.2': [
                {'dependency': 'bpy', 'description': 'Property API updates'}
            ],
            '3.3': [
                {'dependency': 'bpy', 'description': 'Node system changes'}
            ],
            '3.4': [
                {'dependency': 'bpy', 'description': 'Geometry nodes API updates'}
            ],
            '3.5': [
                {'dependency': 'bpy', 'description': 'Asset system changes'}
            ],
            '3.6': [
                {'dependency': 'bpy', 'description': 'Python API improvements'}
            ],
            '4.0': [
                {'dependency': 'bpy', 'description': 'Major API reorganization'},
                {'dependency': 'bmesh', 'description': 'BMesh performance improvements'}
            ]
        }

        version_parts = blender_version.split('.')
        major_minor = f"{version_parts[0]}.{version_parts[1]}"

        if major_minor in breaking_change_db:
            for change in breaking_change_db[major_minor]:
                if change['dependency'] in dependencies:
                    breaking_changes.append(change)

        return breaking_changes

    def _calculate_compatibility(self, blender_version: str,
                               breaking_changes: List[Dict]) -> float:
        """Calculate compatibility score."""
        base_score = 1.0

        # Reduce score based on breaking changes
        base_score -= len(breaking_changes) * 0.2

        # Version distance penalty
        # Newer versions generally have better compatibility
        version_num = self._parse_version(blender_version)
        if version_num < 3.0:
            base_score -= 0.3  # Very old versions

        return max(0.0, base_score)

    def _generate_tags(self, code: str) -> List[str]:
        """Generate relevant tags for the code snippet."""
        tags = []

        # Domain detection
        if 'bpy.ops.mesh' in code:
            tags.append('modeling')
        if 'bpy.context.scene.frame' in code:
            tags.append('animation')
        if 'bpy.data.materials' in code:
            tags.append('materials')
        if 'bpy.ops.rigidbody' in code:
            tags.append('physics')

        # Complexity detection
        lines = len(code.split('\n'))
        if lines < 10:
            tags.append('simple')
        elif lines < 50:
            tags.append('intermediate')
        else:
            tags.append('complex')

        # Functionality detection
        if 'def ' in code:
            tags.append('function')
        if 'class ' in code:
            tags.append('class')
        if 'import bpy' in code:
            tags.append('blender-api')

        return tags

    def _parse_version(self, version_str: str) -> float:
        """Parse version string to comparable number."""
        try:
            parts = version_str.split('.')
            return float(f"{parts[0]}.{parts[1]}")
        except (ValueError, IndexError):
            return 0.0

    def _get_breaking_changes_between(self, from_version: str,
                                    to_version: str) -> List[Dict]:
        """Get breaking changes between two versions."""
        # Implementation for version comparison
        return []

    def _load_metadata_store(self) -> Dict[str, Dict]:
        """Load metadata store from file."""
        if self.store_path.exists():
            with open(self.store_path, 'r') as f:
                return json.load(f)
        return {}

    def _save_metadata_store(self):
        """Save metadata store to file."""
        with open(self.store_path, 'w') as f:
            json.dump(self.metadata_store, f, indent=2, default=str)

# Usage
metadata_manager = MetadataManager()

# Create metadata for new snippet
metadata = metadata_manager.create_metadata(
    code=generated_code,
    blender_version="4.0.0",
    platform="linux",
    python_version="3.10"
)

# Check compatibility
compatibility = metadata_manager.check_compatibility(
    snippet_id=metadata.snippet_id,
    target_blender_version="4.1.0",
    target_platform="windows"
)

print(f"Snippet ID: {metadata.snippet_id}")
print(f"Compatibility: {compatibility['compatible']}")
if not compatibility['compatible']:
    print(f"Issues: {compatibility['issues']}")
```

## RAG for General Coding Tasks

### Indexing

The indexing process transforms source code repositories into searchable knowledge:

- **Code Parsing**: Use Tree-sitter or AST parsing to extract structural elements
- **Symbol Extraction**: Identify functions, classes, imports, and call graphs
- **Metadata Attachment**: Include path, line numbers, dependencies, and usage patterns
- **Hybrid Storage**: Combine dense embeddings (transformers) with sparse indices (BM25/ripgrep)

**Recommended Libraries**:
- **LangGraph**: `/langchain-ai/langgraph` (v0.6.0) - Orchestration framework
- **Pydantic**: `/pydantic/pydantic` - Data validation and modeling
- **FAISS**: `/facebookresearch/faiss` - Vector similarity search
- **Ruff**: `/astral-sh/ruff` - Fast Python linter and formatter

### Retrieval

Multi-stage retrieval ensures comprehensive context gathering:

1. **Query Expansion**: Normalize queries with synonyms and symbol mapping
2. **Hybrid Search**: Combine dense (embeddings) and sparse (term-based) retrieval
3. **Structural Re-ranking**: Prioritize by package proximity and call graph relationships
4. **Context Windowing**: Limit to 8-12 chunks with full provenance tracking

### Graph Structure

The system uses a stateful graph architecture:

```
Router → Retriever → Planner → Code-Actions → Tester → Critic → Loop/End
```

**Key Nodes**:
- **Router**: Classify query type (read/edit/refactor/test/bugfix)
- **Retriever**: Gather relevant code context
- **Planner**: Generate execution plan with target files
- **Code-Actions**: Apply patches using unified diff format
- **Tester**: Run validation tests
- **Critic**: Evaluate results and decide iteration

### Minimal LangGraph Example

```python
from __future__ import annotations
from typing import List, Dict, Tuple, Optional
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# State definitions with comprehensive metadata
class FileSpan(BaseModel):
    path: str
    start: int
    end: int
    symbol: Optional[str] = None
    score: float = 0.0
    preview: Optional[str] = None
    language: str = "python"
    dependencies: List[str] = Field(default_factory=list)

class Plan(BaseModel):
    goal: str
    steps: List[str] = []
    target_files: List[str] = []
    estimated_complexity: str = "medium"

class Patch(BaseModel):
    diff_unified: str
    affected_files: List[str]
    change_summary: str = ""

class RunResult(BaseModel):
    command: str
    stdout: str
    stderr: str
    exit_code: int
    execution_time: float = 0.0

class GraphState(BaseModel):
    query: str
    mode: str = "auto"
    retrieved: List[FileSpan] = Field(default_factory=list)
    plan: Optional[Plan] = None
    answer: Optional[str] = None
    patch: Optional[Patch] = None
    test_result: Optional[RunResult] = None
    iterations: int = 0
    done: bool = False
    error_context: Optional[str] = None
```

### Tool Implementations

```python
import subprocess, textwrap, difflib, pathlib, re, time

def ripgrep_search(query: str, root: str = ".") -> List[FileSpan]:
    """Enhanced ripgrep search with better context extraction."""
    try:
        result = subprocess.run(
            ["rg", "--vimgrep", "-n", "-H", "-S", "-C", "3", query, root],
            capture_output=True, text=True, check=False, timeout=30
        )
        return _parse_ripgrep_output(result.stdout)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []

def read_file_span(span: FileSpan) -> FileSpan:
    """Read file with error handling and metadata extraction."""
    try:
        lines = pathlib.Path(span.path).read_text(encoding="utf-8", errors="ignore").splitlines()
        excerpt = "\n".join(lines[max(0, span.start-1):min(len(lines), span.end)])
        span.preview = excerpt[:7000]

        # Extract additional metadata
        span.language = _detect_language(span.path)
        span.dependencies = _extract_dependencies(excerpt)

        return span
    except Exception as e:
        span.preview = f"Error reading file: {e}"
        return span

def apply_unified_diff(diff_text: str, dry_run: bool = False) -> Dict[str, str]:
    """Apply unified diff with validation."""
    patch_file = pathlib.Path(".rag_patch.diff")
    patch_file.write_text(diff_text, encoding="utf-8")

    try:
        if dry_run:
            result = subprocess.run(
                ["git", "apply", "--check", str(patch_file)],
                capture_output=True, text=True
            )
            return {"status": "checked", "output": result.stdout, "error": result.stderr}

        result = subprocess.run(
            ["git", "apply", "--index", str(patch_file)],
            capture_output=True, text=True
        )

        return {
            "status": "applied" if result.returncode == 0 else "failed",
            "output": result.stdout,
            "error": result.stderr
        }
    finally:
        patch_file.unlink(missing_ok=True)

def run_test_suite(test_cmd: List[str], timeout: int = 60) -> RunResult:
    """Run test suite with timeout and performance metrics."""
    start_time = time.time()
    try:
        result = subprocess.run(
            test_cmd, capture_output=True, text=True, timeout=timeout
        )
        execution_time = time.time() - start_time
        return RunResult(
            command=" ".join(test_cmd),
            stdout=result.stdout[-10000:],
            stderr=result.stderr[-10000:],
            exit_code=result.returncode,
            execution_time=execution_time
        )
    except subprocess.TimeoutExpired:
        return RunResult(
            command=" ".join(test_cmd),
            stdout="",
            stderr=f"Test execution timed out after {timeout}s",
            exit_code=-1,
            execution_time=time.time() - start_time
        )
```

### Graph Nodes Implementation

```python
def route_query_node(state: GraphState) -> GraphState:
    """Intelligent query routing based on content analysis."""
    query_lower = state.query.lower()

    # Classification logic
    if any(keyword in query_lower for keyword in ["bug", "exception", "traceback", "fix", "error"]):
        state.mode = "bugfix"
    elif any(keyword in query_lower for keyword in ["refactor", "rename", "extract", "optimize"]):
        state.mode = "refactor"
    elif any(keyword in query_lower for keyword in ["test", "unit test", "pytest", "coverage"]):
        state.mode = "test"
    elif any(keyword in query_lower for keyword in ["add", "create", "implement", "feature"]):
        state.mode = "feature"
    else:
        state.mode = "read"

    return state

def retrieve_context_node(state: GraphState) -> GraphState:
    """Multi-source context retrieval."""
    # Dense retrieval
    dense_results = VECTOR.search(state.query, k=20)

    # Sparse retrieval
    sparse_results = ripgrep_search(state.query)

    # Combine and deduplicate
    key = lambda s: (s.path, s.start, s.end)
    merged = {key(s): s for s in (dense_results + sparse_results)}.values()

    # Rank and limit
    ranked = sorted(merged, key=lambda s: s.score, reverse=True)[:12]
    state.retrieved = [read_file_span(s) for s in ranked]

    return state

def generate_plan_node(state: GraphState) -> GraphState:
    """Generate execution plan using retrieved context."""
    context = _build_context_string(state.retrieved)

    prompt = f"""Analyze the user request and available context to create a structured plan.

User Request: {state.query}
Mode: {state.mode}

Available Context:
{context}

Generate a JSON plan with:
- goal: Clear objective
- steps: Ordered list of atomic actions
- target_files: Files that will be modified
- estimated_complexity: "low" | "medium" | "high"

Respond with valid JSON only."""

    plan_json = llm_call("auto", prompt)
    state.plan = Plan.parse_raw(plan_json)

    return state

def generate_patch_node(state: GraphState) -> GraphState:
    """Generate unified diff patch."""
    context = _build_context_string(state.retrieved)

    prompt = f"""Generate a minimal, surgical unified diff patch.

Goal: {state.plan.goal}
Target Files: {', '.join(state.plan.target_files)}

Context:
{context}

Requirements:
- Output ONLY a valid unified diff
- Use proper file paths
- Include helpful commit message in diff
- Make changes minimal but complete
- Follow existing code style

Unified Diff:"""

    diff = llm_call("auto", prompt)
    state.patch = Patch(
        diff_unified=diff,
        affected_files=state.plan.target_files,
        change_summary=f"Implementation of: {state.plan.goal}"
    )

    return state

def validate_changes_node(state: GraphState) -> GraphState:
    """Apply and validate changes."""
    # Dry run first
    check_result = apply_unified_diff(state.patch.diff_unified, dry_run=True)

    if check_result["status"] == "failed":
        state.error_context = check_result["error"]
        return state

    # Apply changes
    apply_result = apply_unified_diff(state.patch.diff_unified)

    if apply_result["status"] == "applied":
        # Run tests
        test_cmd = ["pytest", "-q", "--tb=short"] if _has_pytest() else ["python", "-m", "unittest", "discover"]
        state.test_result = run_test_suite(test_cmd)
    else:
        state.error_context = apply_result["error"]

    return state

def critique_results_node(state: GraphState) -> GraphState:
    """Evaluate results and decide next action."""
    if not state.test_result:
        state.done = True
        state.error_context = "No test results available"
        return state

    # Success criteria
    success = (
        state.test_result.exit_code == 0 and
        "FAILED" not in state.test_result.stdout and
        len(state.test_result.stderr.strip()) < 100  # Allow minor warnings
    )

    if success or state.iterations >= 2:
        state.done = True
        state.answer = _format_success_response(state)
    else:
        state.iterations += 1
        state.done = False
        # Add error context for next iteration
        state.error_context = f"Iteration {state.iterations}: {state.test_result.stderr}"

    return state
```

### Graph Construction and Execution

```python
def build_coding_rag_graph() -> StateGraph:
    """Build the complete coding RAG graph."""
    graph = StateGraph(GraphState)

    # Add nodes
    graph.add_node("route", route_query_node)
    graph.add_node("retrieve", retrieve_context_node)
    graph.add_node("plan", generate_plan_node)
    graph.add_node("generate_patch", generate_patch_node)
    graph.add_node("validate", validate_changes_node)
    graph.add_node("critique", critique_results_node)

    # Define flow
    graph.set_entry_point("route")
    graph.add_edge("route", "retrieve")
    graph.add_edge("retrieve", "plan")
    graph.add_edge("plan", "generate_patch")
    graph.add_edge("generate_patch", "validate")
    graph.add_edge("validate", "critique")

    # Conditional edges
    graph.add_conditional_edges(
        "critique",
        lambda s: "retrieve" if not s.done else END,
        {"retrieve": "retrieve", END: END}
    )

    return graph

# Initialize and run
app = build_coding_rag_graph().compile(checkpointer=MemorySaver())

def run_coding_task(query: str) -> Dict:
    """Execute coding task with full RAG pipeline."""
    initial_state = GraphState(query=query)

    final_state = None
    for output in app.stream(initial_state):
        final_state = output

    return {
        "success": final_state.done and final_state.test_result and final_state.test_result.exit_code == 0,
        "result": final_state.answer,
        "iterations": final_state.iterations,
        "patch": final_state.patch.diff_unified if final_state.patch else None,
        "error": final_state.error_context
    }
```

### Enhanced Prompts

**Planner Prompt**:
```python
SYSTEM_PROMPT = """You are an expert software engineer analyzing user requests to create actionable plans.

Consider:
- Code architecture and patterns
- Testing requirements
- Edge cases and error handling
- Performance implications
- Security considerations

Output structured JSON plan only."""

USER_PROMPT_TEMPLATE = """
User Request: {query}
Detected Mode: {mode}

Available Context:
{context}

Create implementation plan:"""
```

**Patcher Prompt**:
```python
SYSTEM_PROMPT = """You are a precision code editor. Generate minimal, correct unified diffs.

Rules:
- Changes must be surgical and complete
- Preserve existing style and patterns
- Include proper error handling
- Add tests for new functionality
- Use git-compatible unified diff format

Output diff only."""

USER_PROMPT_TEMPLATE = """
Task: {goal}

Files to modify: {target_files}

Code Context:
{context}

Generate unified diff:"""
```

### Best Practices

**Retrieval Optimization**:
- Use hybrid search (dense + sparse) for comprehensive coverage
- Implement re-ranking based on structural relationships
- Limit context windows to prevent token overflow
- Cache frequently accessed code regions

**Patch Generation**:
- Always validate diffs before application
- Use dry-run mode for safety checks
- Include rollback mechanisms
- Generate descriptive commit messages

**Testing Integration**:
- Run tests in isolated environments
- Implement timeout protection
- Capture and analyze test output
- Support multiple testing frameworks

**Error Handling**:
- Implement graceful degradation
- Provide detailed error context
- Support manual intervention points
- Log all operations for debugging

This enhanced LangGraph implementation provides a robust foundation for automated code generation and modification tasks.

Evaluating your coding RAG

Retrieval hit@K by oracle questions ("Where is FooError raised?").

Task success rate on a curated set (bugfixes/refactors).

Patch acceptance (applies cleanly, tests pass).

Context efficiency (tokens per success).

Add ablation: dense vs hybrid vs +structure.

Hard-won pitfalls

Don't index huge, generated files; exclude dist/, node_modules/, migrations (unless requested).

Prevent "context drift": keep K small, prefer structural proximity over raw cosine score.

Make the LLM always output diffs; never free-form code blocks for edits.

Sandbox test runs; timeouts and output truncation.

# Blender Python Code Snippets RAG

This section presents a specialized RAG system for generating and validating Blender Python code snippets. Building on the general coding RAG principles, this implementation focuses on Blender's unique API structure and runtime requirements.

## Agentic Blender RAG Architecture

The agentic Blender RAG system incorporates advanced AI capabilities to proactively understand user requirements, generate high-quality code, and ensure robust validation across different Blender versions and platforms.

### Intelligent Query Processing

```python
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from datetime import datetime

class AgenticBlenderQuery(BaseModel):
    original_query: str
    reformulated_queries: List[str]
    domain_analysis: Dict[str, float]  # Domain -> confidence score
    complexity_assessment: str
    blender_version_requirements: List[str]
    platform_considerations: List[str]
    expected_output_types: List[str]  # 'mesh', 'material', 'animation', etc.

class AgenticBlenderRAG:
    """Agentic RAG system specifically designed for Blender Python development."""

    def __init__(self, blender_version: str = "4.0", platform: str = "auto"):
        self.blender_version = blender_version
        self.platform = platform
        self.question_generator = QuestionGenerator()
        self.query_reformulator = QueryReformulator()
        self.quality_assurance = CodeQualityAssurance()
        self.metadata_manager = MetadataManager()

    def process_query(self, user_query: str,
                     context_docs: Optional[List[Dict]] = None) -> AgenticBlenderQuery:
        """Process user query with agentic intelligence."""

        # 1. Reformulate and expand the query
        reformulated = self.query_reformulator.reformulate_query(user_query, context_docs)

        # 2. Generate clarifying questions
        questions = self.question_generator.generate_questions(
            user_query, context_docs or []
        )

        # 3. Analyze domain and requirements
        domain_analysis = self._analyze_blender_domains(user_query)
        complexity = self._assess_complexity(user_query, reformulated)

        # 4. Determine version and platform requirements
        version_reqs = self._determine_version_requirements(user_query, domain_analysis)
        platform_considerations = self._analyze_platform_impact(user_query)

        # 5. Predict expected output types
        output_types = self._predict_output_types(user_query, domain_analysis)

        return AgenticBlenderQuery(
            original_query=user_query,
            reformulated_queries=reformulated.expanded_queries,
            domain_analysis=domain_analysis,
            complexity_assessment=complexity,
            blender_version_requirements=version_reqs,
            platform_considerations=platform_considerations,
            expected_output_types=output_types
        )

    def generate_code_with_quality_assurance(self, processed_query: AgenticBlenderQuery,
                                           context_docs: List[Dict]) -> Dict[str, Any]:
        """Generate code with comprehensive quality assurance."""

        # 1. Build enhanced context
        enhanced_context = self._build_enhanced_context(processed_query, context_docs)

        # 2. Generate code using multiple strategies
        code_candidates = self._generate_code_candidates(processed_query, enhanced_context)

        # 3. Quality assurance for each candidate
        validated_candidates = []
        for candidate in code_candidates:
            quality_report = self.quality_assurance.run_full_quality_check(
                candidate['code'], self.blender_version
            )

            # Create metadata
            metadata = self.metadata_manager.create_metadata(
                code=candidate['code'],
                blender_version=self.blender_version,
                platform=self.platform,
                python_version="3.10"
            )

            # Update with validation results
            self.metadata_manager.update_validation_results(
                metadata.snippet_id,
                quality_report.dict(),
                candidate.get('performance_metrics', {})
            )

            validated_candidates.append({
                'code': candidate['code'],
                'quality_report': quality_report,
                'metadata': metadata,
                'compatibility_check': self._check_environment_compatibility(metadata)
            })

        # 4. Select best candidate
        best_candidate = self._select_best_candidate(validated_candidates)

        return best_candidate

    def _analyze_blender_domains(self, query: str) -> Dict[str, float]:
        """Analyze which Blender domains are relevant to the query."""
        domains = {
            'modeling': ['mesh', 'vertices', 'edges', 'faces', 'modifiers', 'subdivision', 'extrude'],
            'animation': ['keyframes', 'timeline', 'armature', 'bones', 'constraints', 'drivers', 'pose'],
            'rendering': ['camera', 'lighting', 'materials', 'textures', 'render', 'cycles', 'eevee'],
            'physics': ['rigid body', 'soft body', 'cloth', 'simulation', 'collision', 'dynamics'],
            'scripting': ['operators', 'handlers', 'callbacks', 'automation', 'batch', 'addon'],
            'geometry_nodes': ['geometry nodes', 'node', 'attribute', 'field', 'instances'],
            'uv_mapping': ['uv', 'unwrap', 'texture coordinates', 'seams'],
            'sculpting': ['sculpt', 'brush', 'mask', 'dynamic topology']
        }

        scores = {}
        query_lower = query.lower()

        for domain, keywords in domains.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            if score > 0:
                scores[domain] = min(score / len(keywords), 1.0)

        return scores

    def _assess_complexity(self, original_query: str,
                          reformulated: ReformulatedQuery) -> str:
        """Assess the complexity level of the task."""
        complexity_indicators = {
            'basic': ['create', 'add', 'simple', 'basic'],
            'intermediate': ['animate', 'material', 'texture', 'constraint', 'modifier'],
            'advanced': ['simulation', 'procedural', 'optimization', 'complex', 'multi-step']
        }

        query_lower = original_query.lower()
        complexity_score = 0

        for level, indicators in complexity_indicators.items():
            if any(indicator in query_lower for indicator in indicators):
                if level == 'basic':
                    complexity_score = max(complexity_score, 1)
                elif level == 'intermediate':
                    complexity_score = max(complexity_score, 2)
                elif level == 'advanced':
                    complexity_score = max(complexity_score, 3)

        if complexity_score >= 3:
            return 'advanced'
        elif complexity_score >= 2:
            return 'intermediate'
        else:
            return 'basic'

    def _determine_version_requirements(self, query: str,
                                      domain_analysis: Dict[str, float]) -> List[str]:
        """Determine Blender version requirements based on query and domains."""
        requirements = []

        # Version-specific features
        version_features = {
            '3.0': ['geometry nodes', 'fields'],
            '3.1': ['asset browser', 'essentials'],
            '3.2': ['cryptomatte', 'light groups'],
            '3.3': ['hair curves', 'simulation nodes'],
            '3.4': ['node tools', 'menu switches'],
            '3.5': ['real-time compositor'],
            '3.6': ['bake nodes', 'texture baking'],
            '4.0': ['cycles x', 'light linking', 'shadow caustics']
        }

        for version, features in version_features.items():
            for feature in features:
                if feature in query.lower():
                    requirements.append(f"Blender {version}+ required for {feature}")

        # Domain-specific version requirements
        if 'geometry_nodes' in domain_analysis and domain_analysis['geometry_nodes'] > 0.5:
            requirements.append("Blender 3.0+ required for Geometry Nodes")

        if 'hair_curves' in query.lower():
            requirements.append("Blender 3.3+ required for Hair Curves")

        return requirements

    def _analyze_platform_impact(self, query: str) -> List[str]:
        """Analyze platform-specific considerations."""
        considerations = []

        # GPU-specific features
        if any(term in query.lower() for term in ['gpu', 'cycles', 'render', 'viewport']):
            considerations.append("GPU acceleration may vary between platforms")

        # Path-specific operations
        if any(term in query.lower() for term in ['file', 'path', 'import', 'export']):
            considerations.append("File path handling may require platform-specific adjustments")

        # Performance considerations
        if any(term in query.lower() for term in ['performance', 'optimization', 'heavy']):
            considerations.append("Performance characteristics may differ across platforms")

        return considerations

    def _predict_output_types(self, query: str,
                            domain_analysis: Dict[str, float]) -> List[str]:
        """Predict what types of Blender objects will be created."""
        output_types = []

        # Direct mappings
        type_indicators = {
            'mesh': ['cube', 'sphere', 'cylinder', 'mesh', 'object'],
            'material': ['material', 'texture', 'shader', 'surface'],
            'animation': ['animate', 'keyframes', 'timeline', 'motion'],
            'lighting': ['light', 'lamp', 'illumination', 'shadow'],
            'camera': ['camera', 'view', 'perspective', 'orthographic'],
            'curve': ['curve', 'path', 'bezier', 'nurbs'],
            'armature': ['armature', 'bones', 'rig', 'skeleton'],
            'particles': ['particles', 'emitter', 'hair', 'fur']
        }

        query_lower = query.lower()
        for output_type, indicators in type_indicators.items():
            if any(indicator in query_lower for indicator in indicators):
                output_types.append(output_type)

        # Domain-based predictions
        if 'modeling' in domain_analysis and domain_analysis['modeling'] > 0.3:
            if 'mesh' not in output_types:
                output_types.append('mesh')

        if 'animation' in domain_analysis and domain_analysis['animation'] > 0.3:
            if 'animation' not in output_types:
                output_types.append('animation')

        return output_types

    def _build_enhanced_context(self, processed_query: AgenticBlenderQuery,
                              context_docs: List[Dict]) -> Dict[str, Any]:
        """Build enhanced context for code generation."""
        return {
            'original_query': processed_query.original_query,
            'reformulated_queries': processed_query.reformulated_queries,
            'domain_analysis': processed_query.domain_analysis,
            'complexity': processed_query.complexity_assessment,
            'version_requirements': processed_query.blender_version_requirements,
            'platform_considerations': processed_query.platform_considerations,
            'expected_outputs': processed_query.expected_output_types,
            'documentation': context_docs,
            'blender_version': self.blender_version,
            'platform': self.platform
        }

    def _generate_code_candidates(self, processed_query: AgenticBlenderQuery,
                                enhanced_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate multiple code candidates using different strategies."""
        candidates = []

        # Strategy 1: Direct generation
        prompt1 = self._build_generation_prompt(processed_query, enhanced_context, "direct")
        code1 = llm_call("auto", prompt1, temperature=0.3)
        candidates.append({
            'code': code1,
            'strategy': 'direct',
            'confidence': 0.8
        })

        # Strategy 2: Step-by-step generation
        prompt2 = self._build_generation_prompt(processed_query, enhanced_context, "step_by_step")
        code2 = llm_call("auto", prompt2, temperature=0.2)
        candidates.append({
            'code': code2,
            'strategy': 'step_by_step',
            'confidence': 0.9
        })

        # Strategy 3: Template-based generation
        if processed_query.complexity_assessment == 'basic':
            template_code = self._get_basic_template(processed_query.expected_output_types[0])
            prompt3 = self._build_template_prompt(processed_query, enhanced_context, template_code)
            code3 = llm_call("auto", prompt3, temperature=0.1)
            candidates.append({
                'code': code3,
                'strategy': 'template_based',
                'confidence': 0.95
            })

        return candidates

    def _build_generation_prompt(self, processed_query: AgenticBlenderQuery,
                               enhanced_context: Dict[str, Any],
                               strategy: str) -> str:
        """Build generation prompt based on strategy."""

        base_prompt = f"""Generate Blender Python code for: {processed_query.original_query}

Context Analysis:
- Complexity: {processed_query.complexity_assessment}
- Domains: {', '.join(processed_query.domain_analysis.keys())}
- Expected Outputs: {', '.join(processed_query.expected_output_types)}
- Blender Version: {enhanced_context['blender_version']}

Requirements:
{chr(10).join(f"- {req}" for req in processed_query.blender_version_requirements)}
{chr(10).join(f"- {consideration}" for consideration in processed_query.platform_considerations)}

Available Documentation:
{chr(10).join(f"- {doc.get('title', 'Unknown')}" for doc in enhanced_context['documentation'][:3])}

"""

        if strategy == "direct":
            return base_prompt + """
Generate complete, working Blender Python code that addresses the user's request.
Include proper error handling and comments.
"""
        elif strategy == "step_by_step":
            return base_prompt + """
Generate code step by step, explaining each major section.
Include detailed comments explaining the Blender API usage.
"""
        else:
            return base_prompt + """
Generate code following Blender Python best practices.
Ensure compatibility with the specified Blender version.
"""

    def _get_basic_template(self, output_type: str) -> str:
        """Get basic template for simple operations."""
        templates = {
            'mesh': '''
import bpy

# Clear existing objects
bpy.ops.object.select_all(action='DESELECT')
bpy.ops.object.select_by_type(type='MESH')
bpy.ops.object.delete(use_global=False)

# Create basic mesh
bpy.ops.mesh.primitive_cube_add(size=2, location=(0, 0, 0))
obj = bpy.context.active_object

# Add basic material
mat = bpy.data.materials.new(name="BasicMaterial")
mat.diffuse_color = (0.8, 0.8, 0.8, 1)
obj.data.materials.append(mat)
''',
            'material': '''
import bpy

# Create new material
mat = bpy.data.materials.new(name="NewMaterial")
mat.use_nodes = True

# Get nodes
nodes = mat.node_tree.nodes
principled = nodes["Principled BSDF"]

# Modify material properties
principled.inputs["Base Color"].default_value = (1, 0, 0, 1)  # Red
principled.inputs["Metallic"].default_value = 0.5
principled.inputs["Roughness"].default_value = 0.3

# Assign to selected objects
for obj in bpy.context.selected_objects:
    if obj.type == 'MESH':
        obj.data.materials.append(mat)
'''
        }

        return templates.get(output_type, templates['mesh'])

    def _build_template_prompt(self, processed_query: AgenticBlenderQuery,
                             enhanced_context: Dict[str, Any],
                             template: str) -> str:
        """Build prompt for template-based generation."""
        return f"""Starting with this template:

```python
{template}
```

Customize the template to address this specific request: {processed_query.original_query}

Requirements:
- Maintain the basic structure but adapt for the specific use case
- Ensure compatibility with Blender {enhanced_context['blender_version']}
- Add appropriate error handling
- Include relevant comments

Generate the customized code:"""

    def _check_environment_compatibility(self, metadata: SnippetMetadata) -> Dict[str, Any]:
        """Check compatibility with current environment."""
        return self.metadata_manager.check_compatibility(
            metadata.snippet_id,
            self.blender_version,
            self.platform
        )

    def _select_best_candidate(self, validated_candidates: List[Dict]) -> Dict[str, Any]:
        """Select the best candidate based on quality metrics."""
        if not validated_candidates:
            return None

        # Scoring criteria
        best_candidate = None
        best_score = -1

        for candidate in validated_candidates:
            score = 0

            # Quality score (40%)
            score += candidate['quality_report'].quality_score * 0.4

            # Compatibility score (30%)
            score += candidate['compatibility_check']['compatibility_score'] * 0.3

            # Performance score (20%)
            perf_score = candidate['quality_report'].performance_score
            score += perf_score * 0.2

            # Prefer template-based for simple tasks (10%)
            if candidate.get('strategy') == 'template_based':
                score += 0.1

            if score > best_score:
                best_score = score
                best_candidate = candidate

        return best_candidate

# Usage
agentic_rag = AgenticBlenderRAG(blender_version="4.0", platform="linux")

# Process user query
processed = agentic_rag.process_query("Create a red cube with a metallic material")

# Generate code with quality assurance
result = agentic_rag.generate_code_with_quality_assurance(processed, context_docs)

print(f"Generated code quality score: {result['quality_report'].quality_score}")
print(f"Compatibility score: {result['compatibility_check']['compatibility_score']}")
print(f"Snippet ID: {result['metadata'].snippet_id}")
```

### System Architecture

The Blender RAG system extends the general framework with:

- **Blender-specific ingestion** using Sphinx object inventory
- **Runtime validation** in headless Blender environment
- **Automatic healing** for common Blender API issues
- **Visual artifact generation** for result verification
- **Agentic query processing** with intelligent reformulation
- **Comprehensive quality assurance** with linting, testing, and smoke testing
- **Metadata management** for version compatibility and breaking changes

### Ingestion

#### Documentation Processing

Blender's Python API documentation provides comprehensive coverage of all available modules, classes, and functions. The ingestion process leverages Sphinx's object inventory for reliable symbol mapping.

**Key Libraries**:
- **HTTPX**: `/encode/httpx` - Next-generation HTTP client for documentation fetching
- **Selectolax**: Fast HTML parsing library
- **Sphobjinv**: Sphinx object inventory handling (Note: Limited direct matches available)

#### Implementation

```python
# ingest_blender_docs.py
# Dependencies: httpx, selectolax, sphobjinv, faiss-cpu, numpy
from sphobjinv import Inventory
import httpx, json, numpy as np
from typing import List, Dict, Optional
from pydantic import BaseModel

class BlenderDocChunk(BaseModel):
    symbol: str
    url: str
    code: Optional[str] = None
    text: str
    module: str
    blender_version: str = "current"
    examples: List[str] = []

def load_blender_inventory(inv_url: str) -> List[tuple]:
    """Load Blender's Sphinx object inventory."""
    try:
        inv = Inventory(url=inv_url)
        return [(obj.name, obj.uri) for obj in inv.objects]
    except Exception as e:
        print(f"Failed to load inventory: {e}")
        return []

def fetch_documentation_page(base_url: str, uri: str) -> Optional[str]:
    """Fetch and parse documentation page."""
    try:
        url = f"{base_url.rstrip('/')}/{uri}"
        response = httpx.get(url, timeout=30)
        response.raise_for_status()
        return response.text
    except Exception as e:
        print(f"Failed to fetch {url}: {e}")
        return None

def extract_code_examples(html_content: str) -> List[str]:
    """Extract Python code examples from HTML."""
    # Use selectolax or BeautifulSoup for parsing
    # Extract content from <div class="highlight-python"> blocks
    examples = []
    # Implementation for code extraction
    return examples

def chunk_blender_documentation(base_url: str, symbol_uri_pairs: List[tuple]) -> List[BlenderDocChunk]:
    """Process Blender documentation into chunks."""
    chunks = []

    for symbol, uri in symbol_uri_pairs[:100]:  # Limit for demo
        html_content = fetch_documentation_page(base_url, uri)
        if not html_content:
            continue

        # Extract relevant information
        examples = extract_code_examples(html_content)

        # Create chunk
        chunk = BlenderDocChunk(
            symbol=symbol,
            url=f"{base_url.rstrip('/')}/{uri}",
            code=examples[0] if examples else None,
            text=extract_summary_text(html_content),
            module=extract_module_name(symbol),
            examples=examples
        )
        chunks.append(chunk)

    return chunks

def embed_and_index_chunks(chunks: List[BlenderDocChunk]) -> None:
    """Create vector embeddings and FAISS index."""
    # Implementation for embedding and indexing
    # Use sentence-transformers or OpenAI embeddings
    # Store in FAISS with metadata
    pass

# Usage
if __name__ == "__main__":
    base_url = "https://docs.blender.org/api/current"
    inv_url = f"{base_url}/objects.inv"

    symbol_pairs = load_blender_inventory(inv_url)
    chunks = chunk_blender_documentation(base_url, symbol_pairs)
    embed_and_index_chunks(chunks)

    print(f"Processed {len(chunks)} Blender API chunks")
```

#### Chunking Strategy

- **Symbol-centric**: One API symbol per chunk (e.g., `bpy.ops.mesh.primitive_cube_add`)
- **Context preservation**: Include surrounding documentation and examples
- **Version awareness**: Tag chunks with Blender version information
- **Example extraction**: Preserve code examples for generation context

### Retrieval

#### Hybrid Search for Blender API

```python
from typing import List, Dict, Any
import numpy as np
from faiss import IndexFlatIP

class BlenderRetriever:
    def __init__(self, index_path: str, metadata_path: str):
        self.index = self.load_faiss_index(index_path)
        self.metadata = self.load_metadata(metadata_path)

    def search(self, query: str, k: int = 8) -> List[Dict[str, Any]]:
        """Hybrid search combining dense and sparse retrieval."""
        # Dense retrieval
        query_embedding = self.embed_query(query)
        scores, indices = self.index.search(query_embedding.reshape(1, -1), k * 2)

        # Sparse retrieval (symbol matching)
        sparse_results = self.symbol_based_search(query)

        # Combine and rerank
        combined = self.combine_results(scores[0], indices[0], sparse_results)

        return combined[:k]

    def symbol_based_search(self, query: str) -> List[Dict]:
        """Search based on Blender API symbol patterns."""
        results = []
        query_lower = query.lower()

        for i, meta in enumerate(self.metadata):
            symbol = meta.get('symbol', '').lower()
            module = meta.get('module', '').lower()

            # Exact symbol match
            if symbol in query_lower:
                results.append({'index': i, 'score': 1.0, 'type': 'exact_symbol'})

            # Module match
            elif module in query_lower:
                results.append({'index': i, 'score': 0.8, 'type': 'module_match'})

            # Fuzzy matching for related symbols
            elif self.fuzzy_match(symbol, query_lower):
                results.append({'index': i, 'score': 0.6, 'type': 'fuzzy_match'})

        return results

    def combine_results(self, dense_scores: np.ndarray, dense_indices: np.ndarray,
                       sparse_results: List[Dict]) -> List[Dict]:
        """Combine dense and sparse results with reranking."""
        # Implementation for result combination
        # Apply Blender-specific reranking (same module, related operations)
        pass
```

#### Retrieval Features

- **API-aware ranking**: Prioritize symbols from same module (e.g., `bpy.ops.mesh.*`)
- **Context preservation**: Include related symbols and operations
- **Version compatibility**: Filter by Blender version when specified
- **Example prioritization**: Boost chunks containing working code examples

### Validation Toolchain

#### Static Analysis

The system employs multiple static analysis tools for comprehensive code validation:

**Primary Tools**:
- **Ruff**: `/astral-sh/ruff` (v0.x) - Fast Python linter and formatter
- **MyPy**: Static type checker with Blender stubs
- **Pyright**: Microsoft's Python type checker

#### Blender-Specific Validation

```python
# blender_stubs.py - Type stubs for static analysis
from typing import Any, Optional, Union

class bpy:
    class ops:
        class mesh:
            @staticmethod
            def primitive_cube_add(size: float = 2.0, location: tuple = (0, 0, 0)) -> None:
                """Add a cube mesh primitive."""
                pass

        class object:
            @staticmethod
            def modifier_add(type: str) -> None:
                """Add a modifier to the active object."""
                pass

    class data:
        objects: Any
        meshes: Any

    class context:
        active_object: Optional[Any]
        scene: Any

class Object:
    modifiers: Any
    data: Any
```

#### Runtime Validation

Headless Blender execution provides the most reliable validation method:

```python
# blender_validator.py
import subprocess
import tempfile
import pathlib
from typing import Dict, Any
import json

def validate_blender_snippet(code: str, timeout: int = 30) -> Dict[str, Any]:
    """Execute code snippet in headless Blender."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        script_path = f.name

    try:
        result = subprocess.run([
            'blender', '-b', '--python', script_path
        ], capture_output=True, text=True, timeout=timeout)

        return {
            'success': result.returncode == 0,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'exit_code': result.returncode
        }
    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'stdout': '',
            'stderr': f'Execution timed out after {timeout}s',
            'exit_code': -1
        }
    finally:
        pathlib.Path(script_path).unlink(missing_ok=True)
```

### vLLM Critic Integration

**Note**: The system now uses the configurable LLM manager instead of dedicated vLLM endpoints. This provides greater flexibility across different model providers while maintaining the same critic functionality.

#### Critic Architecture

```python
from typing import Dict, Any, List
from pydantic import BaseModel

class ValidationResult(BaseModel):
    score: float  # 0.0 to 1.0
    reasons: List[str]
    suggestions: List[str]
    api_compliance: float
    runtime_safety: float

class BlenderCritic:
    """LLM-based critic for Blender code evaluation."""

    def __init__(self):
        # Uses the global llm_manager for generation
        pass

    def evaluate_snippet(self,
                        snippet: str,
                        context: List[Dict],
                        validation_results: Dict) -> ValidationResult:
        """Evaluate generated Blender code using vLLM critic."""

        prompt = self.build_critic_prompt(snippet, context, validation_results)

        # Call configured LLM endpoint instead of vLLM
        response = llm_manager.generate(prompt, "auto", temperature=0.3, max_tokens=1000)

        return self.parse_critic_response(response)

    def build_critic_prompt(self, snippet: str, context: List[Dict],
                          validation_results: Dict) -> str:
        """Build comprehensive evaluation prompt."""

        context_str = "\n".join([
            f"Symbol: {c.get('symbol', '')}\n"
            f"Documentation: {c.get('text', '')}\n"
            f"Example: {c.get('code', 'N/A')}"
            for c in context[:5]  # Limit context
        ])

        return f"""Evaluate this Blender Python code snippet:

CODE SNIPPET:
```python
{snippet}
```

CONTEXT (API Documentation):
{context_str}

VALIDATION RESULTS:
- Static Analysis: {validation_results.get('static', 'N/A')}
- Runtime Test: {validation_results.get('runtime', 'N/A')}

Evaluate on:
1. API Correctness - Uses proper Blender API calls
2. Context Safety - Avoids GUI-only operations
3. Runtime Safety - Handles errors and edge cases
4. Code Quality - Follows Python/Blender best practices

Provide score (0-1) and detailed feedback in JSON format."""

    def parse_critic_response(self, response: str) -> ValidationResult:
        """Parse vLLM response into structured result."""
        # Implementation for response parsing
        pass
```

### LangGraph Integration

#### Blender-Specific State

```python
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class BlenderChunk(BaseModel):
    symbol: str
    url: str
    text: str
    code: Optional[str] = None
    module: str
    score: float = 0.0

class BlenderValidationReport(BaseModel):
    static_ok: bool
    runtime_success: bool
    runtime_output: str
    runtime_errors: str
    render_available: bool
    render_path: Optional[str] = None

class BlenderGraphState(BaseModel):
    request: str
    processed_query: Optional[AgenticBlenderQuery] = None
    retrieved: List[BlenderChunk] = Field(default_factory=list)
    snippet: Optional[str] = None
    validation: Optional[BlenderValidationReport] = None
    quality_report: Optional[CodeQualityReport] = None
    metadata: Optional[SnippetMetadata] = None
    critic_score: float = 0.0
    critic_feedback: List[str] = Field(default_factory=list)
    done: bool = False
    iterations: int = 0
    healed_snippet: Optional[str] = None
    agentic_rag: Optional[AgenticBlenderRAG] = None
```

#### Graph Nodes

```python
def blender_retrieve_node(state: BlenderGraphState) -> BlenderGraphState:
    """Retrieve Blender API documentation."""
    retriever = BlenderRetriever()
    state.retrieved = retriever.search(state.request, k=10)
    return state

def blender_query_processing_node(state: BlenderGraphState) -> BlenderGraphState:
    """Process user query with agentic intelligence."""
    if not state.agentic_rag:
        state.agentic_rag = AgenticBlenderRAG()

    # Process query with agentic capabilities
    state.processed_query = state.agentic_rag.process_query(
        state.request, [chunk.dict() for chunk in state.retrieved]
    )

    return state

def blender_generate_node(state: BlenderGraphState) -> BlenderGraphState:
    """Generate Blender Python code using agentic RAG."""
    if not state.processed_query:
        # Fallback to simple generation if no processed query
        context = build_blender_context(state.retrieved)
        prompt = f"""Write Blender Python code for: {state.request}

Available API context:
{context}

Requirements:
- Use bpy.data and bpy.ops appropriately
- Handle active object and scene context
- Include error handling
- Follow Blender Python best practices

Generate complete, runnable code:"""

        state.snippet = llm_call("auto", prompt)
    else:
        # Use agentic generation with quality assurance
        result = state.agentic_rag.generate_code_with_quality_assurance(
            state.processed_query, [chunk.dict() for chunk in state.retrieved]
        )

        state.snippet = result['code']
        state.quality_report = result['quality_report']
        state.metadata = result['metadata']

    return state

def blender_validate_node(state: BlenderGraphState) -> BlenderGraphState:
    """Validate generated code with comprehensive quality assurance."""
    if state.quality_report:
        # Use existing quality report from agentic generation
        quality_report = state.quality_report
    else:
        # Run quality assurance if not already done
        if state.agentic_rag:
            quality_report = state.agentic_rag.quality_assurance.run_full_quality_check(
                state.snippet, state.agentic_rag.blender_version
            )
        else:
            # Fallback to basic validation
            quality_assurance = CodeQualityAssurance()
            quality_report = quality_assurance.run_full_quality_check(state.snippet)

    # Extract validation results
    state.validation = BlenderValidationReport(
        static_ok=quality_report.static_analysis_passed,
        runtime_success=quality_report.smoke_test_passed,
        runtime_output=quality_report.smoke_test_output,
        runtime_errors=quality_report.smoke_test_errors,
        render_available=False  # Could be extended for render testing
    )

    # Update metadata with validation results if available
    if state.metadata and state.agentic_rag:
        state.agentic_rag.metadata_manager.update_validation_results(
            state.metadata.snippet_id,
            quality_report.dict(),
            quality_report.performance_score
        )

    return state

def blender_critic_node(state: BlenderGraphState) -> BlenderGraphState:
    """Apply LLM critic evaluation."""
    critic = BlenderCritic()
    result = critic.evaluate_snippet(
        state.snippet,
        [chunk.dict() for chunk in state.retrieved],
        state.validation.dict()
    )

    state.critic_score = result.score
    state.critic_feedback = result.reasons

    # Decide if healing is needed
    needs_healing = (
        not state.validation.static_ok or
        not state.validation.runtime_success or
        state.critic_score < 0.75
    )

    if needs_healing and state.iterations < 2:
        state.done = False
        state.iterations += 1
    else:
        state.done = True

    return state

def blender_heal_node(state: BlenderGraphState) -> BlenderGraphState:
    """Attempt to heal problematic code."""
    error_context = ""
    if state.validation:
        error_context = f"Runtime Errors: {state.validation.runtime_errors}"

    prompt = f"""Fix this Blender Python code:

ORIGINAL CODE:
```python
{state.snippet}
```

ERRORS:
{error_context}

CRITIC FEEDBACK:
{chr(10).join(state.critic_feedback)}

Generate corrected code that addresses these issues:"""

    state.healed_snippet = llm_call("auto", prompt)
    state.snippet = state.healed_snippet  # Update for next iteration

    return state
```

### Complete Blender RAG Pipeline

```python
def build_blender_rag_graph() -> StateGraph:
    """Build complete agentic Blender RAG graph."""
    graph = StateGraph(BlenderGraphState)

    # Add nodes
    graph.add_node("retrieve", blender_retrieve_node)
    graph.add_node("process_query", blender_query_processing_node)
    graph.add_node("generate", blender_generate_node)
    graph.add_node("validate", blender_validate_node)
    graph.add_node("critic", blender_critic_node)
    graph.add_node("heal", blender_heal_node)

    # Define flow
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "process_query")
    graph.add_edge("process_query", "generate")
    graph.add_edge("generate", "validate")
    graph.add_edge("validate", "critic")

    # Conditional healing loop
    graph.add_conditional_edges(
        "critic",
        lambda s: "heal" if not s.done else END,
        {"heal": "heal", END: END}
    )
    graph.add_edge("heal", "validate")

    return graph

# Usage
blender_app = build_blender_rag_graph().compile()

def generate_blender_code(request: str) -> Dict[str, Any]:
    """Generate validated Blender Python code with agentic capabilities."""
    initial_state = BlenderGraphState(request=request)

    final_state = None
    for output in blender_app.stream(initial_state):
        final_state = output

    result = {
        "code": final_state.snippet,
        "validation": final_state.validation,
        "critic_score": final_state.critic_score,
        "iterations": final_state.iterations,
        "success": final_state.done and final_state.critic_score >= 0.75,
        "processed_query": final_state.processed_query,
        "quality_report": final_state.quality_report,
        "metadata": final_state.metadata,
        "compatibility_check": None
    }

    # Add compatibility check if metadata is available
    if final_state.metadata and final_state.agentic_rag:
        result["compatibility_check"] = final_state.agentic_rag._check_environment_compatibility(
            final_state.metadata
        )

    return result
```

**Context**: Top retrieved chunks with symbol documentation and example code.

## Configuration Files

### LLM Provider Configuration

Create a configuration file `llm_config.yaml` or `llm_config.json`:

```yaml
# llm_config.yaml
lmstudio:
  base_url: "http://localhost:1234"
  model: "codellama-34b-instruct"

openrouter:
  api_key: "your-openrouter-api-key"
  model: "anthropic/claude-3-sonnet"

openai:
  api_key: "your-openai-api-key"
  model: "gpt-4-turbo-preview"
```

### Environment Variables

Set API keys as environment variables:

```bash
export OPENROUTER_API_KEY="your-openrouter-key"
export OPENAI_API_KEY="your-openai-key"
```

### Blender-Specific Configuration

```yaml
# blender_config.yaml
llm:
  provider: "auto"  # auto, lmstudio, openrouter, openai
  model_configs:
    lmstudio:
      base_url: "http://localhost:1234"
      model: "codellama-34b-instruct"
    openrouter:
      api_key: "${OPENROUTER_API_KEY}"
      model: "anthropic/claude-3-haiku"
    openai:
      api_key: "${OPENAI_API_KEY}"
      model: "gpt-4"

validation:
  timeout: 30
  enable_static_analysis: true
  enable_runtime_testing: true

ingestion:
  base_url: "https://docs.blender.org/api/current"
  max_chunks: 1000
  embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
```

## Example Usage

### Basic Coding RAG

```python
from langgraph import StateGraph, END
from your_rag_module import build_coding_rag_graph, GraphState

# Initialize with your LLM configuration
app = build_coding_rag_graph().compile()

# Run a coding task
result = app.invoke(GraphState(query="Add error handling to user authentication"))

print(f"Success: {result.done}")
print(f"Generated patch: {result.patch.diff_unified}")
```

### Blender Code Generation

```python
from your_blender_rag import generate_blender_code

# Generate Blender Python code
result = generate_blender_code("Create a cube and add a subdivision modifier")

if result["success"]:
    print(f"Generated code:
{result['code']}")
    print(f"Validation score: {result['critic_score']}")
else:
    print("Code generation failed")
```

### Provider-Specific Usage

```python
from llm_config import llm_manager

# Use specific provider
response = llm_manager.generate(
    "Write a Python function to calculate fibonacci numbers",
    provider="lmstudio",
    temperature=0.7
)

# Auto-select best available provider
response = llm_manager.generate(
    "Explain this code",
    provider="auto"
)
```

## Generation Prompt

**System**: "You write Blender Python that runs headless. Prefer `bpy.data` and explicit contexts; avoid GUI-only calls; cite relevant symbols."

**User**: [task description]

**Context**: Top retrieved chunks with symbol documentation and example code.

## Critic Prompt

"Given docs, code, and linter/runtime outputs, score [0-1] for: API correctness, context safety, idempotence, and side-effect safety. If <0.75, propose a minimal patch."

This comprehensive Blender RAG system provides production-ready code generation with extensive validation and healing capabilities.

## Error Examples & Fixes

This section provides progressively complex Blender Python code examples, from basic operations to advanced workflows. Each example includes error handling patterns, common pitfalls, and instruct-like diffs for fixing issues.

### Simple Operations

#### Example 1: Basic Cube Creation (Beginner Level)

**Task**: Create a simple cube at the origin.

**Simple Prompt**:
```python
# Create a basic cube
import bpy

# Clear existing objects
bpy.ops.object.select_all(action='DESELECT')
bpy.ops.object.select_by_type(type='MESH')
bpy.ops.object.delete(use_global=False)

# Add a cube
bpy.ops.mesh.primitive_cube_add(size=2, location=(0, 0, 0))

# Set material (optional)
mat = bpy.data.materials.new(name="CubeMaterial")
mat.diffuse_color = (1, 0, 0, 1)  # Red color
bpy.context.active_object.data.materials.append(mat)
```

**Common Error & Fix**:
```
Traceback (most recent call last):
  File "<blender_console>", line 1, in <module>
    bpy.ops.mesh.primitive_cube_add(size=2, location=(0, 0, 0))
RuntimeError: Operator bpy.ops.mesh.primitive_cube_add.poll() failed, context is incorrect
```

**Instruct-like Diff Fix**:
```diff
--- a/simple_cube_error.py
+++ b/simple_cube_fixed.py
@@ -1,5 +1,8 @@
 import bpy

 # Clear existing objects
+bpy.ops.object.mode_set(mode='OBJECT')  # Ensure we're in object mode
+bpy.ops.object.select_all(action='DESELECT')
+
 # Add a cube
 bpy.ops.mesh.primitive_cube_add(size=2, location=(0, 0, 0))

```

**Explanation**: The error occurs when trying to add objects while in Edit mode. Always ensure you're in Object mode before using `bpy.ops.mesh.primitive_*` operators.

#### Example 2: Simple Material Assignment

**Task**: Create and assign a basic material to selected objects.

**Code Example**:
```python
import bpy

def create_basic_material(name="SimpleMaterial", color=(0.8, 0.8, 0.8, 1)):
    """Create a basic diffuse material."""
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = False  # Use simple material
    mat.diffuse_color = color
    return mat

def assign_material_to_selection(material):
    """Assign material to all selected objects."""
    for obj in bpy.context.selected_objects:
        if obj.type == 'MESH':
            # Remove existing materials
            obj.data.materials.clear()
            # Add new material
            obj.data.materials.append(material)

# Usage
if bpy.context.selected_objects:
    mat = create_basic_material("MyMaterial", (0, 1, 0, 1))  # Green
    assign_material_to_selection(mat)
    print(f"Assigned material to {len(bpy.context.selected_objects)} objects")
else:
    print("No objects selected")
```

### Intermediate Operations

#### Example 3: Procedural Geometry Creation

**Task**: Create a spiral staircase with proper UV mapping.

**Code Example**:
```python
import bpy
import math
from mathutils import Vector

def create_spiral_staircase(steps=12, radius=3, height_per_step=0.3):
    """Create a procedural spiral staircase."""
    # Create curve for the spiral path
    curve_data = bpy.data.curves.new('SpiralStair', type='CURVE')
    curve_data.dimensions = '3D'
    spline = curve_data.splines.new('BEZIER')

    # Generate spiral points
    points = []
    for i in range(steps * 4):  # 4 points per step for smooth curve
        angle = (i / (steps * 4)) * 2 * math.pi * 2  # 2 full rotations
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        z = (i / (steps * 4)) * steps * height_per_step
        points.append(Vector((x, y, z)))

    # Set spline points
    spline.bezier_points.add(len(points) - 1)
    for i, point in enumerate(points):
        spline.bezier_points[i].co = point
        spline.bezier_points[i].handle_left = point
        spline.bezier_points[i].handle_right = point

    # Create curve object
    curve_obj = bpy.data.objects.new('SpiralStairCurve', curve_data)
    bpy.context.collection.objects.link(curve_obj)

    # Create steps along the curve
    for i in range(steps):
        angle = (i / steps) * 2 * math.pi * 2
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        z = (i / steps) * steps * height_per_step

        # Add step (cube)
        bpy.ops.mesh.primitive_cube_add(size=1.2, location=(x, y, z))
        step = bpy.context.active_object
        step.name = f"Step_{i+1:02d}"

        # Rotate step to follow curve
        step.rotation_euler.z = angle

        # Add tread (thinner cube on top)
        bpy.ops.mesh.primitive_cube_add(size=1.4, location=(x, y, z + 0.1))
        tread = bpy.context.active_object
        tread.scale.z = 0.2  # Make it thinner
        tread.name = f"Tread_{i+1:02d}"

    print(f"Created spiral staircase with {steps} steps")

# Usage
create_spiral_staircase(steps=16, radius=4, height_per_step=0.25)
```

**Error Example & Fix**:
```
Traceback (most recent call last):
  File "spiral_stair.py", line 45, in <module>
    spline.bezier_points[i].co = point
AttributeError: 'BezierSplinePoint' object attribute 'co' is read-only
```

**Instruct-like Diff Fix**:
```diff
--- a/spiral_stair_error.py
+++ b/spiral_stair_fixed.py
@@ -35,7 +35,7 @@ def create_spiral_staircase(steps=12, radius=3, height_per_step=0.3):
     # Set spline points
     spline.bezier_points.add(len(points) - 1)
     for i, point in enumerate(points):
-        spline.bezier_points[i].co = point
+        spline.bezier_points[i].co.xyz = point  # Use .xyz for assignment
         spline.bezier_points[i].handle_left = point
         spline.bezier_points[i].handle_right = point

@@ -45,6 +45,9 @@ def create_spiral_staircase(steps=12, radius=3, height_per_step=0.3):
     # Create curve object
     curve_obj = bpy.data.objects.new('SpiralStairCurve', curve_data)
     bpy.context.collection.objects.link(curve_obj)
+
+    # Set curve to 3D and enable bevel
+    curve_obj.data.bevel_depth = 0.02
+    curve_obj.data.bevel_resolution = 4

     # Create steps along the curve
     for i in range(steps):
```

#### Example 4: Animation Setup with Constraints

**Task**: Create a robotic arm with proper IK constraints and animation.

**Code Example**:
```python
import bpy
from mathutils import Vector, Euler

def create_robotic_arm(segments=4, segment_length=1.0):
    """Create a procedural robotic arm with IK."""
    # Clear scene
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_by_type(type='ARMATURE')
    bpy.ops.object.delete(use_global=False)

    # Create armature
    bpy.ops.object.armature_add(enter_editmode=True, location=(0, 0, 0))
    armature = bpy.context.active_object
    armature.name = "RoboticArm"

    # Enter edit mode
    bpy.ops.object.mode_set(mode='EDIT')

    # Create bones
    bones = []
    for i in range(segments):
        bone_name = f"Bone_{i+1}"
        if i == 0:
            # First bone at origin
            bone = armature.data.edit_bones.new(bone_name)
            bone.head = Vector((0, 0, 0))
            bone.tail = Vector((segment_length, 0, 0))
        else:
            # Subsequent bones
            bone = armature.data.edit_bones.new(bone_name)
            bone.head = bones[i-1].tail
            bone.tail = bone.head + Vector((segment_length, 0, 0))
            bone.parent = bones[i-1]
            bone.use_connect = True

        bones.append(bone)

    # Exit edit mode
    bpy.ops.object.mode_set(mode='OBJECT')

    # Add IK constraint to the last bone
    ik_constraint = bones[-1].constraints.new('IK')
    ik_constraint.target = armature
    ik_constraint.subtarget = "IK_Target"
    ik_constraint.chain_count = segments

    # Create IK target
    bpy.ops.object.empty_add(type='PLAIN_AXES', location=(segments * segment_length, 0, 0))
    ik_target = bpy.context.active_object
    ik_target.name = "IK_Target"

    # Parent IK target to armature for organization
    ik_target.parent = armature

    # Add animation
    create_arm_animation(armature, ik_target, segments)

    return armature, ik_target

def create_arm_animation(armature, ik_target, segments):
    """Create a simple animation for the robotic arm."""
    # Set up animation
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = 120

    # Animate IK target in a circle
    for frame in range(1, 121):
        angle = (frame / 120) * 2 * 3.14159  # Full circle
        radius = segments * 0.8
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        z = 2 + math.sin(angle * 2)  # Add some up/down motion

        ik_target.location = (x, y, z)
        ik_target.keyframe_insert(data_path="location", frame=frame)

    # Add some rotation animation to base
    armature.rotation_euler.z = 0
    armature.keyframe_insert(data_path="rotation_euler", index=2, frame=1)

    armature.rotation_euler.z = 3.14159 / 4  # 45 degrees
    armature.keyframe_insert(data_path="rotation_euler", index=2, frame=60)

    armature.rotation_euler.z = -3.14159 / 4  # -45 degrees
    armature.keyframe_insert(data_path="rotation_euler", index=2, frame=120)

# Usage
arm, target = create_robotic_arm(segments=5, segment_length=0.8)
print("Robotic arm created with IK animation")
```

### Advanced Operations

#### Example 5: Node-Based Material Creation

**Task**: Create a complex PBR material with procedural textures using node editor.

**Code Example**:
```python
import bpy
from bpy.types import NodeTree, Node, NodeSocket

def create_pbr_material(name="AdvancedPBR", base_color=(0.8, 0.2, 0.1, 1)):
    """Create a complex PBR material with procedural textures."""
    # Create new material
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    # Clear default nodes
    nodes.clear()

    # Create Principled BSDF
    principled = nodes.new(type='ShaderNodeBsdfPrincipled')
    principled.location = (0, 0)
    principled.inputs['Base Color'].default_value = base_color
    principled.inputs['Metallic'].default_value = 0.3
    principled.inputs['Roughness'].default_value = 0.4
    principled.inputs['IOR'].default_value = 1.45

    # Create Material Output
    output = nodes.new(type='ShaderNodeOutputMaterial')
    output.location = (400, 0)
    links.new(principled.outputs['BSDF'], output.inputs['Surface'])

    # Create procedural base color texture
    create_procedural_base_color(nodes, links, principled)

    # Create normal map
    create_normal_map(nodes, links, principled)

    # Create roughness variation
    create_roughness_variation(nodes, links, principled)

    return mat

def create_procedural_base_color(nodes, links, principled):
    """Create procedural base color with noise and color ramp."""
    # Noise texture
    noise = nodes.new(type='ShaderNodeTexNoise')
    noise.location = (-600, 200)
    noise.inputs['Scale'].default_value = 15.0
    noise.inputs['Detail'].default_value = 2.0

    # Color ramp for color variation
    color_ramp = nodes.new(type='ShaderNodeValToRGB')
    color_ramp.location = (-400, 200)
    color_ramp.color_ramp.elements[0].color = (0.5, 0.1, 0.05, 1)  # Dark red
    color_ramp.color_ramp.elements[1].color = (1.0, 0.3, 0.1, 1)   # Light red

    # Mix with base color
    mix_rgb = nodes.new(type='ShaderNodeMixRGB')
    mix_rgb.location = (-200, 200)
    mix_rgb.blend_type = 'MULTIPLY'
    mix_rgb.inputs['Fac'].default_value = 0.3

    # Connect nodes
    links.new(noise.outputs['Fac'], color_ramp.inputs['Fac'])
    links.new(color_ramp.outputs['Color'], mix_rgb.inputs['Color1'])
    links.new(principled.inputs['Base Color'], mix_rgb.inputs['Color2'])
    links.new(mix_rgb.outputs['Color'], principled.inputs['Base Color'])

def create_normal_map(nodes, links, principled):
    """Create procedural normal map."""
    # Noise for normal variation
    noise = nodes.new(type='ShaderNodeTexNoise')
    noise.location = (-600, -100)
    noise.inputs['Scale'].default_value = 25.0
    noise.inputs['Detail'].default_value = 3.0

    # Bump node
    bump = nodes.new(type='ShaderNodeBump')
    bump.location = (-200, -100)
    bump.inputs['Strength'].default_value = 0.2
    bump.inputs['Distance'].default_value = 0.1

    # Connect
    links.new(noise.outputs['Fac'], bump.inputs['Height'])
    links.new(bump.outputs['Normal'], principled.inputs['Normal'])

def create_roughness_variation(nodes, links, principled):
    """Create roughness variation."""
    # Voronoi texture for roughness
    voronoi = nodes.new(type='ShaderNodeTexVoronoi')
    voronoi.location = (-600, -300)
    voronoi.inputs['Scale'].default_value = 8.0

    # Map range for roughness
    map_range = nodes.new(type='ShaderNodeMapRange')
    map_range.location = (-400, -300)
    map_range.inputs['From Min'].default_value = 0.0
    map_range.inputs['From Max'].default_value = 1.0
    map_range.inputs['To Min'].default_value = 0.2
    map_range.inputs['To Max'].default_value = 0.8

    # Connect
    links.new(voronoi.outputs['Distance'], map_range.inputs['Value'])
    links.new(map_range.outputs['Result'], principled.inputs['Roughness'])

# Usage
mat = create_pbr_material("ComplexMetal", (0.9, 0.8, 0.2, 1))
print(f"Created advanced PBR material: {mat.name}")
```

#### Example 6: Physics Simulation Setup

**Task**: Create a rigid body physics simulation with constraints and forces.

**Code Example**:
```python
import bpy
import random
from mathutils import Vector

def create_physics_scene():
    """Create a complex physics simulation scene."""
    # Clear scene
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_by_type(type='MESH')
    bpy.ops.object.delete(use_global=False)

    # Enable physics
    bpy.context.scene.rigidbody_world.enabled = True

    # Create ground plane
    bpy.ops.mesh.primitive_plane_add(size=20, location=(0, 0, -1))
    ground = bpy.context.active_object
    ground.name = "Ground"

    # Add rigid body to ground (passive)
    bpy.ops.rigidbody.object_add()
    ground.rigid_body.type = 'PASSIVE'
    ground.rigid_body.collision_shape = 'MESH'

    # Create falling objects
    create_falling_objects()

    # Create launcher mechanism
    create_launcher()

    # Set up animation
    setup_animation()

def create_falling_objects():
    """Create various objects with physics properties."""
    objects = []

    # Create cubes
    for i in range(5):
        bpy.ops.mesh.primitive_cube_add(size=random.uniform(0.5, 1.5),
                                      location=(random.uniform(-5, 5),
                                               random.uniform(-5, 5),
                                               random.uniform(5, 15)))
        obj = bpy.context.active_object
        obj.name = f"Cube_{i+1}"

        # Add rigid body
        bpy.ops.rigidbody.object_add()
        obj.rigid_body.mass = random.uniform(1, 5)
        obj.rigid_body.friction = random.uniform(0.1, 0.9)
        obj.rigid_body.restitution = random.uniform(0.1, 0.8)

        objects.append(obj)

    # Create spheres
    for i in range(3):
        bpy.ops.mesh.primitive_uv_sphere_add(radius=random.uniform(0.3, 0.8),
                                           location=(random.uniform(-5, 5),
                                                    random.uniform(-5, 5),
                                                    random.uniform(5, 15)))
        obj = bpy.context.active_object
        obj.name = f"Sphere_{i+1}"

        # Add rigid body
        bpy.ops.rigidbody.object_add()
        obj.rigid_body.collision_shape = 'SPHERE'
        obj.rigid_body.mass = random.uniform(0.5, 3)
        obj.rigid_body.friction = random.uniform(0.05, 0.3)  # Lower friction for spheres

        objects.append(obj)

    return objects

def create_launcher():
    """Create a launcher mechanism."""
    # Create launcher base
    bpy.ops.mesh.primitive_cube_add(size=2, location=(-8, 0, 1))
    launcher_base = bpy.context.active_object
    launcher_base.name = "LauncherBase"
    launcher_base.scale = (1, 3, 0.5)

    # Create launcher arm
    bpy.ops.mesh.primitive_cube_add(size=1, location=(-6, 0, 3))
    launcher_arm = bpy.context.active_object
    launcher_arm.name = "LauncherArm"
    launcher_arm.scale = (0.2, 2, 0.1)

    # Add hinge constraint
    bpy.ops.rigidbody.constraint_add()
    constraint = bpy.context.active_object.constraints[-1]
    constraint.type = 'HINGE'
    constraint.pivot_type = 'GENERIC'
    constraint.pivot_x = -7
    constraint.pivot_y = 0
    constraint.pivot_z = 2

    return launcher_base, launcher_arm

def setup_animation():
    """Set up animation for the physics simulation."""
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = 300
    bpy.context.scene.rigidbody_world.point_cache.frame_start = 1
    bpy.context.scene.rigidbody_world.point_cache.frame_end = 300

    # Bake physics
    bpy.ops.ptcache.bake_all(bake=True)

# Usage
create_physics_scene()
print("Physics simulation scene created")
```

### Error Patterns & Solutions

#### Pattern 1: Context Incorrect Errors

**Common Error**:
```
RuntimeError: Operator bpy.ops.mesh.primitive_cube_add.poll() failed, context is incorrect
```

**Root Causes**:
- Being in Edit mode when trying to add objects
- No active scene or collection
- Trying to use GUI operators in headless mode

**Solution Pattern**:
```python
def safe_primitive_add(primitive_type, **kwargs):
    """Safely add primitives with proper context."""
    # Ensure object mode
    if bpy.context.active_object:
        bpy.ops.object.mode_set(mode='OBJECT')

    # Deselect all
    bpy.ops.object.select_all(action='DESELECT')

    # Add primitive
    if primitive_type == 'cube':
        bpy.ops.mesh.primitive_cube_add(**kwargs)
    elif primitive_type == 'sphere':
        bpy.ops.mesh.primitive_uv_sphere_add(**kwargs)
    # ... other primitives

    return bpy.context.active_object
```

#### Pattern 2: Attribute Access Errors

**Common Error**:
```
AttributeError: 'Object' object has no attribute 'some_property'
```

**Root Causes**:
- Accessing properties that don't exist on certain object types
- Trying to access mesh data on non-mesh objects
- Version compatibility issues

**Solution Pattern**:
```python
def safe_get_attribute(obj, attr_path, default=None):
    """Safely get nested attributes with error handling."""
    try:
        value = obj
        for attr in attr_path.split('.'):
            value = getattr(value, attr)
        return value
    except AttributeError:
        return default

# Usage
mesh_name = safe_get_attribute(obj, 'data.name', 'Unknown')
vertex_count = safe_get_attribute(obj, 'data.vertices', [])
```

#### Pattern 3: Collection/Link Errors

**Common Error**:
```
ValueError: Object 'MyObject' is not in collection 'Collection'
```

**Root Causes**:
- Objects not properly linked to collections
- Trying to access objects from wrong collection
- Collection hierarchy issues

**Solution Pattern**:
```python
def safe_link_object(obj, collection_name="Collection"):
    """Safely link object to collection."""
    # Get or create collection
    if collection_name in bpy.data.collections:
        collection = bpy.data.collections[collection_name]
    else:
        collection = bpy.data.collections.new(collection_name)
        bpy.context.scene.collection.children.link(collection)

    # Unlink from current collections
    for coll in obj.users_collection:
        coll.objects.unlink(obj)

    # Link to target collection
    collection.objects.link(obj)

    return collection
```

#### Pattern 4: Memory and Performance Issues

**Common Error**:
```
MemoryError: Unable to allocate memory
```

**Root Causes**:
- Large mesh operations without cleanup
- Accumulating too many objects
- Inefficient loops

**Solution Pattern**:
```python
def memory_efficient_operation():
    """Perform operations with memory management."""
    # Disable undo for better performance
    original_undo = bpy.context.preferences.edit.use_global_undo
    bpy.context.preferences.edit.use_global_undo = False

    try:
        # Your operations here
        for i in range(1000):
            # Create object
            bpy.ops.mesh.primitive_cube_add()
            obj = bpy.context.active_object

            # Process object
            # ...

            # Clean up if not needed
            if i % 100 == 0:
                bpy.ops.outliner.orphans_purge()

    finally:
        # Restore undo
        bpy.context.preferences.edit.use_global_undo = original_undo
```

#### Pattern 5: Node Editor Errors

**Common Error**:
```
TypeError: Node sockets are not compatible
```

**Root Causes**:
- Connecting incompatible node socket types
- Missing nodes in the chain
- Wrong node configuration

**Solution Pattern**:
```python
def safe_connect_nodes(from_node, from_socket, to_node, to_socket):
    """Safely connect nodes with type checking."""
    try:
        # Check socket compatibility
        if from_socket.type != to_socket.type:
            print(f"Warning: Connecting {from_socket.type} to {to_socket.type}")

        # Connect nodes
        bpy.context.active_object.data.materials[0].node_tree.links.new(
            from_socket, to_socket
        )
        return True
    except Exception as e:
        print(f"Failed to connect nodes: {e}")
        return False
```

### Blender Area Coverage

#### 1. **Modeling Operations**
- Mesh primitives (cube, sphere, cylinder, etc.)
- Curve and surface creation
- Modifier stack management
- UV mapping and unwrapping

#### 2. **Animation & Rigging**
- Keyframe animation
- Armature creation and posing
- Constraints and drivers
- Shape keys and morphing

#### 3. **Materials & Shading**
- Node-based materials
- PBR workflows
- Procedural textures
- Material assignment and management

#### 4. **Rendering**
- Render engine setup
- Camera positioning
- Lighting setup
- Render layer configuration

#### 5. **Physics & Simulation**
- Rigid body dynamics
- Soft body simulation
- Cloth simulation
- Particle systems

#### 6. **Scripting & Automation**
- Batch operations
- Custom operators
- Add-on development
- UI creation

#### 7. **Data Management**
- Scene organization
- Asset management
- Import/export operations
- File handling

#### 8. **Advanced Features**
- Geometry nodes
- Compositing
- Video editing
- Grease pencil

This comprehensive collection of examples covers the full spectrum of Blender Python development, from simple operations to complex workflows, with robust error handling and debugging patterns.
