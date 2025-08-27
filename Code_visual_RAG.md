# Code Visual RAG: Retrieval-Augmented Generation for Coding Tasks

## Table of Contents

- [Introduction](#introduction)
- [LLM Configuration](#llm-configuration)
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

### System Architecture

The Blender RAG system extends the general framework with:

- **Blender-specific ingestion** using Sphinx object inventory
- **Runtime validation** in headless Blender environment
- **Automatic healing** for common Blender API issues
- **Visual artifact generation** for result verification

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
    retrieved: List[BlenderChunk] = Field(default_factory=list)
    snippet: Optional[str] = None
    validation: Optional[BlenderValidationReport] = None
    critic_score: float = 0.0
    critic_feedback: List[str] = Field(default_factory=list)
    done: bool = False
    iterations: int = 0
    healed_snippet: Optional[str] = None
```

#### Graph Nodes

```python
def blender_retrieve_node(state: BlenderGraphState) -> BlenderGraphState:
    """Retrieve Blender API documentation."""
    retriever = BlenderRetriever()
    state.retrieved = retriever.search(state.request, k=10)
    return state

def blender_generate_node(state: BlenderGraphState) -> BlenderGraphState:
    """Generate Blender Python code."""
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
    return state

def blender_validate_node(state: BlenderGraphState) -> BlenderGraphState:
    """Validate generated code."""
    # Static validation
    static_ok = run_static_validation(state.snippet)

    # Runtime validation
    runtime_result = validate_blender_snippet(state.snippet)

    state.validation = BlenderValidationReport(
        static_ok=static_ok,
        runtime_success=runtime_result['success'],
        runtime_output=runtime_result['stdout'],
        runtime_errors=runtime_result['stderr'],
        render_available='render_path' in runtime_result
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
    """Build complete Blender RAG graph."""
    graph = StateGraph(BlenderGraphState)

    # Add nodes
    graph.add_node("retrieve", blender_retrieve_node)
    graph.add_node("generate", blender_generate_node)
    graph.add_node("validate", blender_validate_node)
    graph.add_node("critic", blender_critic_node)
    graph.add_node("heal", blender_heal_node)

    # Define flow
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
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
    """Generate validated Blender Python code."""
    initial_state = BlenderGraphState(request=request)

    final_state = None
    for output in blender_app.stream(initial_state):
        final_state = output

    return {
        "code": final_state.snippet,
        "validation": final_state.validation,
        "critic_score": final_state.critic_score,
        "iterations": final_state.iterations,
        "success": final_state.done and final_state.critic_score >= 0.75
    }
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
