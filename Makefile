.PHONY: install dev test clean lint format setup-blender help generate-hierarchical generate-basic generate-all

# Default target
help:
	@echo "Available commands:"
	@echo "  install      - Install production dependencies"
	@echo "  dev          - Install development dependencies"
	@echo "  test         - Run all tests (parallel)"
	@echo "  test-fast    - Run fast tests only (unit + integration, no e2e/slow)"
	@echo "  test-slow    - Run slow/e2e/performance tests only"
	@echo "  test-cov     - Run tests with coverage"
	@echo "  test-cov-fast- Run fast tests with coverage"
	@echo "  lint         - Run linting (ruff, mypy)"
	@echo "  format       - Format code (ruff format)"
	@echo "  clean        - Clean up generated files"
	@echo "  setup-blender- Setup Blender for development"
	@echo "  generate-basic - Generate Level 1 basic geometry assets"
	@echo "  generate-hierarchical - Generate assets for all complexity levels (sequential)"
	@echo "  generate-hierarchical-parallel - Generate assets in parallel (4x faster)"
	@echo "  generate-hierarchical-fast - Fast parallel generation (levels 1-3 only)"
	@echo "  generate-level-X - Generate assets for specific level (1-5)"
	@echo "  generate-all - Generate all hierarchical test assets with full exports"
	@echo "  check-config - Check configuration for generation (OpenAI API key, etc.)"
	@echo "  demo-basic   - Demo Level 1 generation (no LLM required)"
	@echo "  demo-hierarchical - Demo all levels generation (no LLM required)"
	@echo "  enhanced-generate - Generate with enhanced output management & interactive refinement"
	@echo "  enhanced-browse - Launch interactive asset browser"
	@echo "  asset-manager - Launch interactive asset management CLI"
	@echo "  cache-stats - Display RAG cache statistics"
	@echo "  cache-test - Test RAG cache retrieval"
	@echo "  cache-cleanup - Clean up low-quality cache entries"

# Installation
install:
	pip install -e .

dev: install
	pip install -e ".[dev]"
	pre-commit install

# Testing
test:
	pytest tests/ -v -n auto

test-fast:
	pytest -c pytest-fast.ini

test-slow:
	pytest -m "slow or performance or e2e" -v --tb=short

test-cov:
	pytest tests/ --cov=src --cov-report=html --cov-report=term-missing -n auto

test-cov-fast:
	pytest -c pytest-fast.ini --cov=src --cov-report=term-missing

# Code quality
lint:
	ruff check src/ tests/
	mypy src/

format:
	ruff format src/ tests/

# Cleanup
clean: clean-assets
	find . -type d -name "__pycache__" -delete
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name "htmlcov" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/

# Blender setup (platform specific)
setup-blender:
	@echo "Setting up Blender..."
	@if [ "$(shell uname)" = "Darwin" ]; then \
		echo "macOS detected"; \
		if [ ! -f "/Applications/Blender.app/Contents/MacOS/Blender" ]; then \
			echo "Blender not found. Please install Blender from https://www.blender.org/download/"; \
		else \
			echo "Blender found at /Applications/Blender.app/Contents/MacOS/Blender"; \
		fi \
	elif [ "$(shell uname)" = "Linux" ]; then \
		echo "Linux detected"; \
		if ! command -v blender &> /dev/null; then \
			echo "Installing Blender..."; \
			sudo apt-get update && sudo apt-get install -y blender; \
		else \
			echo "Blender is already installed"; \
		fi \
	else \
		echo "Windows detected - please install Blender manually"; \
	fi

# Run the application
run-example:
	python -m src.main "Create a red cube with a metallic material"

# Development server (if we add FastAPI later)
dev-server:
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Hierarchical 3D Asset Generation
generate-basic:
	@echo "🎯 Generating Level 1 (Basic Geometry) Assets..."
	@echo "Note: Requires OpenAI API key in .env or USE_LOCAL_LLM=false"
	USE_LOCAL_LLM=false python scripts/run_hierarchical_generation.py --level 1 --max-per-level 3

generate-level-1:
	@echo "🎯 Generating Level 1 (Basic Geometry) Assets..."
	USE_LOCAL_LLM=false python scripts/run_hierarchical_generation.py --level 1

generate-level-2:
	@echo "🎯 Generating Level 2 (Simple Objects) Assets..."
	USE_LOCAL_LLM=false python scripts/run_hierarchical_generation.py --level 2

generate-level-3:
	@echo "🎯 Generating Level 3 (Textured Objects) Assets..."
	USE_LOCAL_LLM=false python scripts/run_hierarchical_generation.py --level 3

generate-level-4:
	@echo "🎯 Generating Level 4 (Lit Scenes) Assets..."
	USE_LOCAL_LLM=false python scripts/run_hierarchical_generation.py --level 4

generate-level-5:
	@echo "🎯 Generating Level 5 (Complex Animations) Assets..."
	USE_LOCAL_LLM=false python scripts/run_hierarchical_generation.py --level 5 --max-per-level 2

generate-hierarchical:
	@echo "🚀 Generating All Hierarchical Test Assets..."
	@echo "This will generate 3D assets for all complexity levels (1-5)"
	@echo "Output: code, renders, .blend files, and exported 3D formats"
	@echo "Starting with basic geometry and progressing to complex animations..."
	@echo "Note: Requires OpenAI API key in .env or USE_LOCAL_LLM=false"
	USE_LOCAL_LLM=false python scripts/run_hierarchical_generation.py --start-level 1 --end-level 5 --max-per-level 2

generate-hierarchical-parallel:
	@echo "🔥 Generating All Hierarchical Test Assets in PARALLEL..."
	@echo "This will generate 3D assets for all complexity levels (1-5) simultaneously"
	@echo "Output: code, renders, .blend files, and exported 3D formats"
	@echo "Parallel processing for maximum speed (4 concurrent tasks by default)"
	@echo "Note: Requires OpenAI API key in .env or USE_LOCAL_LLM=false"
	USE_LOCAL_LLM=false python scripts/run_parallel_hierarchical_generation.py --start-level 1 --end-level 5 --max-per-level 2 --max-concurrent 4

generate-hierarchical-fast:
	@echo "⚡ Fast Parallel Generation (Level 1-3, 1 asset per level)..."
	USE_LOCAL_LLM=false python scripts/run_parallel_hierarchical_generation.py --start-level 1 --end-level 3 --max-per-level 1 --max-concurrent 6

generate-all: generate-hierarchical
	@echo "✅ Full hierarchical asset generation complete!"
	@echo "Check outputs/hierarchical_tests/ for generated assets"

generate-demo:
	@echo "🎭 Running Hierarchical Prompt Demo..."
	python scripts/demo_hierarchical_prompts.py

check-config:
	@echo "🔧 Checking Generation Configuration..."
	python scripts/check_generation_config.py

# Demo targets (no LLM required)
demo-basic:
	@echo "🎭 Demo Level 1 (Basic Geometry) Assets..."
	@echo "No LLM or API key required - creates mock files to show structure"
	python scripts/run_hierarchical_generation_demo.py --level 1 --max-per-level 3

demo-hierarchical:
	@echo "🎭 Demo All Hierarchical Test Assets..."
	@echo "No LLM or API key required - creates mock files to show structure"
	@echo "This demonstrates the hierarchical organization from simple to complex"
	python scripts/run_hierarchical_generation_demo.py --start-level 1 --end-level 5 --max-per-level 2

generate-gangnam:
	@echo "💃 Generating the ultimate complexity test: Gangnam Style!"
	@echo "CharMorph Vitruvian blonde woman in bikini dancing Gangnam Style"
	USE_LOCAL_LLM=false python scripts/run_hierarchical_generation.py --level 5 --max-per-level 1

# Clean generated assets
clean-assets:
	@echo "🧹 Cleaning generated assets..."
	rm -rf outputs/hierarchical_tests/
	@echo "Generated assets cleaned."

# Enhanced Output Management and Interactive Refinement
enhanced-generate:
	@echo "🎨 Enhanced Hierarchical Generation with Interactive Refinement..."
	@echo "Features: Organized output structure, interactive refinement, asset browser"
	@echo "This creates a sophisticated directory structure with projects, sessions, and galleries"
	USE_LOCAL_LLM=false python scripts/enhanced_hierarchical_generation.py --start-level 1 --end-level 3 --max-per-level 1

enhanced-browse:
	@echo "🔍 Launching Interactive Asset Browser..."
	@echo "Navigate and manage your 3D assets with a rich CLI interface"
	python scripts/enhanced_hierarchical_generation.py --browser-only

asset-manager:
	@echo "🛠️ Interactive Asset Management CLI..."
	@echo "Available commands: browse, list, stats, refine, batch, sample, setup"
	@echo "Example: python scripts/interactive_asset_manager.py browse"
	python scripts/interactive_asset_manager.py

asset-manager-browse:
	@echo "🔍 Launching Asset Browser..."
	python scripts/interactive_asset_manager.py browse

asset-manager-stats:
	@echo "📊 System Statistics..."
	python scripts/interactive_asset_manager.py stats

asset-manager-sample:
	@echo "📦 Creating Sample Project..."
	python scripts/interactive_asset_manager.py sample

# Interactive refinement examples
refine-demo:
	@echo "🛠️ Interactive Refinement Demo..."
	@echo "This demonstrates natural language asset refinement"
	@echo "Example: 'make it blue', 'add more detail', 'improve lighting'"
	python scripts/interactive_asset_manager.py sample
	@echo "Sample project created. Use 'make enhanced-browse' to explore and refine assets."

# Clean enhanced outputs
clean-enhanced:
	@echo "🧹 Cleaning enhanced outputs..."
	rm -rf outputs/enhanced/
	@echo "Enhanced outputs cleaned."

# RAG Cache Management
cache-stats:
	@echo "📊 Displaying RAG Cache Statistics..."
	python -m src.cache.cache_manager stats

cache-test:
	@echo "🧪 Testing RAG Cache Retrieval..."
	python -m src.cache.cache_manager test

cache-cleanup:
	@echo "🧹 Cleaning up low-quality cache entries..."
	python -m src.cache.cache_manager cleanup --dry-run

cache-cleanup-force:
	@echo "🧹 Force cleaning up low-quality cache entries..."
	python -m src.cache.cache_manager cleanup

cache-export:
	@echo "📤 Exporting cache to JSON..."
	python -m src.cache.cache_manager export cache_export.json

cache-import:
	@echo "📥 Importing cache from JSON..."
	python -m src.cache.cache_manager import cache_export.json

cache-clear:
	@echo "⚠️  Clearing entire cache (WARNING: This will delete all cached data)..."
	python -m src.cache.cache_manager clear --confirm

cache-rebuild:
	@echo "🔄 Rebuilding all embeddings..."
	python -m src.cache.cache_manager rebuild
