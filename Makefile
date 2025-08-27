.PHONY: install dev test clean lint format setup-blender help

# Default target
help:
	@echo "Available commands:"
	@echo "  install      - Install production dependencies"
	@echo "  dev          - Install development dependencies"
	@echo "  test         - Run tests"
	@echo "  test-cov     - Run tests with coverage"
	@echo "  lint         - Run linting (ruff, mypy)"
	@echo "  format       - Format code (ruff format)"
	@echo "  clean        - Clean up generated files"
	@echo "  setup-blender- Setup Blender for development"

# Installation
install:
	pip install -e .

dev: install
	pip install -e ".[dev]"
	pre-commit install

# Testing
test:
	pytest tests/ -v

test-cov:
	pytest tests/ --cov=src --cov-report=html --cov-report=term-missing

# Code quality
lint:
	ruff check src/ tests/
	mypy src/

format:
	ruff format src/ tests/

# Cleanup
clean:
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
			echo "Blender not found. Please install Blender:"; \
			echo "  Ubuntu/Debian: sudo apt-get install blender"; \
			echo "  Fedora/RHEL:   sudo dnf install blender"; \
			echo "  Arch Linux:    sudo pacman -S blender"; \
			echo "  Snap:          sudo snap install blender --classic"; \
			echo "  Flatpak:       flatpak install flathub org.blender.Blender"; \
			echo "  Or download from: https://www.blender.org/download/"; \
		else \
			echo "Blender is already installed"; \
		fi \
	else \
		echo "Windows detected - please install Blender manually"; \
	fi

# Run the application
run-example:
	OPENAI_API_KEY="sk-example-key" ll3m generate "Create a red cube with a metallic material"

# Development server (placeholder for future FastAPI integration)
dev-server:
	@echo "Development server not implemented yet - placeholder for future FastAPI integration"
