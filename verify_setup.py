#!/usr/bin/env python3
"""Quick verification script for LL3M local setup."""

import asyncio
import subprocess
import sys
from pathlib import Path

import aiohttp


def check_python_version():
    """Check Python version."""
    version = sys.version_info
    if version >= (3, 11):
        print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} (need 3.11+)")
        return False


def check_dependencies():
    """Check if required packages are installed."""
    required_packages = [
        "aiohttp",
        "fastapi",
        "uvicorn",
        "pydantic",
        "structlog",
        "pytest",
    ]

    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} (missing)")
            missing.append(package)

    return len(missing) == 0


def check_blender():
    """Check Blender installation."""
    blender_paths = [
        "/Applications/Blender.app/Contents/MacOS/Blender",
        "/usr/local/bin/blender",
        "/opt/homebrew/bin/blender",
    ]

    for path in blender_paths:
        if Path(path).exists():
            try:
                result = subprocess.run(  # nosec B603
                    [path, "--version"], capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0:
                    version_line = result.stdout.split("\n")[0]
                    print(f"✅ Blender: {version_line}")
                    return path
            except (subprocess.SubprocessError, OSError, FileNotFoundError):
                continue

    print("❌ Blender not found")
    return None


def check_environment_file():
    """Check if .env file exists."""
    env_files = [".env", ".env.local"]

    for env_file in env_files:
        if Path(env_file).exists():
            print(f"✅ Environment file: {env_file}")
            return env_file

    print("❌ No .env file found")
    return None


async def check_lm_studio():
    """Check if LM Studio is running."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                "http://localhost:1234/v1/models",
                timeout=aiohttp.ClientTimeout(total=5),
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    models = [m["id"] for m in data.get("data", [])]
                    if models:
                        print(
                            f"✅ LM Studio running with models: {models[:2]}{'...' if len(models) > 2 else ''}"
                        )
                        return True
                    else:
                        print("⚠️  LM Studio running but no models loaded")
                        return False
                else:
                    print(f"❌ LM Studio API error: {response.status}")
                    return False
    except Exception as e:
        print(f"❌ LM Studio not accessible: {e}")
        return False


async def check_blender_mcp():
    """Check if Blender MCP server is running."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                "http://localhost:3001/health", timeout=aiohttp.ClientTimeout(total=5)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    print(
                        f"✅ Blender MCP server: {data.get('status')} (v{data.get('blender_version', 'unknown')})"
                    )
                    return True
                else:
                    print(f"❌ Blender MCP server error: {response.status}")
                    return False
    except Exception as e:
        print(f"❌ Blender MCP server not accessible: {e}")
        return False


def check_setup_files():
    """Check if setup files exist."""
    setup_files = ["setup/blender_mcp_setup.py", "setup/README.md", "LOCAL_SETUP.md"]

    all_exist = True
    for file_path in setup_files:
        if Path(file_path).exists():
            print(f"✅ Setup file: {file_path}")
        else:
            print(f"❌ Missing: {file_path}")
            all_exist = False

    return all_exist


def check_test_files():
    """Check if test files exist."""
    test_dirs = ["tests/unit", "tests/e2e"]

    all_exist = True
    for test_dir in test_dirs:
        if Path(test_dir).exists():
            test_files = list(Path(test_dir).glob("test_*.py"))
            print(f"✅ Test directory: {test_dir} ({len(test_files)} test files)")
        else:
            print(f"❌ Missing test directory: {test_dir}")
            all_exist = False

    return all_exist


async def main():
    """Main verification function."""
    print("🔍 LL3M Local Setup Verification")
    print("=" * 50)

    checks = []

    # Basic system checks
    print("\n📋 System Requirements:")
    checks.append(check_python_version())
    checks.append(check_dependencies())
    checks.append(check_blender() is not None)
    checks.append(check_environment_file() is not None)

    # File checks
    print("\n📁 Setup Files:")
    checks.append(check_setup_files())
    checks.append(check_test_files())

    # Service checks
    print("\n🌐 Services:")
    checks.append(await check_lm_studio())
    checks.append(await check_blender_mcp())

    # Summary
    print("\n" + "=" * 50)
    passed = sum(checks)
    total = len(checks)

    if passed == total:
        print(f"🎉 All checks passed! ({passed}/{total})")
        print("\nYou're ready to run LL3M E2E tests:")
        print("  python -m pytest tests/e2e/ -v -s")
        return True
    else:
        print(f"⚠️  Some checks failed ({passed}/{total})")
        print("\nNext steps:")

        if not any([check_python_version(), check_dependencies()]):
            print("1. Install Python 3.11+ and required packages:")
            print("   pip install -r requirements.txt")

        if check_blender() is None:
            print("2. Install Blender from https://www.blender.org/download/")

        if check_environment_file() is None:
            print("3. Run setup script:")
            print("   python setup/blender_mcp_setup.py")

        if not await check_lm_studio():
            print("4. Start LM Studio with a model loaded")

        if not await check_blender_mcp():
            print("5. Start Blender MCP server:")
            print("   ./setup/start_blender_mcp.sh")

        print("\nThen run this verification script again.")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    if not success:
        sys.exit(1)
