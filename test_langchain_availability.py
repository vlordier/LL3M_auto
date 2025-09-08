#!/usr/bin/env python3
"""Test script to verify LangChain dependencies are available."""

import sys


def test_langchain_imports():
    """Test that all required LangChain components can be imported."""
    print("🧪 Testing LangChain availability...")

    try:
        from langchain.output_parsers import RetryOutputParser  # noqa: F401
        print("✅ langchain.output_parsers.RetryOutputParser imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import RetryOutputParser: {e}")
        return False

    try:
        from langchain_core.output_parsers import PydanticOutputParser  # noqa: F401
        print("✅ langchain_core.output_parsers.PydanticOutputParser imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import PydanticOutputParser: {e}")
        return False

    try:
        from langchain_openai import ChatOpenAI  # noqa: F401
        print("✅ langchain_openai.ChatOpenAI imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import ChatOpenAI: {e}")
        return False

    return True

def test_llm_client_langchain_availability():
    """Test that the LLM client can now access LangChain."""
    print("\n🧪 Testing LLM client LangChain integration...")

    try:
        from src.utils.llm_client import LANGCHAIN_AVAILABLE
        if LANGCHAIN_AVAILABLE:
            print("✅ LANGCHAIN_AVAILABLE = True")
            return True
        else:
            print("❌ LANGCHAIN_AVAILABLE = False")
            return False
    except ImportError as e:
        print(f"❌ Failed to import from LLM client: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 LangChain Availability Test\n")

    # Test 1: Direct imports
    imports_ok = test_langchain_imports()

    # Test 2: LLM client integration
    client_ok = test_llm_client_langchain_availability()

    if imports_ok and client_ok:
        print("\n🎉 SUCCESS: LangChain is now fully available!")
        print("The message 'LangChain not available, using manual fallback' should no longer appear.")
        return 0
    else:
        print("\n❌ FAILURE: LangChain dependencies are not properly configured.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
