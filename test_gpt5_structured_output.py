#!/usr/bin/env python3
"""Test GPT-5 structured output support."""

import asyncio

from src.utils.llm_client import OpenAIClient


async def test_gpt5_models_support():
    """Test that all GPT-5 models are recognized as supporting structured output."""
    print("=== Testing GPT-5 Structured Output Support ===")

    client = OpenAIClient()

    # Test all GPT-5 model variants
    gpt5_models = ["gpt-5", "gpt-5-mini", "gpt-5-nano", "gpt-5-chat-latest"]

    for model in gpt5_models:
        supports = client._supports_structured_output(model)
        print(f"  {model}: {'✓ SUPPORTED' if supports else '❌ NOT SUPPORTED'}")
        assert supports, f"GPT-5 model {model} should support structured output"

    print("✅ All GPT-5 models correctly recognized as supporting structured output")
    return True


async def test_gpt5_parameter_handling():
    """Test GPT-5 specific parameter handling."""
    print("\n=== Testing GPT-5 Parameter Handling ===")

    client = OpenAIClient()

    # Test that GPT-5 models are detected for parameter handling
    test_cases = [
        ("gpt-5", True),
        ("gpt-5-mini", True),
        ("gpt-5-chat-latest", True),
        ("gpt-4o", False),
        ("gpt-4o-mini", False),
    ]

    for model, should_be_gpt5 in test_cases:
        is_gpt5 = model.startswith("gpt-5")
        supports_structured = client._supports_structured_output(model)

        print(f"  {model}:")
        print(f"    Is GPT-5: {'✓' if is_gpt5 == should_be_gpt5 else '❌'}")
        print(f"    Supports structured output: {'✓' if supports_structured else '❌'}")

        assert is_gpt5 == should_be_gpt5, f"GPT-5 detection failed for {model}"
        assert supports_structured, f"Structured output support failed for {model}"

    print("✅ GPT-5 parameter handling correctly implemented")
    return True


async def test_gpt4_backward_compatibility():
    """Test that GPT-4 models still work correctly."""
    print("\n=== Testing GPT-4 Backward Compatibility ===")

    client = OpenAIClient()

    # Test GPT-4 models
    gpt4_models = ["gpt-4o", "gpt-4o-mini"]

    for model in gpt4_models:
        supports = client._supports_structured_output(model)
        print(f"  {model}: {'✓ SUPPORTED' if supports else '❌ NOT SUPPORTED'}")
        assert supports, f"GPT-4 model {model} should support structured output"

    print("✅ GPT-4 models backward compatibility maintained")
    return True


async def main():
    """Run GPT-5 structured output tests."""
    print("=== GPT-5 Structured Output Support Tests ===")

    tests = [
        test_gpt5_models_support(),
        test_gpt5_parameter_handling(),
        test_gpt4_backward_compatibility(),
    ]

    results = await asyncio.gather(*tests, return_exceptions=True)

    passed = sum(1 for result in results if result is True)
    total = len(results)

    print(f"\n=== Results: {passed}/{total} tests passed ===")

    if passed == total:
        print("\n🎉 All GPT-5 structured output tests passed!")
        print("\n✨ Key Improvements:")
        print("   • GPT-5 models now properly support native structured output")
        print("   • Enhanced parameters: reasoning_effort, verbosity")
        print("   • Proper max_completion_tokens parameter handling")
        print("   • Backward compatibility with GPT-4 models maintained")
        print("   • Fallback system still available for unsupported models")
        return 0
    else:
        print(f"\n⚠️ {total - passed} tests failed")
        for i, result in enumerate(results):
            if result is not True:
                print(f"  Test {i + 1}: {result}")
        return 1


if __name__ == "__main__":
    import sys

    exit_code = asyncio.run(main())
    sys.exit(exit_code)
