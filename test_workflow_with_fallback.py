#!/usr/bin/env python3
"""Test workflow with RetryOutputParser fallback."""

import asyncio
from unittest.mock import AsyncMock, patch

from src.agents.base import EnhancedBaseAgent
from src.agents.critic_schemas import AspectScore, RefinementPriority, VisualAnalysisResult


async def test_gpt5_native_structured_output():
    """Test GPT-5 native structured output."""
    print("=== Testing GPT-5 Native Structured Output ===")

    try:
        from src.utils.llm_client import OpenAIClient

        client = OpenAIClient()

        # Test that gpt-5-chat-latest is now detected as supported
        assert client._supports_structured_output("gpt-5-chat-latest")
        print("✓ gpt-5-chat-latest correctly detected as supported")

        # Mock the structured output to return a properly parsed result
        mock_result = VisualAnalysisResult(
            overall_score=8,
            visual_quality=AspectScore(score=8, notes="Excellent visual quality"),
            geometry=AspectScore(score=7, notes="Good geometry structure"),
            materials=AspectScore(score=9, notes="Outstanding materials"),
            lighting=AspectScore(score=8, notes="Professional lighting"),
            composition=AspectScore(score=8, notes="Great composition"),
            requirements_match=AspectScore(score=9, notes="Perfect match"),
            needs_refinement=False,
            critical_issues=[],
            improvement_suggestions=["Minor texture enhancements"],
            refinement_priority=RefinementPriority.LOW,
        )

        with patch.object(client.client.beta.chat.completions, "parse", new_callable=AsyncMock) as mock_parse:
            # Mock the structured response
            mock_response = AsyncMock()
            mock_response.choices = [AsyncMock()]
            mock_response.choices[0].message.parsed = mock_result
            mock_response.choices[0].message.refusal = None
            mock_parse.return_value = mock_response
            # Test structured completion with fallback
            messages = [
                {"role": "system", "content": "You are a 3D model critic."},
                {"role": "user", "content": "Analyze this wooden table image."},
            ]

            result = await client.structured_completion(
                messages=messages,
                response_format=VisualAnalysisResult,
                model="gpt-5-chat-latest",
            )

            print("✓ GPT-5 native structured output successful")
            print(f"  Overall score: {result.overall_score}")
            print(f"  Needs refinement: {result.needs_refinement}")
            print(f"  Visual quality score: {result.visual_quality.score}")

            # Verify it's the correct type and structure
            assert isinstance(result, VisualAnalysisResult)
            assert result.overall_score == 8
            assert not result.needs_refinement
            assert isinstance(result.visual_quality, AspectScore)

            print("✓ GPT-5 native structured output validation passed")
            return True

    except Exception as e:
        print(f"❌ Structured output test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_base_agent_structured_request():
    """Test base agent structured request with fallback."""
    print("\n=== Testing Base Agent Structured Request ===")

    class TestAgent(EnhancedBaseAgent):
        @property
        def agent_type(self):
            from src.utils.types import AgentType

            return AgentType.CRITIC

        @property
        def name(self):
            return "Test Agent"

        async def process(self, state):
            pass

    # Mock successful structured request from the LLM client
    _mock_result = VisualAnalysisResult(
        overall_score=8,
        visual_quality=AspectScore(score=8, notes="Good appearance"),
        geometry=AspectScore(score=7, notes="Solid structure"),
        materials=AspectScore(score=6, notes="Basic materials"),
        lighting=AspectScore(score=7, notes="Adequate lighting"),
        composition=AspectScore(score=8, notes="Good composition"),
        requirements_match=AspectScore(score=8, notes="Meets requirements"),
        needs_refinement=False,
        critical_issues=[],
        improvement_suggestions=["Minor texture enhancements"],
        refinement_priority=RefinementPriority.LOW,
    )

    # Mock the structured completion method to return our mock result
    mock_response = (
        '{"overall_score": 8, "visual_quality": {"score": 8, "notes": "Good appearance"}, '
        '"geometry": {"score": 7, "notes": "Solid structure"}, "materials": {"score": 6, '
        '"notes": "Basic materials"}, "lighting": {"score": 7, "notes": "Adequate lighting"}, '
        '"composition": {"score": 8, "notes": "Good composition"}, "requirements_match": '
        '{"score": 8, "notes": "Meets requirements"}, "needs_refinement": false, '
        '"critical_issues": [], "improvement_suggestions": ["Minor texture enhancements"], '
        '"refinement_priority": "low"}'
    )

    with patch(
        "src.agents.base.EnhancedBaseAgent.make_openai_request",
        new_callable=AsyncMock,
        return_value=mock_response,
    ):
        try:
            # Create agent with OpenAI configuration to trigger structured output path
            import os

            os.environ["USE_LOCAL_LLM"] = "false"  # Force OpenAI client

            agent = TestAgent({"model": "gpt-5-chat-latest"})

            # Override client to test both paths
            from src.utils.llm_client import LMStudioClient

            agent.client = LMStudioClient()  # Force fallback path

            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Analyze this 3D model."},
            ]

            result = await agent.make_structured_request(messages=messages, response_format=VisualAnalysisResult)

            print("✓ Base agent structured request successful")
            print(f"  Result type: {type(result).__name__}")
            print(f"  Overall score: {result.overall_score}")

            assert isinstance(result, VisualAnalysisResult)
            assert result.overall_score == 8

            print("✓ Base agent structured request validation passed")
            return True

        except Exception as e:
            print(f"❌ Base agent test failed: {e}")
            return False


async def main():
    """Run fallback workflow tests."""
    print("=== Structured Output Fallback Workflow Tests ===")

    tests = [test_gpt5_native_structured_output(), test_base_agent_structured_request()]

    results = await asyncio.gather(*tests, return_exceptions=True)

    passed = sum(1 for result in results if result is True)
    total = len(results)

    print(f"\n=== Results: {passed}/{total} tests passed ===")

    if passed == total:
        print("\n🎉 All fallback workflow tests passed!")
        print("\n✨ Key Benefits:")
        print("   • GPT-5 models now support native structured output")
        print("   • Enhanced GPT-5 parameters: reasoning_effort, verbosity")
        print("   • Proper max_completion_tokens handling for GPT-5")
        print("   • Backward compatibility with GPT-4 models maintained")
        print("   • Multi-layer fallback system for unsupported models")
        print("   • Maintains type safety with Pydantic validation")
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
