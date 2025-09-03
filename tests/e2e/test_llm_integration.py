"""E2E tests for LLM integration (OpenAI and LM Studio)."""

import pytest

from src.utils.llm_client import get_llm_client


@pytest.mark.asyncio
async def test_llm_client_basic_completion(lm_studio_server):
    """Test basic LLM completion functionality."""
    client = get_llm_client()

    messages = [
        {"role": "user", "content": "What is 2 + 2? Respond with only the number."}
    ]

    response = await client.chat_completion(
        messages=messages, temperature=0.1, max_tokens=10
    )

    assert response is not None
    assert "choices" in response
    assert len(response["choices"]) > 0

    content = response["choices"][0]["message"]["content"].strip()
    assert "4" in content

    print(f"✓ LLM responded correctly: {content}")


@pytest.mark.asyncio
async def test_llm_client_code_generation(lm_studio_server):
    """Test LLM code generation capabilities."""
    client = get_llm_client()

    messages = [
        {
            "role": "user",
            "content": """Generate Python code to create a cube in Blender. 
                       Use only bpy operations. Keep it simple and add comments.""",
        }
    ]

    response = await client.chat_completion(
        messages=messages, temperature=0.3, max_tokens=500
    )

    assert response is not None
    assert "choices" in response

    content = response["choices"][0]["message"]["content"]

    # Check for expected Blender code patterns
    assert "bpy" in content.lower()
    assert any(keyword in content.lower() for keyword in ["cube", "primitive", "mesh"])

    print("✓ LLM generated code containing Blender patterns")
    print(f"Generated code preview: {content[:200]}...")


@pytest.mark.asyncio
async def test_llm_streaming_completion(lm_studio_server):
    """Test LLM streaming functionality."""
    client = get_llm_client()

    messages = [{"role": "user", "content": "Count from 1 to 5, one number per line."}]

    chunks = []
    async for chunk in client.stream_chat_completion(
        messages=messages, temperature=0.1, max_tokens=50
    ):
        chunks.append(chunk)
        if len(chunks) >= 10:  # Prevent infinite loops in tests
            break

    assert len(chunks) > 0
    print(f"✓ Received {len(chunks)} streaming chunks")

    # Verify chunk structure
    for chunk in chunks[:3]:  # Check first few chunks
        assert "choices" in chunk
        if chunk["choices"] and chunk["choices"][0]["delta"]:
            print(f"  Chunk content: {chunk['choices'][0]['delta']}")


@pytest.mark.asyncio
async def test_llm_model_detection(lm_studio_server):
    """Test that LM Studio model detection works."""
    from src.utils.llm_client import LMStudioClient

    client = LMStudioClient()
    models = await client._get_available_models()

    assert len(models) > 0
    assert models[0] != "local-model"  # Should detect actual model name

    print(f"✓ Detected models: {models}")


@pytest.mark.asyncio
async def test_llm_error_handling(lm_studio_server):
    """Test LLM error handling."""
    client = get_llm_client()

    # Test with very restrictive settings that might cause issues
    messages = [
        {
            "role": "user",
            "content": "Generate a very long essay about artificial intelligence and machine learning with detailed explanations of neural networks.",
        }
    ]

    try:
        response = await client.chat_completion(
            messages=messages,
            temperature=0.7,
            max_tokens=5,  # Very small limit to potentially trigger truncation
        )

        # Should still succeed but with truncated response
        assert response is not None
        assert "choices" in response

        print("✓ LLM handled restrictive parameters gracefully")

    except Exception as e:
        # If it fails, it should fail gracefully
        print(f"✓ LLM failed gracefully with error: {e}")
        assert "timeout" not in str(e).lower()  # Shouldn't be a timeout


@pytest.mark.asyncio
async def test_llm_context_awareness(lm_studio_server):
    """Test LLM context awareness in conversations."""
    client = get_llm_client()

    # Multi-turn conversation
    messages = [
        {
            "role": "user",
            "content": "I want to create a 3D scene. What's the first step?",
        },
        {
            "role": "assistant",
            "content": "The first step is to set up your 3D environment. In Blender, you would start by opening the application and clearing the default scene.",
        },
        {"role": "user", "content": "What should I add to the scene next?"},
    ]

    response = await client.chat_completion(
        messages=messages, temperature=0.5, max_tokens=200
    )

    content = response["choices"][0]["message"]["content"].lower()

    # Should reference 3D objects or scene elements
    assert any(
        keyword in content
        for keyword in ["object", "cube", "sphere", "light", "camera", "mesh"]
    )

    print("✓ LLM maintained context about 3D scene creation")
    print(f"Response: {content[:150]}...")
