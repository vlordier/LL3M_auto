#!/usr/bin/env python3
"""Simple test of LM Studio connection and basic AI interaction."""

import asyncio
import aiohttp
import json


async def test_lm_studio_simple():
    """Simple test of LM Studio connectivity."""
    base_url = "http://localhost:1234/v1"
    
    async with aiohttp.ClientSession() as session:
        print("🤖 Testing LM Studio Connection")
        print("=" * 40)
        
        # Test 1: Check models
        print("\n📋 Step 1: Check Available Models")
        try:
            async with session.get(f"{base_url}/models") as response:
                if response.status == 200:
                    models = await response.json()
                    model_list = models.get("data", [])
                    if model_list:
                        model_name = model_list[0]["id"]
                        print(f"✅ Model loaded: {model_name}")
                    else:
                        print("❌ No models loaded")
                        return False
                else:
                    print(f"❌ Failed to get models (status: {response.status})")
                    return False
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False
        
        # Test 2: Simple chat completion
        print(f"\n🧠 Step 2: Test AI Chat Completion")
        simple_request = {
            "model": model_name,
            "messages": [
                {"role": "user", "content": "Say 'Hello from Qwen2.5 Coder!' and nothing else."}
            ],
            "temperature": 0.1,
            "max_tokens": 50
        }
        
        try:
            async with session.post(f"{base_url}/chat/completions", json=simple_request) as response:
                if response.status == 200:
                    result = await response.json()
                    ai_response = result["choices"][0]["message"]["content"]
                    print(f"✅ AI Response: {ai_response}")
                else:
                    print(f"❌ Chat completion failed (status: {response.status})")
                    return False
        except Exception as e:
            print(f"❌ Chat completion error: {e}")
            return False
        
        # Test 3: Python code generation
        print(f"\n💻 Step 3: Test Python Code Generation")
        code_request = {
            "model": model_name,
            "messages": [
                {"role": "user", "content": "Write a simple Python function that adds two numbers. Only code, no explanations."}
            ],
            "temperature": 0.1,
            "max_tokens": 100
        }
        
        try:
            async with session.post(f"{base_url}/chat/completions", json=code_request) as response:
                if response.status == 200:
                    result = await response.json()
                    code_response = result["choices"][0]["message"]["content"]
                    print(f"✅ Generated code:")
                    print(f"   {code_response[:100]}...")
                else:
                    print(f"❌ Code generation failed (status: {response.status})")
                    return False
        except Exception as e:
            print(f"❌ Code generation error: {e}")
            return False
        
        print("\n" + "=" * 40)
        print("🎉 LM Studio Integration Test PASSED!")
        print(f"\nSummary:")
        print(f"  ✅ LM Studio Server: Connected")
        print(f"  ✅ Model: {model_name}")
        print(f"  ✅ Chat Completion: Working")
        print(f"  ✅ Code Generation: Working")
        print(f"\n🚀 Ready for Blender integration!")
        
        return True


if __name__ == "__main__":
    success = asyncio.run(test_lm_studio_simple())
    if not success:
        exit(1)