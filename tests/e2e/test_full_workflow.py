"""E2E tests for full LL3M workflow integration."""

import pytest

from src.utils.blender_code_utils import (
    generate_fallback_code,
    get_background_mode_system_prompt,
    transform_interactive_to_background,
    validate_background_compatibility,
)
from src.utils.llm_client import get_llm_client


@pytest.mark.asyncio
async def test_simple_workflow_e2e(
    lm_studio_server, blender_mcp_server, http_session, test_prompt, temp_output_dir
):
    """Test complete workflow: LLM -> Code Generation -> Blender Execution."""

    # Step 1: Generate Blender code using LLM
    print("🤖 Step 1: Generating code with LLM...")

    client = get_llm_client()

    system_prompt = get_background_mode_system_prompt()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Generate Blender Python code for: {test_prompt}"},
    ]

    response = await client.chat_completion(
        messages=messages, temperature=0.3, max_tokens=800
    )

    assert response is not None
    generated_code = response["choices"][0]["message"]["content"]

    # Extract Python code block if wrapped in markdown
    if "```python" in generated_code:
        start = generated_code.find("```python") + 9
        end = generated_code.find("```", start)
        if end > start:
            generated_code = generated_code[start:end].strip()
    elif "```" in generated_code:
        start = generated_code.find("```") + 3
        end = generated_code.find("```", start)
        if end > start:
            generated_code = generated_code[start:end].strip()

    print(f"✓ Generated {len(generated_code)} characters of code")
    print(f"Code preview: {generated_code[:200]}...")

    # Step 1.5: Validate and transform code for background compatibility
    print("\n🔄 Step 1.5: Validating and transforming code for background mode...")

    is_compatible, issues = validate_background_compatibility(generated_code)
    if not is_compatible:
        print(f"⚠️  Code has {len(issues)} compatibility issues:")
        for issue in issues:
            print(f"   - {issue}")

        print("🔧 Transforming code for background compatibility...")
        generated_code = transform_interactive_to_background(generated_code)

        # Validate again
        is_compatible, remaining_issues = validate_background_compatibility(
            generated_code
        )
        if not is_compatible:
            print(f"⚠️  {len(remaining_issues)} issues remain after transformation")
            print("🔄 Using fallback code generation...")
            generated_code = generate_fallback_code(test_prompt)
        else:
            print("✓ Code successfully transformed for background mode")
    else:
        print("✓ Code is already background-mode compatible")

    # Step 2: Execute the generated code in Blender
    print("\n🎨 Step 2: Executing code in Blender...")

    payload = {"code": generated_code, "timeout": 60}

    async with http_session.post(
        f"{blender_mcp_server}/execute", json=payload
    ) as response:
        assert response.status == 200
        execution_result = await response.json()

        if not execution_result["success"]:
            print(f"❌ Blender execution failed: {execution_result.get('error')}")
            print(f"Generated code was:\n{generated_code}")
            pytest.fail(f"Blender execution failed: {execution_result.get('error')}")

        print("✓ Code executed successfully in Blender")
        for log in execution_result.get("logs", []):
            print(f"  Log: {log}")

    # Step 3: Verify the scene was created correctly
    print("\n🔍 Step 3: Verifying scene creation...")

    async with http_session.get(f"{blender_mcp_server}/scene/info") as response:
        assert response.status == 200
        scene_info = await response.json()

        objects = scene_info["objects"]
        assert len(objects) > 0, "Scene should contain at least one object"

        # Check for cube-related objects (since test prompt asks for a cube)
        cube_objects = [obj for obj in objects if "cube" in obj.lower()]
        assert len(cube_objects) > 0, f"Expected cube object, got objects: {objects}"

        print(f"✓ Scene verified with {len(objects)} objects: {objects}")

    # Step 4: Save the result
    print("\n💾 Step 4: Saving scene...")

    output_file = temp_output_dir / "e2e_test_scene.blend"

    async with http_session.post(
        f"{blender_mcp_server}/scene/save", json={"filepath": str(output_file)}
    ) as response:
        assert response.status == 200
        save_result = await response.json()
        assert save_result["success"]

        assert output_file.exists()
        assert output_file.stat().st_size > 0

        print(f"✓ Scene saved to {output_file} ({output_file.stat().st_size} bytes)")

    print("\n🎉 Full E2E workflow completed successfully!")


@pytest.mark.asyncio
async def test_complex_workflow_e2e(
    lm_studio_server,
    blender_mcp_server,
    http_session,
    complex_test_prompt,
    temp_output_dir,
):
    """Test complex workflow with multiple objects and scene setup."""

    print("🚀 Testing complex workflow...")

    client = get_llm_client()

    from src.utils.blender_code_utils import get_enhanced_system_prompt

    system_prompt = get_enhanced_system_prompt("material")

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"Create this Blender scene: {complex_test_prompt}",
        },
    ]

    # Generate code
    response = await client.chat_completion(
        messages=messages,
        temperature=0.2,  # Lower temperature for more consistent code
        max_tokens=1500,
    )

    generated_code = response["choices"][0]["message"]["content"]

    # Clean up code extraction
    if "```python" in generated_code:
        start = generated_code.find("```python") + 9
        end = generated_code.find("```", start)
        if end > start:
            generated_code = generated_code[start:end].strip()
    elif "```" in generated_code:
        start = generated_code.find("```") + 3
        end = generated_code.find("```", start)
        if end > start:
            generated_code = generated_code[start:end].strip()

    print(f"✓ Generated {len(generated_code)} characters of complex scene code")

    # Transform code for background compatibility
    print("🔧 Transforming complex code for background mode...")
    is_compatible, issues = validate_background_compatibility(generated_code)
    if not is_compatible:
        print(f"⚠️  Found {len(issues)} compatibility issues, transforming...")
        generated_code = transform_interactive_to_background(generated_code)

    # Execute in Blender
    payload = {
        "code": generated_code,
        "timeout": 120,
    }  # Longer timeout for complex scene

    async with http_session.post(
        f"{blender_mcp_server}/execute", json=payload
    ) as response:
        assert response.status == 200
        result = await response.json()

        if not result["success"]:
            print(f"❌ Complex scene creation failed: {result.get('error')}")
            print("Generated code:")
            print(generated_code)
            pytest.fail(f"Complex scene creation failed: {result.get('error')}")

        print("✓ Complex scene created successfully")

    # Verify complex scene
    async with http_session.get(f"{blender_mcp_server}/scene/info") as response:
        assert response.status == 200
        scene_info = await response.json()

        objects = scene_info["objects"]

        # Should have multiple objects as requested
        assert (
            len(objects) >= 3
        ), f"Expected at least 3 objects, got {len(objects)}: {objects}"

        # Check for expected object types
        object_names_lower = [obj.lower() for obj in objects]

        # Look for cube, sphere, cylinder
        has_cube = any("cube" in name for name in object_names_lower)
        has_sphere = any("sphere" in name for name in object_names_lower)
        has_cylinder = any(
            "cylinder" in name or "cyl" in name for name in object_names_lower
        )

        print(f"✓ Complex scene has {len(objects)} objects")
        print(f"  Objects: {objects}")
        print(f"  Has cube: {has_cube}, sphere: {has_sphere}, cylinder: {has_cylinder}")

        # At least some of the requested objects should be present
        assert (
            sum([has_cube, has_sphere, has_cylinder]) >= 2
        ), "Should have at least 2 of the 3 requested object types"

    # Save complex scene
    output_file = temp_output_dir / "complex_e2e_scene.blend"

    async with http_session.post(
        f"{blender_mcp_server}/scene/save", json={"filepath": str(output_file)}
    ) as response:
        assert response.status == 200

        assert output_file.exists()
        print(f"✓ Complex scene saved ({output_file.stat().st_size} bytes)")

    print("🎉 Complex E2E workflow completed successfully!")


@pytest.mark.asyncio
async def test_iterative_workflow_e2e(
    lm_studio_server, blender_mcp_server, http_session, temp_output_dir
):
    """Test iterative workflow - create, modify, and refine a scene."""

    print("🔄 Testing iterative workflow...")

    client = get_llm_client()

    # Step 1: Create initial scene
    print("\n📝 Step 1: Creating initial scene...")

    initial_prompt = "Create a simple scene with a blue cube at the center"

    messages = [
        {
            "role": "system",
            "content": get_background_mode_system_prompt(),
        },
        {"role": "user", "content": initial_prompt},
    ]

    response = await client.chat_completion(messages=messages, temperature=0.3)
    initial_code = response["choices"][0]["message"]["content"]

    # Clean code
    if "```python" in initial_code:
        start = initial_code.find("```python") + 9
        end = initial_code.find("```", start)
        if end > start:
            initial_code = initial_code[start:end].strip()

    # Transform and execute initial code
    print("🔧 Transforming initial code for background mode...")
    initial_code = transform_interactive_to_background(initial_code)

    async with http_session.post(
        f"{blender_mcp_server}/execute", json={"code": initial_code, "timeout": 60}
    ) as response:
        assert response.status == 200
        result = await response.json()
        assert result[
            "success"
        ], f"Initial scene creation failed: {result.get('error')}"
        print("✓ Initial scene created")

    # Step 2: Modify the scene
    print("\n🔧 Step 2: Modifying scene...")

    modification_prompt = """Add to the existing scene:
    - A red sphere to the right of the cube (position 3, 0, 0)
    - A yellow light above both objects (position 0, 0, 5)
    Do not clear the existing scene, just add to it."""

    messages = [
        {
            "role": "system",
            "content": get_background_mode_system_prompt()
            + "\n\nIMPORTANT: Do NOT clear the existing scene. Only add new objects to the current scene.",
        },
        {"role": "user", "content": modification_prompt},
    ]

    response = await client.chat_completion(messages=messages, temperature=0.3)
    modification_code = response["choices"][0]["message"]["content"]

    # Clean code
    if "```python" in modification_code:
        start = modification_code.find("```python") + 9
        end = modification_code.find("```", start)
        if end > start:
            modification_code = modification_code[start:end].strip()

    # Transform and execute modification
    print("🔧 Transforming modification code for background mode...")
    modification_code = transform_interactive_to_background(modification_code)

    async with http_session.post(
        f"{blender_mcp_server}/execute", json={"code": modification_code, "timeout": 60}
    ) as response:
        assert response.status == 200
        result = await response.json()
        assert result["success"], f"Scene modification failed: {result.get('error')}"
        print("✓ Scene modified")

    # Step 3: Verify the iterative changes
    print("\n🔍 Step 3: Verifying iterative changes...")

    async with http_session.get(f"{blender_mcp_server}/scene/info") as response:
        assert response.status == 200
        scene_info = await response.json()

        objects = scene_info["objects"]

        # Should have multiple objects now
        assert (
            len(objects) >= 2
        ), f"Expected at least 2 objects after modifications, got: {objects}"

        object_names_lower = [obj.lower() for obj in objects]
        has_cube = any("cube" in name for name in object_names_lower)
        has_sphere = any("sphere" in name for name in object_names_lower)
        has_light = any(
            "light" in name or "sun" in name or "lamp" in name
            for name in object_names_lower
        )

        print(f"✓ Iterative scene has {len(objects)} objects: {objects}")
        print(f"  Has cube: {has_cube}, sphere: {has_sphere}, light: {has_light}")

        # Should have both original and new objects
        assert has_cube, "Original cube should still be present"
        assert (
            has_sphere or len(objects) >= 3
        ), "New sphere or additional objects should be present"

    # Save final iterative scene
    output_file = temp_output_dir / "iterative_e2e_scene.blend"

    async with http_session.post(
        f"{blender_mcp_server}/scene/save", json={"filepath": str(output_file)}
    ) as response:
        assert response.status == 200
        assert output_file.exists()
        print(f"✓ Iterative scene saved ({output_file.stat().st_size} bytes)")

    print("🎉 Iterative E2E workflow completed successfully!")


@pytest.mark.asyncio
async def test_error_recovery_workflow(
    lm_studio_server, blender_mcp_server, http_session
):
    """Test workflow error recovery and handling."""

    print("🛠️  Testing error recovery workflow...")

    client = get_llm_client()

    # Step 1: Try to generate code that might have issues
    problematic_prompt = "Create a complex molecular structure with 100 interconnected spheres and advanced physics simulation"

    messages = [
        {
            "role": "system",
            "content": get_background_mode_system_prompt()
            + "\n\nIf the request is too complex, create a simplified version instead.",
        },
        {"role": "user", "content": problematic_prompt},
    ]

    response = await client.chat_completion(messages=messages, temperature=0.4)
    generated_code = response["choices"][0]["message"]["content"]

    # Clean code
    if "```python" in generated_code:
        start = generated_code.find("```python") + 9
        end = generated_code.find("```", start)
        if end > start:
            generated_code = generated_code[start:end].strip()

    print(f"✓ Generated code for complex request ({len(generated_code)} chars)")

    # Transform the code for background compatibility
    generated_code = transform_interactive_to_background(generated_code)

    # Step 2: Try to execute the code (may fail)
    async with http_session.post(
        f"{blender_mcp_server}/execute", json={"code": generated_code, "timeout": 30}
    ) as response:
        assert response.status == 200
        result = await response.json()

        if not result["success"]:
            print(f"⚠️  Initial code failed as expected: {result.get('error')}")

            # Step 3: Generate simpler fallback code
            fallback_prompt = "The previous code was too complex. Create a simple scene with just 3 colored spheres in a triangle formation."

            messages = [
                {
                    "role": "system",
                    "content": get_background_mode_system_prompt(),
                },
                {"role": "user", "content": fallback_prompt},
            ]

            response = await client.chat_completion(messages=messages, temperature=0.2)
            fallback_code = response["choices"][0]["message"]["content"]

            # Clean fallback code
            if "```python" in fallback_code:
                start = fallback_code.find("```python") + 9
                end = fallback_code.find("```", start)
                if end > start:
                    fallback_code = fallback_code[start:end].strip()

            # Transform and execute fallback code
            fallback_code = transform_interactive_to_background(fallback_code)

            async with http_session.post(
                f"{blender_mcp_server}/execute",
                json={"code": fallback_code, "timeout": 60},
            ) as response:
                assert response.status == 200
                fallback_result = await response.json()

                assert fallback_result[
                    "success"
                ], f"Fallback code should work: {fallback_result.get('error')}"
                print("✓ Fallback code executed successfully")
        else:
            print("✓ Complex code actually worked!")

    # Verify final scene
    async with http_session.get(f"{blender_mcp_server}/scene/info") as response:
        assert response.status == 200
        scene_info = await response.json()

        objects = scene_info["objects"]
        assert (
            len(objects) > 0
        ), "Should have at least some objects after error recovery"

        print(f"✓ Error recovery workflow completed with {len(objects)} objects")

    print("🎉 Error recovery E2E workflow completed successfully!")
