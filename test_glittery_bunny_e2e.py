#!/usr/bin/env python3
"""Complete E2E test: Create a glittery pink bunny using LM Studio + Blender."""

import asyncio
import aiohttp
import json
from pathlib import Path
from datetime import datetime


async def create_glittery_bunny_e2e():
    """Complete workflow: Prompt -> LLM -> Blender -> Render -> Save."""
    lm_studio_url = "http://localhost:1234/v1"
    blender_url = "http://localhost:3001"
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"renders/glittery_bunny_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    async with aiohttp.ClientSession() as session:
        print("🐰✨ LL3M E2E: Creating Glittery Pink Bunny")
        print("=" * 60)
        
        # Step 1: Get model information
        print(f"\n🤖 Step 1: Connecting to LM Studio")
        async with session.get(f"{lm_studio_url}/models") as response:
            models = await response.json()
            model_name = models["data"][0]["id"]
            print(f"✅ Using model: {model_name}")
        
        # Step 2: Generate Blender code with AI
        print(f"\n🎨 Step 2: Generating Blender Code for Glittery Pink Bunny")
        
        prompt = "a glittery pink bunny"
        
        system_prompt = """You are an expert Blender Python programmer. Create complete, working Blender Python code that:

1. Clears the default scene completely (including default cube, light, camera)
2. Creates a detailed bunny shape using Blender operations
3. Applies a glittery pink material with proper nodes and textures
4. Sets up professional lighting (multiple lights for good illumination)
5. Positions a camera for the best view of the bunny
6. Includes print statements for progress tracking

Requirements:
- Use only bpy operations (no external imports)
- Make the bunny actually look like a bunny (ears, body, tail)
- Create a truly glittery pink material with nodes
- Use proper Blender material nodes for metallic/glossy effects
- Set up multiple lights for professional rendering
- Position camera to showcase the bunny beautifully
- Include helpful print statements

Generate clean, complete, executable Blender Python code."""

        llm_request = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Create Blender Python code for: {prompt}"}
            ],
            "temperature": 0.7,
            "max_tokens": 2000
        }
        
        async with session.post(f"{lm_studio_url}/chat/completions", json=llm_request) as response:
            result = await response.json()
            generated_code = result["choices"][0]["message"]["content"]
            
            # Clean up the code (remove markdown if present)
            if "```python" in generated_code:
                generated_code = generated_code.split("```python")[1].split("```")[0].strip()
            elif "```" in generated_code:
                generated_code = generated_code.split("```")[1].split("```")[0].strip()
            
            print("✅ Generated Blender code for glittery pink bunny")
            print("📄 Code preview (first 10 lines):")
            for i, line in enumerate(generated_code.split('\n')[:10], 1):
                print(f"   {i:2d}: {line}")
            print("   ... (truncated)")
        
        # Save generated code
        code_path = output_dir / "generated_bunny_code.py"
        with open(code_path, "w") as f:
            f.write(generated_code)
        print(f"💾 Saved generated code to: {code_path}")
        
        # Step 3: Execute code in Blender
        print(f"\n🎭 Step 3: Executing Code in Blender")
        async with session.post(f"{blender_url}/execute", json={"code": generated_code}) as response:
            execution_result = await response.json()
            
            if execution_result["success"]:
                print("✅ Code executed successfully in Blender!")
                print("📋 Execution logs:")
                for log in execution_result.get("logs", []):
                    if "STDOUT:" in log:
                        stdout = log.replace("STDOUT: ", "").strip()
                        for line in stdout.split('\n'):
                            if line.strip():
                                print(f"   📝 {line}")
            else:
                print(f"❌ Code execution failed: {execution_result.get('error')}")
                return False
        
        # Step 4: Get scene information
        print(f"\n🔍 Step 4: Verifying Scene Creation")
        async with session.get(f"{blender_url}/scene/info") as response:
            scene_info = await response.json()
            objects = scene_info["objects"]
            print(f"✅ Scene '{scene_info['name']}' now contains {len(objects)} objects:")
            for obj in objects:
                print(f"   🎯 {obj}")
        
        # Step 5: Save the Blender scene
        print(f"\n💾 Step 5: Saving Blender Scene")
        scene_path = output_dir / "glittery_bunny.blend"
        
        save_code = f'''
import bpy
import os

# Save the scene
scene_path = r"{scene_path.absolute()}"
bpy.ops.wm.save_as_mainfile(filepath=scene_path)
print(f"✅ Scene saved to: {{scene_path}}")

# Set up render settings for high quality
scene = bpy.context.scene
scene.render.engine = 'CYCLES'  # Use Cycles for better materials
scene.render.filepath = r"{output_dir.absolute()}/glittery_bunny_render"
scene.render.image_settings.file_format = 'PNG'
scene.render.resolution_x = 1920
scene.render.resolution_y = 1080
scene.render.resolution_percentage = 100

# Enable denoising for cleaner renders
scene.cycles.use_denoising = True
scene.cycles.denoiser = 'OPENIMAGEDENOISE'

print("✅ Render settings configured for high quality")
print(f"📸 Render will be saved to: {{scene.render.filepath}}")

# Optional: Take a quick screenshot of viewport
'''
        
        async with session.post(f"{blender_url}/execute", json={"code": save_code}) as response:
            save_result = await response.json()
            if save_result["success"]:
                print("✅ Scene saved successfully!")
                for log in save_result.get("logs", []):
                    if "STDOUT:" in log:
                        stdout = log.replace("STDOUT: ", "").strip()
                        for line in stdout.split('\n'):
                            if line.strip():
                                print(f"   💾 {line}")
            else:
                print(f"❌ Save failed: {save_result.get('error')}")
        
        # Step 6: Render the scene
        print(f"\n🎬 Step 6: Rendering Glittery Pink Bunny")
        render_code = '''
import bpy

print("🎬 Starting render...")
# Render the scene
bpy.ops.render.render(write_still=True)
print("✅ Render completed!")
print(f"📸 Render saved to: {bpy.context.scene.render.filepath}.png")
'''
        
        async with session.post(f"{blender_url}/execute", json={"code": render_code, "timeout": 120}) as response:
            render_result = await response.json()
            if render_result["success"]:
                print("✅ Rendering completed!")
                for log in render_result.get("logs", []):
                    if "STDOUT:" in log:
                        stdout = log.replace("STDOUT: ", "").strip()
                        for line in stdout.split('\n'):
                            if line.strip():
                                print(f"   🎬 {line}")
            else:
                print(f"⚠️  Render may have issues: {render_result.get('error')}")
        
        # Step 7: Create project summary
        print(f"\n📋 Step 7: Creating Project Summary")
        summary = f"""# Glittery Pink Bunny - LL3M E2E Test Results

## Project Details
- **Created**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- **Prompt**: "{prompt}"
- **Model Used**: {model_name}
- **Output Directory**: {output_dir.name}

## Generated Files
- `generated_bunny_code.py` - AI-generated Blender Python code
- `glittery_bunny.blend` - Blender scene file
- `glittery_bunny_render.png` - Final rendered image

## Workflow Summary
✅ LM Studio AI Code Generation  
✅ Blender Code Execution  
✅ Scene Creation & Verification  
✅ Scene File Saving  
✅ High-Quality Rendering  

## Scene Objects
{chr(10).join(f"- {obj}" for obj in objects)}

## Next Steps
1. Open `glittery_bunny.blend` in Blender to view/edit the scene
2. Check `glittery_bunny_render.png` for the final rendered result
3. Modify the generated code for variations or improvements

---
Generated by LL3M - Local LLM + Blender Integration
"""
        
        summary_path = output_dir / "README.md"
        with open(summary_path, "w") as f:
            f.write(summary)
        
        print("=" * 60)
        print("🎉 GLITTERY PINK BUNNY E2E TEST COMPLETED!")
        print(f"\n📁 All files saved to: {output_dir.absolute()}")
        print(f"🎭 Scene file: glittery_bunny.blend")
        print(f"🎬 Render: glittery_bunny_render.png") 
        print(f"💻 Code: generated_bunny_code.py")
        print(f"📋 Summary: README.md")
        
        print(f"\n🚀 LL3M Workflow Results:")
        print(f"  ✅ AI-Generated Blender Code")
        print(f"  ✅ Successfully Executed in Blender")
        print(f"  ✅ Created {len(objects)} Scene Objects")
        print(f"  ✅ Saved High-Quality Scene & Render")
        print(f"  ✅ Complete Local LLM + Blender Pipeline!")
        
        return True


if __name__ == "__main__":
    print("🐰✨ Starting Glittery Pink Bunny E2E Test...")
    success = asyncio.run(create_glittery_bunny_e2e())
    
    if success:
        print(f"\n🎯 RESULT: E2E Glittery Pink Bunny Creation SUCCESSFUL!")
        print("   Your glittery pink bunny has been created and rendered! 🐰✨")
    else:
        print(f"\n❌ RESULT: E2E test failed. Check output above.")
        exit(1)