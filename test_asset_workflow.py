#!/usr/bin/env python3
"""Test script for the complete asset import-edit-export workflow."""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.workflow.orchestrator import LL3MOrchestrator


async def test_asset_workflow():
    """Test the complete asset workflow."""
    print("🧪 Testing LL3M Asset Import-Edit-Export Workflow")
    print("=" * 50)

    # Initialize orchestrator
    orchestrator = LL3MOrchestrator()

    # Test 1: Basic generation
    print("\n1️⃣ Testing basic asset generation...")
    try:
        result = await orchestrator.generate_asset(
            prompt="Create a simple red cube",
            asset_name="test_cube",
            export_formats=["blend", "obj"],
        )

        if result["success"]:
            print("✅ Generation successful!")
            print(f"   Asset path: {result.get('asset_path', 'N/A')}")
            print(f"   Screenshot: {result.get('screenshot_path', 'N/A')}")
            print(f"   Exported files: {len(result.get('exported_files', []))}")
            print(f"   Session ID: {result.get('session_id', 'N/A')}")
        else:
            print(f"❌ Generation failed: {result.get('error', 'Unknown error')}")

    except Exception as e:
        print(f"❌ Generation failed with exception: {e}")

    # Test 2: Asset refinement
    print("\n2️⃣ Testing asset refinement...")
    try:
        result = await orchestrator.refine_asset(
            original_prompt="Create a simple red cube",
            feedback="Make it blue and add a metallic material",
            asset_name="refined_cube",
        )

        if result["success"]:
            print("✅ Refinement successful!")
            print(f"   Asset path: {result.get('asset_path', 'N/A')}")
            print(f"   Screenshot: {result.get('screenshot_path', 'N/A')}")
            print(f"   Session ID: {result.get('session_id', 'N/A')}")
        else:
            print(f"❌ Refinement failed: {result.get('error', 'Unknown error')}")

    except Exception as e:
        print(f"❌ Refinement failed with exception: {e}")

    # Test 3: Mock imported asset editing (since we don't have a real imported asset)
    print("\n3️⃣ Testing imported asset editing (mock)...")
    try:
        # This would normally use a real imported asset path
        mock_asset_path = "/mock/path/to/imported/asset.blend"

        result = await orchestrator.edit_imported_asset(
            asset_path=mock_asset_path,
            edit_prompt="Scale it to be twice as large and add a wood texture",
            asset_name="edited_imported_asset",
            export_formats=["blend", "gltf"],
            preserve_materials=True,
        )

        if result["success"]:
            print("✅ Import editing successful!")
            print(f"   Original asset: {result.get('original_asset_path', 'N/A')}")
            print(f"   Edited asset: {result.get('asset_path', 'N/A')}")
            print(f"   Edit instruction: {result.get('edit_instruction', 'N/A')}")
            print(f"   Session ID: {result.get('session_id', 'N/A')}")
        else:
            print(f"❌ Import editing failed: {result.get('error', 'Unknown error')}")

    except Exception as e:
        print(f"❌ Import editing failed with exception: {e}")

    # Test 4: Cache statistics
    print("\n4️⃣ Testing cache statistics...")
    try:
        stats = await orchestrator.get_cache_statistics()
        print("✅ Cache statistics retrieved:")
        print(f"   Entries: {stats.get('total_entries', 'N/A')}")
        print(f"   Cache hits: {stats.get('cache_hits', 'N/A')}")
        print(f"   Cache misses: {stats.get('cache_misses', 'N/A')}")

    except Exception as e:
        print(f"❌ Cache statistics failed: {e}")

    print("\n" + "=" * 50)
    print("🏁 Asset workflow test complete!")

    # Provide setup instructions for web interface
    print("\n📝 To test the web interface:")
    print("1. Ensure your OpenAI API key is set in .env")
    print("2. Run: make web-run-openai")
    print("3. Open http://localhost:3000")
    print("4. Click on 'Assets' tab to test import functionality")
    print("5. Upload a 3D file (GLTF, OBJ, BLEND, etc.)")
    print("6. Preview the asset and try editing it with natural language")


if __name__ == "__main__":
    asyncio.run(test_asset_workflow())
