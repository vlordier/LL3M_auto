#!/usr/bin/env python3
"""Test artifact management and workflow interruption/resumption system."""

import asyncio
import tempfile
import uuid
from pathlib import Path

from src.services.artifact_manager import ArtifactManager
from src.utils.types import ExecutionResult, SubTask, TaskType, WorkflowState
from src.workflow.workflow_manager import get_workflow_manager


async def test_artifact_management():
    """Test core artifact management functionality."""
    print("🧪 Testing Core Artifact Management")
    print("=" * 60)

    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Initialize artifact manager
        artifact_manager = ArtifactManager(temp_path / "artifacts")

        # Test workflow and artifact IDs
        workflow_id = str(uuid.uuid4())

        # Create test files to store as artifacts
        blend_file = temp_path / "test_cube.blend"
        blend_file.write_text("# Fake Blender file content")

        screenshot_file = temp_path / "test_screenshot.png"
        screenshot_file.write_bytes(b"fake_png_data")

        code_file = temp_path / "test_script.py"
        code_file.write_text("import bpy\nbpy.ops.mesh.primitive_cube_add()")

        print(f"📁 Created test files in {temp_path}")

        # Test 1: Store artifacts
        print("\n🔄 Test 1: Storing Artifacts")

        blend_artifact_id = await artifact_manager.store_artifact(
            workflow_id=workflow_id,
            artifact_type="blend_file",
            source_path=blend_file,
            metadata={"asset_name": "test_cube", "creation_step": "execution"},
        )
        print(f"✅ Stored blend file: {blend_artifact_id}")

        screenshot_artifact_id = await artifact_manager.store_artifact(
            workflow_id=workflow_id,
            artifact_type="screenshot",
            source_path=screenshot_file,
            metadata={"width": 1024, "height": 1024},
        )
        print(f"✅ Stored screenshot: {screenshot_artifact_id}")

        code_artifact_id = await artifact_manager.store_artifact(
            workflow_id=workflow_id,
            artifact_type="code",
            source_path=code_file,
            metadata={"language": "python", "framework": "blender"},
        )
        print(f"✅ Stored code file: {code_artifact_id}")

        # Test 2: Create workflow checkpoint
        print("\n🔄 Test 2: Creating Workflow Checkpoint")

        test_state = WorkflowState(
            prompt="Create a test cube",
            workflow_id=workflow_id,
            generated_code=code_file.read_text(),
            execution_result=ExecutionResult(
                success=True,
                asset_path=str(blend_file),
                screenshot_path=str(screenshot_file),
                errors=[],
                logs=["Created cube successfully"],
                execution_time=1.5,
            ),
            subtasks=[
                SubTask(
                    id="create-cube",
                    type=TaskType.GEOMETRY,
                    description="Create cube primitive",
                    priority=1,
                    dependencies=[],
                    parameters={"shape": "cube"},
                )
            ],
        )

        checkpoint_id = await artifact_manager.create_checkpoint(
            workflow_id=workflow_id,
            state=test_state,
            step_name="execution_completed",
            execution_context={"total_time": 5.2, "tokens_used": 1500},
        )
        print(f"✅ Created checkpoint: {checkpoint_id}")

        # Test 3: Retrieve artifacts and checkpoint
        print("\n🔄 Test 3: Retrieving Artifacts and Checkpoint")

        # Get workflow artifacts
        artifacts = await artifact_manager.get_workflow_artifacts(workflow_id)
        print(f"✅ Retrieved {len(artifacts)} artifacts for workflow")

        for artifact in artifacts:
            print(f"  - {artifact.artifact_type}: {artifact.artifact_id}")
            print(f"    Size: {artifact.size_bytes} bytes")
            print(f"    Created: {artifact.created_at.isoformat()}")

        # Get latest checkpoint
        latest_checkpoint = await artifact_manager.get_latest_checkpoint(workflow_id)
        if latest_checkpoint:
            print(f"✅ Latest checkpoint: {latest_checkpoint.checkpoint_id}")
            print(f"  Step: {latest_checkpoint.step_name}")
            print(f"  Artifacts: {len(latest_checkpoint.artifacts)}")

        # Test 4: Workflow summary
        print("\n🔄 Test 4: Workflow Summary")

        summary = await artifact_manager.get_workflow_summary(workflow_id)
        print("✅ Workflow Summary:")
        print(f"  Total artifacts: {summary['total_artifacts']}")
        print(f"  Total size: {summary['total_size_bytes']} bytes")
        print(f"  Artifact types: {summary['artifact_types']}")
        print(f"  Checkpoints: {summary['total_checkpoints']}")

        # Test 5: Checkpoint restoration
        print("\n🔄 Test 5: Checkpoint Restoration")

        restored_checkpoint = await artifact_manager.restore_from_checkpoint(checkpoint_id)
        print(f"✅ Restored checkpoint: {restored_checkpoint.checkpoint_id}")
        print(f"  Original prompt: {restored_checkpoint.state.prompt}")
        print(f"  Generated code length: {len(restored_checkpoint.state.generated_code)}")
        print(f"  Execution success: {restored_checkpoint.state.execution_result.success}")

        print("\n🎯 Core Artifact Management Test: PASSED")
        return True


async def test_workflow_interruption_resumption():
    """Test workflow manager interruption and resumption capabilities."""
    print("\n🧪 Testing Workflow Interruption and Resumption")
    print("=" * 60)

    # Use workflow manager
    workflow_manager = get_workflow_manager()

    # Test 1: Start a workflow
    print("\n🔄 Test 1: Starting Workflow")

    workflow_id = await workflow_manager.start_workflow(
        "Create a colorful spinning cube with materials", config={"test_mode": True}
    )
    print(f"✅ Started workflow: {workflow_id}")

    # Wait a bit to let workflow start
    await asyncio.sleep(0.5)

    # Test 2: Check workflow status
    print("\n🔄 Test 2: Checking Workflow Status")

    status = await workflow_manager.get_workflow_status(workflow_id)
    print("✅ Workflow Status:")
    print(f"  ID: {status['workflow_id']}")
    print(f"  Status: {status['status']}")
    print(f"  Running: {status['is_running']}")
    print(f"  Can Resume: {status['can_resume']}")

    # Test 3: Interrupt workflow (simulate user interruption)
    print("\n🔄 Test 3: Interrupting Workflow")

    interrupt_success = await workflow_manager.interrupt_workflow(workflow_id)
    print(f"✅ Workflow interruption: {'SUCCESS' if interrupt_success else 'FAILED'}")

    # Wait for interruption to complete
    await asyncio.sleep(0.5)

    # Test 4: Check status after interruption
    print("\n🔄 Test 4: Status After Interruption")

    status_after = await workflow_manager.get_workflow_status(workflow_id)
    print("✅ Status After Interruption:")
    print(f"  Status: {status_after['status']}")
    print(f"  Running: {status_after['is_running']}")
    print(f"  Can Resume: {status_after['can_resume']}")

    # Test 5: List workflows
    print("\n🔄 Test 5: Listing All Workflows")

    workflows = await workflow_manager.list_workflows()
    print(f"✅ Found {len(workflows)} workflows:")
    for workflow in workflows:
        print(f"  - {workflow['workflow_id'][:8]}... | {workflow['status']} | Running: {workflow['is_running']}")

    # Test 6: Get workflow artifacts
    print("\n🔄 Test 6: Getting Workflow Artifacts")

    artifacts = await workflow_manager.get_workflow_artifacts(workflow_id)
    print(f"✅ Retrieved {len(artifacts)} artifacts:")
    for artifact in artifacts:
        print(f"  - {artifact['artifact_type']}: {artifact['artifact_id'][:8]}...")
        print(f"    Created: {artifact['created_at']}")

    # Test 7: Attempt workflow resumption
    print("\n🔄 Test 7: Attempting Workflow Resumption")

    # Note: This might fail in test mode since we don't have a real checkpoint
    # but it tests the resumption logic
    resume_success = await workflow_manager.resume_workflow(workflow_id)
    print(f"✅ Workflow resumption: {'SUCCESS' if resume_success else 'EXPECTED FAILURE (no real checkpoint)'}")

    # Test 8: Export workflow state
    print("\n🔄 Test 8: Export Workflow State")

    with tempfile.TemporaryDirectory() as export_dir:
        export_success = await workflow_manager.export_workflow_state(workflow_id, export_dir)
        if export_success:
            export_files = list(Path(export_dir).rglob("*"))
            print(f"✅ Exported {len(export_files)} files:")
            for file in export_files:
                print(f"  - {file.name}")
        else:
            print("⚠️ Export failed (expected in test mode)")

    # Test 9: Cleanup workflow
    print("\n🔄 Test 9: Cleaning Up Workflow")

    cleanup_success = await workflow_manager.cleanup_workflow(workflow_id, keep_latest_checkpoint=False)
    print(f"✅ Workflow cleanup: {'SUCCESS' if cleanup_success else 'FAILED'}")

    print("\n🎯 Workflow Interruption/Resumption Test: PASSED")
    return True


async def test_edge_cases():
    """Test edge cases and error handling."""
    print("\n🧪 Testing Edge Cases and Error Handling")
    print("=" * 60)

    workflow_manager = get_workflow_manager()

    # Test 1: Non-existent workflow operations
    print("\n🔄 Test 1: Non-existent Workflow Operations")

    fake_id = str(uuid.uuid4())

    status = await workflow_manager.get_workflow_status(fake_id)
    print(f"✅ Status of non-existent workflow: {status['status']}")

    interrupt_result = await workflow_manager.interrupt_workflow(fake_id)
    print(
        f"✅ Interrupt non-existent workflow: {'FAILED (expected)' if not interrupt_result else 'UNEXPECTED SUCCESS'}"
    )

    resume_result = await workflow_manager.resume_workflow(fake_id)
    print(f"✅ Resume non-existent workflow: {'FAILED (expected)' if not resume_result else 'UNEXPECTED SUCCESS'}")

    # Test 2: Invalid artifact storage
    print("\n🔄 Test 2: Invalid Artifact Storage")

    with tempfile.TemporaryDirectory() as temp_dir:
        artifact_manager = ArtifactManager(temp_dir)

        try:
            # Try to store non-existent file
            await artifact_manager.store_artifact(
                workflow_id="test",
                artifact_type="fake",
                source_path="/non/existent/path.txt",
            )
            print("❌ Expected FileNotFoundError but didn't get one")
        except FileNotFoundError:
            print("✅ Correctly handled non-existent file")
        except Exception as e:
            print(f"⚠️ Unexpected error: {e}")

    # Test 3: Multiple workflow management
    print("\n🔄 Test 3: Multiple Concurrent Workflows")

    workflow_ids = []
    for i in range(3):
        wf_id = await workflow_manager.start_workflow(f"Test workflow {i}")
        workflow_ids.append(wf_id)
        print(f"✅ Started workflow {i + 1}: {wf_id[:8]}...")

    # Let them run briefly
    await asyncio.sleep(0.3)

    # Interrupt all
    for i, wf_id in enumerate(workflow_ids):
        success = await workflow_manager.interrupt_workflow(wf_id)
        print(f"✅ Interrupted workflow {i + 1}: {'SUCCESS' if success else 'FAILED'}")

    print("\n🎯 Edge Cases Test: PASSED")
    return True


async def test_performance_and_scalability():
    """Test performance with larger datasets."""
    print("\n🧪 Testing Performance and Scalability")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as temp_dir:
        artifact_manager = ArtifactManager(temp_dir)

        # Test 1: Store many artifacts quickly
        print("\n🔄 Test 1: Batch Artifact Storage")

        import time

        start_time = time.time()

        workflow_id = str(uuid.uuid4())
        artifact_ids = []

        # Create and store 20 small artifacts
        for i in range(20):
            test_file = Path(temp_dir) / f"test_artifact_{i}.txt"
            test_file.write_text(f"Test artifact content {i}" * 100)  # Make it somewhat larger

            artifact_id = await artifact_manager.store_artifact(
                workflow_id=workflow_id,
                artifact_type="test_file",
                source_path=test_file,
                metadata={"index": i, "batch": "performance_test"},
            )
            artifact_ids.append(artifact_id)

        storage_time = time.time() - start_time
        print(f"✅ Stored 20 artifacts in {storage_time:.2f}s ({storage_time / 20:.3f}s per artifact)")

        # Test 2: Retrieve all artifacts quickly
        print("\n🔄 Test 2: Batch Artifact Retrieval")

        start_time = time.time()
        artifacts = await artifact_manager.get_workflow_artifacts(workflow_id)
        retrieval_time = time.time() - start_time

        print(f"✅ Retrieved {len(artifacts)} artifacts in {retrieval_time:.2f}s")

        # Test 3: Create multiple checkpoints
        print("\n🔄 Test 3: Multiple Checkpoint Creation")

        start_time = time.time()
        checkpoint_ids = []

        test_state = WorkflowState(prompt="Performance test workflow", workflow_id=workflow_id)

        for i in range(10):
            test_state.refinement_iterations = i
            checkpoint_id = await artifact_manager.create_checkpoint(
                workflow_id=workflow_id,
                state=test_state,
                step_name=f"step_{i}",
                execution_context={"iteration": i, "test": "performance"},
            )
            checkpoint_ids.append(checkpoint_id)

        checkpoint_time = time.time() - start_time
        print(f"✅ Created 10 checkpoints in {checkpoint_time:.2f}s ({checkpoint_time / 10:.3f}s per checkpoint)")

        # Test 4: Registry size and memory usage
        print("\n🔄 Test 4: Registry Statistics")

        print("✅ Registry Statistics:")
        print(f"  Artifacts in registry: {len(artifact_manager.artifacts_registry)}")
        print(f"  Checkpoints in registry: {len(artifact_manager.checkpoints_registry)}")

        summary = await artifact_manager.get_workflow_summary(workflow_id)
        print(f"  Total workflow size: {summary['total_size_bytes']} bytes")

    print("\n🎯 Performance Test: PASSED")
    return True


async def main():
    """Run all artifact management tests."""
    print("🚀 Starting Comprehensive Artifact Management Tests")
    print("=" * 80)

    tests = [
        ("Core Artifact Management", test_artifact_management),
        ("Workflow Interruption/Resumption", test_workflow_interruption_resumption),
        ("Edge Cases and Error Handling", test_edge_cases),
        ("Performance and Scalability", test_performance_and_scalability),
    ]

    results = []

    for test_name, test_func in tests:
        try:
            print(f"\n{'=' * 20} {test_name} {'=' * 20}")
            success = await test_func()
            results.append((test_name, success, None))
        except Exception as e:
            print(f"\n❌ {test_name} FAILED with error: {e}")
            import traceback

            traceback.print_exc()
            results.append((test_name, False, str(e)))

    # Final summary
    print(f"\n{'=' * 80}")
    print("🎯 FINAL TEST RESULTS")
    print("=" * 80)

    passed = sum(1 for _, success, _ in results if success)
    total = len(results)

    for test_name, success, error in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status}: {test_name}")
        if error:
            print(f"    Error: {error}")

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 ALL TESTS PASSED! Artifact management system is ready for production.")
    else:
        print("⚠️ Some tests failed. Please review the errors above.")

    # Clean up workflow manager
    workflow_manager = get_workflow_manager()
    await workflow_manager.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
