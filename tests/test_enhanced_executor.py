"""Tests for EnhancedBlenderExecutor."""

import ast
import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.blender.enhanced_executor import (
    EnhancedBlenderExecutor,
    BlenderProcessManager,
    PythonCodeValidator,
)


@pytest.fixture
def executor_config():
    """Enhanced executor configuration fixture."""
    return {
        "blender_path": "/usr/bin/blender",
        "headless": True,
        "timeout": 30.0,
        "output_directory": "/tmp/test_output",
    }


@pytest.fixture
def sample_blender_code():
    """Sample safe Blender Python code."""
    return """
import bpy
import bmesh

# Clear scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False, confirm=False)

# Create a cube
bpy.ops.mesh.primitive_cube_add(location=(0, 0, 0))
cube = bpy.context.active_object
cube.name = "TestCube"

# Create material
material = bpy.data.materials.new(name="TestMaterial")
material.use_nodes = True
bsdf = material.node_tree.nodes["Principled BSDF"]
bsdf.inputs['Base Color'].default_value = (1.0, 0.0, 0.0, 1.0)

# Assign material
cube.data.materials.append(material)
"""


@pytest.fixture
def unsafe_code():
    """Sample unsafe Python code."""
    return """
import os
import subprocess

# Unsafe operations
exec("print('dangerous')")
eval("1+1")
os.system("rm -rf /")
subprocess.call(["rm", "-rf", "/"])
"""


class TestPythonCodeValidator:
    """Test cases for PythonCodeValidator."""

    def test_initialization(self):
        """Test validator initialization."""
        validator = PythonCodeValidator()
        
        assert hasattr(validator, 'ALLOWED_MODULES')
        assert hasattr(validator, 'FORBIDDEN_FUNCTIONS')
        assert hasattr(validator, 'FORBIDDEN_ATTRIBUTES')
        
        assert 'bpy' in validator.ALLOWED_MODULES
        assert 'bmesh' in validator.ALLOWED_MODULES
        assert 'exec' in validator.FORBIDDEN_FUNCTIONS
        assert '__builtins__' in validator.FORBIDDEN_ATTRIBUTES

    def test_validate_safe_code(self, sample_blender_code):
        """Test validation of safe Blender code."""
        validator = PythonCodeValidator()
        
        is_valid, issues = validator.validate_code(sample_blender_code)
        
        assert is_valid is True
        assert len(issues) == 0

    def test_validate_unsafe_code(self, unsafe_code):
        """Test validation of unsafe code."""
        validator = PythonCodeValidator()
        
        is_valid, issues = validator.validate_code(unsafe_code)
        
        assert is_valid is False
        assert len(issues) > 0
        assert any("exec" in issue for issue in issues)
        assert any("eval" in issue for issue in issues)
        assert any("subprocess" in issue for issue in issues)

    def test_validate_syntax_error(self):
        """Test validation of code with syntax errors."""
        validator = PythonCodeValidator()
        
        invalid_code = "import bpy\nif True\n  print('invalid')"
        
        is_valid, issues = validator.validate_code(invalid_code)
        
        assert is_valid is False
        assert len(issues) > 0
        assert any("syntax error" in issue.lower() for issue in issues)

    def test_validate_no_blender_api(self):
        """Test validation of code without Blender API usage."""
        validator = PythonCodeValidator()
        
        no_blender_code = "print('hello world')"
        
        is_valid, issues = validator.validate_code(no_blender_code)
        
        assert is_valid is False
        assert any("does not appear to use blender api" in issue.lower() for issue in issues)

    def test_detect_blender_api_usage_bpy(self):
        """Test detection of bpy module usage."""
        validator = PythonCodeValidator()
        
        code = "import bpy\nbpy.ops.mesh.primitive_cube_add()"
        tree = ast.parse(code)
        
        has_blender = validator._has_blender_api_usage(tree)
        assert has_blender is True

    def test_detect_blender_api_usage_bmesh(self):
        """Test detection of bmesh module usage."""
        validator = PythonCodeValidator()
        
        code = "import bmesh\nbm = bmesh.new()"
        tree = ast.parse(code)
        
        has_blender = validator._has_blender_api_usage(tree)
        assert has_blender is True

    def test_analyze_forbidden_function_calls(self):
        """Test detection of forbidden function calls."""
        validator = PythonCodeValidator()
        
        code = "exec('dangerous code')"
        tree = ast.parse(code)
        
        issues = validator._analyze_ast_security(tree)
        
        assert len(issues) > 0
        assert any("exec" in issue for issue in issues)

    def test_analyze_forbidden_imports(self):
        """Test detection of forbidden imports."""
        validator = PythonCodeValidator()
        
        code = "import os\nfrom subprocess import call"
        tree = ast.parse(code)
        
        issues = validator._analyze_ast_security(tree)
        
        assert len(issues) >= 2
        assert any("os" in issue for issue in issues)
        assert any("subprocess" in issue for issue in issues)

    def test_analyze_forbidden_attributes(self):
        """Test detection of forbidden attribute access."""
        validator = PythonCodeValidator()
        
        code = "x = some_obj.__builtins__"
        tree = ast.parse(code)
        
        issues = validator._analyze_ast_security(tree)
        
        assert len(issues) > 0
        assert any("__builtins__" in issue for issue in issues)


class TestBlenderProcessManager:
    """Test cases for BlenderProcessManager."""

    def test_initialization(self):
        """Test process manager initialization."""
        manager = BlenderProcessManager()
        
        assert hasattr(manager, 'active_processes')
        assert hasattr(manager, 'process_counter')
        assert manager.process_counter == 0
        assert len(manager.active_processes) == 0

    @pytest.mark.asyncio
    @patch('asyncio.create_subprocess_exec')
    async def test_start_process_success(self, mock_create_subprocess):
        """Test successful process start."""
        # Mock process
        mock_process = Mock()
        mock_process.pid = 12345
        mock_create_subprocess.return_value = mock_process
        
        manager = BlenderProcessManager()
        
        process_id, process = await manager.start_process(
            ["blender", "--background"],
            timeout=30.0
        )
        
        assert process_id in manager.active_processes
        assert manager.active_processes[process_id] == mock_process
        assert manager.process_counter == 1

    @pytest.mark.asyncio
    @patch('asyncio.create_subprocess_exec')
    async def test_start_process_custom_id(self, mock_create_subprocess):
        """Test process start with custom ID."""
        mock_process = Mock()
        mock_process.pid = 12345
        mock_create_subprocess.return_value = mock_process
        
        manager = BlenderProcessManager()
        
        custom_id = "my_process"
        process_id, process = await manager.start_process(
            ["blender", "--background"],
            process_id=custom_id
        )
        
        assert process_id == custom_id
        assert custom_id in manager.active_processes

    @pytest.mark.asyncio
    async def test_wait_for_process_not_found(self):
        """Test waiting for non-existent process."""
        manager = BlenderProcessManager()
        
        with pytest.raises(ValueError, match="Process nonexistent not found"):
            await manager.wait_for_process("nonexistent")

    @pytest.mark.asyncio
    async def test_terminate_process_not_found(self):
        """Test terminating non-existent process."""
        manager = BlenderProcessManager()
        
        result = await manager.terminate_process("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_cleanup_all_processes_empty(self):
        """Test cleanup with no active processes."""
        manager = BlenderProcessManager()
        
        # Should not raise any exceptions
        await manager.cleanup_all_processes()


class TestEnhancedBlenderExecutor:
    """Test cases for EnhancedBlenderExecutor."""

    @patch('src.blender.enhanced_executor.settings')
    def test_initialization(self, mock_settings):
        """Test executor initialization."""
        mock_settings.blender.path = "/usr/bin/blender"
        mock_settings.blender.headless = True
        mock_settings.blender.timeout = 30
        mock_settings.app.output_directory = "/tmp/output"
        
        with patch.object(Path, 'exists', return_value=True):
            executor = EnhancedBlenderExecutor()
            
            assert executor.blender_path == "/usr/bin/blender"
            assert executor.headless is True
            assert executor.timeout == 30
            assert hasattr(executor, 'process_manager')
            assert hasattr(executor, 'code_validator')
            assert hasattr(executor, 'export_formats')

    def test_export_formats_configuration(self):
        """Test export formats configuration."""
        with patch('src.blender.enhanced_executor.settings') as mock_settings:
            mock_settings.blender.path = "/usr/bin/blender"
            mock_settings.app.output_directory = "/tmp"
            
            with patch.object(Path, 'exists', return_value=True):
                executor = EnhancedBlenderExecutor()
                
                expected_formats = ['blend', 'obj', 'fbx', 'gltf', 'collada', 'stl', 'ply']
                
                for fmt in expected_formats:
                    assert fmt in executor.export_formats
                    assert 'extension' in executor.export_formats[fmt]
                    assert 'requires_addon' in executor.export_formats[fmt]

    @patch('src.blender.enhanced_executor.settings')
    def test_validate_blender_installation_missing(self, mock_settings):
        """Test validation with missing Blender installation."""
        mock_settings.blender.path = "/nonexistent/blender"
        
        with patch.object(Path, 'exists', return_value=False):
            with pytest.raises(RuntimeError, match="Blender not found"):
                EnhancedBlenderExecutor()

    def test_wrap_code_for_execution_single_format(self):
        """Test code wrapping for single export format."""
        with patch('src.blender.enhanced_executor.settings') as mock_settings:
            mock_settings.blender.path = "/usr/bin/blender"
            mock_settings.app.output_directory = "/tmp"
            mock_settings.blender.screenshot_resolution = [1024, 768]
            
            with patch.object(Path, 'exists', return_value=True):
                executor = EnhancedBlenderExecutor()
                
                wrapped = executor._wrap_code_for_execution(
                    "bpy.ops.mesh.primitive_cube_add()",
                    "test_asset",
                    ["blend"]
                )
                
                assert "import bpy" in wrapped
                assert "import bmesh" in wrapped
                assert "test_asset" in wrapped
                assert "save_as_mainfile" in wrapped
                assert "exported_files" in wrapped

    def test_wrap_code_for_execution_multiple_formats(self):
        """Test code wrapping for multiple export formats."""
        with patch('src.blender.enhanced_executor.settings') as mock_settings:
            mock_settings.blender.path = "/usr/bin/blender"
            mock_settings.app.output_directory = "/tmp"
            mock_settings.blender.screenshot_resolution = [1024, 768]
            
            with patch.object(Path, 'exists', return_value=True):
                executor = EnhancedBlenderExecutor()
                
                wrapped = executor._wrap_code_for_execution(
                    "bpy.ops.mesh.primitive_cube_add()",
                    "test_asset",
                    ["blend", "obj", "fbx"]
                )
                
                assert "save_as_mainfile" in wrapped
                assert "export_scene.obj" in wrapped
                assert "export_scene.fbx" in wrapped
                assert wrapped.count("exported_files.append") >= 3

    def test_wrap_code_quality_settings(self):
        """Test code wrapping with quality settings."""
        with patch('src.blender.enhanced_executor.settings') as mock_settings:
            mock_settings.blender.path = "/usr/bin/blender"
            mock_settings.app.output_directory = "/tmp"
            mock_settings.blender.screenshot_resolution = [512, 512]
            
            with patch.object(Path, 'exists', return_value=True):
                executor = EnhancedBlenderExecutor()
                
                quality_settings = {
                    'render_engine': 'CYCLES',
                    'render_samples': 128,
                    'screenshot_resolution': [2048, 2048]
                }
                
                wrapped = executor._wrap_code_for_execution(
                    "bpy.ops.mesh.primitive_cube_add()",
                    "test_asset",
                    ["blend"],
                    quality_settings
                )
                
                assert "CYCLES" in wrapped
                assert "128" in wrapped
                assert "2048" in wrapped

    def test_indent_code(self):
        """Test code indentation utility."""
        with patch('src.blender.enhanced_executor.settings') as mock_settings:
            mock_settings.blender.path = "/usr/bin/blender"
            mock_settings.app.output_directory = "/tmp"
            
            with patch.object(Path, 'exists', return_value=True):
                executor = EnhancedBlenderExecutor()
                
                code = "line1\nline2\n\nline4"
                indented = executor._indent_code(code, "    ")
                
                expected = "    line1\n    line2\n\n    line4"
                assert indented == expected

    def test_parse_enhanced_execution_result_success(self):
        """Test parsing successful execution result."""
        with patch('src.blender.enhanced_executor.settings') as mock_settings:
            mock_settings.blender.path = "/usr/bin/blender"
            mock_settings.app.output_directory = "/tmp"
            
            with patch.object(Path, 'exists', return_value=True):
                executor = EnhancedBlenderExecutor()
                
                mock_result = {
                    "success": True,
                    "asset_path": "/tmp/asset.blend",
                    "screenshot_path": "/tmp/screenshot.png",
                    "logs": ["Test log"],
                    "errors": [],
                    "execution_time": 2.5,
                    "exported_files": ["/tmp/asset.blend", "/tmp/asset.obj"],
                    "export_formats": ["blend", "obj"],
                    "quality_settings": {"render_engine": "EEVEE"}
                }
                
                stdout = f"ENHANCED_EXECUTION_RESULT_JSON: {json.dumps(mock_result)}"
                stderr = ""
                
                result = executor._parse_enhanced_execution_result(stdout, stderr)
                
                assert result.success is True
                assert result.asset_path == "/tmp/asset.blend"
                assert result.screenshot_path == "/tmp/screenshot.png"
                assert result.logs == ["Test log"]
                assert result.errors == []
                assert result.execution_time == 2.5
                assert "exported_files" in result.metadata
                assert "export_formats" in result.metadata

    def test_parse_enhanced_execution_result_failure(self):
        """Test parsing failed execution result."""
        with patch('src.blender.enhanced_executor.settings') as mock_settings:
            mock_settings.blender.path = "/usr/bin/blender"  
            mock_settings.app.output_directory = "/tmp"
            
            with patch.object(Path, 'exists', return_value=True):
                executor = EnhancedBlenderExecutor()
                
                stdout = "No JSON result found"
                stderr = "Error occurred during execution"
                
                result = executor._parse_enhanced_execution_result(stdout, stderr)
                
                assert result.success is False
                assert result.asset_path is None
                assert result.screenshot_path is None

    def test_update_execution_history(self):
        """Test execution history tracking."""
        with patch('src.blender.enhanced_executor.settings') as mock_settings:
            mock_settings.blender.path = "/usr/bin/blender"
            mock_settings.app.output_directory = "/tmp"
            
            with patch.object(Path, 'exists', return_value=True):
                executor = EnhancedBlenderExecutor()
                
                from src.utils.types import ExecutionResult
                
                result = ExecutionResult(
                    success=True,
                    asset_path="/tmp/test.blend",
                    screenshot_path="/tmp/test.png",
                    logs=["log"],
                    errors=[],
                    execution_time=2.0
                )
                
                executor._update_execution_history(
                    "test_asset", result, 2.0, ["blend", "obj"]
                )
                
                assert len(executor.execution_history) == 1
                history_entry = executor.execution_history[0]
                
                assert history_entry["asset_name"] == "test_asset"
                assert history_entry["success"] is True
                assert history_entry["execution_time"] == 2.0
                assert history_entry["export_formats"] == ["blend", "obj"]

    def test_execution_history_limit(self):
        """Test execution history size limit."""
        with patch('src.blender.enhanced_executor.settings') as mock_settings:
            mock_settings.blender.path = "/usr/bin/blender"
            mock_settings.app.output_directory = "/tmp"
            
            with patch.object(Path, 'exists', return_value=True):
                executor = EnhancedBlenderExecutor()
                
                from src.utils.types import ExecutionResult
                
                # Add 150 entries (more than limit of 100)
                for i in range(150):
                    result = ExecutionResult(
                        success=True,
                        asset_path=f"/tmp/test{i}.blend",
                        screenshot_path=f"/tmp/test{i}.png",
                        logs=[],
                        errors=[],
                        execution_time=1.0
                    )
                    
                    executor._update_execution_history(
                        f"test_asset_{i}", result, 1.0, ["blend"]
                    )
                
                # Should keep only last 100
                assert len(executor.execution_history) == 100
                # Should have entries 50-149
                assert executor.execution_history[0]["asset_name"] == "test_asset_50"
                assert executor.execution_history[-1]["asset_name"] == "test_asset_149"

    @pytest.mark.asyncio
    async def test_get_execution_statistics_empty(self):
        """Test execution statistics with no history."""
        with patch('src.blender.enhanced_executor.settings') as mock_settings:
            mock_settings.blender.path = "/usr/bin/blender"
            mock_settings.app.output_directory = "/tmp"
            
            with patch.object(Path, 'exists', return_value=True):
                executor = EnhancedBlenderExecutor()
                
                stats = await executor.get_execution_statistics()
                
                assert stats["total_executions"] == 0

    @pytest.mark.asyncio
    async def test_get_execution_statistics_with_data(self):
        """Test execution statistics with history data."""
        with patch('src.blender.enhanced_executor.settings') as mock_settings:
            mock_settings.blender.path = "/usr/bin/blender"
            mock_settings.app.output_directory = "/tmp"
            
            with patch.object(Path, 'exists', return_value=True):
                executor = EnhancedBlenderExecutor()
                
                # Add some mock history
                executor.execution_history = [
                    {"success": True, "execution_time": 2.0},
                    {"success": False, "execution_time": 1.5},
                    {"success": True, "execution_time": 3.0},
                    {"success": True, "execution_time": 2.5},
                ]
                
                stats = await executor.get_execution_statistics()
                
                assert stats["total_executions"] == 4
                assert stats["success_rate"] == 0.75  # 3 out of 4
                assert stats["avg_execution_time"] == 2.25  # (2.0+1.5+3.0+2.5)/4