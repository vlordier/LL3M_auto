"""Test Blender executor."""

from unittest.mock import patch

import pytest

from src.blender.executor import BlenderExecutor


class TestBlenderExecutor:
    """Test Blender execution."""

    def test_init_with_valid_blender(self) -> None:
        """Test executor initialization with valid Blender path."""
        with patch("pathlib.Path.exists", return_value=True):
            executor = BlenderExecutor()
            assert executor.blender_path is not None

    def test_init_with_invalid_blender(self) -> None:
        """Test executor initialization with invalid Blender path."""
        with patch("pathlib.Path.exists", return_value=False):
            with pytest.raises(RuntimeError, match="Blender not found"):
                BlenderExecutor()

    def test_wrap_code_for_execution(self) -> None:
        """Test code wrapping for execution."""
        with patch("pathlib.Path.exists", return_value=True):
            executor = BlenderExecutor()

            user_code = "bpy.ops.mesh.primitive_cube_add()"
            wrapped = executor._wrap_code_for_execution(
                user_code, "test_asset", "blend"
            )

            assert "import bpy" in wrapped
            assert user_code in wrapped
            assert "test_asset.blend" in wrapped
            assert "EXECUTION_RESULT_JSON" in wrapped

    def test_indent_code(self) -> None:
        """Test code indentation."""
        with patch("pathlib.Path.exists", return_value=True):
            executor = BlenderExecutor()

            code = "line1\nline2\n  indented"
            indented = executor._indent_code(code, "    ")

            expected = "    line1\n    line2\n      indented"
            assert indented == expected

    def test_parse_execution_result_success(self) -> None:
        """Test parsing successful execution result."""
        with patch("pathlib.Path.exists", return_value=True):
            executor = BlenderExecutor()

            json_result = (
                '{"success": true, "asset_path": "/path/asset.blend", '
                '"logs": ["test"], "errors": []}'
            )
            stdout = f"""
Some Blender output
EXECUTION_RESULT_JSON: {json_result}
More output
"""
            stderr = ""

            result = executor._parse_execution_result(stdout, stderr)

            assert result.success is True
            assert result.asset_path == "/path/asset.blend"
            assert result.logs == ["test"]
            assert result.errors == []

    def test_parse_execution_result_failure(self) -> None:
        """Test parsing failed execution result."""
        with patch("pathlib.Path.exists", return_value=True):
            executor = BlenderExecutor()

            json_result = (
                '{"success": false, "asset_path": null, "logs": [], '
                '"errors": ["Error occurred"]}'
            )
            stdout = f"""
EXECUTION_RESULT_JSON: {json_result}
"""
            stderr = ""

            result = executor._parse_execution_result(stdout, stderr)

            assert result.success is False
            assert result.asset_path is None
            assert result.errors == ["Error occurred"]
