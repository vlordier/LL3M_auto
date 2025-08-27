"""Test CLI functionality."""

from unittest.mock import MagicMock, patch

import click
from click.testing import CliRunner

from src.cli import cli, main


class TestCLI:
    """Test CLI commands and functionality."""

    def test_main_function_exists(self) -> None:
        """Test that main function exists and is callable."""
        assert callable(main)

    def test_cli_group_exists(self) -> None:
        """Test that CLI group exists."""
        assert isinstance(cli, click.Group)

    def test_cli_help(self) -> None:
        """Test CLI help command."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])

        assert result.exit_code == 0
        assert "Large Language 3D Modelers" in result.output
        assert "generate" in result.output
        assert "refine" in result.output
        assert "status" in result.output

    def test_cli_version(self) -> None:
        """Test CLI version option."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])

        assert result.exit_code == 0
        assert "0.1.0" in result.output

    def test_generate_command_help(self) -> None:
        """Test generate command help."""
        runner = CliRunner()
        result = runner.invoke(cli, ["generate", "--help"])

        assert result.exit_code == 0
        assert "Generate a 3D asset from a text prompt" in result.output
        assert "--output" in result.output
        assert "--format" in result.output
        assert "--no-refine" in result.output

    def test_generate_command_not_implemented(self) -> None:
        """Test generate command shows not implemented error."""
        runner = CliRunner()
        result = runner.invoke(cli, ["generate", "Create a red cube"])

        assert result.exit_code == 1
        assert "not yet implemented" in result.output

    def test_refine_command_help(self) -> None:
        """Test refine command help."""
        runner = CliRunner()
        result = runner.invoke(cli, ["refine", "--help"])

        assert result.exit_code == 0
        assert "Refine an existing asset with user feedback" in result.output

    def test_refine_command_not_implemented(self) -> None:
        """Test refine command shows not implemented error."""
        runner = CliRunner()
        result = runner.invoke(cli, ["refine", "asset123", "Make it bigger"])

        assert result.exit_code == 1
        assert "not yet implemented" in result.output

    def test_list_assets_command_help(self) -> None:
        """Test list-assets command help."""
        runner = CliRunner()
        result = runner.invoke(cli, ["list-assets", "--help"])

        assert result.exit_code == 0
        assert "List all generated assets" in result.output

    def test_status_command_basic(self) -> None:
        """Test status command basic functionality."""
        runner = CliRunner()
        result = runner.invoke(cli, ["status"])

        assert result.exit_code == 0
        assert "LL3M System Status" in result.output
        assert "Output directory:" in result.output
        assert "Log level:" in result.output

    @patch("pathlib.Path.exists")
    def test_status_command_blender_check_found(self, mock_exists: MagicMock) -> None:
        """Test status command with Blender check - found."""
        mock_exists.return_value = True

        runner = CliRunner()
        result = runner.invoke(cli, ["status", "--check-blender"])

        assert result.exit_code == 0
        assert "Checking Blender installation" in result.output
        assert "Blender found at:" in result.output

    @patch("pathlib.Path.exists")
    def test_status_command_blender_check_not_found(
        self, mock_exists: MagicMock
    ) -> None:
        """Test status command with Blender check - not found."""
        mock_exists.return_value = False

        runner = CliRunner()
        result = runner.invoke(cli, ["status", "--check-blender"])

        assert result.exit_code == 0
        assert "Checking Blender installation" in result.output
        assert "Blender not found at:" in result.output

    @patch("src.cli.settings")
    def test_status_command_openai_check_configured(
        self, mock_settings: MagicMock
    ) -> None:
        """Test status command with OpenAI check - configured."""
        mock_settings.openai.api_key = "sk-test123"

        runner = CliRunner()
        result = runner.invoke(cli, ["status", "--check-openai"])

        assert result.exit_code == 0
        assert "Checking OpenAI API connection" in result.output
        assert "OpenAI API key configured" in result.output

    @patch("src.cli.settings")
    def test_status_command_openai_check_not_configured(
        self, mock_settings: MagicMock
    ) -> None:
        """Test status command with OpenAI check - not configured."""
        mock_settings.openai.api_key = ""

        runner = CliRunner()
        result = runner.invoke(cli, ["status", "--check-openai"])

        assert result.exit_code == 0
        assert "Checking OpenAI API connection" in result.output
        assert "OpenAI API key not configured" in result.output

    def test_main_keyboard_interrupt(self) -> None:
        """Test main function handles keyboard interrupt."""
        with patch("src.cli.cli", side_effect=KeyboardInterrupt()):
            with patch("src.cli.console.print") as mock_print:
                main()
                mock_print.assert_called_with(
                    "\n[yellow]Operation cancelled by user[/yellow]"
                )

    def test_main_unexpected_exception(self) -> None:
        """Test main function handles unexpected exceptions."""
        with patch("src.cli.cli", side_effect=RuntimeError("Test error")):
            with (
                patch("src.cli.console.print") as mock_print,
                patch("src.cli.logger.error") as mock_logger,
            ):
                main()
                mock_print.assert_called_with(
                    "\n[red]Unexpected error: Test error[/red]"
                )
                mock_logger.assert_called_with("CLI error", error="Test error")

    @patch("src.cli.settings")
    def test_generate_command_output_directory_override(
        self,
        mock_settings: MagicMock,  # noqa: ARG002
    ) -> None:
        """Test generate command with custom output directory."""
        runner = CliRunner()
        with runner.isolated_filesystem():
            result = runner.invoke(
                cli, ["generate", "test prompt", "--output", "./custom_output"]
            )

            # Should fail with NotImplementedError but settings should be updated
            assert result.exit_code == 1
            # The settings.app.output_directory would be updated if we could check it

    def test_generate_command_format_validation(self) -> None:
        """Test generate command format validation."""
        runner = CliRunner()
        result = runner.invoke(cli, ["generate", "test prompt", "--format", "invalid"])

        assert result.exit_code == 2  # Click validation error
        assert "invalid" in result.output

    def test_generate_command_valid_formats(self) -> None:
        """Test generate command accepts valid formats."""
        runner = CliRunner()
        valid_formats = ["blend", "gltf", "obj", "fbx"]

        for fmt in valid_formats:
            result = runner.invoke(cli, ["generate", "test prompt", "--format", fmt])
            # Should fail with NotImplementedError but format should be accepted
            assert result.exit_code == 1
            assert "not yet implemented" in result.output
