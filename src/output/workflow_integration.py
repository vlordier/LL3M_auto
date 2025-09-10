"""Integration between the enhanced output system and existing LL3M workflow."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog

from ..workflow.orchestrator import LL3MOrchestrator
from .artifact_manager import ArtifactManager
from .directory_manager import OutputDirectoryManager
from .interactive_refiner import InteractiveRefiner

logger = structlog.get_logger(__name__)


class WorkflowOutputIntegration:
    """Integrates the enhanced output system with LL3M workflows."""

    def __init__(self, base_output_dir: Path | None = None):
        """Initialize the integration.

        Args:
            base_output_dir: Base directory for outputs
        """
        self.directory_manager = OutputDirectoryManager(base_output_dir or Path("outputs"))
        self.artifact_manager = ArtifactManager(base_output_dir or Path("outputs"))
        self.refiner = InteractiveRefiner(self.directory_manager)
        self.orchestrator = LL3MOrchestrator()

    async def generate_with_enhanced_output(
        self,
        prompt: str,
        project_name: str = None,
        complexity_level: int = None,
        enable_interactive: bool = True,
        export_formats: list = None,
    ) -> dict[str, Any]:
        """Generate an asset with enhanced output organization.

        Args:
            prompt: Generation prompt
            project_name: Project name (auto-generated if None)
            complexity_level: Complexity level 1-5 (auto-detected if None)
            enable_interactive: Enable interactive refinement
            export_formats: List of export formats

        Returns:
            Generation result with enhanced metadata
        """
        # Auto-generate project name if not provided
        if not project_name:
            project_name = self._generate_project_name(prompt)

        # Auto-detect complexity level if not provided
        if not complexity_level:
            complexity_level = self._detect_complexity_level(prompt)

        # Create or get existing project
        project_dir = self._get_or_create_project(project_name, prompt)

        # Create session
        session_dir = self.directory_manager.create_session(project_dir)

        # Create asset directory
        asset_name = self._generate_asset_name(prompt)
        asset_dir = self.directory_manager.create_asset_directory(
            session_dir, asset_name, complexity_level, stage="raw"
        )

        logger.info(
            f"Starting enhanced generation: {asset_name}",
            project=project_name,
            asset_dir=str(asset_dir),
        )

        try:
            # Generate the asset using the orchestrator
            result = await self.orchestrator.generate_asset(
                prompt=prompt,
                export_formats=export_formats or ["gltf", "obj"],  # Web-compatible defaults
                asset_name=asset_name,
            )

            # Organize generated files into the enhanced directory structure
            organized_result = await self._organize_generation_result(result, asset_dir, prompt, complexity_level)

            # Update asset metadata with results
            await self._update_asset_metadata(asset_dir, organized_result)

            # Optionally promote high-quality assets to gallery
            if organized_result.get("quality_score", 0) >= 8.0:
                self.directory_manager.promote_to_gallery(asset_dir, "featured")

            # Setup interactive refinement if enabled
            if enable_interactive:
                refinement_info = await self._setup_interactive_refinement(asset_dir, prompt)
                organized_result["refinement_info"] = refinement_info

            logger.info(f"Enhanced generation completed: {asset_name}")
            return organized_result

        except Exception as e:
            logger.exception("Enhanced generation failed")

            # Update asset metadata with error
            await self._update_asset_metadata(asset_dir, {"success": False, "error": str(e), "status": "failed"})

            raise

    async def refine_existing_asset(
        self, asset_path: Path, instructions: list[str], create_variants: bool = True
    ) -> dict[str, Any]:
        """Refine an existing asset with multiple instructions.

        Args:
            asset_path: Path to existing asset
            instructions: List of refinement instructions
            create_variants: Create variants instead of overwriting

        Returns:
            Refinement results
        """
        logger.info(f"Refining asset: {asset_path}", instructions=instructions)

        # Load asset metadata
        metadata_file = asset_path / "asset_metadata.json"
        if not metadata_file.exists():
            raise ValueError(f"Asset metadata not found: {metadata_file}")

        with open(metadata_file) as f:
            asset_meta = json.load(f)

        original_prompt = asset_meta.get("original_prompt", "3D asset")

        # Start refinement session
        session = await self.refiner.start_refinement_session(asset_path, original_prompt)

        refinement_results = []

        for i, instruction_text in enumerate(instructions):
            logger.info(f"Applying instruction {i + 1}/{len(instructions)}: {instruction_text}")

            # Parse and apply instruction
            instruction = self.refiner.parse_instruction(instruction_text)
            result = await self.refiner.apply_instruction(session, instruction)

            refinement_results.append(
                {
                    "instruction": instruction_text,
                    "result": result,
                    "iteration": session.current_iteration,
                }
            )

            # If creating variants, copy successful results to variants directory
            if create_variants and result.get("success"):
                await self._create_variant(asset_path, session.current_iteration, result)

        # Update session with final satisfaction score
        if all(r["result"].get("success") for r in refinement_results):
            session.user_satisfaction = 8  # Auto-rate successful batch refinements

        return {
            "session_id": session.session_id,
            "total_instructions": len(instructions),
            "successful_refinements": sum(1 for r in refinement_results if r["result"].get("success")),
            "refinement_results": refinement_results,
            "final_satisfaction": session.user_satisfaction,
        }

    def get_interactive_suggestions(self, asset_path: Path, context: str | None = None) -> dict[str, Any]:
        """Get AI-generated suggestions for interactive refinement.

        Args:
            asset_path: Path to asset
            context: Additional context for suggestions

        Returns:
            Suggestions and metadata
        """
        # Load asset metadata and analysis
        metadata_file = asset_path / "asset_metadata.json"
        quality_analysis = None

        if metadata_file.exists():
            with open(metadata_file) as f:
                asset_meta = json.load(f)
                quality_analysis = asset_meta.get("quality_analysis")

        # Generate suggestions
        suggestions = self.refiner.get_refinement_suggestions(asset_path, quality_analysis)

        # Add context-specific suggestions
        if context:
            context_suggestions = self._generate_context_suggestions(context, asset_path)
            suggestions.extend(context_suggestions)

        return {
            "suggestions": suggestions[:10],  # Top 10 suggestions
            "asset_info": {
                "name": (asset_meta.get("asset_name") if "asset_meta" in locals() else "Unknown"),
                "complexity": (asset_meta.get("complexity_level") if "asset_meta" in locals() else 1),
                "quality_score": (asset_meta.get("quality_score") if "asset_meta" in locals() else None),
            },
            "refinement_history": self._get_refinement_history(asset_path),
        }

    def _generate_project_name(self, prompt: str) -> str:
        """Generate a project name from the prompt."""
        # Extract key words and create meaningful name
        words = prompt.lower().split()
        key_words = []

        for word in words:
            if len(word) > 3 and word not in ["create", "make", "generate", "build"]:
                key_words.append(word.capitalize())
                if len(key_words) >= 3:
                    break

        if key_words:
            return " ".join(key_words) + " Project"
        else:
            return "3D Generation Project"

    def _detect_complexity_level(self, prompt: str) -> int:
        """Detect complexity level from prompt."""
        prompt_lower = prompt.lower()

        # Keywords for different complexity levels
        complexity_indicators = {
            5: ["animation", "character", "dance", "movement", "rigged", "animated"],
            4: [
                "lighting",
                "scene",
                "environment",
                "multiple",
                "room",
                "interior",
                "exterior",
            ],
            3: [
                "texture",
                "material",
                "wood",
                "metal",
                "fabric",
                "realistic",
                "detailed",
            ],
            2: ["furniture", "object", "table", "chair", "simple", "basic"],
            1: ["cube", "sphere", "primitive", "shape", "geometry", "ovoid"],
        }

        for level in range(5, 0, -1):
            if any(keyword in prompt_lower for keyword in complexity_indicators[level]):
                return level

        return 2  # Default complexity

    def _get_or_create_project(self, project_name: str, description: str) -> Path:
        """Get existing project or create new one."""
        # Look for existing project with same name
        recent_projects = self.directory_manager.get_recent_projects()
        for project in recent_projects:
            if project.get("name") == project_name:
                return Path(project["path"])

        # Create new project
        return self.directory_manager.create_project(project_name, description)

    def _generate_asset_name(self, prompt: str) -> str:
        """Generate asset name from prompt."""
        # Extract main subject
        words = prompt.split()

        # Look for nouns (simple heuristic)
        asset_words = []
        for word in words:
            if len(word) > 2 and word.lower() not in [
                "a",
                "an",
                "the",
                "create",
                "make",
                "generate",
            ]:
                asset_words.append(word.capitalize())
                if len(asset_words) >= 2:
                    break

        if asset_words:
            return "_".join(asset_words)
        else:
            return "3D_Asset"

    async def _organize_generation_result(
        self,
        result: dict[str, Any],
        asset_dir: Path,
        prompt: str,
        complexity_level: int,
    ) -> dict[str, Any]:
        """Organize generation results into enhanced directory structure."""
        organized = {
            "success": result.get("success", False),
            "asset_dir": str(asset_dir),
            "original_prompt": prompt,
            "complexity_level": complexity_level,
            "organized_files": {},
        }

        # Organize files by type
        if result.get("success"):
            # Move generated files to appropriate directories
            source_files = result.get("code_files", [])
            for source_file in source_files:
                if isinstance(source_file, str | Path):
                    source_path = Path(source_file)
                    if source_path.exists():
                        dest_path = asset_dir / "source" / source_path.name
                        source_path.rename(dest_path)
                        organized["organized_files"].setdefault("source", []).append(str(dest_path))

            # Move output files
            output_files = result.get("output_files", [])
            for output_file in output_files:
                if isinstance(output_file, str | Path):
                    output_path = Path(output_file)
                    if output_path.exists():
                        if output_path.suffix.lower() in [
                            ".png",
                            ".jpg",
                            ".jpeg",
                        ] or output_path.suffix.lower() in [".blend"]:
                            dest_path = asset_dir / "outputs" / output_path.name
                        else:
                            dest_path = asset_dir / "exports" / output_path.name

                        output_path.rename(dest_path)
                        organized["organized_files"].setdefault("outputs", []).append(str(dest_path))

        return organized

    async def _update_asset_metadata(self, asset_dir: Path, result_data: dict[str, Any]):
        """Update asset metadata with generation results."""
        metadata_file = asset_dir / "asset_metadata.json"

        if metadata_file.exists():
            with open(metadata_file) as f:
                metadata = json.load(f)
        else:
            metadata = {}

        # Update with result data
        metadata.update(
            {
                "generation_result": result_data,
                "last_updated": datetime.now().isoformat(),
                "status": "completed" if result_data.get("success") else "failed",
            }
        )

        # Add quality analysis if available
        if "quality_analysis" in result_data:
            metadata["quality_score"] = result_data["quality_analysis"].get("overall_score")
            metadata["quality_analysis"] = result_data["quality_analysis"]

        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

    async def _setup_interactive_refinement(self, asset_dir: Path, original_prompt: str) -> dict[str, Any]:
        """Setup interactive refinement for the asset."""
        # Pre-create refinement session
        session = await self.refiner.start_refinement_session(asset_dir, original_prompt)

        return {
            "session_id": session.session_id,
            "refinement_suggestions": self.refiner.get_refinement_suggestions(asset_dir),
            "instructions": [
                "Try changing colors or materials",
                "Adjust the size or proportions",
                "Add more detail or complexity",
                "Modify the lighting",
                "Change the composition or angle",
            ],
        }

    async def _create_variant(self, asset_path: Path, iteration: int, result: dict[str, Any]):
        """Create a variant copy of a successful refinement."""
        variants_dir = asset_path / "variants"
        variants_dir.mkdir(exist_ok=True)

        variant_dir = variants_dir / f"variant_{iteration:02d}"
        variant_dir.mkdir(exist_ok=True)

        # Copy result files to variant directory
        # Implementation depends on result structure

        # Create variant metadata
        variant_metadata = {
            "variant_id": f"variant_{iteration:02d}",
            "source_iteration": iteration,
            "created_at": datetime.now().isoformat(),
            "result": result,
        }

        with open(variant_dir / "variant_metadata.json", "w") as f:
            json.dump(variant_metadata, f, indent=2, default=str)

    def _generate_context_suggestions(
        self,
        context: str,
        asset_path: Path,  # noqa: ARG002
    ) -> list[str]:
        """Generate context-specific suggestions."""
        context_lower = context.lower()
        suggestions = []

        if "game" in context_lower:
            suggestions.extend(
                [
                    "Optimize for game engines",
                    "Reduce polygon count",
                    "Add LOD variants",
                ]
            )
        elif "render" in context_lower or "visualization" in context_lower:
            suggestions.extend(
                [
                    "Increase detail and quality",
                    "Add realistic materials",
                    "Improve lighting setup",
                ]
            )
        elif "3d print" in context_lower:
            suggestions.extend(
                [
                    "Check for printability",
                    "Add support structures",
                    "Ensure manifold geometry",
                ]
            )

        return suggestions

    async def resume_session(self, session_id: str, additional_feedback: str = None) -> dict[str, Any]:
        """Resume a previous generation session.

        Args:
            session_id: Session ID to resume
            additional_feedback: Optional additional user feedback

        Returns:
            Resumption result with updated artifacts
        """
        try:
            # Load session for resumption
            workflow_state, session_dir = self.artifact_manager.load_session_for_resumption(session_id)

            logger.info(
                f"Resuming session: {session_id}",
                original_prompt=workflow_state.original_prompt,
                refinement_iterations=workflow_state.refinement_iterations,
            )

            # Add additional feedback if provided
            if additional_feedback:
                workflow_state.user_feedback = additional_feedback
                workflow_state.refinement_iterations += 1

            # Continue generation with resumed state (placeholder - orchestrator needs this method)
            # result = await self.orchestrator.continue_generation(workflow_state=workflow_state)

            # For now, return resumption info
            return {
                "success": True,
                "resumed_session_id": session_id,
                "session_directory": str(session_dir),
                "artifacts": workflow_state.artifacts.model_dump(),
                "artifact_summary": workflow_state.artifacts.get_complete_artifact_summary(),
                "resume_prompt": workflow_state.generate_resume_prompt(),
                "note": "Session loaded successfully - orchestrator integration needed for continuation",
            }

        except Exception as e:
            logger.exception("Failed to resume session")
            return {
                "success": False,
                "error": str(e),
                "session_id": session_id,
            }

    def get_resumable_sessions(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get list of sessions that can be resumed.

        Args:
            limit: Maximum number of sessions to return

        Returns:
            List of resumable session information
        """
        return self.artifact_manager.list_resumable_sessions(limit)

    def create_session_resume_prompt(self, session_id: str) -> str:
        """Create a human-readable resume prompt for a session.

        Args:
            session_id: Session ID to create prompt for

        Returns:
            Formatted resume prompt
        """
        return self.artifact_manager.create_resumption_prompt(session_id)
