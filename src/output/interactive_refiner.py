"""Interactive refinement system for user-guided 3D asset improvement."""

import asyncio
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import structlog
from pydantic import BaseModel, Field

from ..utils.types import WorkflowState, SubTask, TaskType
from ..workflow.orchestrator import LL3MOrchestrator
from .directory_manager import OutputDirectoryManager

logger = structlog.get_logger(__name__)


class UserInstruction(BaseModel):
    """Represents a user instruction for asset refinement."""
    
    instruction_id: str = Field(default_factory=lambda: f"inst_{datetime.now().strftime('%H%M%S')}")
    timestamp: datetime = Field(default_factory=datetime.now)
    instruction_text: str = Field(..., description="Raw user instruction")
    intent: str = Field(..., description="Parsed intent category")
    target_aspects: List[str] = Field(default_factory=list, description="Which aspects to modify")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Parsed parameters")
    priority: int = Field(default=1, description="Priority level 1-5")
    confidence: float = Field(default=0.8, description="Parsing confidence 0-1")


class RefinementSession(BaseModel):
    """Tracks a refinement session with multiple instructions."""
    
    session_id: str = Field(default_factory=lambda: f"refine_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    asset_path: Path
    original_prompt: str
    instructions: List[UserInstruction] = Field(default_factory=list)
    iterations: List[Dict[str, Any]] = Field(default_factory=list)
    current_iteration: int = Field(default=0)
    status: str = Field(default="active")
    user_satisfaction: Optional[int] = Field(default=None, description="Rating 1-10")


class InteractiveRefiner:
    """Handles interactive refinement of 3D assets based on user instructions."""
    
    def __init__(self, directory_manager: OutputDirectoryManager):
        """Initialize the interactive refiner.
        
        Args:
            directory_manager: Directory manager for organizing outputs
        """
        self.directory_manager = directory_manager
        self.orchestrator = LL3MOrchestrator()
        
        # Instruction parsing patterns
        self.intent_patterns = {
            "color_change": [
                r"make.*(?:more|less)?.*(?:red|blue|green|yellow|orange|purple|pink|brown|black|white|gray)",
                r"change.*color.*to",
                r"(?:darker|lighter|brighter|duller)",
            ],
            "size_change": [
                r"make.*(?:bigger|larger|smaller|tiny|huge|massive)",
                r"(?:scale|resize).*(?:up|down|to)",
                r"(?:increase|decrease).*size",
            ],
            "shape_change": [
                r"make.*(?:rounder|sharper|smoother|rougher)",
                r"add.*(?:curves|edges|details)",
                r"change.*shape",
                r"more.*(?:angular|organic|geometric)",
            ],
            "material_change": [
                r"make.*(?:shiny|matte|glossy|metallic|wooden|plastic|glass)",
                r"change.*(?:material|texture|surface)",
                r"add.*(?:reflection|transparency|roughness)",
            ],
            "lighting_change": [
                r"(?:brighter|darker|more|less).*(?:light|lighting|illumination)",
                r"add.*(?:shadows|highlights)",
                r"change.*(?:lighting|illumination)",
            ],
            "detail_add": [
                r"add.*(?:detail|decoration|pattern|texture)",
                r"more.*(?:complex|detailed|intricate)",
                r"include.*(?:features|elements)",
            ],
            "detail_remove": [
                r"(?:remove|delete|take away)",
                r"(?:simpler|cleaner|minimalist)",
                r"less.*(?:detail|complex|cluttered)",
            ],
            "position_change": [
                r"move.*(?:left|right|up|down|forward|back)",
                r"rotate.*(?:clockwise|counterclockwise)",
                r"(?:center|align|position)",
            ],
            "style_change": [
                r"make.*(?:modern|vintage|futuristic|classical|abstract)",
                r"(?:style|aesthetic|look)",
                r"more.*(?:artistic|realistic|cartoonish)",
            ],
            "composition_change": [
                r"(?:closer|further|zoom)",
                r"(?:camera|view|angle|perspective)",
                r"(?:framing|composition)",
            ],
        }
        
        # Parameter extraction patterns
        self.parameter_patterns = {
            "colors": r"(?:red|blue|green|yellow|orange|purple|pink|brown|black|white|gray|#[0-9a-fA-F]{6})",
            "numbers": r"(\d+(?:\.\d+)?)",
            "percentages": r"(\d+(?:\.\d+)?)%",
            "materials": r"(?:wood|metal|plastic|glass|stone|fabric|leather|concrete)",
            "directions": r"(?:left|right|up|down|forward|back|clockwise|counterclockwise)",
        }
        
    async def start_refinement_session(
        self, 
        asset_path: Path, 
        original_prompt: str
    ) -> RefinementSession:
        """Start a new refinement session for an asset.
        
        Args:
            asset_path: Path to the asset directory
            original_prompt: Original generation prompt
            
        Returns:
            New refinement session
        """
        session = RefinementSession(
            asset_path=asset_path,
            original_prompt=original_prompt
        )
        
        # Create refinement directory
        refinement_dir = asset_path / "refinements"
        refinement_dir.mkdir(exist_ok=True)
        
        # Save session metadata
        session_file = refinement_dir / f"{session.session_id}.json"
        self._save_session(session, session_file)
        
        logger.info(f"Started refinement session: {session.session_id}", asset_path=str(asset_path))
        return session
        
    def parse_instruction(self, instruction_text: str) -> UserInstruction:
        """Parse a user instruction into structured format.
        
        Args:
            instruction_text: Raw user instruction
            
        Returns:
            Structured user instruction
        """
        instruction_text = instruction_text.strip().lower()
        
        # Detect intent
        intent, confidence = self._detect_intent(instruction_text)
        
        # Extract target aspects
        target_aspects = self._extract_target_aspects(instruction_text, intent)
        
        # Extract parameters
        parameters = self._extract_parameters(instruction_text, intent)
        
        # Determine priority (more specific instructions get higher priority)
        priority = self._determine_priority(instruction_text, parameters)
        
        return UserInstruction(
            instruction_text=instruction_text,
            intent=intent,
            target_aspects=target_aspects,
            parameters=parameters,
            priority=priority,
            confidence=confidence
        )
        
    async def apply_instruction(
        self, 
        session: RefinementSession, 
        instruction: UserInstruction
    ) -> Dict[str, Any]:
        """Apply a user instruction to refine the asset.
        
        Args:
            session: Current refinement session
            instruction: User instruction to apply
            
        Returns:
            Results of the refinement
        """
        logger.info(f"Applying instruction: {instruction.instruction_text}")
        
        # Add instruction to session
        session.instructions.append(instruction)
        session.current_iteration += 1
        
        # Create refinement prompt
        refinement_prompt = self._create_refinement_prompt(
            session.original_prompt,
            instruction,
            session.instructions
        )
        
        # Generate refined asset
        asset_name = f"refined_iter_{session.current_iteration:02d}"
        
        try:
            result = await self.orchestrator.generate_asset(
                prompt=refinement_prompt,
                export_formats=['blend', 'obj', 'gltf'],
                asset_name=asset_name
            )
            
            # Create iteration record
            iteration_record = {
                "iteration": session.current_iteration,
                "instruction": instruction.model_dump(),
                "refinement_prompt": refinement_prompt,
                "result": result,
                "timestamp": datetime.now().isoformat(),
                "success": result.get("success", False)
            }
            
            session.iterations.append(iteration_record)
            
            # Save updated session
            refinement_dir = session.asset_path / "refinements"
            session_file = refinement_dir / f"{session.session_id}.json"
            self._save_session(session, session_file)
            
            # Save iteration details
            iteration_file = refinement_dir / f"iteration_{session.current_iteration:02d}.json"
            with open(iteration_file, 'w') as f:
                json.dump(iteration_record, f, indent=2, default=str)
                
            logger.info(f"Applied refinement iteration {session.current_iteration}")
            return iteration_record
            
        except Exception as e:
            logger.error(f"Failed to apply instruction: {e}")
            return {
                "iteration": session.current_iteration,
                "instruction": instruction.model_dump(),
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "success": False
            }
            
    def _detect_intent(self, instruction_text: str) -> Tuple[str, float]:
        """Detect the intent of a user instruction.
        
        Args:
            instruction_text: User instruction text
            
        Returns:
            Tuple of (intent, confidence_score)
        """
        best_intent = "general_change"
        best_score = 0.0
        
        for intent, patterns in self.intent_patterns.items():
            score = 0.0
            for pattern in patterns:
                if re.search(pattern, instruction_text, re.IGNORECASE):
                    score += 1.0
                    
            # Normalize score
            score = score / len(patterns)
            
            if score > best_score:
                best_score = score
                best_intent = intent
                
        # If no specific intent detected, try to infer from keywords
        if best_score < 0.3:
            if any(word in instruction_text for word in ["change", "modify", "adjust", "update"]):
                best_intent = "general_change"
                best_score = 0.5
                
        return best_intent, min(best_score, 1.0)
        
    def _extract_target_aspects(self, instruction_text: str, intent: str) -> List[str]:
        """Extract which aspects of the asset to target.
        
        Args:
            instruction_text: User instruction
            intent: Detected intent
            
        Returns:
            List of target aspects
        """
        aspects = []
        
        aspect_keywords = {
            "geometry": ["shape", "form", "geometry", "mesh", "vertices"],
            "material": ["material", "texture", "surface", "color", "finish"],
            "lighting": ["light", "lighting", "shadow", "brightness", "illumination"],
            "scale": ["size", "scale", "bigger", "smaller", "larger", "tiny"],
            "position": ["position", "location", "move", "place", "rotate"],
            "detail": ["detail", "complexity", "features", "decoration"],
        }
        
        for aspect, keywords in aspect_keywords.items():
            if any(keyword in instruction_text for keyword in keywords):
                aspects.append(aspect)
                
        # Add aspect based on intent if not already present
        intent_to_aspect = {
            "color_change": "material",
            "size_change": "scale", 
            "shape_change": "geometry",
            "material_change": "material",
            "lighting_change": "lighting",
            "detail_add": "detail",
            "detail_remove": "detail",
            "position_change": "position",
        }
        
        if intent in intent_to_aspect:
            aspect = intent_to_aspect[intent]
            if aspect not in aspects:
                aspects.append(aspect)
                
        return aspects if aspects else ["general"]
        
    def _extract_parameters(self, instruction_text: str, intent: str) -> Dict[str, Any]:
        """Extract parameters from the instruction.
        
        Args:
            instruction_text: User instruction
            intent: Detected intent
            
        Returns:
            Dictionary of extracted parameters
        """
        parameters = {}
        
        # Extract colors
        colors = re.findall(self.parameter_patterns["colors"], instruction_text, re.IGNORECASE)
        if colors:
            parameters["colors"] = colors
            
        # Extract numbers
        numbers = re.findall(self.parameter_patterns["numbers"], instruction_text)
        if numbers:
            parameters["numbers"] = [float(n) for n in numbers]
            
        # Extract percentages
        percentages = re.findall(self.parameter_patterns["percentages"], instruction_text)
        if percentages:
            parameters["percentages"] = [float(p) for p in percentages]
            
        # Extract materials
        materials = re.findall(self.parameter_patterns["materials"], instruction_text, re.IGNORECASE)
        if materials:
            parameters["materials"] = materials
            
        # Extract directions
        directions = re.findall(self.parameter_patterns["directions"], instruction_text, re.IGNORECASE)
        if directions:
            parameters["directions"] = directions
            
        # Extract intensity indicators
        intensity_words = {
            "very": 2.0,
            "much": 1.5,
            "slightly": 0.5,
            "little": 0.3,
            "bit": 0.3,
            "way": 2.0,
            "extremely": 3.0,
        }
        
        for word, multiplier in intensity_words.items():
            if word in instruction_text:
                parameters["intensity"] = multiplier
                break
        else:
            parameters["intensity"] = 1.0
            
        return parameters
        
    def _determine_priority(self, instruction_text: str, parameters: Dict[str, Any]) -> int:
        """Determine the priority of an instruction.
        
        Args:
            instruction_text: User instruction
            parameters: Extracted parameters
            
        Returns:
            Priority level (1-5, higher is more important)
        """
        priority = 1
        
        # Higher priority for more specific instructions
        if len(parameters) > 2:
            priority += 1
            
        # Higher priority for urgent language
        urgent_words = ["urgent", "important", "critical", "must", "need"]
        if any(word in instruction_text for word in urgent_words):
            priority += 2
            
        # Higher priority for specific values
        if "numbers" in parameters or "percentages" in parameters:
            priority += 1
            
        # Higher priority for color/material changes (visually important)
        if "colors" in parameters or "materials" in parameters:
            priority += 1
            
        return min(priority, 5)
        
    def _create_refinement_prompt(
        self, 
        original_prompt: str, 
        instruction: UserInstruction,
        all_instructions: List[UserInstruction]
    ) -> str:
        """Create a prompt for refinement based on user instruction.
        
        Args:
            original_prompt: Original generation prompt
            instruction: Current instruction to apply
            all_instructions: All previous instructions in session
            
        Returns:
            Refined prompt
        """
        # Start with original prompt
        refined_prompt = f"Based on this original request: '{original_prompt}'"
        
        # Add cumulative changes from all instructions
        changes = []
        for inst in all_instructions:
            change_desc = self._instruction_to_change_description(inst)
            changes.append(change_desc)
            
        if changes:
            refined_prompt += f"\n\nApply these modifications:\n" + "\n".join(f"- {change}" for change in changes)
            
        refined_prompt += f"\n\nMaintain the overall character and quality of the original design while incorporating these changes."
        
        return refined_prompt
        
    def _instruction_to_change_description(self, instruction: UserInstruction) -> str:
        """Convert an instruction to a change description.
        
        Args:
            instruction: User instruction
            
        Returns:
            Natural language description of the change
        """
        intent = instruction.intent
        params = instruction.parameters
        
        if intent == "color_change" and "colors" in params:
            return f"Change colors to include: {', '.join(params['colors'])}"
        elif intent == "size_change":
            if "bigger" in instruction.instruction_text or "larger" in instruction.instruction_text:
                multiplier = params.get("intensity", 1.0) * 1.5
                return f"Increase size by approximately {multiplier:.1f}x"
            else:
                multiplier = params.get("intensity", 1.0) * 0.7
                return f"Decrease size by approximately {multiplier:.1f}x"
        elif intent == "material_change" and "materials" in params:
            return f"Change material to: {', '.join(params['materials'])}"
        elif intent == "lighting_change":
            if "brighter" in instruction.instruction_text:
                return "Increase lighting brightness and add more illumination"
            else:
                return "Reduce lighting intensity for a softer appearance"
        elif intent == "detail_add":
            return "Add more detail, complexity, and surface features"
        elif intent == "detail_remove":
            return "Simplify design by removing unnecessary details"
        else:
            return instruction.instruction_text
            
    def _save_session(self, session: RefinementSession, file_path: Path):
        """Save refinement session to file.
        
        Args:
            session: Session to save
            file_path: File to save to
        """
        with open(file_path, 'w') as f:
            json.dump(session.model_dump(), f, indent=2, default=str)
            
    def load_session(self, session_file: Path) -> RefinementSession:
        """Load a refinement session from file.
        
        Args:
            session_file: Session file to load
            
        Returns:
            Loaded refinement session
        """
        with open(session_file, 'r') as f:
            data = json.load(f)
            
        # Convert path strings back to Path objects
        data["asset_path"] = Path(data["asset_path"])
        
        return RefinementSession(**data)
        
    def get_refinement_suggestions(
        self, 
        asset_path: Path, 
        quality_analysis: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """Get AI-generated suggestions for refinement.
        
        Args:
            asset_path: Path to asset
            quality_analysis: Optional quality analysis results
            
        Returns:
            List of refinement suggestions
        """
        suggestions = []
        
        if quality_analysis:
            # Use critique analysis to suggest improvements
            overall_score = quality_analysis.get("overall_score", 5)
            
            if overall_score < 6:
                suggestions.extend([
                    "Consider improving the overall visual quality",
                    "Try adjusting the lighting for better visibility",
                    "Add more geometric detail to enhance realism"
                ])
                
            # Specific suggestions based on low scores
            for aspect, score_data in quality_analysis.items():
                if isinstance(score_data, dict) and "score" in score_data:
                    score = score_data["score"]
                    if score < 6:
                        if aspect == "materials":
                            suggestions.append("Try different materials or textures")
                        elif aspect == "lighting":
                            suggestions.append("Adjust lighting setup for better illumination")
                        elif aspect == "geometry":
                            suggestions.append("Refine the geometric structure")
                            
        # General suggestions
        if not suggestions:
            suggestions = [
                "Try changing the color scheme",
                "Adjust the scale or proportions", 
                "Add surface details or textures",
                "Modify the lighting setup",
                "Change the camera angle or composition"
            ]
            
        return suggestions[:5]  # Return top 5 suggestions