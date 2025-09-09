"""Structured output schemas for the critic agent using Pydantic models."""

from enum import Enum

from pydantic import BaseModel, Field


class RefinementPriority(str, Enum):
    """Refinement priority levels."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class QualityChange(str, Enum):
    """Quality change assessment."""

    IMPROVED = "improved"
    SAME = "same"
    DEGRADED = "degraded"


class AspectScore(BaseModel):
    """Individual aspect score with feedback."""

    score: int = Field(..., ge=1, le=10, description="Score from 1-10")
    notes: str = Field(..., description="Specific feedback for this aspect")
    issues: list[str] = Field(default_factory=list, description="Specific issues found")
    suggestions: list[str] = Field(default_factory=list, description="Improvement suggestions")
    is_critical: bool = Field(default=False, description="Whether this aspect has critical issues")


class VisualAnalysisResult(BaseModel):
    """Comprehensive structured output for initial visual quality analysis."""

    # Overall Assessment
    overall_score: int = Field(..., ge=1, le=10, description="Overall quality score from 1-10")

    # Visual Quality Dimensions
    visual_quality: AspectScore = Field(..., description="Visual appearance, aesthetics, professional look")
    geometry: AspectScore = Field(..., description="Shape accuracy, mesh topology, complexity")
    materials: AspectScore = Field(..., description="Surface appearance, textures, shader quality")
    lighting: AspectScore = Field(..., description="Scene illumination, shadows, ambient lighting")
    composition: AspectScore = Field(..., description="Camera angle, framing, scene organization")
    requirements_match: AspectScore = Field(..., description="How well it matches the original request")

    # Technical Analysis Dimensions
    technical_quality: AspectScore = Field(..., description="Technical execution, mesh quality, topology")
    scale_proportions: AspectScore = Field(..., description="Realistic proportions, scale accuracy")
    structural_integrity: AspectScore = Field(..., description="Component connections, stability")

    # Scene and Context Dimensions
    scene_setup: AspectScore = Field(..., description="Environment, background, context")
    presentation_quality: AspectScore = Field(..., description="Professional presentation, render quality")
    artistic_appeal: AspectScore = Field(..., description="Creative design, aesthetic value")

    # Functional Assessment
    functional_accuracy: AspectScore = Field(..., description="Real-world functionality, usability")
    detail_level: AspectScore = Field(..., description="Appropriate level of detail for purpose")
    realism: AspectScore = Field(..., description="Photorealistic quality, believability")

    # Overall Analysis
    needs_refinement: bool = Field(..., description="Whether further refinement is needed")
    critical_issues: list[str] = Field(default_factory=list, description="Critical issues that must be fixed")
    improvement_suggestions: list[str] = Field(default_factory=list, description="Specific improvement recommendations")
    refinement_priority: RefinementPriority = Field(
        default=RefinementPriority.LOW, description="Priority level for refinement"
    )

    # Analysis Metadata
    analysis_focus: list[str] = Field(default_factory=list, description="Primary focus areas of the analysis")
    asset_type_assessment: str = Field(default="", description="Classification of what type of asset this is")
    complexity_level: int = Field(default=1, ge=1, le=5, description="Complexity level from 1-5")


class RefinementComparisonResult(BaseModel):
    """Structured output for before/after comparison analysis."""

    improvement_score: int = Field(..., ge=-5, le=5, description="Improvement score from -5 to +5")
    issues_resolved: list[str] = Field(default_factory=list, description="Issues that were resolved")
    new_issues: list[str] = Field(default_factory=list, description="New issues introduced")
    quality_change: QualityChange = Field(..., description="Overall quality change assessment")
    recommendations: list[str] = Field(default_factory=list, description="Further recommendations")
    continue_refinement: bool = Field(..., description="Whether to continue refinement process")
