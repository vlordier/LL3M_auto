"""Test Blender templates."""

from src.blender.templates import (
    GEOMETRY_TEMPLATES,
    LIGHTING_TEMPLATES,
    MATERIAL_TEMPLATES,
    MODIFIER_TEMPLATES,
    SCENE_TEMPLATES,
)


class TestTemplates:
    """Test template definitions."""

    def test_geometry_templates_exist(self) -> None:
        """Test geometry templates are defined."""
        assert "cube" in GEOMETRY_TEMPLATES
        assert "sphere" in GEOMETRY_TEMPLATES
        assert "cylinder" in GEOMETRY_TEMPLATES
        assert "plane" in GEOMETRY_TEMPLATES
        
        # Verify templates contain expected placeholders
        cube_template = GEOMETRY_TEMPLATES["cube"]
        assert "{x}" in cube_template
        assert "{y}" in cube_template
        assert "{z}" in cube_template
        assert "{name}" in cube_template

    def test_material_templates_exist(self) -> None:
        """Test material templates are defined."""
        assert "basic" in MATERIAL_TEMPLATES
        assert "metallic" in MATERIAL_TEMPLATES
        assert "emission" in MATERIAL_TEMPLATES
        
        basic_template = MATERIAL_TEMPLATES["basic"]
        assert "{name}" in basic_template
        assert "{r}" in basic_template
        assert "{g}" in basic_template
        assert "{b}" in basic_template

    def test_lighting_templates_exist(self) -> None:
        """Test lighting templates are defined."""
        assert "sun" in LIGHTING_TEMPLATES
        assert "point" in LIGHTING_TEMPLATES
        assert "area" in LIGHTING_TEMPLATES
        
        sun_template = LIGHTING_TEMPLATES["sun"]
        assert "{energy}" in sun_template

    def test_scene_templates_exist(self) -> None:
        """Test scene templates are defined."""
        assert "clear_scene" in SCENE_TEMPLATES
        assert "camera_setup" in SCENE_TEMPLATES
        assert "render_settings" in SCENE_TEMPLATES

    def test_modifier_templates_exist(self) -> None:
        """Test modifier templates are defined."""
        assert "subdivision" in MODIFIER_TEMPLATES
        assert "bevel" in MODIFIER_TEMPLATES
        assert "array" in MODIFIER_TEMPLATES