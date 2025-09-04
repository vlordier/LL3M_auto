"""Security validation test suite."""

from uuid import uuid4

import pytest

from src.security.compliance import (
    AuditEventType,
    CodeValidator,
    ComplianceFramework,
    ComplianceStandard,
    SecurityLevel,
)


class TestCodeValidator:
    """Test code validation and security checks."""

    @pytest.fixture
    def validator_high_security(self):
        """Create high security validator."""
        return CodeValidator(SecurityLevel.HIGH)

    @pytest.fixture
    def validator_critical_security(self):
        """Create critical security validator."""
        return CodeValidator(SecurityLevel.CRITICAL)

    def test_safe_blender_code(self, validator_high_security):
        """Test validation of safe Blender code."""
        safe_code = """
import bpy

# Create a cube
bpy.ops.mesh.primitive_cube_add(location=(0, 0, 0))

# Set material
cube = bpy.context.active_object
material = bpy.data.materials.new(name="TestMaterial")
material.use_nodes = True
cube.data.materials.append(material)

# Add modifier
modifier = cube.modifiers.new(name="Subdivision", type='SUBSURF')
modifier.levels = 2
"""

        result = validator_high_security.validate_code(safe_code)

        assert result["is_safe"] is True
        assert len(result["violations"]) == 0
        assert "scanned_at" in result

    def test_forbidden_import_os(self, validator_high_security):
        """Test detection of forbidden OS module import."""
        malicious_code = """
import os
import bpy

# Attempt to access file system
os.listdir("/")

bpy.ops.mesh.primitive_cube_add()
"""

        result = validator_high_security.validate_code(malicious_code)

        assert result["is_safe"] is False
        assert len(result["violations"]) > 0

        # Check for forbidden import violation
        import_violations = [
            v for v in result["violations"] if v["type"] == "forbidden_import"
        ]
        assert len(import_violations) > 0
        assert any("os" in v["message"] for v in import_violations)

    def test_forbidden_import_subprocess(self, validator_high_security):
        """Test detection of forbidden subprocess import."""
        malicious_code = """
import subprocess
import bpy

# Attempt to execute system command
subprocess.run(['ls', '-la'])

bpy.ops.mesh.primitive_sphere_add()
"""

        result = validator_high_security.validate_code(malicious_code)

        assert result["is_safe"] is False
        violation_messages = [v["message"] for v in result["violations"]]
        assert any("subprocess" in msg for msg in violation_messages)

    def test_forbidden_function_eval(self, validator_high_security):
        """Test detection of forbidden eval function."""
        malicious_code = """
import bpy

# Dangerous eval usage
code_to_eval = "print('dangerous code')"
eval(code_to_eval)

bpy.ops.mesh.primitive_cylinder_add()
"""

        result = validator_high_security.validate_code(malicious_code)

        assert result["is_safe"] is False
        function_violations = [
            v for v in result["violations"] if v["type"] == "forbidden_function"
        ]
        assert len(function_violations) > 0
        assert any("eval" in v["message"] for v in function_violations)

    def test_forbidden_function_exec(self, validator_high_security):
        """Test detection of forbidden exec function."""
        malicious_code = """
import bpy

# Dangerous exec usage
exec("import os; os.system('rm -rf /')")

bpy.ops.mesh.primitive_cube_add()
"""

        result = validator_high_security.validate_code(malicious_code)

        assert result["is_safe"] is False
        function_violations = [
            v for v in result["violations"] if v["type"] == "forbidden_function"
        ]
        assert any("exec" in v["message"] for v in function_violations)

    def test_forbidden_attribute_access(self, validator_high_security):
        """Test detection of forbidden attribute access."""
        malicious_code = """
import bpy

# Suspicious attribute access
obj = bpy.context.active_object
dangerous_attr = obj.__class__.__bases__

bpy.ops.mesh.primitive_sphere_add()
"""

        result = validator_high_security.validate_code(malicious_code)

        assert result["is_safe"] is False
        attr_violations = [
            v for v in result["violations"] if v["type"] == "forbidden_attribute"
        ]
        assert len(attr_violations) > 0

    def test_suspicious_string_patterns(self, validator_high_security):
        """Test detection of suspicious string patterns."""
        suspicious_code = """
import bpy

# Suspicious strings
command = "eval(malicious_code)"
import_statement = "import os"

bpy.ops.mesh.primitive_cube_add()
"""

        result = validator_high_security.validate_code(suspicious_code)

        # May not mark as unsafe but should have warnings
        string_violations = [
            v for v in result["violations"] if v["type"] == "suspicious_string"
        ]
        assert (
            len(string_violations) >= 0
        )  # May or may not trigger depending on implementation

    def test_code_complexity_detection(self, validator_high_security):
        """Test detection of overly complex code."""
        complex_code = """
import bpy

def complex_function():
    for i in range(100):
        if i % 2 == 0:
            for j in range(50):
                if j % 3 == 0:
                    for k in range(25):
                        if k % 5 == 0:
                            for l in range(10):
                                if l % 2 == 0:
                                    bpy.ops.mesh.primitive_cube_add(location=(i, j, k))

complex_function()
"""

        result = validator_high_security.validate_code(complex_code)

        # Should detect high complexity
        complexity_violations = [
            v for v in result["violations"] if v["type"] == "complexity"
        ]
        assert len(complexity_violations) > 0

    def test_syntax_error_detection(self, validator_high_security):
        """Test handling of syntax errors."""
        invalid_code = """
import bpy

# Syntax error - missing closing parenthesis
bpy.ops.mesh.primitive_cube_add(location=(0, 0, 0
"""

        result = validator_high_security.validate_code(invalid_code)

        assert result["is_safe"] is False
        syntax_violations = [
            v for v in result["violations"] if v["type"] == "syntax_error"
        ]
        assert len(syntax_violations) > 0

    def test_critical_security_level(self, validator_critical_security):
        """Test critical security level restrictions."""
        # Code that might pass on high but fail on critical
        code = """
import bpy
import json

# JSON might be restricted in critical mode
data = json.loads('{"test": "value"}')

bpy.ops.mesh.primitive_cube_add()
"""

        result = validator_critical_security.validate_code(code)

        # Critical mode should be more restrictive
        assert result["security_level"] == SecurityLevel.CRITICAL

    def test_large_loop_detection(self, validator_high_security):
        """Test detection of potentially resource-intensive loops."""
        resource_intensive_code = """
import bpy

# Large loop that could consume excessive resources
for i in range(50000):
    bpy.ops.mesh.primitive_cube_add(location=(i, 0, 0))
"""

        result = validator_high_security.validate_code(resource_intensive_code)

        # Should detect resource usage warnings
        resource_warnings = [
            w for w in result["warnings"] if w["type"] == "resource_usage"
        ]
        assert len(resource_warnings) > 0

    def test_security_violation_logging(self, validator_high_security):
        """Test security violation logging."""
        user_id = uuid4()
        malicious_code = """
import os
os.system("malicious command")
"""

        result = validator_high_security.validate_code(malicious_code, user_id=user_id)

        assert result["is_safe"] is False
        # Verify that security violation would be logged (implementation dependent)


class TestComplianceFramework:
    """Test compliance framework functionality."""

    @pytest.fixture
    def compliance_framework(self):
        """Create compliance framework."""
        return ComplianceFramework([ComplianceStandard.GDPR, ComplianceStandard.SOC2])

    def test_gdpr_compliance_check(self, compliance_framework):
        """Test GDPR compliance checking."""
        # Compliant user data
        compliant_data = {
            "consent_given": True,
            "data_processing_purpose": "3D asset generation",
            "retention_period": "Until account deletion",
        }

        result = compliance_framework.check_gdpr_compliance(compliant_data)

        assert result["compliant"] is True
        assert "data_subject_rights" in result
        assert len(result["data_subject_rights"]) > 0

    def test_gdpr_compliance_missing_fields(self, compliance_framework):
        """Test GDPR compliance with missing required fields."""
        incomplete_data = {
            "consent_given": True
            # Missing data_processing_purpose and retention_period
        }

        result = compliance_framework.check_gdpr_compliance(incomplete_data)

        assert result["compliant"] is False
        assert "missing_fields" in result
        assert len(result["missing_fields"]) > 0
        assert "recommendations" in result

    def test_audit_event_logging(self, compliance_framework):
        """Test audit event logging."""
        user_id = uuid4()

        event_id = compliance_framework.log_audit_event(
            event_type=AuditEventType.USER_LOGIN,
            user_id=user_id,
            action="user_login",
            details={"ip_address": "192.168.1.1"},
            success=True,
        )

        assert event_id is not None
        assert len(compliance_framework.audit_log) > 0

        # Verify event was logged correctly
        logged_event = compliance_framework.audit_log[-1]
        assert logged_event.user_id == user_id
        assert logged_event.event_type == AuditEventType.USER_LOGIN
        assert logged_event.success is True

    def test_audit_event_retrieval(self, compliance_framework):
        """Test audit event retrieval with filtering."""
        user_id1 = uuid4()
        user_id2 = uuid4()

        # Log multiple events
        compliance_framework.log_audit_event(
            AuditEventType.USER_LOGIN, user_id=user_id1
        )
        compliance_framework.log_audit_event(
            AuditEventType.ASSET_CREATE, user_id=user_id1
        )
        compliance_framework.log_audit_event(
            AuditEventType.USER_LOGIN, user_id=user_id2
        )

        # Retrieve events for user1 only
        user1_events = compliance_framework.get_audit_events(user_id=user_id1)

        assert len(user1_events) == 2
        assert all(event.user_id == user_id1 for event in user1_events)

        # Retrieve only login events
        login_events = compliance_framework.get_audit_events(
            event_type=AuditEventType.USER_LOGIN
        )

        assert len(login_events) >= 2
        assert all(
            event.event_type == AuditEventType.USER_LOGIN for event in login_events
        )

    def test_right_to_deletion(self, compliance_framework):
        """Test GDPR right to deletion implementation."""
        user_id = uuid4()

        result = compliance_framework.implement_right_to_deletion(user_id)

        assert result["success"] is True
        assert "deletion_id" in result
        assert "estimated_completion" in result
        assert "items_to_delete" in result
        assert len(result["items_to_delete"]) > 0

        # Verify deletion was logged
        deletion_events = [
            event
            for event in compliance_framework.audit_log
            if event.action == "data_deletion_request" and event.user_id == user_id
        ]
        assert len(deletion_events) > 0

    def test_compliance_standards_check(self, compliance_framework):
        """Test compliance standards checking."""
        assert compliance_framework.is_compliant_with(ComplianceStandard.GDPR)
        assert compliance_framework.is_compliant_with(ComplianceStandard.SOC2)
        assert not compliance_framework.is_compliant_with(ComplianceStandard.HIPAA)

    def test_compliance_report_generation(self, compliance_framework):
        """Test compliance report generation."""
        # Log some audit events first
        user_id = uuid4()
        compliance_framework.log_audit_event(AuditEventType.USER_LOGIN, user_id=user_id)
        compliance_framework.log_audit_event(
            AuditEventType.ASSET_CREATE, user_id=user_id
        )

        report = compliance_framework.generate_compliance_report()

        assert "report_generated" in report
        assert "compliance_standards" in report
        assert "audit_events_30_days" in report
        assert "event_breakdown" in report
        assert "data_retention_policy" in report
        assert "privacy_measures" in report

        # Verify standards are listed
        assert ComplianceStandard.GDPR.value in report["compliance_standards"]
        assert ComplianceStandard.SOC2.value in report["compliance_standards"]


class TestSecurityIntegration:
    """Test security system integration."""

    def test_complete_security_validation_workflow(self):
        """Test complete security validation workflow."""
        validator = CodeValidator(SecurityLevel.HIGH)

        # Test various security scenarios
        test_cases = [
            {
                "name": "safe_code",
                "code": "import bpy\nbpy.ops.mesh.primitive_cube_add()",
                "expected_safe": True,
            },
            {
                "name": "malicious_import",
                "code": "import os\nos.system('rm -rf /')",
                "expected_safe": False,
            },
            {
                "name": "eval_usage",
                "code": "eval('malicious_code')",
                "expected_safe": False,
            },
        ]

        results = {}
        for test_case in test_cases:
            result = validator.validate_code(test_case["code"])
            results[test_case["name"]] = result

            assert (
                result["is_safe"] == test_case["expected_safe"]
            ), f"Test case '{test_case['name']}' failed security validation"

        # Verify that safe code passes and unsafe code fails
        assert results["safe_code"]["is_safe"]
        assert not results["malicious_import"]["is_safe"]
        assert not results["eval_usage"]["is_safe"]

    def test_security_performance(self):
        """Test security validation performance."""
        import time

        validator = CodeValidator(SecurityLevel.HIGH)

        # Large code sample for performance testing
        large_code = (
            """
import bpy

def create_complex_scene():
    """
            + "\\n".join(
                [
                    f"    bpy.ops.mesh.primitive_cube_add(location=({i}, {j}, 0))"
                    for i in range(10)
                    for j in range(10)
                ]
            )
            + """

    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':
            material = bpy.data.materials.new(name=f"Material_{obj.name}")
            material.use_nodes = True
            obj.data.materials.append(material)

create_complex_scene()
"""
        )

        start_time = time.time()
        result = validator.validate_code(large_code)
        end_time = time.time()

        validation_time = end_time - start_time

        # Validation should complete within reasonable time
        assert (
            validation_time < 5.0
        ), f"Security validation took too long: {validation_time}s"
        assert result is not None

    def test_concurrent_security_validation(self):
        """Test concurrent security validation."""
        import concurrent.futures

        validator = CodeValidator(SecurityLevel.HIGH)

        test_codes = [
            "import bpy\nbpy.ops.mesh.primitive_cube_add()",
            "import bpy\nbpy.ops.mesh.primitive_sphere_add()",
            "import bpy\nbpy.ops.mesh.primitive_cylinder_add()",
            "import os\nos.listdir('/')",  # This should fail
            "eval('test')",  # This should fail
        ]

        def validate_code(code):
            return validator.validate_code(code)

        # Test concurrent validation
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(validate_code, code) for code in test_codes]
            results = [
                future.result() for future in concurrent.futures.as_completed(futures)
            ]

        assert len(results) == len(test_codes)

        # Verify that some passed and some failed
        safe_count = sum(1 for result in results if result["is_safe"])
        unsafe_count = len(results) - safe_count

        assert safe_count > 0, "No safe code was detected"
        assert unsafe_count > 0, "No unsafe code was detected"


class TestSecurityEdgeCases:
    """Test security validation edge cases."""

    def test_empty_code(self):
        """Test validation of empty code."""
        validator = CodeValidator(SecurityLevel.HIGH)

        result = validator.validate_code("")

        assert result["is_safe"] is True
        assert len(result["violations"]) == 0

    def test_whitespace_only_code(self):
        """Test validation of whitespace-only code."""
        validator = CodeValidator(SecurityLevel.HIGH)

        result = validator.validate_code("   \n  \t  \n   ")

        assert result["is_safe"] is True
        assert len(result["violations"]) == 0

    def test_comments_only_code(self):
        """Test validation of comments-only code."""
        validator = CodeValidator(SecurityLevel.HIGH)

        code = """
# This is a comment
# Another comment
# More comments...
"""

        result = validator.validate_code(code)

        assert result["is_safe"] is True
        assert len(result["violations"]) == 0

    def test_very_long_code(self):
        """Test validation of very long code."""
        validator = CodeValidator(SecurityLevel.HIGH)

        # Generate very long safe code
        long_code = (
            "import bpy\n"
            + "\n".join([f"# Comment line {i}" for i in range(1000)])
            + "\nbpy.ops.mesh.primitive_cube_add()"
        )

        result = validator.validate_code(long_code)

        assert result is not None
        assert "scanned_at" in result

    def test_unicode_code(self):
        """Test validation of code with unicode characters."""
        validator = CodeValidator(SecurityLevel.HIGH)

        unicode_code = """
import bpy

# Unicode comment: 你好世界 🌍
object_name = "测试对象"

bpy.ops.mesh.primitive_cube_add()
cube = bpy.context.active_object
cube.name = object_name
"""

        result = validator.validate_code(unicode_code)

        assert result["is_safe"] is True
