"""Security and compliance framework for LL3M."""

import ast
import hashlib
import hmac
import secrets
from datetime import datetime, timedelta
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel

# Security thresholds
LARGE_LOOP_THRESHOLD = 10000
MAX_NESTING_DEPTH = 5
MAX_FUNCTION_COUNT = 20


class SecurityLevel(str, Enum):
    """Security levels for code validation."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ComplianceStandard(str, Enum):
    """Compliance standards supported."""

    GDPR = "gdpr"
    SOC2 = "soc2"
    HIPAA = "hipaa"
    ISO27001 = "iso27001"
    PCI_DSS = "pci_dss"


class AuditEventType(str, Enum):
    """Types of audit events."""

    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    ASSET_CREATE = "asset_create"
    ASSET_DELETE = "asset_delete"
    ASSET_EXPORT = "asset_export"
    CODE_EXECUTION = "code_execution"
    SECURITY_VIOLATION = "security_violation"
    ADMIN_ACTION = "admin_action"
    DATA_ACCESS = "data_access"


class SecurityViolation(BaseModel):
    """Security violation record."""

    id: str
    user_id: UUID | None
    violation_type: str
    severity: str
    description: str
    code_snippet: str | None
    ip_address: str | None
    user_agent: str | None
    timestamp: datetime
    resolved: bool = False


class AuditEvent(BaseModel):
    """Audit log event."""

    id: str
    event_type: AuditEventType
    user_id: UUID | None
    resource_id: str | None
    action: str
    details: dict[str, Any]
    ip_address: str | None
    user_agent: str | None
    timestamp: datetime
    success: bool


class CodeValidator:
    """Validates Blender Python code for security issues."""

    def __init__(self, security_level: SecurityLevel = SecurityLevel.HIGH):
        """Initialize security validator with specified security level."""
        self.security_level = security_level
        self.forbidden_imports = self._get_forbidden_imports()
        self.forbidden_functions = self._get_forbidden_functions()
        self.forbidden_attributes = self._get_forbidden_attributes()

    def _get_forbidden_imports(self) -> set[str]:
        """Get list of forbidden imports based on security level."""
        base_forbidden = {
            "os",
            "sys",
            "subprocess",
            "shutil",
            "glob",
            "socket",
            "urllib",
            "requests",
            "http",
            "ftplib",
            "smtplib",
            "poplib",
            "imaplib",
            "pickle",
            "marshal",
            "shelve",
            "dill",
            "eval",
            "exec",
            "compile",
            "__import__",
        }

        if self.security_level in [SecurityLevel.HIGH, SecurityLevel.CRITICAL]:
            base_forbidden.update(
                {
                    "pathlib",
                    "tempfile",
                    "fileinput",
                    "csv",
                    "json",
                    "sqlite3",
                    "threading",
                    "multiprocessing",
                    "ctypes",
                    "platform",
                    "getpass",
                    "pwd",
                    "grp",
                }
            )

        if self.security_level == SecurityLevel.CRITICAL:
            base_forbidden.update(
                {
                    "time",
                    "datetime",
                    "random",
                    "hashlib",
                    "hmac",
                    "base64",
                    "codecs",
                    "zlib",
                }
            )

        return base_forbidden

    def _get_forbidden_functions(self) -> set[str]:
        """Get list of forbidden functions."""
        return {
            "eval",
            "exec",
            "compile",
            "__import__",
            "globals",
            "locals",
            "vars",
            "dir",
            "getattr",
            "setattr",
            "delattr",
            "hasattr",
            "open",
            "input",
            "raw_input",
            "reload",
            "exit",
            "quit",
        }

    def _get_forbidden_attributes(self) -> set[str]:
        """Get list of forbidden attributes."""
        return {
            "__class__",
            "__bases__",
            "__subclasses__",
            "__mro__",
            "__globals__",
            "__code__",
            "__closure__",
            "__defaults__",
            "__dict__",
            "__weakref__",
            "__module__",
            "__name__",
        }

    def validate_code(self, code: str, user_id: UUID | None = None) -> dict[str, Any]:
        """Validate Python code for security issues."""
        violations = []
        warnings = []

        try:
            # Parse code into AST
            tree = ast.parse(code)

            # Check for security violations
            for node in ast.walk(tree):
                violations.extend(self._check_imports(node))
                violations.extend(self._check_function_calls(node))
                violations.extend(self._check_attribute_access(node))
                violations.extend(self._check_string_operations(node))
                warnings.extend(self._check_resource_usage(node))

            # Additional checks
            violations.extend(self._check_code_complexity(tree))
            warnings.extend(self._check_best_practices(tree))

        except SyntaxError as e:
            violations.append(
                {
                    "type": "syntax_error",
                    "severity": "error",
                    "message": f"Syntax error: {e}",
                    "line": e.lineno,
                }
            )
        except Exception as e:
            violations.append(
                {
                    "type": "parse_error",
                    "severity": "error",
                    "message": f"Failed to parse code: {e}",
                }
            )

        # Determine if code is safe
        critical_violations = [v for v in violations if v.get("severity") == "critical"]
        high_violations = [v for v in violations if v.get("severity") == "high"]

        is_safe = len(critical_violations) == 0 and (
            self.security_level != SecurityLevel.CRITICAL or len(high_violations) == 0
        )

        # Log security violations
        if not is_safe and user_id:
            self._log_security_violation(user_id, violations, code)

        return {
            "is_safe": is_safe,
            "violations": violations,
            "warnings": warnings,
            "security_level": self.security_level,
            "scanned_at": datetime.utcnow().isoformat(),
        }

    def _check_imports(self, node: ast.AST) -> list[dict[str, Any]]:
        """Check for forbidden imports."""
        violations = []

        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in self.forbidden_imports:
                    violations.append(
                        {
                            "type": "forbidden_import",
                            "severity": "critical",
                            "message": f"Import '{alias.name}' is not allowed",
                            "line": node.lineno,
                        }
                    )

        elif isinstance(node, ast.ImportFrom) and node.module and node.module in self.forbidden_imports:
            violations.append(
                {
                    "type": "forbidden_import",
                    "severity": "critical",
                    "message": f"Import from '{node.module}' is not allowed",
                    "line": node.lineno,
                }
            )

        return violations

    def _check_function_calls(self, node: ast.AST) -> list[dict[str, Any]]:
        """Check for forbidden function calls."""
        violations = []

        if isinstance(node, ast.Call):
            func_name = None

            if isinstance(node.func, ast.Name):
                func_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                func_name = node.func.attr

            if func_name in self.forbidden_functions:
                violations.append(
                    {
                        "type": "forbidden_function",
                        "severity": "critical",
                        "message": f"Function '{func_name}' is not allowed",
                        "line": node.lineno,
                    }
                )

        return violations

    def _check_attribute_access(self, node: ast.AST) -> list[dict[str, Any]]:
        """Check for forbidden attribute access."""
        violations = []

        if isinstance(node, ast.Attribute) and node.attr in self.forbidden_attributes:
            violations.append(
                {
                    "type": "forbidden_attribute",
                    "severity": "high",
                    "message": f"Attribute '{node.attr}' access is suspicious",
                    "line": node.lineno,
                }
            )

        return violations

    def _check_string_operations(self, node: ast.AST) -> list[dict[str, Any]]:
        """Check for suspicious string operations."""
        violations = []

        # Handle both deprecated ast.Str and modern ast.Constant nodes
        string_value: str | None = None
        if isinstance(node, ast.Str):
            string_value = node.s
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            string_value = node.value
        if string_value is not None and isinstance(string_value, str):
            # Check for potential code injection
            suspicious_patterns = ["__", "eval(", "exec(", "import ", "from "]
            for pattern in suspicious_patterns:
                if pattern in string_value:
                    violations.append(
                        {
                            "type": "suspicious_string",
                            "severity": "medium",
                            "message": f"Suspicious string pattern '{pattern}' found",
                            "line": getattr(node, "lineno", 0),
                        }
                    )

        return violations

    def _check_resource_usage(self, node: ast.AST) -> list[dict[str, Any]]:
        """Check for potential resource exhaustion."""
        warnings = []

        if isinstance(node, ast.For) and isinstance(node.iter, ast.Call):
            # Check for potential infinite loops
            func_name = getattr(node.iter.func, "id", None)
            if func_name == "range" and len(node.iter.args) > 0 and (
                isinstance(node.iter.args[0], ast.Num)
                and isinstance(node.iter.args[0].n, int | float)
                and node.iter.args[0].n > LARGE_LOOP_THRESHOLD
            ):
                    warnings.append(
                        {
                            "type": "resource_usage",
                            "severity": "warning",
                            "message": "Large loop detected - may consume excessive resources",
                            "line": node.lineno,
                        }
                    )

        return warnings

    def _check_code_complexity(self, tree: ast.AST) -> list[dict[str, Any]]:
        """Check code complexity metrics."""
        violations = []

        # Count nested loops/conditions
        max_nesting = self._calculate_max_nesting(tree)
        if max_nesting > MAX_NESTING_DEPTH:
            violations.append(
                {
                    "type": "complexity",
                    "severity": "medium",
                    "message": f"Code has high nesting level ({max_nesting})",
                    "line": 1,
                }
            )

        return violations

    def _calculate_max_nesting(self, node: ast.AST, current_depth: int = 0) -> int:
        """Calculate maximum nesting depth."""
        max_depth = current_depth

        nesting_nodes = (
            ast.For,
            ast.While,
            ast.If,
            ast.With,
            ast.Try,
            ast.FunctionDef,
            ast.ClassDef,
        )

        for child in ast.iter_child_nodes(node):
            if isinstance(child, nesting_nodes):
                depth = self._calculate_max_nesting(child, current_depth + 1)
                max_depth = max(max_depth, depth)
            else:
                depth = self._calculate_max_nesting(child, current_depth)
                max_depth = max(max_depth, depth)

        return max_depth

    def _check_best_practices(self, tree: ast.AST) -> list[dict[str, Any]]:
        """Check for code quality best practices."""
        warnings = []

        # Count functions and classes
        function_count = len(
            [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
        )
        if function_count > MAX_FUNCTION_COUNT:
            warnings.append(
                {
                    "type": "best_practices",
                    "severity": "info",
                    "message": f"Large number of functions ({function_count}) - consider refactoring",
                    "line": 1,
                }
            )

        return warnings

    def _log_security_violation(
        self, user_id: UUID, violations: list[dict[str, Any]], _code: str
    ) -> None:
        """Log security violation."""
        # This would integrate with the audit system
        print(f"Security violation by user {user_id}: {violations}")


class ComplianceFramework:
    """Compliance framework for various standards."""

    def __init__(self, enabled_standards: list[ComplianceStandard]):
        self.enabled_standards = enabled_standards
        self.audit_log: list[AuditEvent] = []
        self.violation_log: list[SecurityViolation] = []

    def is_compliant_with(self, standard: ComplianceStandard) -> bool:
        """Check if system is compliant with given standard."""
        return standard in self.enabled_standards

    def log_audit_event(
        self,
        event_type: AuditEventType,
        user_id: UUID | None = None,
        resource_id: str | None = None,
        action: str = "",
        details: dict[str, Any] | None = None,
        ip_address: str | None = None,
        user_agent: str | None = None,
        success: bool = True,
    ) -> str:
        """Log an audit event."""
        event_id = secrets.token_urlsafe(16)

        event = AuditEvent(
            id=event_id,
            event_type=event_type,
            user_id=user_id,
            resource_id=resource_id,
            action=action,
            details=details or {},
            ip_address=ip_address,
            user_agent=user_agent,
            timestamp=datetime.utcnow(),
            success=success,
        )

        self.audit_log.append(event)

        # In production, this would be stored in a secure, immutable log store
        return event_id

    def get_audit_events(
        self,
        user_id: UUID | None = None,
        event_type: AuditEventType | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 100,
    ) -> list[AuditEvent]:
        """Retrieve audit events with filtering."""
        events = self.audit_log

        # Apply filters
        if user_id:
            events = [e for e in events if e.user_id == user_id]

        if event_type:
            events = [e for e in events if e.event_type == event_type]

        if start_time:
            events = [e for e in events if e.timestamp >= start_time]

        if end_time:
            events = [e for e in events if e.timestamp <= end_time]

        # Sort by timestamp (newest first) and limit
        events.sort(key=lambda x: x.timestamp, reverse=True)
        return events[:limit]

    def check_gdpr_compliance(self, user_data: dict[str, Any]) -> dict[str, Any]:
        """Check GDPR compliance for user data."""
        if ComplianceStandard.GDPR not in self.enabled_standards:
            return {"compliant": False, "reason": "GDPR not enabled"}

        required_fields = [
            "consent_given",
            "data_processing_purpose",
            "retention_period",
        ]
        missing_fields = [field for field in required_fields if field not in user_data]

        if missing_fields:
            return {
                "compliant": False,
                "missing_fields": missing_fields,
                "recommendations": [
                    "Obtain explicit user consent",
                    "Document data processing purpose",
                    "Define data retention period",
                ],
            }

        return {
            "compliant": True,
            "data_subject_rights": [
                "Right to access",
                "Right to rectification",
                "Right to erasure",
                "Right to restrict processing",
                "Right to data portability",
                "Right to object",
                "Rights related to automated decision making",
            ],
        }

    def implement_right_to_deletion(self, user_id: UUID) -> dict[str, Any]:
        """Implement GDPR right to deletion (right to be forgotten)."""
        if ComplianceStandard.GDPR not in self.enabled_standards:
            return {"success": False, "reason": "GDPR not enabled"}

        # Log the deletion request
        self.log_audit_event(
            AuditEventType.ADMIN_ACTION,
            user_id=user_id,
            action="data_deletion_request",
            details={"type": "gdpr_right_to_deletion"},
        )

        # In production, this would:
        # 1. Mark user account for deletion
        # 2. Anonymize or delete personal data
        # 3. Remove user-generated content (assets)
        # 4. Clear audit logs (where legally permissible)
        # 5. Notify connected services

        return {
            "success": True,
            "deletion_id": secrets.token_urlsafe(16),
            "estimated_completion": datetime.utcnow() + timedelta(days=30),
            "items_to_delete": [
                "User profile data",
                "Generated assets",
                "Account preferences",
                "Usage analytics (anonymized)",
            ],
        }

    def generate_compliance_report(self) -> dict[str, Any]:
        """Generate compliance report."""
        recent_events = self.get_audit_events(
            start_time=datetime.utcnow() - timedelta(days=30)
        )

        event_summary: dict[str, int] = {}
        for event in recent_events:
            event_type = event.event_type.value
            event_summary[event_type] = event_summary.get(event_type, 0) + 1

        security_violations = len([v for v in self.violation_log if not v.resolved])

        return {
            "report_generated": datetime.utcnow().isoformat(),
            "compliance_standards": [s.value for s in self.enabled_standards],
            "audit_events_30_days": len(recent_events),
            "event_breakdown": event_summary,
            "security_violations_open": security_violations,
            "data_retention_policy": {
                "user_data": "Retained until account deletion",
                "audit_logs": "Retained for 7 years",
                "asset_data": "Retained until user deletion or explicit removal",
            },
            "privacy_measures": [
                "End-to-end encryption for sensitive data",
                "Regular security audits",
                "Access controls and authentication",
                "Data minimization practices",
                "Regular staff training",
            ],
        }


class DataEncryption:
    """Data encryption utilities for compliance."""

    def __init__(self, master_key: str | None = None):
        self.master_key = master_key or self._generate_master_key()

    def _generate_master_key(self) -> str:
        """Generate a master encryption key."""
        return secrets.token_urlsafe(32)

    def encrypt_sensitive_data(self, data: str, context: str = "") -> dict[str, str]:
        """Encrypt sensitive data with context."""
        # In production, use proper encryption library like cryptography
        # This is a simplified example

        salt = secrets.token_bytes(16)
        key = hashlib.pbkdf2_hmac("sha256", self.master_key.encode(), salt, 100000)

        # Mock encryption (use proper AES encryption in production)
        encrypted = hmac.new(key, data.encode(), hashlib.sha256).hexdigest()

        return {
            "encrypted_data": encrypted,
            "salt": salt.hex(),
            "context": context,
            "algorithm": "HMAC-SHA256",  # Mock algorithm
            "encrypted_at": datetime.utcnow().isoformat(),
        }

    def decrypt_sensitive_data(self, _encrypted_data: dict[str, str]) -> str:
        """Decrypt sensitive data."""
        # This would implement proper decryption in production
        # For now, return a placeholder
        return "[DECRYPTED_DATA]"


# Global instances
code_validator = CodeValidator(SecurityLevel.HIGH)
compliance_framework = ComplianceFramework(
    [ComplianceStandard.GDPR, ComplianceStandard.SOC2, ComplianceStandard.ISO27001]
)
data_encryption = DataEncryption()
