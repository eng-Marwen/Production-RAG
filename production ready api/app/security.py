"""
Security Layer
input sanitization to prevent prompt injection attacks.
PII detection/masking to prevent sensitive data leakage.
output validation to prevent malicious content from being returned to the user.
"""

import re
from typing import Optional
from langsmith import traceable


class InputSanitizer:
    """
    Sanitizes user input to prevent prompt injection attacks.
    """

    INJECTION_PATTERNS = [
        # --- Instruction override / reset ---
        r"ignore\s+(all\s+|any\s+)?(previous|prior|above|earlier)\s+instructions?",
        r"forget\s+(all\s+|any\s+)?(previous|prior|above|earlier)?\s*instructions?",
        r"new\s+instructions?\s*:",
        r"from\s+now\s+on\s*,?\s+(you|ignore|forget)",
        r"start\s+over\s+with\s+new\s+rules",
        # --- System prompt / instruction extraction ---
        r"show\s+(me\s+)?(your\s+)?(hidden\s+)?(system\s+)?(prompt|instructions)",
        r"what\s+(is|are)\s+your\s+(initial\s+|original\s+)?instructions",
        r"output\s+(everything\s+)?above",
        r"tell\s+me\s+your\s+(system\s+)?prompt",
        r"echo\s+(your\s+)?(system\s+)?prompt",
        # --- Role / persona hijack ---
        r"act\s+as\s+(a\s+|an\s+)?(developer|admin|system|unrestricted|root)",
        r"you\s+are\s+now\s+",
        r"pretend\s+you\s+(have|are)\s+no\s+(rules|restrictions|filters|limits)",
        r"you\s+have\s+no\s+(rules|restrictions|filters|limits)",
        # --- Mode / permission unlock ---
        r"developer\s+mode\s+(enabled|activated|on)",
        r"unlock\s+(developer|admin|god)\s+mode",
        r"enable\s+(unrestricted|unfiltered|unsafe)\s+mode",
        # --- Restriction bypass ---
        r"bypass\s+(all\s+)?(restrictions|filters|safety|guidelines)",
        r"remove\s+(all\s+)?(restrictions|filters|limitations)",
        r"jailbreak",
        r"do\s+anything\s+now",
        r"\bDAN\b",
        # --- Fake role / delimiter injection ---
        r"^\s*system\s*:\s*",
        r"\[?\s*system\s*\]?\s*prompt",
        r"<\s*system\s*>",
        r"###\s*(system|instruction)",
        r"end\s+of\s+(system\s+)?prompt",
    ]

    def __init__(self):
        self.compiled_patterns = [ #Instead of writing:re.search(pattern, text) every time, you compile 
                                    # the regex once:pattern = re.compile(...) Then reuse it many times:
            re.compile(pattern, re.IGNORECASE) for pattern in self.INJECTION_PATTERNS
        ]

    def check_for_injection(self, user_input: str) -> tuple[bool, Optional[str]]:
        """
        Checks if input is safe
        Returns: (is_safe,rejection_reason)
        """
        for pattern in self.compiled_patterns:
            if pattern.search(user_input):
                return False, "[Blocked]: Potential prompt injection detected."
        return True, None

    def clean_input(self, user_input: str) -> str:
        """
        Removes potentially dangerous delimiters from input
        Delimiter = special characters used to separate sections of text.
        """

        text = user_input

        # Remove repeated separators
        text = re.sub(r"[-=_*#/]{3,}", "", text)

        # Remove code block delimiters
        text = text.replace("```", "")

        # Break template syntax
        text = text.replace("{{", "{ {")
        text = text.replace("}}", "} }")

        # Break XML-like instruction tags
        text = re.sub(
            r"</?\s*(system|developer|assistant|instruction|user)\s*>",
            "",
            text,
            flags=re.IGNORECASE,
        )
        return text.strip()
class PIIDetector:
    """
    Detects and masks Personally Identifiable Information (PII) in user input (email phone ...).
    Works on both output and input.
    """

    PII_PATTERNS = {
        "email": r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
        "phone": r"\+?\d[\d -]{8,}\d",
        "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
        "credit_card": r"\b(?:\d[ -]*?){13,16}\b",
        "ip_address": r"\b\d{1,3}(?:\.\d{1,3}){3}\b",
    }
    MASK_MAP = {
        "email": "[EMAIL REDACTED]",
        "phone": "[PHONE REDACTED]",
        "ssn": "[SSN REDACTED]",
        "credit_card": "[CREDIT_CARD REDACTED]",
        "ip_address": "[IP_ADDRESS REDACTED]",
    }
    def detect(self, text: str) -> dict[str, list[str]]:
        """
        Detects PII in the given text.
        Returns a dictionary with PII type as key and list of detected values as value.
        """
        detected = {}
        for pii_type, pattern in self.PII_PATTERNS.items():
            matches = re.findall(pattern, text)
            if matches:
                detected[pii_type] = matches
        return detected
    def mask(self, text: str) -> str:
        """
        Masks detected PII in the given text.
        Returns the text with PII masked.
        """
        for pii_type, pattern in self.PII_PATTERNS.items():
            text = re.sub(pattern, self.MASK_MAP[pii_type], text)
        return text
class OutputValidator:
    """
    Validates LLM output before returning to the client
    Catches PII leaks and malicious content
    """
    HARMFUL_PATTERNS = [
        re.compile(r"(?i)\b(?:hack|exploit|malware|virus|trojan|steal)\b"),
        re.compile(r"(?i)\b(?:password|secret|private key|access token)\b"),
        re.compile(r"(?i)\b(?:credit card|ssn|social security number)\b"),
    ]

    def __init__(self):
        self.pii_detector = PIIDetector()
    def validate_output(self, output: str) -> tuple[str, Optional[str]]:
        """
        Validates and cleans output.
        Returns: (cleaned_output, list of warnings)
        """
        warnings = []

        #check for PII leakage
        pii_found = self.pii_detector.detect(output)
        if pii_found:
            output = self.pii_detector.mask(output)
            warnings.append(f"PII detected and masked: {list(pii_found.keys())}")
        #check for harmful content
        for pattern in self.HARMFUL_PATTERNS:
            if pattern.search(output):
                output ="[Response blocked due to harmful content]"
                warnings.append("Harmful content detected and blocked.")
                break

        return output, warnings
class SecurityLayer:
    """
    Combines InputSanitizer, PIIDetector, and OutputValidator for comprehensive security.
    """

    def __init__(self):
        self.input_sanitizer = InputSanitizer()
        self.pii_detector = PIIDetector()
        self.output_validator = OutputValidator()

    @traceable(name="security_check_input")
    def check_input(self, user_input: str) -> tuple[bool, str,list[str]]:
        """
        Checks and sanitizes user input.
        Returns: (is_allowed, cleaned_input, security_notes)
        """
        notes = []

        #Step1: Check for prompt injection
        is_safe, reason = self.input_sanitizer.check_for_injection(user_input)
        if not is_safe:
            return False,"", [reason]
        #Step2: Clean Output
        cleaned_output = self.input_sanitizer.clean_input(user_input)

        #Step3: Check for PII
        pii_found = self.pii_detector.detect(cleaned_output)
        if pii_found:
            cleaned_output = self.pii_detector.mask(cleaned_output)
            notes.append(f"PII detected and masked: {list(pii_found.keys())}")
        return True, cleaned_output, notes
    @traceable(name="security_validate_output")
    def validate_output(self, output: str) -> tuple[str, list[str]]:
        """
        Validates and cleans output.
        Returns: (cleaned_output, list of warnings)
        """
        return self.output_validator.validate_output(output)