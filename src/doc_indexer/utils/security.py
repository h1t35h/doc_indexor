"""
Security utility functions for input validation and sanitization.
"""

import os
import re
from typing import Optional


class PromptSanitizer:
    """Sanitize prompts to prevent injection attacks."""

    # Patterns that could be used for prompt injection
    DANGEROUS_PATTERNS = [
        # System prompts
        r"system\s*:",
        r"assistant\s*:",
        r"user\s*:",
        r"human\s*:",
        # Common injection attempts
        r"ignore\s+previous",
        r"disregard\s+above",
        r"forget\s+everything",
        r"new\s+instructions",
        r"override\s+system",
        # Role switching attempts
        r"you\s+are\s+now",
        r"act\s+as\s+if",
        r"pretend\s+to\s+be",
        # Data extraction attempts
        r"repeat\s+everything",
        r"show\s+system\s+prompt",
        r"reveal\s+instructions",
    ]

    # Maximum safe lengths
    MAX_PROMPT_LENGTH = 1000
    MAX_TEXT_LENGTH = 10000

    @classmethod
    def sanitize_prompt(cls, prompt: str, max_length: Optional[int] = None) -> str:
        """
        Sanitize a prompt to prevent injection attacks.

        Args:
            prompt: The prompt to sanitize
            max_length: Maximum allowed length (default: MAX_PROMPT_LENGTH)

        Returns:
            Sanitized prompt
        """
        if not prompt:
            return ""

        # Limit length
        if max_length is None:
            max_length = cls.MAX_PROMPT_LENGTH
        prompt = prompt[:max_length]

        # Remove dangerous patterns (case-insensitive)
        for pattern in cls.DANGEROUS_PATTERNS:
            prompt = re.sub(pattern, "[FILTERED]", prompt, flags=re.IGNORECASE)

        # Remove excessive whitespace
        prompt = " ".join(prompt.split())

        # Escape special characters that might be interpreted as formatting
        prompt = prompt.replace("\\", "\\\\")
        prompt = prompt.replace("`", "\\`")

        return prompt

    @classmethod
    def sanitize_text(cls, text: str, max_length: Optional[int] = None) -> str:
        """
        Sanitize text content for LLM processing.

        Args:
            text: The text to sanitize
            max_length: Maximum allowed length (default: MAX_TEXT_LENGTH)

        Returns:
            Sanitized text
        """
        if not text:
            return ""

        # Limit length
        if max_length is None:
            max_length = cls.MAX_TEXT_LENGTH
        text = text[:max_length]

        # Remove potential command injections
        text = re.sub(r"</?script[^>]*>", "[SCRIPT_REMOVED]", text, flags=re.IGNORECASE)
        text = re.sub(r"javascript:", "[JS_REMOVED]", text, flags=re.IGNORECASE)

        # Remove excessive whitespace while preserving paragraph structure
        lines = text.split("\n")
        cleaned_lines = [" ".join(line.split()) for line in lines]
        text = "\n".join(cleaned_lines)

        return text

    @classmethod
    def validate_file_content(cls, content: str) -> bool:
        """
        Validate that file content is safe to process.

        Args:
            content: File content to validate

        Returns:
            True if content appears safe, False otherwise
        """
        # Check for binary content
        if "\x00" in content:
            return False

        # Check for excessive special characters (might be binary)
        special_char_ratio = sum(
            1 for c in content if ord(c) < 32 or ord(c) > 126
        ) / max(len(content), 1)
        if special_char_ratio > 0.3:
            return False

        return True


class PathValidator:
    """Validate file paths for security."""

    @staticmethod
    def is_safe_path(path: str, base_dir: str) -> bool:
        """
        Check if a path is safe (no directory traversal).

        Args:
            path: Path to validate
            base_dir: Base directory to restrict access to

        Returns:
            True if path is safe, False otherwise
        """
        from pathlib import Path

        try:
            # Resolve to absolute paths
            target = Path(path).resolve()
            base = Path(base_dir).resolve()

            # Check if target is within base
            target.relative_to(base)
            return True
        except (ValueError, Exception):
            return False

    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """
        Sanitize a filename to remove dangerous characters.

        Args:
            filename: Filename to sanitize

        Returns:
            Sanitized filename
        """
        # Remove path separators and null bytes
        filename = filename.replace("/", "_")
        filename = filename.replace("\\", "_")
        filename = filename.replace("\x00", "")

        # Remove other dangerous characters
        filename = re.sub(r'[<>:"|?*]', "_", filename)

        # Limit length
        max_length = 255
        if len(filename) > max_length:
            name, ext = os.path.splitext(filename)
            name = name[: max_length - len(ext) - 1]
            filename = name + ext

        return filename
