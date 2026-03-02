"""Universal type-aware comparator MCP server.

This server provides comparison capabilities that automatically detect
and handle different data types (text, int, float, json) with lenient
parsing and multiple comparison strategies.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from difflib import SequenceMatcher
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field

from hud.server import MCPServer
from hud.tools import BaseTool, SubmitTool
from hud.tools.submit import get_submission
from hud.tools.types import EvaluationResult

# Configure logging
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


class DataType(str, Enum):
    """Detected data types."""

    TEXT = "text"
    INTEGER = "integer"
    FLOAT = "float"
    JSON = "json"
    BOOLEAN = "boolean"


class ComparisonMode(str, Enum):
    """Available comparison modes."""

    AUTO = "auto"  # Auto-detect based on type
    EXACT = "exact"  # Exact match
    FUZZY = "fuzzy"  # Fuzzy text matching with similarity threshold
    NUMERIC = "numeric"  # Numeric comparison with tolerance
    SEMANTIC = "semantic"  # Semantic equivalence (JSON structures)


class ComparisonResult(BaseModel):
    """Result of a comparison operation."""

    matches: bool
    similarity: float = Field(ge=0.0, le=1.0)
    detected_type: DataType
    comparison_mode: ComparisonMode
    details: dict[str, Any] = Field(default_factory=dict)


# Extraction functions for handling LLM outputs
def extract_json(text: str) -> str:
    """Extract the last valid JSON object or array from text."""
    if not text:
        return text

    # First, try if the whole string is valid JSON
    try:
        json.loads(text)
        return text
    except (json.JSONDecodeError, TypeError):
        pass

    # Strategy: Find all { or [ characters and try to parse from each one
    candidates = []

    # Find all potential JSON starting points
    for i, char in enumerate(text):
        if char in "{[":
            # Try to find matching closing bracket
            bracket_count = 0
            in_string = False
            escape_next = False

            for j in range(i, len(text)):
                current_char = text[j]

                if escape_next:
                    escape_next = False
                    continue

                if current_char == "\\":
                    escape_next = True
                    continue

                if current_char == '"' and not escape_next:
                    in_string = not in_string
                    continue

                if not in_string:
                    if current_char in "{[":
                        bracket_count += 1
                    elif current_char in "}]":
                        bracket_count -= 1

                        if bracket_count == 0:
                            # Found matching bracket
                            candidate = text[i : j + 1]
                            try:
                                json.loads(candidate)
                                candidates.append((j + 1, candidate))
                            except (json.JSONDecodeError, TypeError):
                                pass
                            break

    # Return the last valid JSON found
    if candidates:
        candidates.sort(key=lambda x: x[0])
        return candidates[-1][1]

    return text


def extract_number(text: str, number_type: Literal["int", "float"] = "float") -> str:
    """Extract the last number from text."""
    if not text:
        return text

    # Pattern for numbers (including scientific notation)
    number_pattern = r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?"

    matches = list(re.finditer(number_pattern, text))
    if matches:
        last_number = matches[-1].group()

        # For int type, ensure we don't have decimals
        if number_type == "int":
            try:
                # Check if it's actually an integer
                float_val = float(last_number)
                if float_val.is_integer():
                    return str(int(float_val))
            except ValueError:
                pass

        return last_number

    return text


def extract_boolean(text: str) -> str:
    """Extract the last boolean value from text."""
    if not text:
        return text

    # Look for boolean values (case insensitive)
    bool_pattern = r"\b(true|false|True|False|TRUE|FALSE)\b"

    matches = list(re.finditer(bool_pattern, text))
    if matches:
        return matches[-1].group().lower()

    return text


def extract_list(text: str) -> str:
    """Extract the last list/array from text."""
    # For lists, we use the same logic as JSON extraction
    # since lists are JSON arrays
    return extract_json(text)


def _compare_exact(value: Any, reference: Any, **kwargs: Any) -> tuple[bool, float, dict]:
    """Exact comparison."""
    matches = value == reference
    return matches, 1.0 if matches else 0.0, {}


def _compare_fuzzy(
    value: Any, reference: Any, threshold: float = 0.8, **kwargs: Any
) -> tuple[bool, float, dict]:
    """Fuzzy text comparison."""
    str_value = str(value).strip()
    str_reference = str(reference).strip()
    similarity = SequenceMatcher(None, str_value, str_reference).ratio()
    return similarity >= threshold, similarity, {"threshold": threshold}


def _compare_numeric(
    value: Any, reference: Any, tolerance: float = 1e-6, **kwargs: Any
) -> tuple[bool, float, dict]:
    """Numeric comparison with tolerance."""
    try:
        num_value = float(value)
        num_reference = float(reference)

        abs_diff = abs(num_value - num_reference)
        rel_diff = abs_diff / max(abs(num_reference), 1e-10)

        matches = abs_diff <= tolerance or rel_diff <= tolerance
        similarity = max(0.0, 1.0 - rel_diff)

        return (
            matches,
            similarity,
            {
                "absolute_difference": abs_diff,
                "relative_difference": rel_diff,
                "tolerance": tolerance,
            },
        )
    except (ValueError, TypeError):
        return False, 0.0, {"error": "non-numeric values"}


def _compare_semantic(
    value: Any, reference: Any, threshold: float = 0.8, tolerance: float = 1e-6, **kwargs: Any
) -> tuple[bool, float, dict]:
    """Semantic comparison for JSON-parseable structures."""
    # Try to parse as JSON
    try:
        v_obj = json.loads(value) if isinstance(value, str) else value
        r_obj = json.loads(reference) if isinstance(reference, str) else reference
    except (json.JSONDecodeError, TypeError):
        # Not valid JSON - fall back to fuzzy text comparison
        return _compare_fuzzy(value, reference, threshold)

    # Now dispatch based on parsed types
    if isinstance(v_obj, dict) and isinstance(r_obj, dict):
        # Dictionary comparison
        try:
            norm_value = json.dumps(v_obj, sort_keys=True)
            norm_reference = json.dumps(r_obj, sort_keys=True)
            matches = norm_value == norm_reference

            # Calculate similarity based on keys and values
            common_keys = set(v_obj.keys()) & set(r_obj.keys())
            all_keys = set(v_obj.keys()) | set(r_obj.keys())
            key_similarity = len(common_keys) / len(all_keys) if all_keys else 1.0

            if common_keys:
                matching_values = sum(1 for k in common_keys if v_obj.get(k) == r_obj.get(k))
                value_similarity = matching_values / len(common_keys)
                similarity = (key_similarity + value_similarity) / 2
            else:
                similarity = key_similarity

            return matches, similarity, {"type": "dict"}
        except Exception as e:
            return False, 0.0, {"error": str(e)}

    elif isinstance(v_obj, list) and isinstance(r_obj, list):
        # List comparison - element by element
        matches = v_obj == r_obj
        if len(v_obj) == len(r_obj) == 0:
            return True, 1.0, {"type": "list", "length": 0}

        # Similarity based on matching positions
        if len(v_obj) == len(r_obj):
            matching = sum(1 for a, b in zip(v_obj, r_obj, strict=False) if a == b)
            similarity = matching / len(v_obj)
        else:
            similarity = 0.0

        return matches, similarity, {"type": "list", "length": len(v_obj)}

    elif isinstance(v_obj, (int | float)) and isinstance(r_obj, (int | float)):
        # Numeric comparison
        return _compare_numeric(v_obj, r_obj, tolerance)

    elif isinstance(v_obj, bool) and isinstance(r_obj, bool):
        # Boolean comparison
        return _compare_exact(v_obj, r_obj)

    elif isinstance(v_obj, str) and isinstance(r_obj, str):
        # String comparison - could be fuzzy or exact
        return _compare_fuzzy(v_obj, r_obj, threshold)

    else:
        # Different types or other cases - exact comparison
        return _compare_exact(v_obj, r_obj)


# Map modes to comparison functions
COMPARISON_FUNCTIONS = {
    ComparisonMode.EXACT: _compare_exact,
    ComparisonMode.FUZZY: _compare_fuzzy,
    ComparisonMode.NUMERIC: _compare_numeric,
    ComparisonMode.SEMANTIC: _compare_semantic,
}


def detect_type(value: str | None = None) -> DataType:
    """Detect the data type of a string value."""
    if value is None:
        return DataType.TEXT

    # Try boolean
    if value.lower() in ("true", "false"):
        return DataType.BOOLEAN

    # Try JSON (dict or list)
    try:
        parsed = json.loads(value)
        if isinstance(parsed, (dict | list)):
            return DataType.JSON
        # Continue checking if it's a JSON primitive
        value = str(parsed)
    except (json.JSONDecodeError, TypeError):
        pass

    # Try integer
    try:
        int(value)
        return DataType.INTEGER
    except ValueError:
        pass

    # Try float
    try:
        float(value)
        return DataType.FLOAT
    except ValueError:
        pass

    # Default to text
    return DataType.TEXT


def auto_select_mode(value_type: DataType, ref_type: DataType) -> ComparisonMode:
    """Auto-select comparison mode based on detected types."""
    # If either is JSON, use semantic
    if DataType.JSON in (value_type, ref_type):
        return ComparisonMode.SEMANTIC

    # If either is numeric, use numeric
    if value_type in (DataType.INTEGER, DataType.FLOAT) or ref_type in (
        DataType.INTEGER,
        DataType.FLOAT,
    ):
        return ComparisonMode.NUMERIC

    # Booleans use exact
    if value_type == DataType.BOOLEAN or ref_type == DataType.BOOLEAN:
        return ComparisonMode.EXACT

    # Default to fuzzy for text
    return ComparisonMode.FUZZY


class CompareTool(BaseTool):
    """Universal comparison tool with mode selection."""

    name = "compare"
    title = "Compare Tool"
    description = "Compare values with explicit or automatic mode selection"

    async def __call__(
        self,
        value: Any | list[Any] | None = None,
        reference: Any | list[Any] | None = None,
        mode: ComparisonMode = ComparisonMode.AUTO,
        threshold: float = 0.8,
        tolerance: float = 1e-6,
    ) -> EvaluationResult:
        """Compare values with specified or auto-detected mode."""
        # Get value from submission if not provided
        if value is None:
            value = get_submission()

        # Normalize inputs to lists
        if value is None or reference is None:
            return EvaluationResult(
                reward=0.0, done=False, content="Missing value or reference", isError=True
            )

        # Convert to lists
        val_list = value if isinstance(value, list) else [value]
        ref_list = reference if isinstance(reference, list) else [reference]

        # Check list compatibility
        comp_type = "scalar"
        if isinstance(value, list) and isinstance(reference, list):
            if len(val_list) != len(ref_list):
                return EvaluationResult(
                    reward=0.0,
                    done=False,
                    content=f"Error: Mismatched lengths - {len(val_list)} values vs {len(ref_list)} references",  # noqa: E501
                )
            comp_type = "batch"
        elif isinstance(value, list) and not isinstance(reference, list):
            ref_list = [reference] * len(val_list)
            comp_type = "broadcast"
        elif not isinstance(value, list) and isinstance(reference, list):
            val_list = [value] * len(ref_list)
            comp_type = "broadcast"

        # Process each pair
        results = []
        for v, r in zip(val_list, ref_list, strict=False):
            # Convert to strings
            v_str, r_str = str(v), str(r)

            # Determine comparison mode
            if mode == ComparisonMode.AUTO:
                v_type = detect_type(v_str)
                r_type = detect_type(r_str)
                comparison_mode = auto_select_mode(v_type, r_type)
                detected_type = v_type if v_type == r_type else DataType.TEXT
            else:
                comparison_mode = mode
                detected_type = DataType.TEXT

            # For exact mode, skip parsing and compare raw strings
            if comparison_mode == ComparisonMode.EXACT and mode == ComparisonMode.EXACT:
                matches = v_str == r_str
                result = ComparisonResult(
                    matches=matches,
                    similarity=1.0 if matches else 0.0,
                    detected_type=detected_type,
                    comparison_mode=comparison_mode,
                )
            else:
                # Get comparison function and run it
                compare_fn = COMPARISON_FUNCTIONS[comparison_mode]
                matches, similarity, details = compare_fn(
                    v_str, r_str, threshold=threshold, tolerance=tolerance
                )

                result = ComparisonResult(
                    matches=matches,
                    similarity=similarity,
                    detected_type=detected_type,
                    comparison_mode=comparison_mode,
                    details=details,
                )

            results.append(result)

        # Aggregate results
        if not results:
            return EvaluationResult(reward=0.0, done=False, content="No comparisons performed")

        total_similarity = sum(r.similarity for r in results)
        avg_similarity = total_similarity / len(results)
        all_match = all(r.matches for r in results)
        match_count = sum(1 for r in results if r.matches)

        # Format content
        prefix = {"scalar": "Single", "batch": "Batch", "broadcast": "Broadcast"}.get(
            comp_type, "Unknown"
        )
        mode_name = results[0].comparison_mode.value.capitalize()

        return EvaluationResult(
            reward=avg_similarity,
            done=all_match,
            content=f"{prefix} {mode_name}: {match_count}/{len(results)} matches, avg={avg_similarity:.3f}",  # noqa: E501
        )


# Map of specific aliases to their preprocessing needs
ALIAS_PREPROCESSORS = {
    "compare_json": lambda v: extract_json(v),
    "compare_int": lambda v: extract_number(v, "int"),
    "compare_float": lambda v: extract_number(v, "float"),
    "compare_boolean": lambda v: extract_boolean(v),
    "compare_list": lambda v: extract_list(v),
}


# Helper to create alias tool classes
def make_alias_tool(name: str, preset_mode: ComparisonMode, description: str) -> type[BaseTool]:
    """Create an alias tool class that presets the mode."""

    class AliasTool(BaseTool):
        def __init__(self) -> None:
            super().__init__(
                name=name,
                title=f"Compare ({preset_mode.capitalize()})",
                description=description + " (auto-handles lists, extracts from outputs)",
            )

        async def __call__(
            self,
            value: Any | list[Any] | None = None,
            reference: Any | list[Any] | None = None,
            threshold: float = 0.8,
            tolerance: float = 1e-6,
        ) -> EvaluationResult:
            """Alias that calls compare with preset mode."""
            # Apply specific preprocessing if this alias has one
            if value is not None and name in ALIAS_PREPROCESSORS:
                preprocessor = ALIAS_PREPROCESSORS[name]
                if isinstance(value, list):
                    value = [preprocessor(str(v)) for v in value]
                else:
                    value = preprocessor(str(value))

            tool = CompareTool()
            return await tool(
                value=value,
                reference=reference,
                mode=preset_mode,
                threshold=threshold,
                tolerance=tolerance,
            )

    return AliasTool


# Create MCP server
comparator = MCPServer(name="comparator")

# Register main tool
comparator.add_tool(SubmitTool())
comparator.add_tool(CompareTool())

# Register aliases - these are just thin wrappers
ALIASES = [
    ("compare_exact", ComparisonMode.EXACT, "Exact string comparison"),
    ("compare_text", ComparisonMode.FUZZY, "Fuzzy text comparison"),
    ("compare_string", ComparisonMode.FUZZY, "Fuzzy string comparison (alias for text)"),
    ("compare_numeric", ComparisonMode.NUMERIC, "Numeric comparison with tolerance"),
    ("compare_float", ComparisonMode.NUMERIC, "Float comparison (alias for numeric)"),
    ("compare_int", ComparisonMode.NUMERIC, "Integer comparison (alias for numeric)"),
    ("compare_json", ComparisonMode.SEMANTIC, "Semantic JSON comparison"),
    ("compare_boolean", ComparisonMode.EXACT, "Boolean comparison (exact match)"),
    ("compare_list", ComparisonMode.SEMANTIC, "List comparison (alias for CompareTool)"),
]

for name, mode, desc in ALIASES:
    AliasTool = make_alias_tool(name, mode, desc)
    comparator.add_tool(AliasTool())

# Export for mounting
__all__ = ["comparator"]


if __name__ == "__main__":
    # Run as standalone server
    logger.info("Starting Comparator MCP Server...")
    comparator.run()
