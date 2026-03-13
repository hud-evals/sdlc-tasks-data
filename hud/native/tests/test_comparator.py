"""Tests for the comparator module."""

from __future__ import annotations

import pytest
from fastmcp.tools.tool import FunctionTool

from hud.native.comparator import (
    CompareTool,
    ComparisonMode,
    ComparisonResult,
    DataType,
    auto_select_mode,
    comparator,
    detect_type,
    extract_boolean,
    extract_json,
    extract_list,
    extract_number,
)
from hud.tools.submit import set_submission
from hud.tools.types import EvaluationResult


class TestTypeDetection:
    """Test type detection functionality."""

    @pytest.mark.parametrize(
        "value,expected_type",
        [
            # Booleans
            ("true", DataType.BOOLEAN),
            ("True", DataType.BOOLEAN),
            ("false", DataType.BOOLEAN),
            ("False", DataType.BOOLEAN),
            # Integers
            ("42", DataType.INTEGER),
            ("-123", DataType.INTEGER),
            ("0", DataType.INTEGER),
            # Floats
            ("3.14", DataType.FLOAT),
            ("-2.5", DataType.FLOAT),
            ("1e-6", DataType.FLOAT),
            # JSON
            ('{"key": "value"}', DataType.JSON),
            ("[1, 2, 3]", DataType.JSON),
            ('{"nested": {"value": 42}}', DataType.JSON),
            # Text (fallback)
            ("hello world", DataType.TEXT),
            ("not a number", DataType.TEXT),
            ("{invalid json", DataType.TEXT),
            ("", DataType.TEXT),
        ],
    )
    def test_detect_type(self, value, expected_type):
        """Test type detection for various inputs."""
        assert detect_type(value) == expected_type

    def test_detect_type_none(self):
        """Test type detection for None."""
        assert detect_type(None) == DataType.TEXT


class TestAutoSelectMode:
    """Test automatic mode selection based on types."""

    @pytest.mark.parametrize(
        "value_type,ref_type,expected_mode",
        [
            # JSON gets semantic
            (DataType.JSON, DataType.JSON, ComparisonMode.SEMANTIC),
            (DataType.JSON, DataType.TEXT, ComparisonMode.SEMANTIC),
            (DataType.TEXT, DataType.JSON, ComparisonMode.SEMANTIC),
            # Numeric gets numeric
            (DataType.INTEGER, DataType.INTEGER, ComparisonMode.NUMERIC),
            (DataType.FLOAT, DataType.FLOAT, ComparisonMode.NUMERIC),
            (DataType.INTEGER, DataType.FLOAT, ComparisonMode.NUMERIC),
            (DataType.FLOAT, DataType.TEXT, ComparisonMode.NUMERIC),
            # Boolean gets exact
            (DataType.BOOLEAN, DataType.BOOLEAN, ComparisonMode.EXACT),
            (DataType.BOOLEAN, DataType.TEXT, ComparisonMode.EXACT),
            # Text gets fuzzy
            (DataType.TEXT, DataType.TEXT, ComparisonMode.FUZZY),
        ],
    )
    def test_auto_select_mode(self, value_type, ref_type, expected_mode):
        """Test mode selection logic."""
        assert auto_select_mode(value_type, ref_type) == expected_mode


class TestCompareTool:
    """Test the main CompareTool class."""

    @pytest.mark.asyncio
    async def test_scalar_comparison(self):
        """Test comparing scalar values."""
        tool = CompareTool()

        # Exact match
        result = await tool("hello", "hello", mode=ComparisonMode.EXACT)
        assert isinstance(result, EvaluationResult)
        assert result.done
        assert result.reward == 1.0
        assert "Single Exact: 1/1 matches" in result.content if result.content else False

        # Fuzzy match
        result = await tool("hello world", "hello wrld", mode=ComparisonMode.FUZZY)
        assert isinstance(result, EvaluationResult)
        assert result.done
        assert 0.8 < result.reward < 1.0
        assert "Single Fuzzy: 1/1 matches" in result.content if result.content else False

    @pytest.mark.asyncio
    async def test_list_comparison(self):
        """Test comparing lists of values."""
        tool = CompareTool()

        # Exact lists
        result = await tool(["a", "b", "c"], ["a", "b", "c"], mode=ComparisonMode.EXACT)
        assert result.done
        assert result.reward == 1.0
        assert "Batch Exact: 3/3 matches" in result.content if result.content else False

        # Partial match
        result = await tool(["a", "b", "c"], ["a", "x", "c"], mode=ComparisonMode.EXACT)
        assert not result.done
        assert result.reward < 1.0
        assert "Batch Exact: 2/3 matches" in result.content if result.content else False

    @pytest.mark.asyncio
    async def test_broadcast_comparison(self):
        """Test broadcasting single value against list."""
        tool = CompareTool()

        # Single value vs list
        result = await tool("a", ["a", "a", "a"], mode=ComparisonMode.EXACT)
        assert result.done
        assert result.reward == 1.0
        assert "Broadcast Exact: 3/3 matches" in result.content if result.content else False

        # List vs single value
        result = await tool(["x", "x", "x"], "x", mode=ComparisonMode.EXACT)
        assert result.done
        assert result.reward == 1.0
        assert "Broadcast Exact: 3/3 matches" in result.content if result.content else False

    @pytest.mark.asyncio
    async def test_submission_fallback(self):
        """Test using submission when value is None."""
        tool = CompareTool()

        # Set submission
        set_submission("test value")

        # Compare without providing value
        result = await tool(None, "test value", mode=ComparisonMode.EXACT)
        assert result.done
        assert result.reward == 1.0

        # Clear submission
        set_submission(None)

    @pytest.mark.asyncio
    async def test_error_cases(self):
        """Test error handling."""
        tool = CompareTool()

        # Missing reference
        result = await tool("value", None)
        assert result.isError
        assert "Missing value or reference" in result.content if result.content else False

        # Mismatched list lengths
        result = await tool([1, 2], [1, 2, 3])
        assert not result.done
        assert result.reward == 0.0
        assert "Mismatched lengths" in result.content if result.content else False


class TestCompareToolModes:
    """Test different comparison modes in CompareTool."""

    @pytest.mark.asyncio
    async def test_auto_mode(self):
        """Test automatic mode detection."""
        tool = CompareTool()

        # Numbers should use numeric
        result = await tool("42", "42.0", mode=ComparisonMode.AUTO)
        assert result.done
        assert result.reward == 1.0
        assert "Numeric" in result.content if result.content else False

        # JSON should use semantic
        result = await tool('{"a": 1}', '{"a": 1}', mode=ComparisonMode.AUTO)
        assert result.done
        assert result.reward == 1.0
        assert "Semantic" in result.content if result.content else False

    @pytest.mark.asyncio
    async def test_exact_mode(self):
        """Test exact string comparison."""
        tool = CompareTool()

        # Exact match
        result = await tool("hello", "hello", mode=ComparisonMode.EXACT)
        assert result.done
        assert result.reward == 1.0

        # Different strings
        result = await tool("42", "42.0", mode=ComparisonMode.EXACT)
        assert not result.done
        assert result.reward == 0.0

    @pytest.mark.asyncio
    async def test_fuzzy_mode(self):
        """Test fuzzy text matching."""
        tool = CompareTool()

        # High similarity
        result = await tool("hello world", "hello wrld", mode=ComparisonMode.FUZZY, threshold=0.8)
        assert result.done
        assert result.reward > 0.8

        # Low similarity
        result = await tool("hello", "goodbye", mode=ComparisonMode.FUZZY, threshold=0.9)
        assert not result.done
        assert result.reward < 0.5

    @pytest.mark.asyncio
    async def test_numeric_mode(self):
        """Test numeric comparison with tolerance."""
        tool = CompareTool()

        # Within tolerance
        result = await tool("1.0", "1.000001", mode=ComparisonMode.NUMERIC, tolerance=1e-5)
        assert result.done
        assert result.reward > 0.99

        # Outside tolerance
        result = await tool("1.0", "2.0", mode=ComparisonMode.NUMERIC, tolerance=0.1)
        assert not result.done
        assert result.reward <= 0.5

    @pytest.mark.asyncio
    async def test_semantic_mode(self):
        """Test semantic comparison for various types."""
        tool = CompareTool()

        # JSON objects (same structure)
        result = await tool('{"b": 2, "a": 1}', '{"a": 1, "b": 2}', mode=ComparisonMode.SEMANTIC)
        assert result.done
        assert result.reward == 1.0

        # JSON arrays
        result = await tool("[1, 2, 3]", "[1, 2, 3]", mode=ComparisonMode.SEMANTIC)
        assert result.done
        assert result.reward == 1.0

        # Numbers via semantic (uses numeric comparison)
        result = await tool("42", "42.0", mode=ComparisonMode.SEMANTIC, tolerance=1e-6)
        assert result.done
        assert result.reward == 1.0

        # Booleans via semantic
        result = await tool("true", "true", mode=ComparisonMode.SEMANTIC)
        assert result.done
        assert result.reward == 1.0

        # Text fallback when not JSON
        result = await tool(
            "hello world", "hello wrld", mode=ComparisonMode.SEMANTIC, threshold=0.8
        )
        assert result.done
        assert result.reward > 0.8


class TestComparisonResult:
    """Test ComparisonResult model."""

    def test_comparison_result_fields(self):
        """Test ComparisonResult has all expected fields."""
        result = ComparisonResult(
            matches=True,
            similarity=0.95,
            detected_type=DataType.TEXT,
            comparison_mode=ComparisonMode.FUZZY,
            details={"threshold": 0.8},
        )

        assert result.matches is True
        assert result.similarity == 0.95
        assert result.detected_type == DataType.TEXT
        assert result.comparison_mode == ComparisonMode.FUZZY
        assert result.details["threshold"] == 0.8

    def test_comparison_result_validation(self):
        """Test ComparisonResult validation."""
        # Valid similarity
        result = ComparisonResult(
            matches=True,
            similarity=1.0,
            detected_type=DataType.TEXT,
            comparison_mode=ComparisonMode.EXACT,
        )
        assert result.similarity == 1.0

        # Invalid similarity should raise
        with pytest.raises(ValueError):
            ComparisonResult(
                matches=True,
                similarity=1.5,  # > 1.0
                detected_type=DataType.TEXT,
                comparison_mode=ComparisonMode.EXACT,
            )


class TestAliasTools:
    """Test the alias tools created by make_alias_tool."""

    @pytest.mark.asyncio
    async def test_aliases_work(self):
        """Test that aliases are properly registered and work."""
        from hud.native.comparator import comparator

        # Check that aliases are registered
        tool_names = [t.name for t in comparator._tool_manager._tools.values()]

        expected_aliases = [
            "compare_exact",
            "compare_text",
            "compare_string",
            "compare_numeric",
            "compare_float",
            "compare_int",
            "compare_json",
            "compare_boolean",
            "compare_list",
        ]

        for alias in expected_aliases:
            assert alias in tool_names, f"Alias {alias} not found in registered tools"


class TestExtraction:
    """Test extraction functions for handling LLM outputs."""

    def test_json_extraction(self):
        """Test JSON extraction from text."""
        # Already valid JSON
        assert extract_json('{"a": 1}') == '{"a": 1}'
        assert extract_json("[1, 2, 3]") == "[1, 2, 3]"

        # JSON embedded in text
        assert extract_json('The answer is {"result": 42}') == '{"result": 42}'
        assert extract_json('First {"a": 1} then {"b": 2}') == '{"b": 2}'  # Last one
        assert extract_json("The list is [1, 2, 3] and done") == "[1, 2, 3]"

        # Complex nested JSON
        complex_json = """{
            "status": "success",
            "data": {
                "values": [1, 2, 3],
                "metadata": {
                    "timestamp": "2024-01-01",
                    "version": "1.0"
                }
            }
        }"""

        llm_output = f"""
        Let me analyze this request.
        
        The final result is:
        {complex_json}
        
        This completes the analysis.
        """

        extracted = extract_json(llm_output)
        import json

        assert json.loads(extracted) == json.loads(complex_json)

        # No JSON
        assert extract_json("No JSON here") == "No JSON here"

    def test_number_extraction(self):
        """Test number extraction from text."""
        # Plain numbers
        assert extract_number("42") == "42"
        assert extract_number("3.14") == "3.14"
        assert extract_number("-123") == "-123"
        assert extract_number("1.5e-10") == "1.5e-10"

        # Numbers in text
        assert extract_number("The answer is 42") == "42"
        assert extract_number("First 10 then 20") == "20"  # Last one
        assert extract_number("Value: 3.14159") == "3.14159"

        # Integer extraction
        assert extract_number("42.0", "int") == "42"
        assert extract_number("The count is 42.0 items", "int") == "42"

        # No numbers
        assert extract_number("No numbers here") == "No numbers here"

    def test_boolean_extraction(self):
        """Test boolean extraction from text."""
        # Plain booleans
        assert extract_boolean("true") == "true"
        assert extract_boolean("false") == "false"
        assert extract_boolean("True") == "true"
        assert extract_boolean("FALSE") == "false"

        # Booleans in text
        assert extract_boolean("The answer is True") == "true"
        assert extract_boolean("First false then TRUE") == "true"  # Last one

        # No booleans
        assert extract_boolean("No booleans here") == "No booleans here"

    def test_list_extraction(self):
        """Test list extraction (uses JSON extraction)."""
        assert extract_list("[1, 2, 3]") == "[1, 2, 3]"
        assert extract_list('The array is ["a", "b", "c"]') == '["a", "b", "c"]'
        assert extract_list("No lists here") == "No lists here"


class TestAliasPreprocessing:
    """Test that alias tools correctly preprocess LLM outputs."""

    @pytest.mark.asyncio
    async def test_json_alias_preprocessing(self):
        """Test JSON extraction in compare_json tool."""
        tools = {t.name: t for t in comparator._tool_manager._tools.values()}
        json_tool = tools["compare_json"]

        assert isinstance(json_tool, FunctionTool)
        result = await json_tool.fn(
            value='The model thinks the answer is {"result": 42, "confidence": 0.9}',
            reference='{"result": 42, "confidence": 0.9}',
        )
        assert result.done
        assert result.reward == 1.0
        assert "Semantic" in result.content

    @pytest.mark.asyncio
    async def test_numeric_alias_preprocessing(self):
        """Test number extraction in numeric tools."""
        tools = {t.name: t for t in comparator._tool_manager._tools.values()}

        # Float tool
        float_tool = tools["compare_float"]
        assert isinstance(float_tool, FunctionTool)
        result = await float_tool.fn(
            value="After careful calculation, the answer is 3.14159", reference="3.14159"
        )
        assert result.done
        assert result.reward == 1.0
        assert "Numeric" in result.content

        # Integer tool
        int_tool = tools["compare_int"]
        assert isinstance(int_tool, FunctionTool)
        result = await int_tool.fn(value="The count is exactly 42 items", reference="42")
        assert result.done
        assert result.reward == 1.0
        assert "Numeric" in result.content

    @pytest.mark.asyncio
    async def test_boolean_alias_preprocessing(self):
        """Test boolean extraction in compare_boolean tool."""
        tools = {t.name: t for t in comparator._tool_manager._tools.values()}
        bool_tool = tools["compare_boolean"]

        assert isinstance(bool_tool, FunctionTool)
        result = await bool_tool.fn(
            value="Based on the analysis, the statement is TRUE", reference="true"
        )
        assert result.done
        assert result.reward == 1.0
        assert "Exact" in result.content

    @pytest.mark.asyncio
    async def test_list_alias_preprocessing(self):
        """Test list extraction in compare_list tool."""
        tools = {t.name: t for t in comparator._tool_manager._tools.values()}
        list_tool = tools["compare_list"]

        assert isinstance(list_tool, FunctionTool)
        result = await list_tool.fn(
            value="The sorted results are [1, 2, 3, 4, 5]", reference="[1, 2, 3, 4, 5]"
        )
        assert result.done
        assert result.reward == 1.0
        assert "Semantic" in result.content

    @pytest.mark.asyncio
    async def test_complex_llm_output(self):
        """Test extraction from complex LLM outputs with reasoning."""
        tools = {t.name: t for t in comparator._tool_manager._tools.values()}
        json_tool = tools["compare_json"]

        llm_output = """
        Let me analyze this request step by step.

        First, I'll process the data:
        - Item 1: processed
        - Item 2: processed with value 42

        After careful consideration, the final result is:
        {
            "status": "success",
            "data": {
                "values": [1, 2, 3],
                "metadata": {
                    "timestamp": "2024-01-01",
                    "version": "1.0"
                }
            }
        }
        
        This completes the analysis. The JSON above contains all the required information.
        """

        reference = """
        {"status": "success", "data": {"values": [1, 2, 3],
        "metadata": {"timestamp": "2024-01-01", "version": "1.0"}}}
        """

        assert isinstance(json_tool, FunctionTool)
        result = await json_tool.fn(value=llm_output, reference=reference)
        assert result.done
        assert result.reward == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
