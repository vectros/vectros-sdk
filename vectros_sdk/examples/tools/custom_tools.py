"""
Custom Tools for AIOS Vectros SDK Agents.

Demonstrates creating custom tools subclassing `BaseTool` with JSON schema
definitions (`get_tool_call_format`), parameter validation, and execution logic (`run`).
"""

import json
import math
from typing import Any, Dict, Optional, Type

from vectros_sdk.tool.core.base import BaseTool
from vectros_sdk.tool.core.registry import (
    TOOL_REGISTRY,
    get_tool,
    list_registered_tools,
    register_tool,
)


class MathEvaluatorTool(BaseTool):
    """
    Custom tool for safely calculating mathematical expressions and statistical operations.
    """

    name = "math_evaluator"
    description = "Evaluates mathematical expressions and statistical summaries (sum, avg, min, max, std)."

    def get_tool_call_format(self) -> Dict[str, Any]:
        """
        Return the JSON schema definition for tool call validation.
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "operation": {
                            "type": "string",
                            "enum": ["eval", "sum", "average", "min", "max", "variance"],
                            "description": "Mathematical operation to perform.",
                        },
                        "numbers": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": "List of numbers for array-based operations.",
                        },
                        "expression": {
                            "type": "string",
                            "description": "Arithmetic expression (for operation='eval').",
                        },
                    },
                    "required": ["operation"],
                },
            },
        }

    def get_schema(self) -> Dict[str, Any]:
        """Alias for get_tool_call_format."""
        return self.get_tool_call_format()

    def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the math tool with provided parameters dictionary.
        """
        op = params.get("operation", "eval")
        numbers = params.get("numbers", [])
        expression = params.get("expression", "")

        if op == "eval":
            if not expression:
                return {"success": False, "error": "Expression is required for eval operation"}
            allowed_names = {
                "sin": math.sin,
                "cos": math.cos,
                "sqrt": math.sqrt,
                "pow": pow,
                "pi": math.pi,
                "e": math.e,
                "abs": abs,
                "round": round,
            }
            try:
                result = eval(expression, {"__builtins__": None}, allowed_names)  # pylint: disable=eval-used
                return {"success": True, "operation": op, "result": result}
            except Exception as exc:
                return {"success": False, "error": f"Evaluation error: {exc}"}

        if not numbers:
            return {"success": False, "error": "Numbers array required for statistical operations"}

        if op == "sum":
            return {"success": True, "operation": op, "result": sum(numbers)}
        elif op == "average":
            return {"success": True, "operation": op, "result": sum(numbers) / len(numbers)}
        elif op == "min":
            return {"success": True, "operation": op, "result": min(numbers)}
        elif op == "max":
            return {"success": True, "operation": op, "result": max(numbers)}
        elif op == "variance":
            mean = sum(numbers) / len(numbers)
            var = sum((x - mean) ** 2 for x in numbers) / len(numbers)
            return {"success": True, "operation": op, "result": var}

        return {"success": False, "error": f"Unsupported operation: {op}"}

    def execute(self, **kwargs: Any) -> Dict[str, Any]:
        """Convenience execution wrapper accepting keyword arguments."""
        return self.run(kwargs)


class DataFormatterTool(BaseTool):
    """
    Custom tool for formatting raw structured datasets into Markdown tables or formatted JSON.
    """

    name = "data_formatter"
    description = "Formats structured dictionary and list data into clean Markdown tables or formatted JSON."

    def get_tool_call_format(self) -> Dict[str, Any]:
        """
        Return the JSON schema definition for tool parameter validation.
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "format_type": {
                            "type": "string",
                            "enum": ["markdown_table", "pretty_json", "csv"],
                            "description": "Target output format.",
                        },
                        "data": {
                            "type": "array",
                            "items": {"type": "object"},
                            "description": "List of dictionary records to format.",
                        },
                    },
                    "required": ["format_type", "data"],
                },
            },
        }

    def get_schema(self) -> Dict[str, Any]:
        """Alias for get_tool_call_format."""
        return self.get_tool_call_format()

    def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute formatting operation.
        """
        format_type = params.get("format_type", "markdown_table")
        data = params.get("data", [])

        if not isinstance(data, list) or not data:
            return {"success": False, "error": "Data must be a non-empty list of dictionaries"}

        if format_type == "pretty_json":
            return {"success": True, "formatted": json.dumps(data, indent=2)}

        headers = list(data[0].keys())
        if format_type == "csv":
            rows = [",".join(headers)]
            for item in data:
                rows.append(",".join(str(item.get(h, "")) for h in headers))
            return {"success": True, "formatted": "\n".join(rows)}

        if format_type == "markdown_table":
            header_row = "| " + " | ".join(headers) + " |"
            sep_row = "| " + " | ".join(["---"] * len(headers)) + " |"
            data_rows = ["| " + " | ".join(str(item.get(h, "")) for h in headers) + " |" for item in data]
            table = "\n".join([header_row, sep_row] + data_rows)
            return {"success": True, "formatted": table}

        return {"success": False, "error": f"Unsupported format_type: {format_type}"}

    def execute(self, **kwargs: Any) -> Dict[str, Any]:
        """Convenience execution wrapper accepting keyword arguments."""
        return self.run(kwargs)


class SentimentAnalyzerTool(BaseTool):
    """
    Custom tool for analyzing sentiment, confidence, and extracting key tags from text.
    """

    name = "sentiment_analyzer"
    description = "Analyzes text content for positive/neutral/negative sentiment and key themes."

    def get_tool_call_format(self) -> Dict[str, Any]:
        """
        Return the JSON schema definition for parameter validation.
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "The text to analyze.",
                        },
                    },
                    "required": ["text"],
                },
            },
        }

    def get_schema(self) -> Dict[str, Any]:
        """Alias for get_tool_call_format."""
        return self.get_tool_call_format()

    def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute sentiment analysis.
        """
        text = params.get("text", "").lower()
        if not text:
            return {"success": False, "error": "Text is required for analysis"}

        pos_words = {"great", "excellent", "good", "fast", "success", "optimal", "superb", "secure", "passed"}
        neg_words = {"bad", "poor", "slow", "error", "fail", "failed", "bug", "crash", "corrupted", "critical"}

        words = set(text.replace(".", " ").replace(",", " ").split())
        pos_matches = words.intersection(pos_words)
        neg_matches = words.intersection(neg_words)

        score = len(pos_matches) - len(neg_matches)
        if score > 0:
            sentiment = "positive"
            confidence = min(0.6 + 0.1 * score, 0.99)
        elif score < 0:
            sentiment = "negative"
            confidence = min(0.6 + 0.1 * abs(score), 0.99)
        else:
            sentiment = "neutral"
            confidence = 0.75

        return {
            "success": True,
            "sentiment": sentiment,
            "score": score,
            "confidence": round(confidence, 2),
            "positive_indicators": list(pos_matches),
            "negative_indicators": list(neg_matches),
        }

    def execute(self, **kwargs: Any) -> Dict[str, Any]:
        """Convenience execution wrapper accepting keyword arguments."""
        return self.run(kwargs)


def register_custom_tools(target_registry: Optional[Dict[str, Type[Any]]] = None) -> None:
    """
    Register default custom tool classes into the global or specified registry.

    Args:
        target_registry: Optional target registry dictionary (defaults to TOOL_REGISTRY).
    """
    if target_registry is not None:
        target_registry["math_evaluator"] = MathEvaluatorTool
        target_registry["data_formatter"] = DataFormatterTool
        target_registry["sentiment_analyzer"] = SentimentAnalyzerTool
    else:
        register_tool("math_evaluator", MathEvaluatorTool)
        register_tool("data_formatter", DataFormatterTool)
        register_tool("sentiment_analyzer", SentimentAnalyzerTool)
