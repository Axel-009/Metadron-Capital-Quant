"""LLM factory for creating language model instances.

This module provides a factory function to create LLM instances with intelligent model selection.
"""

import logging
import os
from typing import Any

from langchain_community.llms import FakeListLLM

from maverick_mcp.providers.openrouter_provider import (
    TaskType,
    get_openrouter_llm,
)

logger = logging.getLogger(__name__)


def get_llm(
    task_type: TaskType = TaskType.GENERAL,
    prefer_fast: bool = False,
    prefer_cheap: bool = True,  # Default to cost-effective
    prefer_quality: bool = False,
    model_override: str | None = None,
) -> Any:
    """Create and return an LLM instance with intelligent model selection.

    Args:
        task_type: Type of task to optimize model selection for
        prefer_fast: Prioritize speed over quality
        prefer_cheap: Prioritize cost over quality (default True)
        prefer_quality: Use premium models regardless of cost
        model_override: Override automatic model selection

    Returns:
        An LLM instance optimized for the task.

    Priority order:
    1. Anthropic ChatAnthropic (primary — unified platform LLM)
    2. FakeListLLM as fallback for testing
    """
    # Primary: Anthropic Claude Opus 4.6
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    if anthropic_api_key:
        logger.info(f"Using Anthropic Claude Opus 4.6 for task: {task_type}")
        try:
            from langchain_anthropic import ChatAnthropic

            model = model_override or "claude-opus-4-6"
            return ChatAnthropic(model=model, temperature=0.3)
        except ImportError:
            logger.warning("langchain_anthropic not installed, falling back")

    # Final fallback to fake LLM for testing
    logger.warning("ANTHROPIC_API_KEY not found - using FakeListLLM for testing")
    return FakeListLLM(
        responses=[
            "Mock analysis response for testing purposes.",
            "This is a simulated LLM response.",
            "Market analysis: Moderate bullish sentiment detected.",
        ]
    )
