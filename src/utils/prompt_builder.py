"""Prompt builder utilities for LLM."""

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def build_photos_context(photos: list[Path]) -> str:
    """Build context string about uploaded photos.

    Args:
        photos: List of photo paths

    Returns:
        Context string describing photos
    """
    count = len(photos)
    if count == 0:
        return "Фотографии не загружены"
    elif count == 1:
        return "Загружена 1 фотография товара"
    else:
        return f"Загружено {count} фотографии товара"


def build_llm_prompt(user_brief: str, photos_context: str = "") -> str:
    """Build prompt for LLM normalization.

    Args:
        user_brief: Raw user brief
        photos_context: Context about photos

    Returns:
        Formatted prompt for LLM
    """
    parts = [f"Запрос пользователя:\n{user_brief}"]
    if photos_context:
        parts.append(f"\nКонтекст фотографий: {photos_context}")

    parts.append(
        "\nПреобразуй этот запрос в структурированный формат для генерации изображения товара."
    )

    return "\n".join(parts)

