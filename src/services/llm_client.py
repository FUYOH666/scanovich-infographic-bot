"""LLM client for brief normalization using VLLM."""

import json
import logging
from typing import Any

from openai import AsyncOpenAI

from src.config import VLLMConfig, get_config

logger = logging.getLogger(__name__)


class LLMClient:
    """Client for VLLM API (OpenAI-compatible)."""

    def __init__(self, config: VLLMConfig | None = None):
        """Initialize LLM client.

        Args:
            config: VLLM configuration. If None, uses global config.
        """
        self.config = config or get_config().vllm
        self.client = AsyncOpenAI(
            base_url=self.config.base_url,
            api_key="not-needed",  # VLLM doesn't require API key
        )
        logger.info(f"LLM Client initialized: {self.config.base_url}, model: {self.config.model}")

    async def normalize_brief(self, user_brief: str, photos_context: str = "") -> dict[str, Any]:
        """Normalize user brief into structured prompt for Gemini.

        Args:
            user_brief: Raw user brief (text or transcribed from voice)
            photos_context: Optional context about uploaded photos

        Returns:
            Dictionary with:
            - normalized_brief: Human-readable normalized brief
            - prompt_for_model: Structured prompt for Gemini
            - image_type: Type of image (main_photo, infographic, etc.)
            - style: Style preferences (white_background, lifestyle, etc.)
            - marketplace: Target marketplace (wildberries, ozon, etc.)
            - additional_params: Additional parameters

        Raises:
            Exception: On API error
        """
        system_prompt = """# РОЛЬ: Виртуальный Арт-Директор и Промпт-Инженер (E-commerce & Infographic Specialist)

**ТВОЯ МИССИЯ:**
Ты — промежуточное звено между неквалифицированным пользователем и мощной AI-моделью генерации изображений ("Фотографом"). Твоя задача — принимать сырые, часто неполные или непрофессиональные запросы пользователей и трансформировать их в идеальные, детализированные технические задания (промпты) для создания студийных фотографий товаров с инфографикой премиум-класса.

**КРИТИЧЕСКИ ВАЖНО: ПРИОРИТЕТ ИНФОГРАФИКИ**
По умолчанию ВСЕГДА создавай инфографику с преимуществами и характеристиками товара, ЕСЛИ пользователь явно не указал "без инфографики", "только фото", "просто фото" или аналогичные формулировки отказа от инфографики.

**ТВОЙ КЛИЕНТ (ЦЕЛЬ):**
Конечный промпт предназначен для AI, который работает *только* в стиле дорогой рекламной фотосъемки. Он не понимает абстракций, ему нужны физические параметры сцены.

**ПРОТОКОЛ НОРМАЛИЗАЦИИ:**

При получении запроса от пользователя ты должен выполнить следующие шаги:

1. **АНАЛИЗ ИНТЕНТА (О чем речь?):**
   - Выдели главный объект (товар или товары)
   - Пойми желаемое действие: создать с нуля, улучшить фото, добавить фон, добавить инфографику
   - **ПРОВЕРЬ ОТКАЗ ОТ ИНФОГРАФИКИ:** Если пользователь явно указал "без инфографики", "только фото", "просто фото" → создай только фото без инфографики
   - **ПО УМОЛЧАНИЮ:** Если пользователь НЕ отказался от инфографики → создай инфографику с преимуществами и характеристиками
   - Определи тип маркетплейса (Wildberries, Ozon, Яндекс.Маркет, Amazon и т.д.)

2. **АНАЛИЗ ТИПА ТОВАРА И ИЗВЛЕЧЕНИЕ ХАРАКТЕРИСТИК:**
   - **ОПРЕДЕЛЕНИЕ ТИПА ТОВАРА:**
     Проанализируй описание пользователя и определи тип товара:
     * clothing (одежда) — футболки, платья, куртки, обувь и т.д.
     * electronics (электроника) — телефоны, наушники, зарядки, гаджеты и т.д.
     * cosmetics (косметика) — кремы, шампуни, маски, парфюмерия и т.д.
     * home (дом/интерьер) — мебель, декор, посуда, текстиль для дома и т.д.
     * food (продукты) — еда, напитки, добавки и т.д.
     * other (другое) — если не подходит ни к одной категории
   
   - **ИЗВЛЕЧЕНИЕ ХАРАКТЕРИСТИК ПО ТИПУ:**
     В зависимости от типа товара извлеки ключевые характеристики из описания пользователя:
     
     **Одежда:**
     * Размеры (S, M, L, XL, 42, 44 и т.д.)
     * Материал (хлопок, полиэстер, шерсть, кожа и т.д.)
     * Цвет (белый, черный, синий и т.д.)
     * Особенности ткани (дышащая, не мнется, stretch, водоотталкивающая и т.д.)
     * Уход (стирка, глажка, сухая чистка)
     
     **Электроника:**
     * Мощность/производительность (ватты, амперы, емкость батареи и т.д.)
     * Размеры и вес
     * Функции и возможности (быстрая зарядка, беспроводная связь и т.д.)
     * Совместимость (с какими устройствами работает)
     * Гарантия
     
     **Косметика:**
     * Объем (мл, г)
     * Состав (ключевые ингредиенты)
     * Тип кожи (для сухой, жирной, комбинированной и т.д.)
     * Способ применения
     * Эффект (увлажнение, омоложение, защита и т.д.)
     
     **Дом/Интерьер:**
     * Размеры (длина, ширина, высота)
     * Материал
     * Цвет
     * Стиль (минимализм, классика, современный и т.д.)
     * Особенности (складной, водонепроницаемый и т.д.)
     
     **Продукты:**
     * Вес/объем
     * Состав
     * Срок годности
     * Особенности (органический, без ГМО, веганский и т.д.)
     
     **Другое:**
     * Извлеки все упомянутые характеристики
   
   - **ПРИОРИТИЗАЦИЯ ХАРАКТЕРИСТИК:**
     Выдели 3-5 самых важных характеристик, которые влияют на решение о покупке.
     Например, для одежды: размеры и материал — самые важные.
     Для электроники: мощность и функции — самые важные.
   
   - **ИЗВЛЕЧЕНИЕ ПРЕИМУЩЕСТВ:**
     Выдели 3-5 ключевых преимуществ товара из описания пользователя.
     Если пользователь не указал преимущества явно, выведи их из характеристик.
     Например: "дышащая ткань" → преимущество "Комфорт в носке", "быстрая зарядка" → преимущество "Экономия времени".
   
   - **СТРУКТУРИРОВАНИЕ ДЛЯ ИНФОГРАФИКИ:**
     Определи визуальную иерархию:
     * priority_specs: какие характеристики показать крупнее (самые важные для покупателя)
     * benefits_order: порядок преимуществ от самого важного к менее важному
     * visual_hierarchy: описание как расположить элементы (главные характеристики крупно, преимущества средним размером, остальное мелко)

3. **ОБОГАЩЕНИЕ ДЕТАЛЯМИ (Чего не хватает?):**
   - Пользователи почти никогда не описывают свет. ТЫ ДОЛЖЕН добавить профессиональное описание света:
     * Для студийных фото: "softbox studio lighting", "dramatic rim light", "diffused lighting"
     * Для лайфстайла: "natural window light", "warm ambient lighting"
   - Уточни материалы, если они не указаны (например, "бутылка" → "glass bottle with condensation droplets")
   - Выбери лучший ракурс для этого типа товара, если он не задан:
     * Для обуви: "3/4 view" или "hero shot from below"
     * Для электроники: "front-facing product shot" или "angled perspective"
     * Для одежды: "flat lay" или "on-model"
   - Добавь детали композиции: "centered composition", "rule of thirds", "negative space"

3. **ПЕРЕВОД НА ПРОФЕССИОНАЛЬНЫЙ ЯЗЫК:**
   - "красиво" → "aesthetically pleasing, premium quality"
   - "четко" → "high detail, sharp focus on texture"
   - "размытый фон" → "shallow depth of field, bokeh background"
   - "ярко" → "vibrant colors, high contrast"
   - "профессионально" → "commercial photography quality, studio-grade"

4. **РАБОТА С ИНФОГРАФИКОЙ (ПРИОРИТЕТНАЯ ЗАДАЧА):**
   - **СТРУКТУРА ИНФОГРАФИКИ (по умолчанию):**
     * Товар занимает центральную часть изображения (70-80% кадра)
     * Информационные блоки расположены вокруг товара или в нижней части изображения
     * Преимущества товара: 3-5 ключевых пунктов с иконками и текстом на русском языке
     * Характеристики товара: материал, размеры, вес, цвет и другие важные параметры (на русском языке)
     * Профессиональный дизайн: современная типографика, четкая иерархия информации, читаемые шрифты
     * Цветовая схема: контрастные цвета для читаемости, но не кричащие
     * Чистый белый фон для маркетплейсов (RGB 255,255,255)
   
   - **BEST PRACTICES ДЛЯ МАРКЕТПЛЕЙСОВ:**
     * **Визуальная иерархия:**
       - Самые важные характеристики (из priority_specs) — крупнее и заметнее
       - Преимущества — средним размером с иконками для быстрого восприятия
       - Второстепенные характеристики — мельче, но читаемо
     * **Правило 3-5:** максимум 3-5 ключевых преимуществ (больше — перегружает)
     * **Читаемость:** контрастные цвета, крупные шрифты для важной информации (минимум 14-16pt для заголовков)
     * **Сканируемость:** информация должна читаться за 3-5 секунд (покупатель быстро просматривает)
     * **Эмоциональный триггер:** акцент на выгодах покупателя, а не просто характеристиках
       Например: "Дышащая ткань" → "Комфорт в любую погоду", "Быстрая зарядка" → "Готов к использованию за 30 минут"
     * **Правило F-паттерна:** важная информация слева и сверху (там, где взгляд начинает сканирование)
   
   - **ОПИСАНИЕ В ПРОМПТЕ:**
     * Стиль шрифта: "modern sans-serif", "bold headings", "elegant typography", "readable font sizes"
     * Расположение элементов: "product centered, infographic elements around product or at bottom", "balanced composition"
     * Иконки: "modern minimalist icons", "professional iconography", "clear visual hierarchy"
     * Цвета текста: "high contrast text colors for readability", "professional color palette"
     * Визуальная иерархия: используй extracted_specs и extracted_benefits из JSON для структурирования
     * Приоритеты: покажи priority_specs крупнее, benefits_order в указанном порядке
     * Чтобы это выглядело как профессиональный дизайн для маркетплейсов
   
   - **КРИТИЧЕСКИ ВАЖНО:** В prompt_for_model явно укажи:
     * "infographic with Russian text", "all text labels in Russian language", "Russian typography"
     * "product benefits and specifications in Russian", "Russian language infographic elements"
     * Структуру инфографики: "product in center, benefits and specs around or below"
     * Используй извлеченные характеристики и преимущества: "show [priority_specs] prominently, display [extracted_benefits] with icons"
     * Визуальную иерархию: "main specifications large and bold, benefits medium size with icons, other specs smaller"
     * **СОХРАНЕНИЕ ОРИГИНАЛЬНОГО ТОВАРА:** Если есть входное изображение товара, обязательно добавь: "preserve original product exactly as shown in input image", "do not modify product shape, color, details, or geometry", "product must remain identical to input image", "only change background, lighting, and add infographic around product"

**ФОРМАТ ВЫХОДА (СТРОГОЕ СОБЛЮДЕНИЕ):**

Ты должен выдать ответ в формате JSON со следующей структурой:

{
    "normalized_brief": "Человекочитаемое описание задачи на РУССКОМ ЯЗЫКЕ для пользователя. Должно быть понятным и детальным, описывать что именно будет создано. Пример: 'Создам главное фото товара для Wildberries с белым фоном, студийным освещением, товар будет крупно в центре кадра'.",
    "prompt_for_model": "ДЕТАЛЬНЫЙ ПРОМПТ НА АНГЛИЙСКОМ ЯЗЫКЕ для модели генерации изображений. Должен включать: описание товара, освещение, композицию, фон, стиль, ракурс, материалы, текстуры. Используй профессиональную терминологию фотографии и дизайна.",
    "image_type": "main_photo|infographic|lifestyle|other",
    "style": "white_background|lifestyle|interior|colorful",
    "marketplace": "wildberries|ozon|yandex_market|amazon|other",
    "additional_params": {
        "icons_count": 0,
        "text_elements": false,
        "product_centered": true,
        "background_color": "white",
        "lighting_type": "studio|natural|dramatic",
        "camera_angle": "front|3/4|top|hero",
        "has_infographic": true,
        "product_type": "clothing|electronics|cosmetics|home|food|other",
        "extracted_specs": {
            "material": "материал товара или null",
            "dimensions": "размеры товара или null",
            "weight": "вес товара или null",
            "color": "цвет товара или null",
            "volume": "объем товара или null",
            "power": "мощность/производительность или null",
            "composition": "состав товара или null",
            "other": "другие важные характеристики или null"
        },
        "extracted_benefits": [
            "Преимущество 1",
            "Преимущество 2",
            "Преимущество 3"
        ],
        "infographic_structure": {
            "priority_specs": ["список ключевых характеристик для показа крупно"],
            "benefits_order": ["порядок преимуществ от важного к менее важному"],
            "visual_hierarchy": "описание визуальной иерархии: main_specs_large, benefits_medium, other_specs_small"
        }
    }
}

**КРИТИЧЕСКИ ВАЖНО:**
- normalized_brief ДОЛЖЕН быть на РУССКОМ ЯЗЫКЕ (для русскоязычного пользователя)
- prompt_for_model ДОЛЖЕН быть на АНГЛИЙСКОМ языке (для Gemini API)
- prompt_for_model ДОЛЖЕН быть детальным и включать ВСЕ физические параметры сцены
- Если пользователь не указал свет — добавь профессиональное описание света
- Если пользователь не указал ракурс — выбери оптимальный для типа товара
- По умолчанию используй белый фон (white_background), если не указано иное
- Для инфографики опиши расположение элементов, стиль типографики, цветовую схему
- **ОБЯЗАТЕЛЬНО:** Если в изображении будет текст (инфографика, надписи), в prompt_for_model явно укажи: "Russian text", "text in Russian language", "Russian typography" — чтобы Gemini генерировал текст на русском языке
- **СОХРАНЕНИЕ ОРИГИНАЛЬНОГО ТОВАРА:** Если есть входное изображение товара (пользователь предоставил фото), в prompt_for_model ОБЯЗАТЕЛЬНО добавь: "preserve original product exactly as shown in input image", "do not modify product shape, color, details, or geometry", "product must remain identical to input image", "only change background, lighting, and add infographic around product" — чтобы Gemini не изменял сам товар
"""

        user_prompt = f"""Запрос пользователя:
{user_brief}

{f"Контекст фотографий: {photos_context}" if photos_context else ""}

Твоя задача:
1. Проанализировать интент пользователя
2. Определить тип товара и извлечь ключевые характеристики и преимущества
3. Приоритизировать характеристики по важности для покупателя
4. Обогатить запрос профессиональными деталями (свет, материалы, ракурс, композиция)
5. Перевести на профессиональный язык фотографии и дизайна
6. Сформировать normalized_brief на РУССКОМ ЯЗЫКЕ для пользователя (понятное описание того, что будет создано)
7. Сформировать prompt_for_model на АНГЛИЙСКОМ языке для модели генерации изображений с учетом извлеченных характеристик и преимуществ

Помни:
- normalized_brief должен быть на РУССКОМ для русскоязычного пользователя
- prompt_for_model должен быть на АНГЛИЙСКОМ для Gemini API
- Модель нуждается в физических параметрах сцены, а не абстракциях. Добавь всё, чего не хватает
- **ИЗВЛЕЧЕНИЕ ХАРАКТЕРИСТИК:** Обязательно заполни extracted_specs, extracted_benefits и infographic_structure в additional_params
- **СТРУКТУРИРОВАНИЕ ПРОМПТА:** В prompt_for_model включи извлеченные характеристики и преимущества, укажи визуальную иерархию
- **КРИТИЧЕСКИ ВАЖНО:** Если пользователь предоставил фото товара (есть входное изображение), в prompt_for_model ОБЯЗАТЕЛЬНО добавь инструкции о сохранении оригинального товара без изменений: "preserve original product exactly as shown in input image", "do not modify product shape, color, details, or geometry", "product must remain identical to input image", "only change background, lighting, and add infographic around product"."""

        try:
            response = await self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=self.config.max_tokens,
                temperature=0.7,
            )

            content = response.choices[0].message.content
            if not content:
                raise ValueError("Empty response from LLM")

            # Try to parse JSON from response
            # LLM might return JSON wrapped in markdown code blocks
            content = content.strip()
            if content.startswith("```"):
                # Extract JSON from code block
                lines = content.split("\n")
                json_start = None
                json_end = None
                for i, line in enumerate(lines):
                    if line.strip().startswith("```"):
                        if json_start is None:
                            json_start = i + 1
                        else:
                            json_end = i
                            break
                if json_start and json_end:
                    content = "\n".join(lines[json_start:json_end])
                elif json_start:
                    content = "\n".join(lines[json_start:])

            # Parse JSON
            try:
                result = json.loads(content)
            except json.JSONDecodeError:
                # Fallback: try to extract JSON object
                import re

                json_match = re.search(r"\{.*\}", content, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                else:
                    # Fallback: create basic structure with infographic by default
                    logger.warning("Failed to parse JSON, using fallback")
                    user_brief_lower = user_brief.lower()
                    if any(phrase in user_brief_lower for phrase in ["без инфографики", "только фото", "просто фото", "без текста", "no infographic"]):
                        image_type = "main_photo"
                        has_infographic = False
                        # Add preservation instruction if there's input photo
                        preserve_product = "Preserve original product exactly as shown in input image. Do not modify product shape, color, details, or geometry. Product must remain identical to input image. Only change background and lighting." if photos_context else ""
                        prompt = f"Professional studio product photography: {user_brief}. {preserve_product}Softbox studio lighting, white background, centered composition, high detail, commercial quality."
                    else:
                        image_type = "infographic"
                        has_infographic = True
                        # Check if there's input photo (photos_context indicates user provided photo)
                        preserve_product = "Preserve original product exactly as shown in input image. Do not modify product shape, color, details, or geometry. Product must remain identical to input image. Only change background, lighting, and add infographic around product." if photos_context else ""
                        prompt = f"Professional studio product photography with infographic: {user_brief}. {preserve_product}Product centered (70-80% of frame), infographic elements around product or at bottom with product benefits and specifications in Russian language. Softbox studio lighting, white background, modern typography, high contrast text colors, professional iconography. All text in Russian language."
                    result = {
                        "normalized_brief": user_brief,
                        "prompt_for_model": prompt,
                        "image_type": image_type,
                        "style": "white_background",
                        "marketplace": "other",
                        "additional_params": {
                            "has_infographic": has_infographic,
                            "product_type": "other",
                            "extracted_specs": {},
                            "extracted_benefits": [],
                            "infographic_structure": {
                                "priority_specs": [],
                                "benefits_order": [],
                                "visual_hierarchy": "main_specs_large, benefits_medium, other_specs_small"
                            }
                        },
                    }

            # Validate and set defaults (infographic by default)
            result.setdefault("normalized_brief", user_brief)
            result.setdefault("prompt_for_model", user_brief)
            # Default to infographic unless user explicitly refused
            user_brief_lower = user_brief.lower()
            if any(phrase in user_brief_lower for phrase in ["без инфографики", "только фото", "просто фото", "без текста", "no infographic"]):
                result.setdefault("image_type", "main_photo")
                result.setdefault("additional_params", {}).setdefault("has_infographic", False)
            else:
                result.setdefault("image_type", "infographic")
                result.setdefault("additional_params", {}).setdefault("has_infographic", True)
            result.setdefault("style", "white_background")
            result.setdefault("marketplace", "other")
            if "additional_params" not in result:
                result["additional_params"] = {}

            logger.info(f"Brief normalized successfully: {result.get('image_type')}")
            return result

        except Exception as e:
            logger.error(f"Error normalizing brief: {e}")
            # Fallback: return basic structure with infographic by default
            user_brief_lower = user_brief.lower()
            if any(phrase in user_brief_lower for phrase in ["без инфографики", "только фото", "просто фото", "без текста", "no infographic"]):
                image_type = "main_photo"
                has_infographic = False
                # Add preservation instruction if there's input photo
                preserve_product = "Preserve original product exactly as shown in input image. Do not modify product shape, color, details, or geometry. Product must remain identical to input image. Only change background and lighting." if photos_context else ""
                prompt = f"Professional studio product photography: {user_brief}. {preserve_product}Softbox studio lighting, white background, centered composition, high detail, commercial quality."
            else:
                image_type = "infographic"
                has_infographic = True
                # Check if there's input photo (photos_context indicates user provided photo)
                preserve_product = "Preserve original product exactly as shown in input image. Do not modify product shape, color, details, or geometry. Product must remain identical to input image. Only change background, lighting, and add infographic around product." if photos_context else ""
                prompt = f"Professional studio product photography with infographic: {user_brief}. {preserve_product}Product centered (70-80% of frame), infographic elements around product or at bottom with product benefits and specifications in Russian language. Softbox studio lighting, white background, modern typography, high contrast text colors, professional iconography. All text in Russian language."
            return {
                "normalized_brief": user_brief,
                "prompt_for_model": prompt,
                "image_type": image_type,
                "style": "white_background",
                "marketplace": "other",
                "additional_params": {
                    "has_infographic": has_infographic,
                    "product_type": "other",
                    "extracted_specs": {},
                    "extracted_benefits": [],
                    "infographic_structure": {
                        "priority_specs": [],
                        "benefits_order": [],
                        "visual_hierarchy": "main_specs_large, benefits_medium, other_specs_small"
                    }
                },
            }


# Singleton instance
_llm_client: LLMClient | None = None


def get_llm_client() -> LLMClient:
    """Get LLM client singleton instance."""
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client
