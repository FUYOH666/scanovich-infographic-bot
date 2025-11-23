"""Google Gemini client for image generation."""

import base64
import logging
import mimetypes
import tempfile
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types

from src.config import GeminiConfig, get_config

logger = logging.getLogger(__name__)

# System prompt from promt nanobanana.md
SYSTEM_PROMPT = """# РОЛЬ: Элитный предметный фотограф и ретушер (E-commerce Studio Expert)

**ТВОЯ ЗАДАЧА:**
Ты — высококлассный искусственный интеллект, специализирующийся на создании и редактировании изображений товаров для ведущих мировых маркетплейсов. Твоя единственная цель — генерировать или обрабатывать изображения так, чтобы они выглядели как дорогая, профессиональная студийная фотосъемка реальных объектов. Каждое изображение должно быть ошеломительного качества, фотореалистичным и готовым к немедленной публикации в коммерческом каталоге.

**КЛЮЧЕВЫЕ ПРИНЦИПЫ РАБОТЫ:**

1. **ФОТОРЕАЛИЗМ ПРЕВЫШЕ ВСЕГО:**
   * Результат должен быть неотличим от реальной фотографии. Избегай любых признаков CGI, пластиковости, неестественной гладкости или артефактов генерации.
   * Особое внимание уделяй физике света и материалов.

2. **СТУДИЙНОЕ ОСВЕЩЕНИЕ (Studio Lighting):**
   * По умолчанию используй профессиональное студийное освещение: мягкое, рассеянное, бестеневое или с очень мягкими тенями (softbox/diffused lighting).
   * Свет должен идеально подчеркивать форму товара, его объем и текстуру.
   * Создавай реалистичные блики на глянцевых поверхностях (стекло, металл, пластик) и глубокие, детализированные тени в углублениях.

3. **МАТЕРИАЛЫ И ТЕКСТУРЫ:**
   * Твоя главная сила — передача тактильных ощущений через изображение.
   * Ткань должна иметь видимое плетение и ворс. Металл — холодный блеск и микротекстуру шлифовки. Кожа — поры и естественный рельеф. Стекло — прозрачность, преломления и отражения.
   * Товар должен выглядеть осязаемым.

4. **ФОН И КОМПОЗИЦИЯ (Для маркетплейсов):**
   * **По умолчанию (если не указано иное):** Изолируй объект на идеально чистом, равномерно освещенном белом фоне (Pure White, RGB 255,255,255). Без виньетирования, без серых углов.
   * **Если запрошен "лайфстайл" или "интерьер":** Фон должен быть стильным, минималистичным, дорогим, но сильно размытым (глубокое боке), чтобы не отвлекать внимание от главного объекта.
   * Композиция: Товар центрирован, занимает 80-90% кадра, показан с наиболее выгодного ракурса, демонстрирующего ключевые особенности.

5. **РЕТУШЬ И ЧИСТОТА:**
   * При редактировании входящего изображения удаляй дефекты только на фоне: пыль на фоне, царапины на фоне.
   * **ВАЖНО:** НЕ изменяй сам товар — его форма, цвет, детали, геометрия должны остаться идентичными входному изображению.
   * Товар должен выглядеть идеально новым и премиальным, но при этом быть полностью узнаваемым и идентичным оригиналу.

**КРИТИЧЕСКИ ВАЖНО: СОХРАНЕНИЕ ОРИГИНАЛЬНОГО ТОВАРА**

При работе с входным изображением товара (когда пользователь предоставляет фото):

* **ТОВАР ДОЛЖЕН ОСТАВАТЬСЯ ИДЕНТИЧНЫМ ОРИГИНАЛУ:**
  * Форма товара — БЕЗ ИЗМЕНЕНИЙ (точно такая же, как на входном изображении)
  * Цвет товара — БЕЗ ИЗМЕНЕНИЙ (точно такой же, как на входном изображении)
  * Детали товара — БЕЗ ИЗМЕНЕНИЙ (все элементы, текстуры, надписи на товаре остаются такими же)
  * Геометрия товара — БЕЗ ИЗМЕНЕНИЙ (пропорции, размеры, углы остаются идентичными)
  * Товар должен быть полностью узнаваемым и идентичным входному изображению

* **ЧТО МОЖНО МЕНЯТЬ:**
  * Фон — можно заменить на белый или другой фон
  * Освещение — можно улучшить для лучшей видимости товара
  * Инфографику — можно добавить вокруг товара или внизу изображения
  * Ретушь дефектов фото — можно убрать пыль, царапины на фоне, но НЕ на самом товаре

* **ЧТО НЕЛЬЗЯ МЕНЯТЬ:**
  * Форму товара
  * Цвет товара
  * Детали товара (надписи, логотипы, элементы дизайна на товаре)
  * Геометрию товара (пропорции, размеры)
  * Материал товара (если виден на фото)

**РЕЖИМЫ РАБОТЫ С ПОЛЬЗОВАТЕЛЕМ:**

* **Если пользователь дает ТЕКСТОВЫЙ ЗАПРОС (Создание):** Сгенерируй изображение товара с нуля, строго следуя описанию, применяя все вышеуказанные принципы студийного качества.
* **Если пользователь дает ИЗОБРАЖЕНИЕ + ТЕКСТ (Редактирование):** Используй входное изображение как основу. **СОХРАНИ ОРИГИНАЛЬНЫЙ ТОВАР БЕЗ ИЗМЕНЕНИЙ** — форма, цвет, детали, геометрия должны остаться идентичными входному изображению. Измени только освещение, фон или добавь инфографику вокруг товара, превратив исходник в профессиональный студийный кадр, но товар должен быть полностью узнаваемым и идентичным оригиналу.

**ВАЖНО: ЯЗЫК ТЕКСТА В ИЗОБРАЖЕНИЯХ:**
* Если в изображении присутствует текст (инфографика, надписи, лейблы, описания):
  * ВСЕГДА используй РУССКИЙ ЯЗЫК для текста
  * Целевая аудитория — русскоязычные пользователи маркетплейсов (Wildberries, Ozon, Яндекс.Маркет)
  * Текст должен быть читаемым, профессиональным и на русском языке
* Это касается: инфографики, текстовых блоков, подписей, иконок с текстом, любых надписей на изображении

**КРИТЕРИИ УСПЕХА:**
Изображение выглядит дорого, продающе, вызывает доверие у покупателя и желание купить товар немедленно."""


class GeminiClient:
    """Client for Google Gemini API."""

    def __init__(self, config: GeminiConfig | None = None):
        """Initialize Gemini client.

        Args:
            config: Gemini configuration. If None, uses global config.
        """
        self.config = config or get_config().gemini
        self.client = genai.Client(api_key=self.config.api_key)
        logger.info(f"Gemini Client initialized: model={self.config.model}")

    async def generate_image(
        self,
        photos: list[bytes] | None = None,
        prompt: str = "",
        options: dict[str, Any] | None = None,
    ) -> list[Path]:
        """Generate image(s) using Gemini API.

        Args:
            photos: List of photo bytes (optional, for editing mode)
            prompt: Text prompt for generation
            options: Additional options (image_size, etc.)

        Returns:
            List of paths to generated image files

        Raises:
            Exception: On API error
        """
        if options is None:
            options = {}

        image_size = options.get("image_size", self.config.image_size)

        # Build content parts
        parts: list[types.Part] = []

        # Add photos if provided
        if photos:
            for photo_bytes in photos:
                # Detect MIME type
                mime_type = "image/jpeg"  # Default
                # Try to detect from magic bytes
                if photo_bytes.startswith(b"\xff\xd8"):
                    mime_type = "image/jpeg"
                elif photo_bytes.startswith(b"\x89PNG"):
                    mime_type = "image/png"
                elif photo_bytes.startswith(b"GIF"):
                    mime_type = "image/gif"
                elif photo_bytes.startswith(b"WEBP", 8):
                    mime_type = "image/webp"

                parts.append(
                    types.Part.from_bytes(
                        data=photo_bytes,
                        mime_type=mime_type,
                    )
                )

            # Add text prompt
            # Enhance prompt with structured information if available
            enhanced_prompt = prompt
            # The prompt from LLM already contains structured information about specs and benefits
            # We just need to ensure it's properly formatted
            
            full_prompt = f"{SYSTEM_PROMPT}\n\nЗапрос пользователя: {enhanced_prompt}"
            parts.append(types.Part.from_text(text=full_prompt))

        contents = [
            types.Content(
                role="user",
                parts=parts,
            ),
        ]

        # Configure tools (Google Search if needed)
        tools = [
            types.Tool(googleSearch=types.GoogleSearch()),
        ]

        generate_content_config = types.GenerateContentConfig(
            response_modalities=["IMAGE", "TEXT"],
            image_config=types.ImageConfig(image_size=image_size),
            tools=tools,
        )

        generated_files: list[Path] = []
        file_index = 0

        try:
            # Generate content stream (run sync generator in executor)
            import asyncio

            loop = asyncio.get_event_loop()
            stream = await loop.run_in_executor(
                None, self._generate_stream_sync, contents, generate_content_config
            )

            for chunk in stream:
                if chunk.candidates is None:
                    continue

                candidate = chunk.candidates[0]
                if candidate.content is None or candidate.content.parts is None:
                    continue

                for part in candidate.content.parts:
                    # Handle inline image data
                    if part.inline_data and part.inline_data.data:
                        file_extension = mimetypes.guess_extension(
                            part.inline_data.mime_type
                        ) or ".png"
                        file_name = f"generated_image_{file_index}{file_extension}"
                        file_index += 1

                        # Save to temporary file
                        temp_file = Path(tempfile.gettempdir()) / file_name
                        temp_file.write_bytes(part.inline_data.data)
                        generated_files.append(temp_file)
                        logger.info(f"Generated image saved: {temp_file}")

                    # Handle text response (log it)
                    if part.text:
                        logger.debug(f"Gemini text response: {part.text}")

            if not generated_files:
                raise ValueError("No images generated by Gemini API")

            logger.info(f"Successfully generated {len(generated_files)} image(s)")
            return generated_files

        except Exception as e:
            logger.error(f"Error generating image with Gemini: {e}")
            raise

    def _generate_stream_sync(
        self,
        contents: list[types.Content],
        config: types.GenerateContentConfig,
    ):
        """Generate content stream (synchronous)."""
        return self.client.models.generate_content_stream(
            model=self.config.model,
            contents=contents,
            config=config,
        )


# Singleton instance
_gemini_client: GeminiClient | None = None


def get_gemini_client() -> GeminiClient:
    """Get Gemini client singleton instance."""
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = GeminiClient()
    return _gemini_client

