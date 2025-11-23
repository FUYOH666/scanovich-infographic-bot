"""Handler for /gen command and generation logic."""

import logging
from pathlib import Path

from aiogram import Bot, Router
from aiogram.filters import Command
from aiogram import F
from aiogram.fsm.context import FSMContext
from aiogram.types import FSInputFile, Message

from src.bot.states import GenerationStates
from src.config import get_config
from src.services.asr_client import ASRClient, get_asr_client
from src.services.gemini_client import GeminiClient, get_gemini_client
from src.services.llm_client import LLMClient, get_llm_client
from src.utils.analytics import FREE_REQUESTS_LIMIT, increment_user_request, register_user
from src.utils.file_handler import cleanup_file, download_voice, read_file_bytes

logger = logging.getLogger(__name__)

router = Router()


@router.message(Command("gen"))
@router.message(lambda m: m.text and m.text.lower() in ["ген"])
async def cmd_gen(message: Message, state: FSMContext) -> None:
    """Handle /gen command or 'ген' text."""
    user_id = message.from_user.id if message.from_user else None
    username = message.from_user.username if message.from_user else None

    # Register user for analytics
    if user_id:
        await register_user(user_id, username)

    logger.info(f"[USER {user_id}] [COMMAND: /gen] Starting new generation process")

    # Reset state
    await state.set_state(GenerationStates.WAITING_PHOTOS)
    await state.update_data(photos=[], brief="", normalized_brief=None)
    logger.info(f"[USER {user_id}] [STATE: WAITING_PHOTOS] State reset, ready for photos")

    await message.answer(
        "✨ Давай сделаем картинку для твоего товара с инфографикой!\n\n"
        "1️⃣ Сначала отправь фото товара как есть. Можно просто сфотать на телефон.\n"
        "2️⃣ Потом в отдельном сообщении опиши, что хочешь получить — текстом или голосовым.\n\n"
        "⚠️ <b>При загрузке фото не снимайте галочку \"Сжать изображение\"</b>\n\n"
        "📊 По умолчанию я создам фото с инфографикой (преимущества и характеристики товара).\n\n"
        "Примеры запросов:\n"
        "• \"Нужно фото для Wildberries с инфографикой\"\n"
        "• \"Создай картинку с преимуществами товара\"\n"
        "• \"Только фото без инфографики\" (если не нужна инфографика)\n\n"
        "Просто опиши задачу своими словами — я пойму!",
        parse_mode="HTML"
    )


@router.message(GenerationStates.WAITING_BRIEF)
async def handle_brief(message: Message, bot: Bot, state: FSMContext) -> None:
    """Handle brief (text or voice) in WAITING_BRIEF state."""
    user_id = message.from_user.id if message.from_user else None
    logger.info(f"[USER {user_id}] [STATE: WAITING_BRIEF] Received message type: {message.content_type}")
    
    try:
        user_brief = ""
        data = await state.get_data()
        photos_raw = data.get("photos", [])
        # Convert strings back to Path objects
        photos: list[Path] = [Path(p) if isinstance(p, str) else p for p in photos_raw]
        logger.info(f"[USER {user_id}] Current photos in state: {len(photos)}")

        if not photos:
            logger.warning(f"[USER {user_id}] No photos in state, but in WAITING_BRIEF")
            await message.answer(
                "❌ Сначала отправь фото товара. Используй /gen для начала."
            )
            return

        # Check if it's voice message
        if message.voice:
            logger.info(f"[USER {user_id}] Processing voice message...")
            await message.answer("🎤 Обрабатываю голосовое сообщение...")
            await state.set_state(GenerationStates.PROCESSING)

            voice_path = None
            try:
                # Download voice
                logger.info(f"[USER {user_id}] Downloading voice message...")
                logger.info(f"[USER {user_id}] Voice file_id: {message.voice.file_id}, duration: {message.voice.duration}s, mime_type: {message.voice.mime_type}")
                voice_path = await download_voice(bot, message)
                if not voice_path:
                    logger.error(f"[USER {user_id}] Failed to download voice - download_voice returned None")
                    await message.answer("❌ Ошибка при загрузке голосового сообщения.")
                    await state.set_state(GenerationStates.WAITING_BRIEF)
                    return

                logger.info(f"[USER {user_id}] Voice downloaded to: {voice_path}, exists: {voice_path.exists()}")
                if voice_path.exists():
                    logger.info(f"[USER {user_id}] Voice file size: {voice_path.stat().st_size} bytes")

                # Transcribe with ASR
                logger.info(f"[USER {user_id}] Starting ASR transcription for file: {voice_path}")
                user_brief = await get_asr_client().transcribe(str(voice_path))

                if not user_brief:
                    logger.error(f"[USER {user_id}] ASR returned empty transcript")
                    await message.answer(
                        "❌ Не удалось распознать речь. Попробуй отправить текстом."
                    )
                    await state.set_state(GenerationStates.WAITING_BRIEF)
                    return

                logger.info(f"[USER {user_id}] ASR transcription successful: {len(user_brief)} chars - {user_brief[:100]}...")
                await message.answer(f"📝 Распознано: {user_brief}")
                
                # Process generation immediately after successful transcription
                logger.info(f"[USER {user_id}] Starting generation process with {len(photos)} photos and brief: {user_brief[:50]}...")
                await process_generation(message, bot, state, photos, user_brief)
                return  # Exit early after successful processing

            except ValueError as e:
                logger.error(f"[USER {user_id}] ValueError in voice processing: {e}", exc_info=True)
                await message.answer(
                    f"❌ Формат файла не поддерживается: {e}. Попробуй отправить текстом."
                )
                await state.set_state(GenerationStates.WAITING_BRIEF)
                return
            except FileNotFoundError as e:
                logger.error(f"[USER {user_id}] FileNotFoundError in voice processing: {e}", exc_info=True)
                await message.answer(
                    "❌ Файл не найден после загрузки. Попробуй отправить текстом."
                )
                await state.set_state(GenerationStates.WAITING_BRIEF)
                return
            except Exception as e:
                logger.error(f"[USER {user_id}] Error processing voice: {type(e).__name__}: {e}", exc_info=True)
                await message.answer(
                    f"❌ Ошибка при обработке голосового сообщения: {type(e).__name__}. Попробуй отправить текстом."
                )
                await state.set_state(GenerationStates.WAITING_BRIEF)
                return
            finally:
                # Always cleanup voice file
                if voice_path:
                    cleanup_file(voice_path)
                    logger.info(f"[USER {user_id}] Cleaned up voice file: {voice_path}")

        elif message.text:
            user_brief = message.text.strip()
            logger.info(f"[USER {user_id}] Received text brief: {user_brief[:100]}...")
            if not user_brief:
                await message.answer("Пожалуйста, опиши задачу текстом или голосом.")
                return
            await state.set_state(GenerationStates.PROCESSING)
            # Process generation for text
            logger.info(f"[USER {user_id}] Starting generation process with {len(photos)} photos and brief: {user_brief[:50]}...")
            await process_generation(message, bot, state, photos, user_brief)
            return  # Exit early after successful processing
        else:
            logger.warning(f"[USER {user_id}] Unexpected message type in WAITING_BRIEF: {message.content_type}")
            await message.answer("Пожалуйста, отправь текст или голосовое сообщение.")
            return
    except Exception as e:
        logger.error(f"Error in handle_brief: {e}", exc_info=True)
        # Try to preserve state
        try:
            data = await state.get_data()
            photos_raw = data.get("photos", [])
            photos = [Path(p) if isinstance(p, str) else p for p in photos_raw]
            if photos:
                # Convert back to strings for storage
                await state.update_data(photos=[str(p) for p in photos])
                await state.set_state(GenerationStates.WAITING_BRIEF)
                await message.answer(
                    "❌ Произошла ошибка при обработке запроса, но твои фото сохранены. "
                    "Попробуй описать задачу ещё раз текстом или голосом."
                )
            else:
                await message.answer(
                    "❌ Произошла ошибка. Используй /gen для начала заново."
                )
        except Exception:
            await message.answer(
                "❌ Произошла ошибка. Используй /gen для начала заново."
            )


async def process_generation(
    message: Message,
    bot: Bot,
    state: FSMContext,
    photos: list[Path],
    user_brief: str,
) -> None:
    """Process image generation."""
    config = get_config()
    user_id = message.from_user.id if message.from_user else None
    logger.info(f"[USER {user_id}] [STATE: PROCESSING] Starting generation process")
    logger.info(f"[USER {user_id}] Photos count: {len(photos)}, Brief: {user_brief[:100]}...")
    
    await message.answer("🔄 Обрабатываю запрос...")

    try:
        # Normalize brief with LLM
        logger.info(f"[USER {user_id}] Step 1/3: Normalizing brief with LLM (VLLM)...")
        photos_context = f"Загружено {len(photos)} фото(графий) товара"
        normalized = await get_llm_client().normalize_brief(user_brief, photos_context)
        logger.info(f"[USER {user_id}] LLM normalization complete. Image type: {normalized.get('image_type')}, Style: {normalized.get('style')}")

        await state.update_data(
            brief=user_brief,
            normalized_brief=normalized,
        )

        await message.answer("🎨 Генерирую изображение...")

        # Read photos as bytes
        logger.info(f"[USER {user_id}] Step 2/3: Reading photos as bytes...")
        photo_bytes_list = [read_file_bytes(photo) for photo in photos]
        logger.info(f"[USER {user_id}] Photos read: {[len(b) for b in photo_bytes_list]} bytes each")

        # Generate image with Gemini
        logger.info(f"[USER {user_id}] Step 3/3: Generating image with Gemini (NanoBanana)...")
        logger.info(f"[USER {user_id}] Prompt for Gemini: {normalized['prompt_for_model'][:200]}...")
        generated_images = await get_gemini_client().generate_image(
            photos=photo_bytes_list,
            prompt=normalized["prompt_for_model"],
            options={"image_size": "1K"},
        )

        if not generated_images:
            raise ValueError("No images generated")

        logger.info(f"[USER {user_id}] Gemini generated {len(generated_images)} image(s)")

        # Send images to user
        logger.info(f"[USER {user_id}] Sending images to user...")
        for i, img_path in enumerate(generated_images[:3]):  # Max 3 images
            photo_file = FSInputFile(str(img_path))
            await message.answer_photo(photo_file)
            cleanup_file(img_path)
            logger.info(f"[USER {user_id}] Sent image {i+1}/{len(generated_images)}")

        # Cleanup input photos
        for photo_path in photos:
            cleanup_file(photo_path)

        await state.set_state(GenerationStates.SHOW_RESULT)

        # Increment request counter for analytics (only after successful generation)
        # Skip increment for owner (developer) - unlimited usage
        try:
            if user_id == config.telegram.owner_id:
                logger.info(f"[USER {user_id}] Owner - skipping request counter increment (unlimited)")
                success_message = (
                    "✅ Готово! (Безлимит для разработчика)\n\n"
                    "Напиши /gen для новой генерации."
                )
            else:
                request_count = await increment_user_request(user_id)
                remaining = FREE_REQUESTS_LIMIT - request_count
                logger.info(f"[USER {user_id}] Request count incremented: {request_count}/{FREE_REQUESTS_LIMIT}")

                # Show remaining requests in success message
                if remaining > 0:
                    success_message = (
                        f"✅ Готово! Осталось бесплатных запросов: {remaining}/{FREE_REQUESTS_LIMIT}\n\n"
                        "Напиши /gen для новой генерации."
                    )
                else:
                    success_message = (
                        "✅ Готово! Вы использовали все бесплатные запросы.\n\n"
                        "Для продолжения использования свяжитесь с владельцем:\n"
                        f"📧 Telegram: @{config.telegram.owner_username}\n\n"
                        "Напиши /gen для новой генерации."
                    )
            await message.answer(success_message)
        except Exception as e:
            logger.error(f"[USER {user_id}] Failed to increment request counter: {e}", exc_info=True)
            await message.answer(
                "✅ Готово!\n\n"
                "Напиши /gen для новой генерации."
            )

        logger.info(f"[USER {user_id}] [STATE: SHOW_RESULT] Successfully completed generation. Generated {len(generated_images)} image(s)")

    except Exception as e:
        logger.error(f"Error during generation: {e}", exc_info=True)
        await message.answer(
            "❌ Произошла ошибка при генерации изображения.\n"
            "Попробуй ещё раз с командой /gen"
        )
        await state.set_state(GenerationStates.IDLE)

        # Cleanup on error
        for photo_path in photos:
            cleanup_file(photo_path)


@router.message(GenerationStates.SHOW_RESULT)
async def handle_result_actions(message: Message, bot: Bot, state: FSMContext) -> None:
    """Handle actions after showing result."""
    user_id = message.from_user.id if message.from_user else None
    text = message.text or ""

    # Only allow /gen command to start new generation
    if text.lower().startswith("/gen") or text.lower() == "ген":
        # Reset state and start new generation
        await state.set_state(GenerationStates.WAITING_PHOTOS)
        await state.update_data(photos=[], brief="", normalized_brief=None)
        await message.answer(
            "✨ Давай сделаем картинку для твоего товара с инфографикой!\n\n"
            "1️⃣ Сначала отправь фото товара как есть. Можно просто сфотать на телефон.\n"
            "2️⃣ Потом в отдельном сообщении опиши, что хочешь получить — текстом или голосовым.\n\n"
            "📊 По умолчанию я создам фото с инфографикой (преимущества и характеристики товара).\n\n"
            "Просто опиши задачу своими словами — я пойму!"
        )
    else:
        # Unknown command, remind about /gen
        logger.info(f"[USER {user_id}] Unknown action in SHOW_RESULT: {text}")
        await message.answer(
            "Напиши /gen для начала новой генерации."
        )

