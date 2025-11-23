"""Handler for photo messages."""

import logging
from pathlib import Path

from aiogram import Bot, Router
from aiogram.fsm.context import FSMContext
from aiogram.types import Message

from src.bot.states import GenerationStates
from src.utils.file_handler import download_photo

logger = logging.getLogger(__name__)

router = Router()


@router.message(GenerationStates.WAITING_PHOTOS)
async def handle_photos(message: Message, bot: Bot, state: FSMContext) -> None:
    """Handle photo messages in WAITING_PHOTOS state."""
    user_id = message.from_user.id if message.from_user else None
    logger.info(f"[USER {user_id}] [STATE: WAITING_PHOTOS] Received message type: {message.content_type}")
    
    try:
        # Get current photos from state
        data = await state.get_data()
        photos_raw = data.get("photos", [])
        # Convert strings back to Path objects
        photos: list[Path] = [Path(p) if isinstance(p, str) else p for p in photos_raw]
        logger.info(f"[USER {user_id}] Current photos in state: {len(photos)}")

        # If user sends text or voice and already has photo, process it
        if photos and (message.text or message.voice):
            logger.info(f"[USER {user_id}] User has photo, received {message.content_type}, switching to WAITING_BRIEF")
            # User already has photo, switch to WAITING_BRIEF and redirect to handle_brief
            await state.set_state(GenerationStates.WAITING_BRIEF)
            # Re-process message in new state - router will catch it
            # We need to manually call the handler
            from src.bot.handlers import gen
            await gen.handle_brief(message, bot, state)
            return

        # If no photo in message
        if not message.photo:
            logger.info(f"[USER {user_id}] No photo in message, current photos: {len(photos)}")
            if photos:
                # User already has photos, remind about brief
                await message.answer(
                    "✅ У тебя уже есть фото. Теперь опиши задачу текстом или голосовым сообщением."
                )
            else:
                await message.answer(
                    "📸 Пожалуйста, отправь фото товара."
                )
            return

        # Download photo
        logger.info(f"[USER {user_id}] Starting photo download...")
        photo_path = await download_photo(bot, message)
        if not photo_path:
            logger.error(f"[USER {user_id}] Failed to download photo")
            await message.answer("❌ Ошибка при загрузке фото. Попробуй ещё раз.")
            return

        logger.info(f"[USER {user_id}] Photo downloaded successfully: {photo_path}")
        
        # Convert Path to string for Redis storage (Path objects are not JSON serializable)
        photo_path_str = str(photo_path)
        photos.append(photo_path_str)
        
        logger.info(f"[USER {user_id}] Photo path added to list: {photo_path_str}")

        # Save state immediately after successful download
        logger.info(f"[USER {user_id}] Saving state with {len(photos)} photos...")
        await state.update_data(photos=photos)
        logger.info(f"[USER {user_id}] State saved successfully")

        # After receiving one photo, switch to WAITING_BRIEF
        logger.info(f"[USER {user_id}] Got 1 photo, switching to WAITING_BRIEF")
        await state.set_state(GenerationStates.WAITING_BRIEF)
        await message.answer(
            "✅ Фото получено!\n\n"
            "📝 Теперь опиши свой товар — текстом или голосовым сообщением.\n\n"
            "📊 <b>Для максимально эффективной инфографики укажи:</b>\n\n"
            "🔹 <b>Материал:</b> из чего сделан товар\n"
            "   (стекло, металл, пластик, ткань, кожа и т.д.)\n\n"
            "🔹 <b>Характеристики:</b> размеры, вес, цвет, объем\n"
            "   и другие важные параметры\n\n"
            "🔹 <b>Преимущества:</b> 3-5 ключевых преимуществ\n"
            "   (что делает товар особенным)\n\n"
            "🔹 <b>Маркетплейс:</b> для какого маркетплейса\n"
            "   (Wildberries, Ozon, Яндекс.Маркет и т.д.)\n\n"
            "💡 <b>Пример:</b>\n"
            "<i>\"Футболка из хлопка, размеры S-XL, цвета: белый, черный, синий. "
            "Преимущества: дышащая ткань, не садится после стирки, удобный крой. "
            "Для Wildberries.\"</i>\n\n"
            "ℹ️ Если не нужна инфографика — скажи <b>\"только фото\"</b> или <b>\"без инфографики\"</b>.",
            parse_mode="HTML"
        )
        logger.info(f"[USER {user_id}] Successfully uploaded photo, now in WAITING_BRIEF state")

    except Exception as e:
        logger.error(f"Error in handle_photos: {e}", exc_info=True)
        # Try to preserve state
        try:
            data = await state.get_data()
            photos_raw = data.get("photos", [])
            photos = [Path(p) if isinstance(p, str) else p for p in photos_raw]
            if photos:
                await state.update_data(photos=photos)
                await message.answer(
                    "❌ Произошла ошибка при обработке фото, но твоё фото сохранено. "
                    "Попробуй описать задачу текстом или голосом."
                )
            else:
                await message.answer(
                    "❌ Произошла ошибка при обработке фото. Попробуй ещё раз или используй /gen для начала."
                )
        except Exception:
            await message.answer(
                "❌ Произошла ошибка. Используй /gen для начала заново."
            )

