"""Middleware for logging and error handling."""

import logging
import traceback
from typing import Any, Awaitable, Callable

from aiogram import BaseMiddleware, Bot
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.types import Message, TelegramObject

from src.config import get_config
from src.utils.analytics import check_user_limit, FREE_REQUESTS_LIMIT, register_user

logger = logging.getLogger(__name__)


class LoggingMiddleware(BaseMiddleware):
    """Middleware for logging user actions."""

    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        """Log user action."""
        if isinstance(event, Message):
            user_id = event.from_user.id if event.from_user else None
            username = event.from_user.username if event.from_user else None
            text = event.text or (event.caption or "")
            
            # Get current FSM state
            state: FSMContext | None = data.get("state")
            current_state = "UNKNOWN"
            if state:
                try:
                    current_state = await state.get_state()
                    current_state = str(current_state) if current_state else "NONE"
                except Exception:
                    pass

            logger.info(
                f"[USER {user_id}] (@{username}) [STATE: {current_state}] "
                f"Content: {event.content_type} - {text[:100]}"
            )

        return await handler(event, data)


class ErrorHandlerMiddleware(BaseMiddleware):
    """Middleware for error handling and owner notifications."""

    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        """Handle errors and notify owner."""
        try:
            return await handler(event, data)
        except Exception as e:
            logger.error(f"Error in handler: {e}", exc_info=True)

            # Try to send error message to user
            if isinstance(event, Message):
                try:
                    bot: Bot = data.get("bot")
                    if bot:
                        await event.answer(
                            "❌ Произошла внутренняя ошибка. "
                            "Попробуй ещё раз или используй /start для перезапуска."
                        )
                except Exception:
                    pass

            # Notify owner about critical errors
            try:
                bot: Bot = data.get("bot")
                if bot:
                    error_msg = (
                        f"⚠️ Ошибка в боте:\n"
                        f"Тип: {type(e).__name__}\n"
                        f"Сообщение: {str(e)}\n"
                        f"Пользователь: {event.from_user.id if isinstance(event, Message) and event.from_user else 'N/A'}\n"
                        f"Текст: {event.text[:200] if isinstance(event, Message) else 'N/A'}"
                    )
                    config = get_config()
                    await bot.send_message(
                        chat_id=config.telegram.owner_id, text=error_msg
                    )
            except Exception as notify_error:
                logger.error(f"Failed to notify owner: {notify_error}")

            # Re-raise to let aiogram handle it
            raise


class LimitCheckMiddleware(BaseMiddleware):
    """Middleware for checking request limits before generation."""

    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        """Check request limit before processing generation."""
        if not isinstance(event, Message):
            return await handler(event, data)

        # Register user for analytics
        if event.from_user:
            await register_user(event.from_user.id, event.from_user.username)

        # Skip limit check for commands that don't count as requests
        if event.text:
            text_lower = event.text.lower()
            if text_lower.startswith("/start") or text_lower.startswith("/stats"):
                return await handler(event, data)

        # Check if this is a generation request
        # We check limit when user sends brief (text/voice) in WAITING_BRIEF state
        # or when user tries to generate more variants
        state: FSMContext | None = data.get("state")
        if state:
            try:
                current_state = await state.get_state()
                state_str = str(current_state) if current_state else ""

                # Check limit for generation requests
                # WAITING_BRIEF: user sends text/voice brief
                # SHOW_RESULT: user requests "ещё" or "исправь"
                is_generation_request = (
                    "WAITING_BRIEF" in state_str
                    and (event.text or event.voice)
                    and not event.text.startswith("/")  # Don't count commands
                ) or (
                    "SHOW_RESULT" in state_str
                    and event.text
                    and (event.text.lower() == "ещё" or event.text.lower().startswith("исправь:"))
                )

                if is_generation_request:
                    user_id = event.from_user.id if event.from_user else None
                    if user_id:
                        # Skip limit check for owner (developer) - unlimited usage
                        config = get_config()
                        if user_id == config.telegram.owner_id:
                            logger.info(f"Owner {user_id} bypassing limit check (unlimited)")
                        else:
                            can_request = await check_user_limit(user_id, FREE_REQUESTS_LIMIT)
                            if not can_request:
                                bot: Bot = data.get("bot")
                                owner_username = config.telegram.owner_username

                                limit_message = (
                                    f"❌ Вы использовали все бесплатные запросы ({FREE_REQUESTS_LIMIT}).\n\n"
                                    f"Для продолжения использования бота свяжитесь с владельцем:\n"
                                    f"📧 Telegram: @{owner_username}\n\n"
                                    f"Или напишите /start для информации о тарифах."
                                )

                                await event.answer(limit_message)
                                logger.info(f"User {user_id} exceeded free limit ({FREE_REQUESTS_LIMIT})")
                                return  # Don't process the request
            except Exception as e:
                logger.warning(f"Error checking limit: {e}")

        return await handler(event, data)

