"""Handler for /stats command (owner only)."""

import logging

from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message

from src.config import get_config
from src.utils.analytics import get_stats, get_top_users, get_user_meta, get_user_request_count

logger = logging.getLogger(__name__)

router = Router()


@router.message(Command("stats"))
async def cmd_stats(message: Message) -> None:
    """Handle /stats command - show bot statistics (owner only)."""
    config = get_config()
    user_id = message.from_user.id if message.from_user else None

    # Check if user is owner
    if user_id != config.telegram.owner_id:
        await message.answer("❌ Эта команда доступна только владельцу бота.")
        logger.warning(f"User {user_id} tried to access /stats command")
        return

    try:
        # Get statistics
        stats = await get_stats()
        top_users = await get_top_users(limit=10)

        # Format statistics message
        stats_text = "📊 <b>Статистика бота</b>\n\n"
        stats_text += f"👥 Всего пользователей: {stats['total_users']}\n"
        stats_text += f"🔄 Всего запросов: {stats['total_requests']}\n"
        stats_text += f"📅 Активных сегодня: {stats['active_today']}\n\n"

        if top_users:
            stats_text += "🏆 <b>Топ-10 пользователей по запросам:</b>\n"
            for i, user in enumerate(top_users, 1):
                username = f"@{user['username']}" if user.get("username") else f"ID: {user['user_id']}"
                first_seen = user.get("first_seen", "")[:10] if user.get("first_seen") else "N/A"
                last_seen = user.get("last_seen", "")[:10] if user.get("last_seen") else "N/A"
                stats_text += (
                    f"{i}. {username}: {user['requests']} запросов\n"
                    f"   📅 Первый визит: {first_seen}, Последний: {last_seen}\n"
                )

        await message.answer(stats_text, parse_mode="HTML")
        logger.info(f"Owner {user_id} requested statistics")

    except Exception as e:
        logger.error(f"Error getting statistics: {e}", exc_info=True)
        await message.answer("❌ Ошибка при получении статистики.")


@router.message(Command("user"))
async def cmd_user_stats(message: Message) -> None:
    """Handle /user <user_id> command - show detailed stats for specific user (owner only)."""
    config = get_config()
    user_id = message.from_user.id if message.from_user else None

    # Check if user is owner
    if user_id != config.telegram.owner_id:
        await message.answer("❌ Эта команда доступна только владельцу бота.")
        return

    # Parse user_id from message
    text = message.text or ""
    parts = text.split()
    if len(parts) < 2:
        await message.answer(
            "📊 Использование: <code>/user &lt;user_id&gt;</code>\n\n"
            "Пример: <code>/user 123456789</code>",
            parse_mode="HTML"
        )
        return

    try:
        target_user_id = int(parts[1])
        request_count = await get_user_request_count(target_user_id)
        user_meta = await get_user_meta(target_user_id)

        if not user_meta:
            await message.answer(f"❌ Пользователь {target_user_id} не найден в базе.")
            return

        stats_text = f"👤 <b>Статистика пользователя</b>\n\n"
        stats_text += f"🆔 ID: <code>{target_user_id}</code>\n"
        username = user_meta.get('username', 'N/A')
        if username and username != 'N/A':
            stats_text += f"📝 Username: @{username}\n"
        else:
            stats_text += f"📝 Username: N/A\n"
        stats_text += f"🔄 Запросов: {request_count}\n"
        first_seen = user_meta.get('first_seen', 'N/A')
        if first_seen != 'N/A' and len(first_seen) > 19:
            first_seen = first_seen[:19]
        last_seen = user_meta.get('last_seen', 'N/A')
        if last_seen != 'N/A' and len(last_seen) > 19:
            last_seen = last_seen[:19]
        stats_text += f"📅 Первый визит: {first_seen}\n"
        stats_text += f"📅 Последний визит: {last_seen}\n"

        await message.answer(stats_text, parse_mode="HTML")
        logger.info(f"Owner {user_id} requested stats for user {target_user_id}")

    except ValueError:
        await message.answer("❌ Неверный формат user_id. Используйте число.")
    except Exception as e:
        logger.error(f"Error getting user stats: {e}", exc_info=True)
        await message.answer("❌ Ошибка при получении статистики пользователя.")

