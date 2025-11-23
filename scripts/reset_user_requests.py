#!/usr/bin/env python3
"""Скрипт для обнуления счетчиков запросов всех пользователей."""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.bot.storage import create_redis_storage


async def reset_all_user_requests():
    """Обнулить счетчики запросов для всех пользователей."""
    storage = create_redis_storage()
    redis = storage.redis

    print("🔍 Поиск всех пользователей с запросами...")

    # Find all user request keys
    pattern = "user_requests:*"
    keys_to_delete = []
    async for key in redis.scan_iter(match=pattern):
        keys_to_delete.append(key)
        user_id = key.decode().split(":")[-1] if isinstance(key, bytes) else key.split(":")[-1]
        print(f"  Найден пользователь: {user_id}")

    if not keys_to_delete:
        print("✅ Нет пользователей с запросами для обнуления")
        return

    print(f"\n🗑️  Удаление счетчиков для {len(keys_to_delete)} пользователей...")

    # Delete all user request keys
    if keys_to_delete:
        deleted = await redis.delete(*keys_to_delete)
        print(f"✅ Удалено {deleted} счетчиков запросов")

    # Optionally reset total requests counter
    print("\n❓ Обнулить общий счетчик запросов? (y/n): ", end="")
    reset_total = input().strip().lower()
    if reset_total == "y":
        await redis.set("stats:total_requests", "0")
        print("✅ Общий счетчик запросов обнулен")
    else:
        print("ℹ️  Общий счетчик запросов оставлен без изменений")

    print("\n✅ Готово! Все счетчики запросов пользователей обнулены.")


if __name__ == "__main__":
    asyncio.run(reset_all_user_requests())

