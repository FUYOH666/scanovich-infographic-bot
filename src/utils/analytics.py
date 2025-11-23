"""Analytics and request tracking utilities."""

import logging
from datetime import datetime, timezone
from typing import Any

from redis.asyncio import Redis

from src.config import get_config

logger = logging.getLogger(__name__)

# Redis key patterns
USER_REQUESTS_KEY = "user_requests:{user_id}"
USER_META_KEY = "user_meta:{user_id}"
STATS_TOTAL_USERS_KEY = "stats:total_users"
STATS_TOTAL_REQUESTS_KEY = "stats:total_requests"
STATS_ACTIVE_TODAY_KEY = "stats:active_today:{date}"

# Free requests limit
FREE_REQUESTS_LIMIT = 10


_redis_client: Redis | None = None


async def get_redis() -> Redis:
    """Get Redis client instance."""
    global _redis_client
    if _redis_client is None:
        from src.bot.storage import create_redis_storage

        storage = create_redis_storage()
        _redis_client = storage.redis
    return _redis_client


async def register_user(user_id: int, username: str | None = None) -> None:
    """Register a new user in analytics.

    Args:
        user_id: Telegram user ID
        username: Telegram username (optional)
    """
    redis = await get_redis()
    now = datetime.now(timezone.utc).isoformat()

    # Check if user already exists
    exists = await redis.exists(USER_META_KEY.format(user_id=user_id))
    if not exists:
        # New user - set metadata
        await redis.hset(
            USER_META_KEY.format(user_id=user_id),
            mapping={
                "first_seen": now,
                "last_seen": now,
                "username": username or "",
                "total_requests": "0",
            },
        )
        # Increment total users counter
        await redis.incr(STATS_TOTAL_USERS_KEY)
        logger.info(f"Registered new user: {user_id} (@{username})")
    else:
        # Update last seen
        await redis.hset(
            USER_META_KEY.format(user_id=user_id),
            "last_seen",
            now,
        )
        if username:
            await redis.hset(
                USER_META_KEY.format(user_id=user_id),
                "username",
                username,
            )


async def increment_user_request(user_id: int) -> int:
    """Increment request counter for user.

    Args:
        user_id: Telegram user ID

    Returns:
        New request count
    """
    redis = await get_redis()
    key = USER_REQUESTS_KEY.format(user_id=user_id)

    # Increment counter
    count = await redis.incr(key)

    # Update user metadata
    now = datetime.now(timezone.utc).isoformat()
    await redis.hset(USER_META_KEY.format(user_id=user_id), "last_seen", now)
    await redis.hincrby(USER_META_KEY.format(user_id=user_id), "total_requests", 1)

    # Increment global stats
    await redis.incr(STATS_TOTAL_REQUESTS_KEY)

    # Increment daily active users
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    await redis.incr(STATS_ACTIVE_TODAY_KEY.format(date=today))

    logger.info(f"User {user_id} request count: {count}")
    return count


async def get_user_request_count(user_id: int) -> int:
    """Get request count for user.

    Args:
        user_id: Telegram user ID

    Returns:
        Number of requests made by user
    """
    redis = await get_redis()
    key = USER_REQUESTS_KEY.format(user_id=user_id)
    count = await redis.get(key)
    return int(count) if count else 0


async def check_user_limit(user_id: int, limit: int = FREE_REQUESTS_LIMIT) -> bool:
    """Check if user has exceeded request limit.

    Args:
        user_id: Telegram user ID
        limit: Maximum allowed requests (default: 10)

    Returns:
        True if user can make requests, False if limit exceeded
    """
    count = await get_user_request_count(user_id)
    return count < limit


async def get_user_meta(user_id: int) -> dict[str, Any]:
    """Get user metadata.

    Args:
        user_id: Telegram user ID

    Returns:
        Dictionary with user metadata
    """
    redis = await get_redis()
    key = USER_META_KEY.format(user_id=user_id)
    meta = await redis.hgetall(key)

    if not meta:
        return {}

    # Decode bytes to strings
    return {k.decode() if isinstance(k, bytes) else k: v.decode() if isinstance(v, bytes) else v for k, v in meta.items()}


async def get_stats() -> dict[str, Any]:
    """Get overall bot statistics.

    Returns:
        Dictionary with statistics
    """
    redis = await get_redis()

    # Get total users
    total_users = await redis.get(STATS_TOTAL_USERS_KEY)
    total_users = int(total_users) if total_users else 0

    # Get total requests
    total_requests = await redis.get(STATS_TOTAL_REQUESTS_KEY)
    total_requests = int(total_requests) if total_requests else 0

    # Get active users today
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    active_today = await redis.get(STATS_ACTIVE_TODAY_KEY.format(date=today))
    active_today = int(active_today) if active_today else 0

    return {
        "total_users": total_users,
        "total_requests": total_requests,
        "active_today": active_today,
    }


async def get_top_users(limit: int = 10) -> list[dict[str, Any]]:
    """Get top users by request count.

    Args:
        limit: Number of top users to return

    Returns:
        List of user dictionaries with request counts
    """
    redis = await get_redis()

    # Get all user request keys
    pattern = USER_REQUESTS_KEY.format(user_id="*")
    keys = []
    async for key in redis.scan_iter(match=pattern):
        keys.append(key)

    # Get counts for all users
    users = []
    for key in keys:
        user_id_str = key.decode().split(":")[-1] if isinstance(key, bytes) else key.split(":")[-1]
        try:
            user_id = int(user_id_str)
            count = await get_user_request_count(user_id)
            meta = await get_user_meta(user_id)
            users.append(
                {
                    "user_id": user_id,
                    "username": meta.get("username", ""),
                    "requests": count,
                    "first_seen": meta.get("first_seen", ""),
                    "last_seen": meta.get("last_seen", ""),
                }
            )
        except ValueError:
            continue

    # Sort by request count descending
    users.sort(key=lambda x: x["requests"], reverse=True)
    return users[:limit]

