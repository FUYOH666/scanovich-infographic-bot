"""Storage configuration for FSM."""

import logging
from typing import Any

from aiogram.fsm.storage.redis import RedisStorage
from redis.asyncio import Redis

from src.config import RedisConfig, get_config

logger = logging.getLogger(__name__)


def create_redis_storage() -> RedisStorage:
    """Create Redis storage for FSM.

    Returns:
        Configured RedisStorage instance
    """
    config = get_config().redis

    redis_client = Redis(
        host=config.host,
        port=config.port,
        db=config.db,
        decode_responses=False,  # aiogram RedisStorage expects bytes
    )

    storage = RedisStorage(redis=redis_client)
    logger.info(f"Redis storage initialized: {config.host}:{config.port}/{config.db}")
    return storage

