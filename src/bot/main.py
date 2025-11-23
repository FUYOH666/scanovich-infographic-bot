"""Main bot file."""

import asyncio
import logging
import sys
from pathlib import Path

from aiogram import Bot, Dispatcher
from aiogram.fsm.storage.redis import RedisStorage
from dotenv import load_dotenv

from src.bot.handlers import gen, photos, start, stats
from src.bot.middleware import ErrorHandlerMiddleware, LimitCheckMiddleware, LoggingMiddleware
from src.bot.storage import create_redis_storage
from src.bot.states import GenerationStates
from src.config import get_config

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s · %(levelname)s · %(name)s · %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


async def main() -> None:
    """Main entry point."""
    try:
        # Load configuration
        config = get_config()

        # Set logging level from config
        log_level = getattr(logging, config.logging.level.upper(), logging.INFO)
        logging.getLogger().setLevel(log_level)

        logger.info("Starting Scanovich Content Bot...")

        # Initialize bot and dispatcher
        bot = Bot(token=config.telegram.bot_token)
        storage = create_redis_storage()
        dp = Dispatcher(storage=storage)

        # Register middleware (order matters!)
        dp.message.middleware(LoggingMiddleware())
        dp.message.middleware(LimitCheckMiddleware())  # Check limits before processing
        dp.message.middleware(ErrorHandlerMiddleware())  # Error handling last

        # Register routers
        dp.include_router(start.router)
        dp.include_router(stats.router)  # Stats command for owner
        dp.include_router(gen.router)
        dp.include_router(photos.router)

        logger.info("Bot initialized successfully")

        # Start polling
        await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())

    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        logger.info("Shutting down...")


if __name__ == "__main__":
    asyncio.run(main())

