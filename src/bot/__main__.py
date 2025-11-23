"""Entry point for running bot as module."""

from src.bot.main import main

if __name__ == "__main__":
    import asyncio

    asyncio.run(main())

