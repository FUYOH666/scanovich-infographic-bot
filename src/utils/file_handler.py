"""File handling utilities for Telegram bot."""

import logging
import tempfile
from pathlib import Path
from typing import BinaryIO

from aiogram import Bot
from aiogram.types import Message

logger = logging.getLogger(__name__)


async def download_photo(bot: Bot, message: Message) -> Path | None:
    """Download photo from Telegram message.

    Args:
        bot: Telegram bot instance
        message: Message with photo

    Returns:
        Path to downloaded photo file, or None if error
    """
    if not message.photo:
        return None

    try:
        # Get largest photo
        photo = message.photo[-1]
        file_info = await bot.get_file(photo.file_id)

        # Create temporary file
        temp_file = Path(tempfile.gettempdir()) / f"photo_{photo.file_id}.jpg"

        # Download file
        await bot.download_file(file_info.file_path, temp_file)

        logger.info(f"Photo downloaded: {temp_file}, size: {temp_file.stat().st_size}")
        return temp_file

    except Exception as e:
        logger.error(f"Error downloading photo: {e}", exc_info=True)
        return None


async def download_voice(bot: Bot, message: Message) -> Path | None:
    """Download voice message from Telegram.

    Args:
        bot: Telegram bot instance
        message: Message with voice

    Returns:
        Path to downloaded audio file, or None if error
    """
    if not message.voice:
        return None

    try:
        file_info = await bot.get_file(message.voice.file_id)

        # Determine file extension
        mime_type = message.voice.mime_type or "audio/ogg"
        ext = ".ogg"
        if "mpeg" in mime_type or "mp3" in mime_type:
            ext = ".mp3"
        elif "wav" in mime_type:
            ext = ".wav"
        elif "m4a" in mime_type:
            ext = ".m4a"

        # Create temporary file
        temp_file = Path(tempfile.gettempdir()) / f"voice_{message.voice.file_id}{ext}"

        # Download file
        await bot.download_file(file_info.file_path, temp_file)

        logger.info(f"Voice downloaded: {temp_file}, size: {temp_file.stat().st_size}")
        return temp_file

    except Exception as e:
        logger.error(f"Error downloading voice: {e}", exc_info=True)
        return None


def read_file_bytes(file_path: Path) -> bytes:
    """Read file as bytes.

    Args:
        file_path: Path to file

    Returns:
        File contents as bytes

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    return file_path.read_bytes()


def cleanup_file(file_path: Path) -> None:
    """Delete temporary file.

    Args:
        file_path: Path to file to delete
    """
    try:
        if file_path.exists():
            file_path.unlink()
            logger.debug(f"Cleaned up file: {file_path}")
    except Exception as e:
        logger.warning(f"Error cleaning up file {file_path}: {e}")

