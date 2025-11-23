"""ASR client for audio transcription."""

import logging
from pathlib import Path
from typing import Any

import httpx

from src.config import ASRConfig, get_config

logger = logging.getLogger(__name__)


class ASRClient:
    """Client for ASR API."""

    def __init__(self, config: ASRConfig | None = None):
        """Initialize ASR client.

        Args:
            config: ASR configuration. If None, uses global config.
        """
        self.config = config or get_config().asr
        self.timeout = httpx.Timeout(self.config.timeout)
        logger.info(f"ASR Client initialized: {self.config.api_url}")

    async def transcribe(self, audio_file_path: str | Path) -> str:
        """Transcribe audio file.

        Args:
            audio_file_path: Path to audio file

        Returns:
            Transcribed text

        Raises:
            FileNotFoundError: If file not found
            ValueError: If format not supported
            httpx.HTTPError: On API error
        """
        audio_path = Path(audio_file_path)
        logger.info(f"ASR: Starting transcription for file: {audio_path}")
        
        if not audio_path.exists():
            logger.error(f"ASR: File not found: {audio_path}")
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        file_size = audio_path.stat().st_size
        logger.info(f"ASR: File size: {file_size} bytes")

        # Check format
        file_ext = audio_path.suffix.lstrip(".").lower()
        logger.info(f"ASR: File extension: {file_ext}")
        
        if file_ext not in self.config.supported_formats:
            logger.error(f"ASR: Unsupported format: {file_ext}. Supported: {self.config.supported_formats}")
            raise ValueError(
                f"Unsupported format: {file_ext}. "
                f"Supported: {', '.join(self.config.supported_formats)}"
            )

        # Read file
        logger.info(f"ASR: Reading file data...")
        with audio_path.open("rb") as f:
            audio_data = f.read()
        logger.info(f"ASR: Read {len(audio_data)} bytes from file")

        # Send request
        logger.info(f"ASR: Sending request to {self.config.api_url}...")
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                files = {"file": (audio_path.name, audio_data, f"audio/{file_ext}")}
                logger.info(f"ASR: Uploading file: {audio_path.name}, size: {len(audio_data)}, mime: audio/{file_ext}")
                response = await client.post(self.config.api_url, files=files)
                logger.info(f"ASR: Response status: {response.status_code}")
                
                response.raise_for_status()

                # Parse response
                result: dict[str, Any] = response.json()
                logger.info(f"ASR: Response JSON keys: {list(result.keys())}")
                logger.info(f"ASR: Full response: {result}")
                
                transcript = (
                    result.get("text")
                    or result.get("transcript")
                    or result.get("transcription", "")
                )

                if not transcript:
                    logger.warning(f"ASR: Empty transcript in API response. Full response: {result}")
                    return ""

                logger.info(f"ASR: Transcription successful: {len(transcript)} characters. Text: {transcript[:100]}...")
                return transcript

            except httpx.TimeoutException as e:
                logger.error(f"ASR: Timeout during transcription: {e}", exc_info=True)
                raise
            except httpx.HTTPStatusError as e:
                error_text = e.response.text if e.response else "No response text"
                logger.error(
                    f"ASR: HTTP error during transcription: {e.response.status_code} - {error_text}",
                    exc_info=True
                )
                raise
            except Exception as e:
                logger.error(f"ASR: Error during transcription: {e}", exc_info=True)
                raise

    async def health_check(self) -> bool:
        """Check ASR API availability.

        Returns:
            True if service is available, False otherwise
        """
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.get(self.config.health_url, timeout=5)
                return response.status_code < 500
        except Exception as e:
            logger.warning(f"ASR health check failed: {e}")
            return False


# Singleton instance
_asr_client: ASRClient | None = None


def get_asr_client() -> ASRClient:
    """Get ASR client singleton instance."""
    global _asr_client
    if _asr_client is None:
        _asr_client = ASRClient()
    return _asr_client

