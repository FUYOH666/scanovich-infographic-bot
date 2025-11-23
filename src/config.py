"""Configuration module using pydantic-settings."""

import logging
from pathlib import Path
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class ASRConfig(BaseSettings):
    """ASR service configuration."""

    host: str = Field(..., description="ASR service host")
    port: int = Field(8001, ge=1, le=65535, description="ASR service port")
    timeout: int = Field(60, ge=1, description="Request timeout in seconds")
    supported_formats: list[str] = Field(
        default=["wav", "mp3", "ogg", "m4a", "flac", "webm"],
        description="Supported audio formats",
    )

    model_config = SettingsConfigDict(env_prefix="ASR_", case_sensitive=False)

    @property
    def api_url(self) -> str:
        """Get full ASR API URL."""
        return f"http://{self.host}:{self.port}/transcribe"

    @property
    def health_url(self) -> str:
        """Get ASR health check URL."""
        return f"http://{self.host}:{self.port}/health"


class VLLMConfig(BaseSettings):
    """VLLM service configuration."""

    host: str = Field(..., description="VLLM service host")
    port: int = Field(8002, ge=1, le=65535, description="VLLM service port")
    model: str = Field(
        "models/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit",
        description="VLLM model name",
    )
    timeout: int = Field(30, ge=1, description="Request timeout in seconds")
    max_tokens: int = Field(2000, ge=1, le=16384, description="Maximum tokens in response")

    model_config = SettingsConfigDict(env_prefix="VLLM_", case_sensitive=False)

    @property
    def base_url(self) -> str:
        """Get VLLM base URL."""
        return f"http://{self.host}:{self.port}/v1"


class GeminiConfig(BaseSettings):
    """Google Gemini API configuration."""

    api_key: str = Field(..., description="Gemini API key")
    model: str = Field(
        "gemini-3-pro-image-preview",
        description="Gemini model name",
    )
    timeout: int = Field(120, ge=1, description="Request timeout in seconds")
    image_size: str = Field("1K", description="Generated image size")

    model_config = SettingsConfigDict(env_prefix="GEMINI_", case_sensitive=False)


class TelegramConfig(BaseSettings):
    """Telegram bot configuration."""

    bot_token: str = Field(..., description="Telegram bot token")
    owner_id: int = Field(8347160745, description="Bot owner user ID")
    owner_username: str = Field("WuWeiBuild", description="Bot owner username")

    model_config = SettingsConfigDict(env_prefix="TELEGRAM_", case_sensitive=False)


class RedisConfig(BaseSettings):
    """Redis configuration."""

    host: str = Field(..., description="Redis host")
    port: int = Field(6380, ge=1, le=65535, description="Redis port")
    db: int = Field(0, ge=0, description="Redis database number")

    model_config = SettingsConfigDict(env_prefix="REDIS_", case_sensitive=False)


class LoggingConfig(BaseSettings):
    """Logging configuration."""

    level: str = Field("INFO", description="Logging level")
    format: str = Field("text", description="Log format (text or json)")

    model_config = SettingsConfigDict(env_prefix="LOG_", case_sensitive=False)


class AppConfig(BaseSettings):
    """Main application configuration."""

    asr: ASRConfig
    vllm: VLLMConfig
    gemini: GeminiConfig
    telegram: TelegramConfig
    redis: RedisConfig
    logging: LoggingConfig

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        # Note: We use custom from_yaml() method instead of YamlConfigSettingsSource
        # so yaml_file and yaml_file_encoding are not needed here
    )

    @classmethod
    def from_yaml(cls, yaml_path: Path | str) -> "AppConfig":
        """Load configuration from YAML file."""
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")

        # Load YAML and create nested configs
        import yaml

        with yaml_path.open("r", encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f) or {}

        # Create nested configs from YAML data
        config_data: dict[str, Any] = {}
        for key, value in yaml_data.items():
            if isinstance(value, dict):
                config_data[key] = value
            else:
                config_data[key] = value

        # Create config instances for nested configs (with env vars support)
        # Pydantic Settings automatically loads from env vars with matching prefix
        if "asr" in config_data:
            config_data["asr"] = ASRConfig(**config_data["asr"])
        if "vllm" in config_data:
            config_data["vllm"] = VLLMConfig(**config_data["vllm"])
        if "gemini" in config_data:
            # GeminiConfig will load api_key from GEMINI_API_KEY env var
            config_data["gemini"] = GeminiConfig(**config_data["gemini"])
        if "telegram" in config_data:
            config_data["telegram"] = TelegramConfig(**config_data["telegram"])
        if "redis" in config_data:
            config_data["redis"] = RedisConfig(**config_data["redis"])
        if "logging" in config_data:
            config_data["logging"] = LoggingConfig(**config_data["logging"])

        return cls(**config_data)



# Global config instance
_config: AppConfig | None = None


def get_config() -> AppConfig:
    """Get application configuration (singleton)."""
    global _config
    if _config is None:
        # Try to load from YAML first, then fallback to env-only
        config_path = Path("config.yaml")
        try:
            if config_path.exists():
                _config = AppConfig.from_yaml(config_path)
            else:
                _config = AppConfig()
        except Exception as e:
            logger.warning(f"Failed to load from YAML, using env only: {e}")
            _config = AppConfig()
        logger.info("Configuration loaded successfully")
    return _config

