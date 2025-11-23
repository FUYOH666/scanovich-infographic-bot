# Multi-stage Dockerfile for Scanovich Content Bot
# Используем uv для управления зависимостями (как указано в правилах проекта)

FROM python:3.12-slim as builder

# Установка системных зависимостей для сборки
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Установка uv
RUN pip install --no-cache-dir uv

# Рабочая директория
WORKDIR /app

# Копирование файлов зависимостей
COPY pyproject.toml uv.lock ./

# Установка зависимостей в виртуальное окружение
# Используем --frozen для детерминированной сборки из uv.lock
RUN uv sync --frozen --no-dev --no-editable

# Финальный образ
FROM python:3.12-slim

# Создание непривилегированного пользователя
RUN groupadd -r botuser && useradd -r -g botuser botuser

# Установка только runtime зависимостей
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Копирование виртуального окружения из builder
COPY --from=builder /app/.venv /app/.venv

# Копирование кода приложения
WORKDIR /app
COPY src/ ./src/
COPY config.yaml ./
COPY .env.example ./.env.example

# Установка прав доступа
RUN chown -R botuser:botuser /app

# Переключение на непривилегированного пользователя
USER botuser

# Добавление .venv в PATH
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import sys; sys.exit(0)" || exit 1

# Точка входа
CMD ["python", "-m", "src.bot.main"]

