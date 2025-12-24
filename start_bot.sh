#!/bin/bash
# Скрипт запуска Telegram бота Scanovich Content

# Переход в директорию скрипта
cd "$(dirname "$0")"

# Активация виртуального окружения
source .venv/bin/activate

# Запуск бота
PYTHONPATH=. python -m src.bot.main
