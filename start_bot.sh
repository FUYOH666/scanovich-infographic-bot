#!/bin/bash
# Скрипт запуска Telegram бота Scanovich Content

cd /Users/aleksandrmordvinov/development/Scanovich-Content
source .venv/bin/activate
PYTHONPATH=. python -m src.bot.main

