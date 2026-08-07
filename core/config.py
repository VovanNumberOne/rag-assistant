"""
Конфигурация проекта RAG-ассистента.

Загружает переменные окружения из .env и предоставляет
единые настройки для бота, RAG-pipeline, логирования и путей.
"""

import os
from pathlib import Path
from dotenv import load_dotenv


# Корень проекта
# Файл находится здесь: project_root/core/config.py
# Поэтому parent.parent даёт project_root
BASE_DIR = Path(__file__).resolve().parent.parent

ENV_PATH = BASE_DIR / ".env"

# Загружаем .env из корня проекта.
# Если файла нет, пытаемся загрузить переменные окружения из текущего окружения.
if ENV_PATH.exists():
    load_dotenv(ENV_PATH, override=False)
else:
    load_dotenv(override=False)


def _resolve_path(env_name: str, default_relative_path: str) -> Path:
    """
    Возвращает абсолютный путь на основе переменной окружения.

    Если путь относительный, он считается относительно корня проекта.
    """
    raw_path = os.getenv(env_name, default_relative_path)
    path = Path(raw_path)

    if path.is_absolute():
        return path

    return (BASE_DIR / path).resolve()


# ------------------------------------------------------------------
# Секреты и токены
# ------------------------------------------------------------------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")


# ------------------------------------------------------------------
# Настройки RAG
# ------------------------------------------------------------------

MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "api_rag_collection")

DATA_DIR = _resolve_path("DATA_DIR", "data")
SUPPORTED_DATA_EXTENSIONS = (".txt", ".md")
CACHE_DB_PATH = _resolve_path("CACHE_DB_PATH", "api_rag_cache.db")
CHROMA_DIR = _resolve_path("CHROMA_DIR", "chroma_db")


# ------------------------------------------------------------------
# Логирование
# ------------------------------------------------------------------

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "bot.log"


# ------------------------------------------------------------------
# Вспомогательные функции
# ------------------------------------------------------------------

def mask_secret(value: str) -> str:
    """
    Маскирует секрет для безопасного вывода в логи.
    """
    if not value:
        return "NOT SET"

    if len(value) <= 8:
        return "*" * len(value)

    return value[:4] + "*" * 8 + value[-4:]


def validate_rag() -> list:
    """
    Проверяет минимальные настройки для работы RAG-системы.
    Возвращает список ошибок. Если список пустой - всё в порядке.
    """
    errors = []

    if not OPENAI_API_KEY:
        errors.append("OPENAI_API_KEY не установлен")

    if not DATA_DIR.exists():
        errors.append(f"Папка базы знаний не найдена: {DATA_DIR}")
    elif not DATA_DIR.is_dir():
        errors.append(f"Путь базы знаний не является папкой: {DATA_DIR}")
    else:
        data_files = [
            file_path
            for file_path in DATA_DIR.iterdir()
            if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_DATA_EXTENSIONS
        ]

        if not data_files:
            errors.append(
                f"В папке {DATA_DIR} не найдены файлы знаний. "
                f"Поддерживаемые расширения: {', '.join(SUPPORTED_DATA_EXTENSIONS)}"
            )

    return errors


def validate_bot() -> list:
    """
    Проверяет настройки для запуска Telegram-бота.
    Возвращает список ошибок. Если список пустой - всё в порядке.
    """
    errors = validate_rag()

    if not TELEGRAM_BOT_TOKEN:
        errors.append("TELEGRAM_BOT_TOKEN не установлен")

    return errors


def get_debug_info() -> dict:
    """
    Возвращает безопасную информацию о конфигурации для отладки.
    Секреты маскируются.
    """
    return {
        "BASE_DIR": str(BASE_DIR),
        "ENV_PATH": str(ENV_PATH),
        "ENV_EXISTS": ENV_PATH.exists(),
        "OPENAI_API_KEY": mask_secret(OPENAI_API_KEY),
        "TELEGRAM_BOT_TOKEN": mask_secret(TELEGRAM_BOT_TOKEN),
        "MODEL_NAME": MODEL_NAME,
        "COLLECTION_NAME": COLLECTION_NAME,
        "DATA_DIR": str(DATA_DIR),
        "DATA_DIR_EXISTS": DATA_DIR.exists(),
        "DATA_DIR_IS_DIR": DATA_DIR.is_dir(),
        "CACHE_DB_PATH": str(CACHE_DB_PATH),
        "CHROMA_DIR": str(CHROMA_DIR),
        "LOG_LEVEL": LOG_LEVEL,
        "LOG_FILE": str(LOG_FILE),
    }


if __name__ == "__main__":
    print("=" * 70)
    print("ПРОВЕРКА КОНФИГУРАЦИИ RAG-АССИСТЕНТА")
    print("=" * 70)

    debug_info = get_debug_info()

    for key, value in debug_info.items():
        print(f"{key}: {value}")

    print()
    print("Ошибки конфигурации для бота:")

    bot_errors = validate_bot()

    if bot_errors:
        for error in bot_errors:
            print(f"- {error}")
    else:
        print("Нет ошибок. Конфигурация готова к запуску бота.")

    print("=" * 70)