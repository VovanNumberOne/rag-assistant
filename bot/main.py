"""
Точка запуска Telegram-бота на базе pyTelegramBotAPI (telebot).
"""

import logging
from telebot import TeleBot

from bot.handlers import register_handlers, set_bot_commands
from core import config
from core.rag_service import RAGService


logger = logging.getLogger(__name__)


def setup_logging() -> None:
    """
    Настройка логирования.
    """
    logging.basicConfig(
        level=config.LOG_LEVEL,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler(config.LOG_FILE, encoding="utf-8"),
            logging.StreamHandler()
        ]
    )


def validate_configuration() -> None:
    """
    Проверяет конфигурацию перед запуском бота.
    """
    errors = config.validate_bot()

    if errors:
        for error in errors:
            print(f"❌ {error}")

        raise SystemExit("Конфигурация бота некорректна. Исправьте ошибки и попробуйте снова.")


def main() -> None:
    """
    Главная функция запуска бота.
    """
    setup_logging()

    logger.info("Проверка конфигурации...")
    validate_configuration()

    logger.info("Инициализация RAGService...")
    service = RAGService()

    logger.info("Создание Telegram bot...")
    bot = TeleBot(config.TELEGRAM_BOT_TOKEN)

    set_bot_commands(bot)
    register_handlers(bot, service)

    logger.info("Telegram bot запущен. Ожидание сообщений...")

    try:
        # infinity_polling автоматически переподключается при разрывах связи
        bot.infinity_polling(
            timeout=60,
            long_polling_timeout=60,
            skip_pending=True
        )
    except KeyboardInterrupt:
        logger.info("Бот остановлен пользователем.")
    except Exception as e:
        logger.error(f"Критическая ошибка polling: {e}", exc_info=True)


if __name__ == "__main__":
    main()