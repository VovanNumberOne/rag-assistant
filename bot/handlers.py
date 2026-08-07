"""
Обработчики Telegram-бота на базе pyTelegramBotAPI (telebot).
"""

import logging
from telebot import TeleBot
from telebot.types import Message, BotCommand

from bot.history import add_exchange, clear_history, get_history
from core.rag_service import RAGService


logger = logging.getLogger(__name__)

# Telegram ограничивает длину одного сообщения.
# Берём запас меньше максимального лимита.
TELEGRAM_MESSAGE_LIMIT = 3500


def split_text(text: str, limit: int = TELEGRAM_MESSAGE_LIMIT) -> list:
    """
    Разбивает длинный текст на части, чтобы Telegram принял сообщение.
    """
    chunks = []

    while len(text) > limit:
        chunks.append(text[:limit])
        text = text[limit:]

    if text:
        chunks.append(text)

    return chunks


def format_stats(stats: dict) -> str:
    """
    Форматирует статистику RAG-системы для отправки в Telegram.
    """
    vector_stats = stats.get("vector_store", {})
    cache_stats = stats.get("cache", {})

    lines = [
        "📊 Статистика системы",
        "",
        "🗄 Векторное хранилище:",
        f"Коллекция: {vector_stats.get('name')}",
        f"Документов: {vector_stats.get('count')}",
        "",
        "💾 Кеш:",
        f"Записей: {cache_stats.get('total_entries')}",
        f"Размер БД: {cache_stats.get('db_size_mb', 0):.2f} MB",
        "",
        f"🤖 Модель: {stats.get('model')}",
        f"🌐 Режим: {stats.get('mode')}",
    ]

    return "\n".join(lines)


def register_handlers(bot: TeleBot, service: RAGService) -> None:
    """
    Регистрирует все обработчики бота.

    Args:
        bot: экземпляр TeleBot
        service: экземпляр RAGService
    """

    @bot.message_handler(commands=["start"])
    def handle_start(message: Message) -> None:
        """Обработка команды /start."""
        text = (
            "👋 Привет! Я RAG-ассистент.\n\n"
            "Я отвечаю на вопросы по загруженной базе знаний.\n\n"
            "Просто отправь мне вопрос текстом.\n\n"
            "Команды:\n"
            "/help - справка\n"
            "/stats - статистика системы\n"
            "/reset - очистить историю диалога\n"
            "/clear_cache - очистить кеш ответов"
        )
        bot.send_message(message.chat.id, text)

    @bot.message_handler(commands=["help"])
    def handle_help(message: Message) -> None:
        """Обработка команды /help."""
        text = (
            "ℹ️ Справка\n\n"
            "Я работаю по принципу RAG:\n"
            "1. Ищу релевантные фрагменты в базе знаний.\n"
            "2. Передаю их в языковую модель.\n"
            "3. Формирую ответ на основе найденного контекста.\n\n"
            "Отправь мне вопрос обычным текстовым сообщением.\n\n"
            "Команды:\n"
            "/start - начать работу\n"
            "/stats - статистика системы\n"
            "/reset - очистить историю диалога\n"
            "/clear_cache - очистить кеш ответов"
        )
        bot.send_message(message.chat.id, text)

    @bot.message_handler(commands=["stats"])
    def handle_stats(message: Message) -> None:
        """Обработка команды /stats."""
        try:
            stats = service.get_stats()
            text = format_stats(stats)
            bot.send_message(message.chat.id, text)
        except Exception as e:
            logger.error(f"Ошибка при получении статистики: {e}", exc_info=True)
            bot.send_message(
                message.chat.id,
                "Не удалось получить статистику системы."
            )

    @bot.message_handler(commands=["reset"])
    def handle_reset(message: Message) -> None:
        """Обработка команды /reset. Очищает историю диалога пользователя."""
        user_id = message.from_user.id
        clear_history(user_id)
        bot.send_message(message.chat.id, "✅ История диалога очищена.")

    @bot.message_handler(commands=["clear_cache"])
    def handle_clear_cache(message: Message) -> None:
        """Обработка команды /clear_cache. Очищает кеш ответов."""
        try:
            service.clear_cache()
            bot.send_message(message.chat.id, "✅ Кеш ответов очищен.")
        except Exception as e:
            logger.error(f"Ошибка при очистке кеша: {e}", exc_info=True)
            bot.send_message(message.chat.id, "Не удалось очистить кеш.")

    @bot.message_handler(content_types=["text"])
    def handle_text(message: Message) -> None:
        """Обработка обычного текстового вопроса."""
        user_id = message.from_user.id
        question = (message.text or "").strip()

        if not question:
            bot.send_message(
                message.chat.id,
                "Пожалуйста, отправьте вопрос текстовым сообщением."
            )
            return

        # Игнорируем команды, которые не распознал telebot
        if question.startswith("/"):
            return

        logger.info(f"user_id={user_id} question={question}")

        history = get_history(user_id)

        try:
            # telebot синхронный, поэтому вызываем RAGService напрямую
            result = service.answer_question(
                question=question,
                history=history,
                use_cache=True
            )

            answer = result.get("answer", "")

            # Сохраняем обмен в историю после успешного ответа
            add_exchange(user_id, question, answer)

            source = "💾 Кеш" if result.get("from_cache") else "🌐 OpenAI API"
            duration = result.get("duration_sec")

            footer = f"\n\n{source}"

            if duration is not None:
                footer += f"\n⏱ Время ответа: {duration} сек."

            full_answer = answer + footer

            for chunk in split_text(full_answer):
                bot.send_message(message.chat.id, chunk)

            logger.info(
                f"user_id={user_id} answer_sent=True "
                f"from_cache={result.get('from_cache')} "
                f"duration={duration}"
            )

        except ValueError as e:
            logger.warning(f"user_id={user_id} validation_error={e}")
            bot.send_message(message.chat.id, str(e))

        except Exception as e:
            logger.error(f"user_id={user_id} error={e}", exc_info=True)
            bot.send_message(
                message.chat.id,
                "❌ Произошла ошибка при обработке вопроса. Попробуйте позже."
            )


def set_bot_commands(bot: TeleBot) -> None:
    """
    Устанавливает меню команд бота в Telegram.
    """
    commands = [
        BotCommand("start", "Начать работу"),
        BotCommand("help", "Справка"),
        BotCommand("stats", "Статистика системы"),
        BotCommand("reset", "Очистить историю диалога"),
        BotCommand("clear_cache", "Очистить кеш ответов"),
    ]
    bot.set_my_commands(commands)