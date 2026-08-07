"""
Хранилище истории диалога пользователей для Telegram-бота.

Для MVP история хранится в памяти.
В production лучше использовать Redis или PostgreSQL.
"""

from collections import defaultdict
from typing import Dict, List


# Максимальное количество сообщений в истории.
# 6 сообщений = 3 вопроса пользователя + 3 ответа ассистента.
MAX_HISTORY_MESSAGES = 6

# user_id -> список сообщений
user_histories: Dict[int, List[Dict[str, str]]] = defaultdict(list)


def get_history(user_id: int) -> List[Dict[str, str]]:
    """
    Получить историю диалога пользователя.

    Args:
        user_id: Telegram ID пользователя

    Returns:
        список сообщений в формате:
        [
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."}
        ]
    """
    return user_histories[user_id]


def add_exchange(user_id: int, question: str, answer: str) -> None:
    """
    Добавить в историю вопрос пользователя и ответ ассистента.

    Args:
        user_id: Telegram ID пользователя
        question: вопрос пользователя
        answer: ответ ассистента
    """
    user_histories[user_id].append(
        {
            "role": "user",
            "content": question.strip()
        }
    )

    user_histories[user_id].append(
        {
            "role": "assistant",
            "content": answer.strip()
        }
    )

    _trim_history(user_id)


def clear_history(user_id: int) -> None:
    """
    Очистить историю диалога пользователя.

    Args:
        user_id: Telegram ID пользователя
    """
    if user_id in user_histories:
        user_histories.pop(user_id)


def _trim_history(user_id: int) -> None:
    """
    Ограничить историю последними MAX_HISTORY_MESSAGES сообщениями.

    Args:
        user_id: Telegram ID пользователя
    """
    if len(user_histories[user_id]) > MAX_HISTORY_MESSAGES:
        user_histories[user_id] = user_histories[user_id][-MAX_HISTORY_MESSAGES:]