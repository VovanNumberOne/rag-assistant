"""
Сервисный слой для RAG-ассистента.

Этот модуль использует Telegram-бот или любой другой интерфейс.
Он скрывает детали работы RAGPipeline и предоставляет удобный API.
"""

import time
from typing import Any, Dict, List, Optional

from core import config
from rag_pipeline import RAGPipeline


class RAGService:
    """
    Сервис для работы с RAG-pipeline.
    """

    def __init__(self):
        """
        Инициализация RAG-сервиса.
        Использует настройки из core/config.py.
        """
        self.pipeline = RAGPipeline(
            collection_name=config.COLLECTION_NAME,
            cache_db_path=str(config.CACHE_DB_PATH),
            data_file=str(config.DATA_DIR),
            model=config.MODEL_NAME
        )

    def answer_question(
        self,
        question: str,
        history: Optional[List[Dict[str, str]]] = None,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Ответить на вопрос пользователя через RAG-pipeline.

        Args:
            question: вопрос пользователя
            history: история диалога
            use_cache: использовать ли кеш

        Returns:
            словарь с ответом и метаданными
        """
        question = (question or "").strip()

        if not question:
            raise ValueError("Пустой вопрос")

        history = history or []

        start_time = time.time()

        result = self.pipeline.query(
            user_query=question,
            history=history,
            use_cache=use_cache
        )

        duration = time.time() - start_time

        result = self._normalize_context_docs(result)
        result["duration_sec"] = round(duration, 2)

        return result

    def get_stats(self) -> Dict[str, Any]:
        """
        Получить статистику системы.

        Returns:
            словарь со статистикой RAG-системы
        """
        return self.pipeline.get_stats()

    def clear_cache(self) -> None:
        """
        Очистить кеш ответов.
        """
        self.pipeline.cache.clear()

    @staticmethod
    def _normalize_context_docs(result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Приводит контекст к единому формату.

        В текущем RAGPipeline ответ из кеша может содержать контекст
        как список строк, а ответ из API - как список словарей.

        Этот метод приводит всё к виду:
        [
            {"text": "..."},
            {"text": "..."}
        ]
        """
        context_docs = result.get("context_docs") or []
        normalized_docs = []

        for doc in context_docs:
            if isinstance(doc, str):
                normalized_docs.append({"text": doc})
            elif isinstance(doc, dict):
                normalized_docs.append(doc)

        result["context_docs"] = normalized_docs
        return result


if __name__ == "__main__":
    print("=" * 70)
    print("ТЕСТ RAGService")
    print("=" * 70)

    print("\nИнициализация RAGService...")
    service = RAGService()

    print("\nСтатистика системы:")
    stats = service.get_stats()
    print(stats)

    test_question = "Что такое RAG?"

    print("\n" + "=" * 70)
    print(f"Тестовый вопрос: {test_question}")
    print("=" * 70)

    try:
        result = service.answer_question(
            question=test_question,
            history=[],
            use_cache=True
        )

        print("\nРезультат:")
        print(f"Источник: {'Кеш' if result.get('from_cache') else 'OpenAI API'}")
        print(f"Время ответа: {result.get('duration_sec')} сек.")
        print(f"Количество документов контекста: {len(result.get('context_docs', []))}")

        print("\nОтвет:")
        print(result["answer"][:500])

    except Exception as e:
        print(f"\nОшибка при тестовом запросе: {e}")