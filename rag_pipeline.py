"""
Основной RAG pipeline для API режима.
Управляет потоком: запрос -> кеш -> vector search -> LLM -> ответ -> кеш.
"""

from typing import Dict, Any, List, Optional
import os
from openai import OpenAI

from vector_store import VectorStore
from cache import RAGCache

import logging

logger = logging.getLogger(__name__)

class RAGPipeline:
    """Основной pipeline для RAG системы в API режиме."""
    
    def __init__(
            self,
            collection_name: str = "rag_collection",
            cache_db_path: str = "rag_cache.db",
            data_file: str = "data/docs.txt",
            model: str = "gpt-4o-mini"):
        """
        Инициализация RAG pipeline.
        """
        # Проверка API ключа
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY не установлен")

        self.model = model
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Инициализация компонентов
        logger.info("Инициализация векторного хранилища...")
        self.vector_store = VectorStore(collection_name=collection_name)

        # Загрузка документов, если коллекция пустая
        if self.vector_store.collection.count() == 0:
            logger.info(f"Загрузка документов из {data_file}...")
            self.vector_store.load_documents(data_file)

        logger.info("Инициализация кеша...")
        self.cache = RAGCache(db_path=cache_db_path)
        logger.info("RAG Pipeline инициализирован (API mode)")
        
    def _create_prompt(
        self,
        query: str,
        context_docs: List[Dict[str, Any]],
        history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Создание промпта для LLM с контекстом и историей диалога.
        Args:
            query: вопрос пользователя
            context_docs: релевантные документы из векторного хранилища
            history: история предыдущих сообщений
        Returns:
            сформированный промпт
        """
        # Формирование контекста из документов
        context_parts = []
        for i, doc in enumerate(context_docs, 1):
            context_parts.append(f"Документ {i}:\n{doc['text']}\n")

        context = "\n".join(context_parts)

        # Формирование текста истории диалога
        history_text = ""

        if history:
            history_lines = []

            # Берём только последние 6 сообщений, чтобы не раздувать промпт
            for message in history[-6:]:
                role = message.get("role")
                content = (message.get("content") or "").strip()

                if not content:
                    continue

                if role == "user":
                    history_lines.append(f"Пользователь: {content}")
                elif role == "assistant":
                    history_lines.append(f"Ассистент: {content}")

            if history_lines:
                history_text = (
                    "История предыдущего диалога:\n"
                    + "\n".join(history_lines)
                    + "\n\n"
                )

        # Создание промпта
        prompt = f"""Ты - полезный AI ассистент. Ответь на вопрос пользователя на основе предоставленного контекста.

Истрория диалога: 
{history_text}
Контекст: 
{context}

Вопрос: {query}

Инструкции:
- Учитывай историю диалога, если она передана.
- Отвечай только на основе предоставленного контекста и истории.
- Если в контексте нет информации для ответа, скажи об этом.
- Будь точным и кратким.
- Отвечай на русском языке.

Ответ:"""

        return prompt
    
    def _generate_answer(self, prompt: str) -> str:
        """
        Генерация ответа через OpenAI API.
        
        Args:
            prompt: промпт для модели
            
        Returns:
            сгенерированный ответ
        """
        response = self.openai_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "Ты - полезный AI ассистент, который отвечает на вопросы на основе предоставленного контекста."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,  # Низкая температура для более точных ответов
            max_tokens=500
        )
        
        return response.choices[0].message.content.strip()
    
    def query(self, user_query: str, use_cache: bool = True, history: list = None) -> Dict[str, Any]:
        logger.info(f"Запрос: {user_query}")
        
        # Шаг 1: Проверка кеша
        if use_cache:
            logger.debug("Проверка кеша...")
            cached_result = self.cache.get(user_query)
            if cached_result:
                logger.info("Ответ найден в кеше")
                return {
                    "query": user_query,
                    "answer": cached_result["answer"],
                    "from_cache": True,
                    "context_docs": cached_result.get("context"),
                    "cached_at": cached_result.get("created_at")
                }
            else:
                logger.debug("Ответ не найден в кеше")
                
        # Шаг 2: Поиск релевантных документов
        logger.debug("Поиск релевантных документов через API...")
        context_docs = self.vector_store.search(user_query, top_k=3)
        logger.info(f"Найдено {len(context_docs)} релевантных документов")
        
        # Шаг 3: Формирование промпта
        logger.debug("Формирование промпта...")
        prompt = self._create_prompt(user_query, context_docs, history)
        
        # Шаг 4: Генерация ответа через API
        logger.info(f"Генерация ответа через OpenAI API ({self.model})...")
        answer = self._generate_answer(prompt)
        logger.info("Ответ получен от API")
        
        # Шаг 5: Сохранение в кеш
        if use_cache:
            logger.debug("Сохранение в кеш...")
            context_for_cache = [doc['text'] for doc in context_docs]
            self.cache.set(user_query, answer, context_for_cache)
            logger.debug("Сохранено в кеш")
            
        return {
            "query": user_query,
            "answer": answer,
            "from_cache": False,
            "context_docs": context_docs,
            "model": self.model,
            "mode": "API"
        }    
    def get_stats(self) -> Dict[str, Any]:
        """
        Получение статистики системы.
        
        Returns:
            словарь со статистикой
        """
        return {
            "vector_store": self.vector_store.get_collection_stats(),
            "cache": self.cache.get_stats(),
            "model": self.model,
            "mode": "API"
        }


if __name__ == "__main__":
    # Тестирование RAG pipeline в API режиме
    import sys
    
    try:
        pipeline = RAGPipeline()
        
        # Тестовые запросы
        test_queries = [
            "Что такое машинное обучение?",
            "Что такое RAG?",
            "Как работают трансформеры?"
        ]
        
        for query in test_queries:
            result = pipeline.query(query)
            logger.info(f"\n{'='*60}")
            logger.info(f"Вопрос: {result['query']}")
            logger.info(f"Из кеша: {result['from_cache']}")
            logger.info(f"Ответ: {result['answer']}")
            logger.info(f"{'='*60}\n")
        
        # Повторный запрос (должен быть из кеша)
        logger.info("\n--- Повторный запрос ---")
        result = pipeline.query(test_queries[0])
        logger.info(f"Из кеша: {result['from_cache']}")
        
        # Статистика
        stats = pipeline.get_stats()
        logger.info(f"\nСтатистика системы:")
        logger.info(f"Векторное хранилище: {stats['vector_store']}")
        logger.info(f"Кеш: {stats['cache']}")
        logger.info(f"Режим: {stats['mode']}")
        
    except Exception as e:
        logger.info(f"Ошибка: {e}")
        sys.exit(1)

