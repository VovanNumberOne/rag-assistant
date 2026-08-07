"""
Модуль работы с векторным хранилищем ChromaDB.
Обрабатывает загрузку документов, chunking и поиск по векторам.
"""

import chromadb
from typing import List, Dict, Any
import os
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path

import logging

logger = logging.getLogger(__name__)

load_dotenv()


class VectorStore:
    """Векторное хранилище на основе ChromaDB."""
    
    def __init__(self, collection_name: str = "rag_collection", persist_directory: str = "./chroma_db"):
        """
        Инициализация векторного хранилища.
        
        Args:
            collection_name: имя коллекции в ChromaDB
            persist_directory: директория для хранения данных
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        
        # Инициализация ChromaDB клиента
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Получение или создание коллекции
        try:
            self.collection = self.client.get_collection(name=collection_name)
            logger.info(f"Коллекция '{collection_name}' загружена. Документов: {self.collection.count()}")
        except Exception:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Создана новая коллекция '{collection_name}'")
        
        # OpenAI клиент для создания embeddings
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def _chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
        """
        Умное разбиение текста на чанки с учётом семантики.
        
        Стратегия:
        1. Приоритет абзацам (разделение по \n\n)
        2. Разбиение длинных абзацев по предложениям
        3. Сохранение контекста через overlap
        4. Учёт минимального и максимального размера чанка
        
        Args:
            text: исходный текст
            chunk_size: целевой размер чанка в символах
            overlap: размер перекрытия между чанками
            
        Returns:
            список чанков
        """
        # Разделяем текст на абзацы
        paragraphs = text.split('\n\n')
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # Если абзац помещается в текущий чанк
            if len(current_chunk) + len(paragraph) + 2 <= chunk_size:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
            
            # Если текущий чанк не пустой и добавление абзаца превысит размер
            elif current_chunk:
                chunks.append(current_chunk)
                # Добавляем overlap из конца предыдущего чанка
                overlap_text = self._get_overlap_text(current_chunk, overlap)
                current_chunk = overlap_text + "\n\n" + paragraph if overlap_text else paragraph
            
            # Если абзац слишком большой, разбиваем его на предложения
            else:
                if len(paragraph) > chunk_size:
                    # Разбиваем длинный абзац на предложения
                    sentence_chunks = self._split_long_paragraph(paragraph, chunk_size, overlap)
                    
                    # Добавляем все чанки кроме последнего
                    if sentence_chunks:
                        chunks.extend(sentence_chunks[:-1])
                        current_chunk = sentence_chunks[-1]
                else:
                    current_chunk = paragraph
        
        # Добавляем последний чанк
        if current_chunk:
            chunks.append(current_chunk)
        
        # Пост-обработка: фильтруем слишком короткие чанки
        chunks = [chunk for chunk in chunks if len(chunk) >= 50]
        
        return chunks
    
    def _get_overlap_text(self, text: str, overlap_size: int) -> str:
        """
        Получение текста для overlap из конца предыдущего чанка.
        Пытается взять целые предложения.
        
        Args:
            text: текст для извлечения overlap
            overlap_size: желаемый размер overlap
            
        Returns:
            текст overlap
        """
        if len(text) <= overlap_size:
            return text
        
        # Берём последние overlap_size символов
        overlap_candidate = text[-overlap_size:]
        
        # Ищем начало предложения в overlap
        sentence_starts = ['. ', '! ', '? ', '\n']
        best_start = 0
        
        for delimiter in sentence_starts:
            pos = overlap_candidate.find(delimiter)
            if pos != -1 and pos > best_start:
                best_start = pos + len(delimiter)
        
        if best_start > 0:
            return overlap_candidate[best_start:].strip()
        
        return overlap_candidate.strip()
    
    def _split_long_paragraph(self, paragraph: str, chunk_size: int, overlap: int) -> List[str]:
        """
        Разбиение длинного абзаца на чанки по предложениям.
        
        Args:
            paragraph: абзац для разбиения
            chunk_size: целевой размер чанка
            overlap: размер перекрытия
            
        Returns:
            список чанков
        """
        # Разделяем на предложения
        import re
        sentences = re.split(r'([.!?]+\s+)', paragraph)
        
        # Собираем предложения обратно с их разделителями
        full_sentences = []
        for i in range(0, len(sentences) - 1, 2):
            if i + 1 < len(sentences):
                full_sentences.append(sentences[i] + sentences[i + 1])
            else:
                full_sentences.append(sentences[i])
        
        # Если осталось что-то в конце без разделителя
        if len(sentences) % 2 == 1:
            full_sentences.append(sentences[-1])
        
        chunks = []
        current_chunk = ""
        
        for sentence in full_sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Если предложение помещается в текущий чанк
            if len(current_chunk) + len(sentence) + 1 <= chunk_size:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
            else:
                # Сохраняем текущий чанк
                if current_chunk:
                    chunks.append(current_chunk)
                    # Добавляем overlap
                    overlap_text = self._get_overlap_text(current_chunk, overlap)
                    current_chunk = overlap_text + " " + sentence if overlap_text else sentence
                else:
                    # Если одно предложение больше chunk_size, всё равно добавляем его
                    current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def load_documents(self, path: str):
        """
        Загрузка документов из файла или папки.

        Если path - файл, загружает один файл.
        Если path - папка, загружает все поддерживаемые файлы из неё.

        Args:
            path: путь к файлу или папке с документами
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Путь {path} не найден")

        # Если документы уже загружены, не дублируем их
        if self.collection.count() > 0:
            logger.info("Документы уже загружены в коллекцию")
            return

        if path.is_dir():
            self._load_documents_from_dir(path)
        else:
            self._load_single_document(path)

    def _load_single_document(self, file_path: Path):
        """
        Загрузка одного текстового файла.

        Args:
            file_path: путь к файлу
        """
        logger.info(f"Чтение файла {file_path.name}...")

        text = file_path.read_text(encoding="utf-8")
        chunks = self._chunk_text(text)

        logger.info(f"Файл {file_path.name} разбит на {len(chunks)} чанков")

        self._add_chunks(
            chunks,
            source_name=file_path.name
        )


    def _load_documents_from_dir(self, dir_path: Path):
        """
        Загрузка всех поддерживаемых файлов из папки.

        Сейчас поддерживаются:
        - .txt
        - .md

        Args:
            dir_path: путь к папке с документами
        """
        extensions = (".txt", ".md")

        files = sorted(
            file_path
            for file_path in dir_path.iterdir()
            if file_path.is_file() and file_path.suffix.lower() in extensions
        )

        if not files:
            raise FileNotFoundError(
                f"В папке {dir_path} не найдены файлы форматов {', '.join(extensions)}"
            )

        all_chunks = []
        metadatas = []

        for file_path in files:
            logger.info(f"Обработка файла {file_path.name}...")

            try:
                text = file_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                logger.info(f"[!] Пропускаем файл {file_path.name}: не удалось прочитать как UTF-8")
                continue

            chunks = self._chunk_text(text)

            logger.info(f"  [+] Файл {file_path.name}: {len(chunks)} чанков")

            all_chunks.extend(chunks)
            metadatas.extend([{"source": file_path.name} for _ in chunks])

        if not all_chunks:
            raise ValueError(f"Не удалось получить текст из файлов в папке {dir_path}")

        self._add_chunks(
            all_chunks,
            source_name=dir_path.name,
            metadatas=metadatas
        )


    def _add_chunks(self, chunks: List[str], source_name: str, metadatas: List[Dict[str, Any]] = None):
        """
        Добавление чанков в ChromaDB.

        Args:
            chunks: список текстовых чанков
            source_name: имя источника для метаданных
            metadatas: список метаданных для чанков
        """
        if not chunks:
            logger.info("Нет чанков для загрузки")
            return

        if metadatas is None:
            metadatas = [{"source": source_name} for _ in chunks]

        documents = []
        ids = []
        embeddings = []

        for i, chunk in enumerate(chunks):
            embedding = self._create_embedding(chunk)

            documents.append(chunk)
            ids.append(f"doc_{i}")
            embeddings.append(embedding)

            if (i + 1) % 10 == 0:
                logger.info(f"Обработано {i + 1}/{len(chunks)} чанков")

        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas
        )

        logger.info(f"Загружено {len(chunks)} документов в коллекцию '{self.collection_name}'")
    
    def _create_embedding(self, text: str) -> List[float]:
        """
        Создание векторного представления текста через OpenAI.
        
        Args:
            text: текст для векторизации
            
        Returns:
            вектор embeddings
        """
        response = self.openai_client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding
    
    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Поиск релевантных документов по запросу.
        
        Args:
            query: текст запроса
            top_k: количество документов для возврата
            
        Returns:
            список документов с метаданными
        """
        # Создание embedding для запроса
        query_embedding = self._create_embedding(query)
        
        # Поиск в ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        # Форматирование результатов
        documents = []
        if results['documents'] and len(results['documents']) > 0:
            for i in range(len(results['documents'][0])):
                documents.append({
                    'id': results['ids'][0][i],
                    'text': results['documents'][0][i],
                    'distance': results['distances'][0][i] if 'distances' in results else None
                })
        
        return documents
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Получение статистики коллекции.
        
        Returns:
            словарь со статистикой
        """
        return {
            'name': self.collection_name,
            'count': self.collection.count(),
            'persist_directory': self.persist_directory
        }


if __name__ == "__main__":
    # Тестирование векторного хранилища
    import sys
    
    if not os.getenv("OPENAI_API_KEY"):
        logger.info("Ошибка: установите переменную окружения OPENAI_API_KEY")
        sys.exit(1)
    
    vector_store = VectorStore(collection_name="test_collection")
    
    # Загрузка документов
    if os.path.exists("data"):
        vector_store.load_documents("data")
    
    # Поиск
    results = vector_store.search("Что такое машинное обучение?", top_k=3)
    logger.info("\nРезультаты поиска:")
    for i, doc in enumerate(results, 1):
        logger.info(f"\n{i}. {doc['text'][:200]}...")
        logger.info(f"   Distance: {doc['distance']}")
    
    # Статистика
    stats = vector_store.get_collection_stats()
    logger.info(f"\nСтатистика: {stats}")

