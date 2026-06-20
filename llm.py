from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers.document_compressors import DocumentCompressorPipeline, CrossEncoderReranker, EmbeddingsFilter
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_classic.retrievers.document_compressors import EmbeddingsFilter
# from langchain.agents import create_agent
# from langfuse.callback import CallbackHandler
# from langfuse.decorators import observe, langfuse_context
from docx import Document
import camelot
import pandas as pd
import os
import re
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Dict, Any
import json
import sys
import argparse


# ---------------------CONFIG---------------------
BASE_DIR = Path(__file__).resolve().parent  
faiss_index_path = str(BASE_DIR / "faiss_index")
SELECTED_MODEL = "openai/gpt-oss-20b:free"               # openai/gpt-4o-mini   openrouter/free
openai_api_base="https://openrouter.ai/api/v1"   
temperature=0.5
max_tokens=1024

# Инициализация модели эмбеддингов
embeddings_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={'device': "cpu"},
    encode_kwargs={'normalize_embeddings': True}
)
# Langchain-обёртку для CrossEncoderReranker
hf_cross_encoder = HuggingFaceCrossEncoder(
    model_name="BAAI/bge-reranker-v2-m3"            # cross-encoder/ms-marco-MiniLM-L-6-v2  BAAI/bge-reranker-v2-m3      jinaai/jina-reranker-v2-base-multilingual
)
#-------------------------------------------------

# Загружаем переменные
load_dotenv()

# Получаем API‑ключ
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise ValueError("OPENROUTER_API_KEY не найден в переменных окружения")


# Загружаем векторную БД
index_dir = Path(faiss_index_path)
if not index_dir.exists() or not (index_dir / "index.faiss").is_file():
    raise SystemExit(
        f"Индекс FAISS не найден в '{faiss_index_path}'. "
        f"Запустите rag.py, чтобы создать его."
    )
vectorstore = FAISS.load_local(
    faiss_index_path,
    embeddings_model,
    allow_dangerous_deserialization=True,
)
print(f" Успешно подключено к базе. Количество векторов: {vectorstore.index.ntotal}")

# Создаем LLM через OpenRouter 
llm = ChatOpenAI(
        openai_api_key=api_key,
        openai_api_base=openai_api_base,
        model=SELECTED_MODEL,
        temperature=temperature,
        max_tokens=max_tokens,
)

# Создадим интерфейс для доступа к векторному хранилищу
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 15})

# Embeddings filter для быстрой фильтрации
embeddings_filter = EmbeddingsFilter(
    embeddings=embeddings_model,
    similarity_threshold=0.3
)

# Cross-encoder reranker для точного ранжирования
reranker_compressor = CrossEncoderReranker(
    model=hf_cross_encoder,
    top_n=3
)

# Комбинируем filter + reranker
compressor_pipeline = DocumentCompressorPipeline(
    transformers=[embeddings_filter, reranker_compressor]
)

# Создание интерфейса к БД через пайплайн ContextualCompressionRetriever с цепочкой filter + reranker
retriever = ContextualCompressionRetriever(
    base_compressor=compressor_pipeline,
    base_retriever=base_retriever
)

#---------- Извлечение нужных данных из файлов ----------
# Функция извлекает данные из таблицы в файле .docx
def extract_tables_from_docx(file_path):
    doc = Document(file_path)
    tables = []
    for table in doc.tables:
        table_data = []
        for row in table.rows:
            row_data = []
            for cell in row.cells:
                row_data.append(cell.text.strip())
            table_data.append(row_data)
        tables.append(table_data)
    return tables

# Функция извлекает данные из таблицы в файле .pdf
def extract_tables_from_pdf(file_path):
    table_list = []
    tables = camelot.read_pdf(file_path, pages='all', flavor='lattice')
    for table in tables:
        table_list.append(table.df.values.tolist())
    return table_list

def get_tables_text(file_path: Path):
    ext = file_path.suffix.lower()
    if ext == '.docx':
        return extract_tables_from_docx(file_path)
    elif ext == '.pdf':
        return extract_tables_from_pdf(file_path)
    else:
        raise ValueError("Неподдерживаемый формат")

# Извлекаем полезную инфомрацию
def parse_procurement_document(file_path):
    text = get_tables_text(file_path)
    if not text.strip():
        raise ValueError("В документе не найдено подходящих данных")

    # Промпт для извлечения позиций
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Ты — эксперт по закупочной документации. Проанализируй содержимое данных из документа и извлеки все позиции,
        относящиеся к светотехническому оборудованию (светильники, лампы, прожекторы, светодиодные ленты и т.п.).
        Для каждой позиции укажи:
        - name: точное наименование
        - quantity: количество (число или диапазон)
        - specifications: технические характеристики (мощность, напряжение, цветовая температура, материал, степень защиты и т.д.)

        Верни результат строго в формате JSON по схеме:
        {{
          "items": [
            {{"name": "...", "quantity": "...", "specifications": "..."}},
            ...
          ]
        }}
        Если позиции не найдены, верни {{"items": []}}.
        Не добавляй пояснений, только JSON."""),
        ("human", "{text}")
    ])

    chain = prompt | llm
    response = chain.invoke({"text": text}) 

    # Парсим JSON из ответа
    content = response.content.strip()
    json_match = re.search(r'\{.*\}', content, re.DOTALL)
    if not json_match:
        raise ValueError("Не удалось извлечь JSON из ответа модели")
    try:
        data = json.loads(json_match.group())
        items = data.get("items", [])
    except json.JSONDecodeError as e:
        raise ValueError(f"Ошибка парсинга JSON: {e}")

    return items

# ---------- Поиск соответствий в каталоге ----------
def find_best_match(item: Dict[str, str], retriever, llm):
    """
    Для каждой позиции из файла с запросом ищет наиболее подходящую запись в каталоге.
    Возвращает словарь с результатами.
    """
    query = f"{item['name']} {item['specifications']}".strip()
    if not query:
        query = item['name']

    docs = retriever.invoke(query)
    if not docs:
        return {
            "match_found": False,
            "catalog_name": None,
            "catalog_specs": None,
            "comment": "Нет совпадений в каталоге"
        }

    # Если несколько кандидатов, используем LLM для выбора лучшего
    if len(docs) == 1:
        best_doc = docs[0]
    else:
        candidates = []
        for i, doc in enumerate(docs):
            candidates.append(f"Кандидат {i+1}:\n{doc.page_content}")

        prompt = ChatPromptTemplate.from_messages([
            ("system", """Ты — эксперт по светотехнике. Сравни позицию из документа закупки с предложенными кандидатами из каталога.
            Выбери тот, который наиболее точно соответствует по наименованию и характеристикам, если подходящей позиции нет в каталоге, то найди 
            наиболее близкую и укажи на этот факт в столбце Комментарий.
            Ответь только номером кандидата (например, "1").
            Если ни один не подходит, ответь "0"."""),
            ("human", f"""Позиция из закупки:
Название: {item['name']}
Характеристики: {item['specifications']}

Кандидаты из каталога:
{chr(10).join(candidates)}
""")
        ])
        chain = prompt | llm
        response = chain.invoke({})
        choice_text = response.content.strip()
        try:
            choice = int(choice_text)
        except ValueError:
            choice = 0
        if choice <= 0 or choice > len(docs):
            best_doc = None
        else:
            best_doc = docs[choice - 1]

    if best_doc is None:
        return {
            "match_found": False,
            "catalog_name": None,
            "catalog_specs": None,
            "comment": "Ни один кандидат не подходит"
        }

    # Извлекаем из документа каталога наименование и характеристики (упрощённо)
    content = best_doc.page_content
    catalog_name = content.split('.')[0] if '.' in content else content[:50]
    catalog_specs = content

    return {
        "match_found": True,
        "catalog_name": catalog_name,
        "catalog_specs": catalog_specs,
        "comment": "Найдено соответствие"
    }
