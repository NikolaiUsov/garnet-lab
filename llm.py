from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers.document_compressors import DocumentCompressorPipeline, CrossEncoderReranker, EmbeddingsFilter
from docx import Document
import pdfplumber
import pandas as pd
import os
import re
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict
import json
import sys
import argparse


# ---------------------CONFIG---------------------
BASE_DIR = Path(__file__).resolve().parent  
faiss_index_path = str(BASE_DIR / "faiss_index")
SELECTED_MODEL = "openai/gpt-4o-mini"               # openai/gpt-4o-mini   openrouter/free
openai_api_base="https://openrouter.ai/api/v1"   
temperature=0
max_tokens=4096

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
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_tables = page.extract_tables()
            for table in page_tables:
                if table:
                    table_list.append(table)
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
    tables = get_tables_text(file_path)          
    if not tables:                               
        raise ValueError("В документе не найдено подходящих данных")
    # Преобразуем все таблицы в одну строку
    text_lines = []
    for table in tables:
        for row in table:
            row_text = " ".join(cell.strip() for cell in row if cell and cell.strip())
            if row_text:
                text_lines.append(row_text)
    text = "\n".join(text_lines)

    if not text.strip():
        raise ValueError("В документе не найдено подходящих данных")

    # Промпт для извлечения позиций — все фигурные скобки экранированы двойными {{ }}
    prompt = ChatPromptTemplate.from_messages([
    ("system", """Ты — эксперт по закупочной документации в области светотехники. Твоя задача — извлечь все позиции, относящиеся к осветительному оборудованию (светильники, лампы, прожекторы, опоры, кронштейны и т.п.) из текста документа закупки.

**ВАЖНЫЕ ПРАВИЛА:**

1. Для каждой найденной позиции заполни поля:
   - `name` — точное наименование из документа (как в таблице или описании).
   - `quantity` — количество:
        * если указано число — верни число (int).
        * если указан диапазон (например, "10-15") — верни строку с диапазоном.
        * если не указано — `null`.
   - `specifications_raw` — **все** технические характеристики, перечисленные в документе для данной позиции, одной строкой (без изменений, с сохранением единиц измерения).
   - `specs` — структурированные ключевые параметры для сопоставления с каталогом. **Заполняй только те поля, которые явно указаны в документе.** Если параметр не указан или указан неоднозначно — ставь `null`.

2. **Нормализация значений для `specs`:**
   - `power_w` — мощность в ваттах (число). Извлеки из текста, например: "40 Вт", "40W", "мощность 40 Вт" → 40. Если указан диапазон (например, "35–40 Вт") — возьми  минимальное значение (35). 
   - `cct_k` — цветовая температура в кельвинах (число). Извлеки из текста: "4000 K", "4000К", "4000K" → 4000. Если диапазон — бери минимальное (например, "3000-4000" → 3000).
   - `luminous_flux_lm` — световой поток в люменах (число). Аналогично.
   - `dimensions_mm` — габариты в миллиметрах, **строго в формате "ДxШxВ"** или "ДxШ" (для плоских светильников). Приводи к нижнему регистру, без пробелов и без "мм". Например: "600×600 мм" → "600x600", "1200*300*80" → "1200x300x80". Если в документе несколько вариантов размеров — выбери тот, который относится к данной позиции (обычно указан в той же строке). Если неясно — `null`.

3. **Если в документе нет светотехнических позиций** — верни `{{"items": []}}`.

4. **Выходной формат** — строго JSON по схеме:
{{
  "items": [
    {{
      "name": "Светильник потолочный LED 600x600",
      "quantity": 12,
      "specifications_raw": "Светильник светодиодный, 40 Вт, 4000 К, 3600 лм, размер 600x600 мм, IP20",
      "specs": {{
        "dimensions_mm": "600x600",
        "power_w": 40,
        "cct_k": 4000,
        "luminous_flux_lm": 3600
      }}
    }}
  ]
}}

**Важно:** Верни только JSON, без пояснений, без markdown. 
Пример:
{{"items": [{{"name": "Светильник LED", "quantity": 10, "specifications_raw": "40W 4000K", "specs": {{"power_w": 40, "cct_k": 4000, "luminous_flux_lm": null, "dimensions_mm": null}}}}]}}"""),
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
    query = f"{item['name']} {item['specs']}".strip()
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
        candidates_text = "\n".join(candidates)

        # Используем ChatPromptTemplate с переменными
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Ты — эксперт по светотехнике. Сравни позицию из документа закупки с предложенными кандидатами из каталога. Выбери тот, который наиболее точно соответствует по наименованию и характеристикам, если подходящей позиции нет в каталоге, то найди наиболее близкую и укажи на этот факт в столбце Комментарий. Ответь только номером кандидата (например, '1'). Если ни один не подходит, ответь '0'."),
            ("human", "Позиция из закупки:\nНазвание: {name}\nХарактеристики: {specs}\n\nКандидаты из каталога:\n{candidates}")
        ])
        chain = prompt | llm
        # передаём значения явно
        response = chain.invoke({
            "name": item['name'],
            "specs": str(item.get('specs', {})),   # преобразуем словарь в строку
            "candidates": candidates_text
        })
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

    # Извлекаем из документа каталога наименование и характеристики
    content = best_doc.page_content
    catalog_name = content.split('.')[0] if '.' in content else content[:50]
    catalog_specs = content

    return {
        "match_found": True,
        "catalog_name": catalog_name,
        "catalog_specs": catalog_specs,
        "comment": "Найдено соответствие"
    }


# ---------- Основная функция ----------
def process_procurement_file(file_path: Path) -> pd.DataFrame:
    print(f"📄 Обработка файла: {file_path}")
    items = parse_procurement_document(file_path)
    if not items:
        print("⚠️ Позиции не найдены.")
        return pd.DataFrame()

    print(f"🔍 Найдено {len(items)} позиций. Ищем соответствия в каталоге...")
    results = []
    for idx, item in enumerate(items, 1):
        print(f"  Обработка позиции {idx}: {item['name']}")
        match = find_best_match(item, retriever, llm)
        results.append({
            "Название позиции": item['name'],
            "Требуемое кол-во": item['quantity'],
            "Характеристики": item.get("specs") or item.get("specifications_raw", ""),
            "Найденное сопоставление в каталоге (наименование)": match.get("catalog_name"),
            "Найденное сопоставление в каталоге (характеристики)": match.get("catalog_specs"),
            "Комментарий": match.get("comment", "")
        })

    return pd.DataFrame(results)


def pick_file_via_dialog() -> Path | None:
    """Открывает системный диалог выбора файла."""
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    selected = filedialog.askopenfilename(
        title="Выберите файл закупки",
        filetypes=[
            ("Документы Word/PDF", "*.docx *.doc *.pdf"),
            ("Word", "*.docx *.doc"),
            ("PDF", "*.pdf"),
            ("Все файлы", "*.*"),
        ],
    )
    root.destroy()
    return Path(selected) if selected else None


# ... (предыдущий код остаётся без изменений) ...

if __name__ == "__main__":
    import argparse

    # Создаём папку для результатов, если её нет
    results_dir = BASE_DIR / "results"
    results_dir.mkdir(exist_ok=True)

    parser = argparse.ArgumentParser(description="Обработка файла закупки и поиск соответствий в каталоге.")
    parser.add_argument("-f", "--file", type=str, help="Путь к файлу закупки (.docx или .pdf)")
    args = parser.parse_args()

    if args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"❌ Файл не найден: {file_path}")
            sys.exit(1)
    else:
        print("🖱️ Выберите файл закупки в диалоговом окне...")
        file_path = pick_file_via_dialog()
        if file_path is None:
            print("❌ Файл не выбран.")
            sys.exit(0)

    df = process_procurement_file(file_path)
    if df.empty:
        print("⚠️ Нет данных для сохранения.")
        sys.exit(0)

    # Формируем имена файлов
    base_name = file_path.stem
    xlsx_path = results_dir / f"{base_name}_result.xlsx"

    # Сохраняем
    df.to_excel(xlsx_path, index=False)

    print(f"✅ Результаты сохранены: 📊 XLSX: {xlsx_path}")