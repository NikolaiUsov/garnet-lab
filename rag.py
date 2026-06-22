import os
import pandas as pd
from pathlib import Path
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# CONFIG
BASE_DIR = Path(__file__).resolve().parent
EXCEL_PATH = BASE_DIR / "docs" / "КАТАЛОГ_ред_18.03.26.xlsx"
faiss_index_path = str(BASE_DIR / "faiss_index")
os.makedirs(faiss_index_path, exist_ok=True)

embeddings_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)
print("✅ Модель загружена!")
# Смотрим размерность эмбеддингов модели
test_embedding = embeddings_model.embed_query("проверка размерности")
print(f"📏 Размерность: {len(test_embedding)}")


# ---------- ЗАГРУЗКА И ПРЕОБРАЗОВАНИЕ EXCEL ----------
def excel_row_to_text(row, sheet_name):
    """
    Превращает строку DataFrame в читаемый текст.
    Игнорирует колонки с изображениями и пустые значения.
    """

    parts = []

    # Отдельно обрабатываем колонку "Наименование"
    if "Наименование" in row.index:
        name = row["Наименование"]
        if pd.notna(name) and str(name).strip():
            parts.append(f"Наименование: {str(name).strip()}")

    # Перебираем все колонки, кроме "Наименование" и "Изображение"
    for col in row.index:
        if col in ("Наименование", "Изображение"):
            continue
        val = row[col]
        if pd.notna(val) and str(val).strip():
            # Обрабатываем многострочные данные в ячейке - меняем <br> на пробелы
            text_val = str(val).replace('<br>', ' ').strip()
            text_val = ' '.join(text_val.split())
            parts.append(f"{col}: {text_val}")
    # Склеиваем все фрагменты через точку с пробелом
    return ". ".join(parts) if parts else ""

print(f"📂 Читаем Excel: {EXCEL_PATH}")
if not EXCEL_PATH.exists():
    raise FileNotFoundError(f"Файл не найден: {EXCEL_PATH}")

# Читаем все листы
xl = pd.ExcelFile(EXCEL_PATH)
documents = []

for sheet_name in xl.sheet_names:
    print(f"   Обработка листа: {sheet_name}")
    df = pd.read_excel(xl, sheet_name=sheet_name, header=0, dtype=str)
    # Удаляем полностью пустые строки
    df = df.dropna(how='all')

    for idx, row in df.iterrows():
        # Формируем текст
        text = excel_row_to_text(row, sheet_name)
        if not text:
            continue   # пропускаем пустые строки

        # Создаём Document
        metadata = {
            "source": EXCEL_PATH.name,
            "sheet": sheet_name,
            "row_index": int(idx)
        }
        doc = Document(page_content=text, metadata=metadata)
        documents.append(doc)

print(f"✅ Загружено {len(documents)} документов.")
if not documents:
    raise SystemExit("Нет данных для индексации. Проверьте файлы.")

# Проверяем, существует ли папка с индексом и есть ли в ней файлы
if os.path.exists(faiss_index_path) and os.path.isfile(os.path.join(faiss_index_path, "index.faiss")):
    print("✅ Индекс найден!")
else:
    print(f"\n🆕 Индекс не найден. Создаём новый в '{faiss_index_path}'...")

    # Создание FAISS индекса
    vectorstore = FAISS.from_documents(documents, embeddings_model)
    print(f"📊 Количество векторов: {vectorstore.index.ntotal}")
    vectorstore.save_local(faiss_index_path)
    print("✅ Индекс сохранён и готов к использованию!")