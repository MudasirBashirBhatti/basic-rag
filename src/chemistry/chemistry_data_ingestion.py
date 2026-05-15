import re
from collections import Counter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
import os
from dotenv import load_dotenv

from langchain_core.documents import Document


load_dotenv()

# Load pdf

loader = PyMuPDF4LLMLoader("chemistry-10th.pdf")
pages = loader.load()

# Detect repeating headers

line_counter = Counter()

for page in pages:
    lines = page.page_content.split("\n")

    for line in lines:
        clean = line.strip()

        if clean:
            line_counter[clean] += 1

# Lines repeated many times are likely headers/footers
repeated_lines = {
    line for line, count in line_counter.items()
    if count >= 8
}

# Cleaning function

def clean_text(text):

    # Remove repeated headers/footers
    lines = []

    for line in text.split("\n"):
        stripped = line.strip()

        if stripped in repeated_lines:
            continue

        lines.append(line)

    text = "\n".join(lines)

    # Remove standalone page numbers
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)

    # Remove hyphen line breaks
    text = re.sub(r'-\n', '', text)

    # Fix broken newlines inside sentences
    text = re.sub(r'(?<![.!?:])\n(?!\n)', ' ', text)

    # Remove excessive spaces
    text = re.sub(r'\s+', ' ', text)

    return text.strip()

# Metadata detection

def detect_chapter(text):
    match = re.search(r'Chapter[-\s]*(\d+)', text, re.IGNORECASE)

    if match:
        return match.group(1)

    return None


def detect_section(text):
    match = re.search(r'(\d+\.\d+\s+[A-Za-z].*)', text)

    if match:
        return match.group(1).strip()

    return None


def detect_topic(text):

    # Example heuristic
    topic_keywords = [
        "boiling",
        "evaporation",
        "condensation",
        "internal energy",
        "kinetic theory",
        "diffusion",
    ]

    lower = text.lower()

    for topic in topic_keywords:
        if topic in lower:
            return topic

    return None

# Metadata Propagation

docs = []

current_chapter = None
current_section = None

for page_num, page in enumerate(pages):

    raw_text = page.page_content

    cleaned_text = clean_text(raw_text)

    chapter = detect_chapter(cleaned_text)
    section = detect_section(cleaned_text)

    # Propagate metadata
    if chapter:
        current_chapter = chapter

    if section:
        current_section = section

    metadata = {
        "source": "chemistry-10th.pdf",
        "subject": "chemistry",
        "class": 10,
        "board": "Punjab Board",
        "language": "english",
        "page": page_num + 1,
        "chapter": current_chapter,
        "section": current_section,
        "topic": detect_topic(cleaned_text),
        "content_type": "textbook"
    }

    docs.append(
        Document(
            page_content=cleaned_text,
            metadata=metadata
        )
    )

# Better chunking

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=180,
    separators=[
        "\n\n",
        "\n",
        ". ",
        "? ",
        "! ",
        " "
    ]
)

chunks = splitter.split_documents(docs)

# Remove duplicates based on cleaned content

unique_chunks = []
seen = set()

for chunk in chunks:

    normalized = chunk.page_content.strip().lower()

    if normalized not in seen:
        seen.add(normalized)
        unique_chunks.append(chunk)

# Embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

if os.path.exists("./physics_db"):
    vector_db = Chroma(
        persist_directory="./physics_db",
        embedding_function=embeddings
    )
    print("DB loaded!")
else:
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./physics_db"
    )
    vector_db.persist()
    print("DB created!")

print("Vector Database save ho gaya!")