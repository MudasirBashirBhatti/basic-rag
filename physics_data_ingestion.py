import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

loader = PyPDFLoader("physics_9.pdf") # Initialize the PDF loader with the path to your PDF file
pages = loader.load() # Load the PDF and split it into pages

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150) # Initialize the text splitter with a chunk size of 1000 characters and an overlap of 150 characters between chunks
chunks = text_splitter.split_documents(pages) # Split the pages into smaller chunks of text

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2") # Initialize the embedding model using a pre-trained model from Hugging Face

# 3. chroma setup (Ye chunks aur model dono ko use kar ke DB banaye ga)

if os.path.exists("./physics_db"):
    vector_db = Chroma(
        persist_directory="./physics_db",
        embedding_function=embedding_model
    )
    print("DB loaded!")
else:
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory="./physics_db"
    )
    vector_db.persist()
    print("DB created!")

print("Vector Database save ho gaya!")

# 4. Search test
query = "What is my name"
docs = vector_db.similarity_search(query, k=2)

print("\n--- Search Result ---")
for doc in docs:
    print(f"\nChunk found: {doc.page_content[:200]}...")