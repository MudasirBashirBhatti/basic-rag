import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

loader = PyPDFLoader("resume.pdf") # Initialize the PDF loader with the path to your PDF file
pages = loader.load() # Load the PDF and split it into pages

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50) # Initialize the text splitter with a chunk size of 500 characters and an overlap of 50 characters between chunks
chunks = text_splitter.split_documents(pages) # Split the pages into smaller chunks of text

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2") # Initialize the embedding model using a pre-trained model from Hugging Face

# 3. chroma setup (Ye chunks aur model dono ko use kar ke DB banaye ga)

if os.path.exists("./my_db"):
    vector_db = Chroma(
        persist_directory="./my_db",
        embedding_function=embedding_model
    )
    print("DB loaded!")
else:
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory="./my_db"
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